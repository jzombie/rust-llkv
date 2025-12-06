use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, mpsc};

use arrow::compute::concat_batches;
use arrow::array::{ArrayRef, Array, BooleanArray};
use arrow::datatypes::{Field as ArrowField, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{OwnedRow, RowConverter, SortField};
use llkv_aggregate::AggregateStream;
use llkv_column_map::gather::gather_optional_projected_indices_from_batches;
use llkv_column_map::store::Projection;

use llkv_compute::eval::ScalarEvaluator;
use llkv_expr::AggregateCall;
use llkv_expr::expr::{CompareOp, Expr as LlkvExpr, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_expr::{Expr, Filter, InList, Operator, BinaryOp};
use llkv_join::JoinType;
use llkv_plan::logical_planner::{LogicalPlan, LogicalPlanner, ResolvedJoin};
use llkv_plan::physical::table::{ExecutionTable, TableProvider};
use llkv_plan::planner::PhysicalPlanner;
use llkv_plan::plans::JoinPlan;
use llkv_plan::plans::SelectPlan;
use llkv_plan::plans::{AggregateExpr, AggregateFunction};
use llkv_plan::plans::{CompoundOperator, CompoundQuantifier, CompoundSelectPlan};
use llkv_plan::plans::FilterSubquery;
use llkv_plan::schema::{PlanColumn, PlanSchema};
use llkv_result::Error;
use llkv_scan::{RowIdFilter, ScanProjection};
use llkv_storage::pager::Pager;
use llkv_types::FieldId;
use llkv_types::LogicalFieldId;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use simd_r_drive_entry_handle::EntryHandle;

use crate::ExecutorResult;
use crate::types::{ExecutorTable, ExecutorTableProvider};
use llkv_join::vectorized::VectorizedHashJoinStream;

pub type BatchIter = Box<dyn Iterator<Item = ExecutorResult<RecordBatch>> + Send>;
/// Plan-driven SELECT executor bridging planner output to storage.
pub struct QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    logical_planner: LogicalPlanner<P>,
    physical_planner: PhysicalPlanner<P>,
    provider: Arc<dyn ExecutorTableProvider<P>>,
}







impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn ExecutorTableProvider<P>>) -> Self {
        let planner_provider = Arc::new(PlannerTableProvider { inner: provider.clone() });
        Self {
            logical_planner: LogicalPlanner::new(planner_provider),
            physical_planner: PhysicalPlanner::new(),
            provider,
        }
    }

    fn get_scalar(array: &ArrayRef, index: usize) -> ExecutorResult<Literal> {
        if array.is_null(index) {
            return Ok(Literal::Null);
        }
        match array.data_type() {
            arrow::datatypes::DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                Ok(Literal::Boolean(arr.value(index)))
            }
            arrow::datatypes::DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            arrow::datatypes::DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                Ok(Literal::Float64(arr.value(index)))
            }
            arrow::datatypes::DataType::Utf8 => {
                let arr = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                Ok(Literal::String(arr.value(index).to_string()))
            }
             arrow::datatypes::DataType::LargeUtf8 => {
                let arr = array.as_any().downcast_ref::<arrow::array::LargeStringArray>().unwrap();
                Ok(Literal::String(arr.value(index).to_string()))
            }
            arrow::datatypes::DataType::Null => Ok(Literal::Null),
            _ => Err(Error::Internal(format!("Unsupported scalar subquery return type: {:?}", array.data_type()))),
        }
    }

    fn rewrite_scalar_subqueries<F: Clone>(
        expr: &ScalarExpr<F>,
        results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
    ) -> ScalarExpr<F> {
        match expr {
            ScalarExpr::ScalarSubquery(s) => {
                if let Some(lit) = results.get(&s.id) {
                    ScalarExpr::Literal(lit.clone())
                } else {
                    ScalarExpr::ScalarSubquery(s.clone())
                }
            }
            ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
                left: Box::new(Self::rewrite_scalar_subqueries(left, results)),
                op: *op,
                right: Box::new(Self::rewrite_scalar_subqueries(right, results)),
            },
            ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(Self::rewrite_scalar_subqueries(e, results))),
            ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
                expr: Box::new(Self::rewrite_scalar_subqueries(expr, results)),
                negated: *negated,
            },
            ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
                expr: Box::new(Self::rewrite_scalar_subqueries(expr, results)),
                data_type: data_type.clone(),
            },
            ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
                left: Box::new(Self::rewrite_scalar_subqueries(left, results)),
                op: *op,
                right: Box::new(Self::rewrite_scalar_subqueries(right, results)),
            },
            ScalarExpr::Coalesce(exprs) => ScalarExpr::Coalesce(
                exprs.iter().map(|e| Self::rewrite_scalar_subqueries(e, results)).collect()
            ),
            ScalarExpr::Case { operand, branches, else_expr } => ScalarExpr::Case {
                operand: operand.as_ref().map(|o| Box::new(Self::rewrite_scalar_subqueries(o, results))),
                branches: branches.iter().map(|(w, t)| (
                    Self::rewrite_scalar_subqueries(w, results),
                    Self::rewrite_scalar_subqueries(t, results)
                )).collect(),
                else_expr: else_expr.as_ref().map(|e| Box::new(Self::rewrite_scalar_subqueries(e, results))),
            },
            ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
                base: Box::new(Self::rewrite_scalar_subqueries(base, results)),
                field_name: field_name.clone(),
            },
            _ => expr.clone(),
        }
    }

    fn rewrite_expr_subqueries<F: Clone>(
        expr: &Expr<'_, F>,
        results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
    ) -> Expr<'static, F> {
        match expr {
            Expr::And(list) => Expr::And(list.iter().map(|e| Self::rewrite_expr_subqueries(e, results)).collect()),
            Expr::Or(list) => Expr::Or(list.iter().map(|e| Self::rewrite_expr_subqueries(e, results)).collect()),
            Expr::Not(e) => Expr::Not(Box::new(Self::rewrite_expr_subqueries(e, results))),
            Expr::Pred(f) => {
                let op = match &f.op {
                    Operator::Equals(l) => Operator::Equals(l.clone()),
                    Operator::GreaterThan(l) => Operator::GreaterThan(l.clone()),
                    Operator::GreaterThanOrEquals(l) => Operator::GreaterThanOrEquals(l.clone()),
                    Operator::LessThan(l) => Operator::LessThan(l.clone()),
                    Operator::LessThanOrEquals(l) => Operator::LessThanOrEquals(l.clone()),
                    Operator::Range { lower, upper } => Operator::Range { lower: lower.clone(), upper: upper.clone() },
                    Operator::IsNull => Operator::IsNull,
                    Operator::IsNotNull => Operator::IsNotNull,
                    Operator::StartsWith { pattern, case_sensitive } => Operator::StartsWith { pattern: pattern.clone(), case_sensitive: *case_sensitive },
                    Operator::EndsWith { pattern, case_sensitive } => Operator::EndsWith { pattern: pattern.clone(), case_sensitive: *case_sensitive },
                    Operator::Contains { pattern, case_sensitive } => Operator::Contains { pattern: pattern.clone(), case_sensitive: *case_sensitive },
                    Operator::In(list) => {
                        let owned: Vec<Literal> = list.iter().cloned().collect();
                        Operator::In(InList::shared(owned))
                    },
                };
                Expr::Pred(llkv_expr::Filter { field_id: f.field_id.clone(), op })
            },
            Expr::Compare { left, op, right } => Expr::Compare {
                left: Self::rewrite_scalar_subqueries(left, results),
                op: *op,
                right: Self::rewrite_scalar_subqueries(right, results),
            },
            Expr::InList { expr, list, negated } => Expr::InList {
                expr: Self::rewrite_scalar_subqueries(expr, results),
                list: list.iter().map(|e| Self::rewrite_scalar_subqueries(e, results)).collect(),
                negated: *negated,
            },
            Expr::IsNull { expr, negated } => Expr::IsNull {
                expr: Self::rewrite_scalar_subqueries(expr, results),
                negated: *negated,
            },
            Expr::Literal(b) => Expr::Literal(*b),
            Expr::Exists(s) => Expr::Exists(s.clone()),
        }
    }

    pub fn execute_scalar_subqueries(
        &self,
        subqueries: &[llkv_plan::plans::ScalarSubquery],
    ) -> ExecutorResult<FxHashMap<llkv_expr::expr::SubqueryId, Literal>> {
        let mut results = FxHashMap::default();
        for sub in subqueries {
            if !sub.correlated_columns.is_empty() {
                continue;
            }
            let execution = self.execute_select(*sub.plan.clone())?;
            let batches = execution.collect()?;
            
            let mut result_val = None;
            for batch in batches {
                if batch.num_rows() > 0 {
                    if result_val.is_some() || batch.num_rows() > 1 {
                         return Err(Error::Internal("Scalar subquery returned more than one row".into()));
                    }
                    if batch.num_columns() != 1 {
                         return Err(Error::Internal("Scalar subquery returned more than one column".into()));
                    }
                    let col = batch.column(0);
                    let scalar = Self::get_scalar(col, 0)?;
                    result_val = Some(scalar);
                }
            }
            
            let val = result_val.unwrap_or(Literal::Null);
            results.insert(sub.id, val);
        }
        Ok(results)
    }

    fn split_predicate(expr: &Expr<'static, String>) -> (Option<Expr<'static, String>>, Option<Expr<'static, String>>) {
        if !Self::contains_exists(expr) {
            return (Some(expr.clone()), None);
        }
        
        match expr {
            Expr::And(list) => {
                let mut pushable = Vec::new();
                let mut residual = Vec::new();
                for e in list {
                    let (p, r) = Self::split_predicate(e);
                    if let Some(p) = p { pushable.push(p); }
                    if let Some(r) = r { residual.push(r); }
                }
                let p = if pushable.is_empty() { None } else { Some(Expr::And(pushable)) };
                let r = if residual.is_empty() { None } else { Some(Expr::And(residual)) };
                (p, r)
            }
            _ => (None, Some(expr.clone()))
        }
    }

    fn contains_exists(expr: &Expr<'static, String>) -> bool {
        match expr {
            Expr::Exists(_) => true,
            Expr::And(l) | Expr::Or(l) => l.iter().any(Self::contains_exists),
            Expr::Not(e) => Self::contains_exists(e),
            _ => false,
        }
    }

    fn rewrite_expr_placeholders(
        expr: &Expr<'static, String>,
        replacements: &HashMap<String, Literal>,
    ) -> Expr<'static, String> {
        match expr {
            Expr::And(l) => Expr::And(l.iter().map(|e| Self::rewrite_expr_placeholders(e, replacements)).collect()),
            Expr::Or(l) => Expr::Or(l.iter().map(|e| Self::rewrite_expr_placeholders(e, replacements)).collect()),
            Expr::Not(e) => Expr::Not(Box::new(Self::rewrite_expr_placeholders(e, replacements))),
            Expr::Pred(f) => Expr::Pred(f.clone()),
            Expr::Compare { left, op, right } => Expr::Compare {
                left: Self::rewrite_scalar_expr_placeholders(left.clone(), replacements),
                op: *op,
                right: Self::rewrite_scalar_expr_placeholders(right.clone(), replacements),
            },
            Expr::InList { expr, list, negated } => Expr::InList {
                expr: Self::rewrite_scalar_expr_placeholders(expr.clone(), replacements),
                list: list.iter().map(|e| Self::rewrite_scalar_expr_placeholders(e.clone(), replacements)).collect(),
                negated: *negated,
            },
            Expr::IsNull { expr, negated } => Expr::IsNull {
                expr: Self::rewrite_scalar_expr_placeholders(expr.clone(), replacements),
                negated: *negated,
            },
            Expr::Literal(b) => Expr::Literal(*b),
            Expr::Exists(sub) => Expr::Exists(sub.clone()),
        }
    }

    fn rewrite_scalar_expr_placeholders(
        expr: ScalarExpr<String>,
        replacements: &HashMap<String, Literal>,
    ) -> ScalarExpr<String> {
        match expr {
            ScalarExpr::Column(name) => {
                if let Some(lit) = replacements.get(&name) {
                    ScalarExpr::Literal(lit.clone())
                } else {
                    ScalarExpr::Column(name)
                }
            }
            ScalarExpr::Literal(l) => ScalarExpr::Literal(l),
            ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
                left: Box::new(Self::rewrite_scalar_expr_placeholders(*left, replacements)),
                op,
                right: Box::new(Self::rewrite_scalar_expr_placeholders(*right, replacements)),
            },
            ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(Self::rewrite_scalar_expr_placeholders(*e, replacements))),
            ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
                expr: Box::new(Self::rewrite_scalar_expr_placeholders(*expr, replacements)),
                negated,
            },
            ScalarExpr::Aggregate(_) => expr,
            ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
                base: Box::new(Self::rewrite_scalar_expr_placeholders(*base, replacements)),
                field_name,
            },
            ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
                expr: Box::new(Self::rewrite_scalar_expr_placeholders(*expr, replacements)),
                data_type,
            },
            ScalarExpr::ScalarSubquery(_) => expr,
             ScalarExpr::Case { operand, branches, else_expr } => ScalarExpr::Case {
                operand: operand.map(|o| Box::new(Self::rewrite_scalar_expr_placeholders(*o, replacements))),
                branches: branches.into_iter().map(|(w, t)| (
                    Self::rewrite_scalar_expr_placeholders(w, replacements),
                    Self::rewrite_scalar_expr_placeholders(t, replacements)
                )).collect(),
                else_expr: else_expr.map(|e| Box::new(Self::rewrite_scalar_expr_placeholders(*e, replacements))),
            },
            _ => expr,
        }
    }

    fn rewrite_select_plan_placeholders(
        plan: &SelectPlan,
        replacements: &HashMap<String, Literal>,
    ) -> SelectPlan {
        let mut new_plan = plan.clone();
        
        if let Some(filter) = &mut new_plan.filter {
            filter.predicate = Self::rewrite_expr_placeholders(&filter.predicate, replacements);
            for sub in &mut filter.subqueries {
                sub.plan = Box::new(Self::rewrite_select_plan_placeholders(&sub.plan, replacements));
            }
        }
        
        new_plan.projections = new_plan.projections.into_iter().map(|proj| match proj {
            llkv_plan::plans::SelectProjection::Computed { expr, alias } => {
                llkv_plan::plans::SelectProjection::Computed {
                    expr: Self::rewrite_scalar_expr_placeholders(expr, replacements),
                    alias: alias,
                }
            }
            llkv_plan::plans::SelectProjection::Column { name, alias } => {
                if let Some(lit) = replacements.get(&name) {
                    llkv_plan::plans::SelectProjection::Computed {
                        expr: ScalarExpr::Literal(lit.clone()),
                        alias: alias.unwrap_or_else(|| name.clone()),
                    }
                } else {
                    llkv_plan::plans::SelectProjection::Column { name, alias }
                }
            }
            other => other,
        }).collect();
        
        if let Some(having) = &mut new_plan.having {
             *having = Self::rewrite_expr_placeholders(having, replacements);
        }
        
        new_plan
    }

    fn expr_to_scalar_expr(expr: &Expr<'static, String>) -> ScalarExpr<String> {
        match expr {
            Expr::And(list) => {
                if list.is_empty() {
                    return ScalarExpr::Literal(Literal::Boolean(true));
                }
                let mut iter = list.iter();
                let first = Self::expr_to_scalar_expr(iter.next().unwrap());
                iter.fold(first, |acc, e| ScalarExpr::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::And,
                    right: Box::new(Self::expr_to_scalar_expr(e)),
                })
            }
            Expr::Or(list) => {
                if list.is_empty() {
                    return ScalarExpr::Literal(Literal::Boolean(false));
                }
                let mut iter = list.iter();
                let first = Self::expr_to_scalar_expr(iter.next().unwrap());
                iter.fold(first, |acc, e| ScalarExpr::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::Or,
                    right: Box::new(Self::expr_to_scalar_expr(e)),
                })
            }
            Expr::Not(e) => ScalarExpr::Not(Box::new(Self::expr_to_scalar_expr(e))),
            Expr::Pred(f) => {
                let col = ScalarExpr::Column(f.field_id.clone());
                match &f.op {
                    Operator::Equals(l) => ScalarExpr::Compare {
                        left: Box::new(col),
                        op: CompareOp::Eq,
                        right: Box::new(ScalarExpr::Literal(l.clone())),
                    },
                    Operator::GreaterThan(l) => ScalarExpr::Compare {
                        left: Box::new(col),
                        op: CompareOp::Gt,
                        right: Box::new(ScalarExpr::Literal(l.clone())),
                    },
                    Operator::GreaterThanOrEquals(l) => ScalarExpr::Compare {
                        left: Box::new(col),
                        op: CompareOp::GtEq,
                        right: Box::new(ScalarExpr::Literal(l.clone())),
                    },
                    Operator::LessThan(l) => ScalarExpr::Compare {
                        left: Box::new(col),
                        op: CompareOp::Lt,
                        right: Box::new(ScalarExpr::Literal(l.clone())),
                    },
                    Operator::LessThanOrEquals(l) => ScalarExpr::Compare {
                        left: Box::new(col),
                        op: CompareOp::LtEq,
                        right: Box::new(ScalarExpr::Literal(l.clone())),
                    },
                    Operator::IsNull => ScalarExpr::IsNull {
                        expr: Box::new(col),
                        negated: false,
                    },
                    Operator::IsNotNull => ScalarExpr::IsNull {
                        expr: Box::new(col),
                        negated: true,
                    },
                    Operator::In(list) => {
                        if list.is_empty() {
                            return ScalarExpr::Literal(Literal::Boolean(false));
                        }
                        let mut iter = list.iter();
                        let first = ScalarExpr::Compare {
                            left: Box::new(col.clone()),
                            op: CompareOp::Eq,
                            right: Box::new(ScalarExpr::Literal(iter.next().unwrap().clone())),
                        };
                        iter.fold(first, |acc, l| ScalarExpr::Binary {
                            left: Box::new(acc),
                            op: BinaryOp::Or,
                            right: Box::new(ScalarExpr::Compare {
                                left: Box::new(col.clone()),
                                op: CompareOp::Eq,
                                right: Box::new(ScalarExpr::Literal(l.clone())),
                            }),
                        })
                    }
                    _ => ScalarExpr::Literal(Literal::Boolean(true)),
                }
            }
            Expr::Compare { left, op, right } => ScalarExpr::Compare {
                left: Box::new(left.clone()),
                op: *op,
                right: Box::new(right.clone()),
            },
            Expr::InList { expr, list, negated } => {
                if list.is_empty() {
                    return ScalarExpr::Literal(Literal::Boolean(*negated));
                }
                let mut iter = list.iter();
                let first = ScalarExpr::Compare {
                    left: Box::new(expr.clone()),
                    op: CompareOp::Eq,
                    right: Box::new(iter.next().unwrap().clone()),
                };
                let combined = iter.fold(first, |acc, l| ScalarExpr::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::Or,
                    right: Box::new(ScalarExpr::Compare {
                        left: Box::new(expr.clone()),
                        op: CompareOp::Eq,
                        right: Box::new(l.clone()),
                    }),
                });
                if *negated {
                    ScalarExpr::Not(Box::new(combined))
                } else {
                    combined
                }
            }
            Expr::IsNull { expr, negated } => ScalarExpr::IsNull {
                expr: Box::new(expr.clone()),
                negated: *negated,
            },
            Expr::Literal(b) => ScalarExpr::Literal(Literal::Boolean(*b)),
            Expr::Exists(_) => ScalarExpr::Literal(Literal::Boolean(false)),
        }
    }

    fn remap_string_expr_to_indices(
        expr: &ScalarExpr<String>,
        schema: &Schema,
        name_overrides: Option<&FxHashMap<String, usize>>,
    ) -> ExecutorResult<ScalarExpr<(usize, u32)>> {
        match expr {
            ScalarExpr::Column(name) => {
                let lowered = name.to_ascii_lowercase();
                let idx = name_overrides
                    .and_then(|m| m.get(&lowered).copied())
                    .or_else(|| resolve_schema_index(name, schema))
                    .ok_or_else(|| Error::Internal(format!("Column not found: {}", name)))?;
                Ok(ScalarExpr::Column((0, idx as u32)))
            }
            ScalarExpr::Literal(l) => Ok(ScalarExpr::Literal(l.clone())),
            ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
                left: Box::new(Self::remap_string_expr_to_indices(left, schema, name_overrides)?),
                op: *op,
                right: Box::new(Self::remap_string_expr_to_indices(right, schema, name_overrides)?),
            }),
            ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(Self::remap_string_expr_to_indices(e, schema, name_overrides)?))),
            ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
                expr: Box::new(Self::remap_string_expr_to_indices(expr, schema, name_overrides)?),
                negated: *negated,
            }),
            ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
                left: Box::new(Self::remap_string_expr_to_indices(left, schema, name_overrides)?),
                op: *op,
                right: Box::new(Self::remap_string_expr_to_indices(right, schema, name_overrides)?),
            }),
            ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
                expr: Box::new(Self::remap_string_expr_to_indices(expr, schema, name_overrides)?),
                data_type: data_type.clone(),
            }),
            ScalarExpr::Case { operand, branches, else_expr } => Ok(ScalarExpr::Case {
                operand: operand.as_ref().map(|o| Self::remap_string_expr_to_indices(o, schema, name_overrides)).transpose()?.map(Box::new),
                branches: branches.iter().map(|(w, t)| {
                    Ok((
                        Self::remap_string_expr_to_indices(w, schema, name_overrides)?,
                        Self::remap_string_expr_to_indices(t, schema, name_overrides)?
                    ))
                }).collect::<ExecutorResult<Vec<_>>>()?,
                else_expr: else_expr.as_ref().map(|e| Self::remap_string_expr_to_indices(e, schema, name_overrides)).transpose()?.map(Box::new),
            }),
            ScalarExpr::Coalesce(items) => Ok(ScalarExpr::Coalesce(
                items.iter().map(|e| Self::remap_string_expr_to_indices(e, schema, name_overrides)).collect::<ExecutorResult<Vec<_>>>()?
            )),
            ScalarExpr::ScalarSubquery(s) => Ok(ScalarExpr::ScalarSubquery(s.clone())),
            _ => Err(Error::Internal(format!("Unsupported expression for remapping: {:?}", expr))),
        }
    }

    fn evaluate_row_predicate(
        batch: &RecordBatch,
        row_index: usize,
        predicate: &Expr<'static, String>,
        subqueries: &[FilterSubquery],
        provider: &Arc<dyn ExecutorTableProvider<P>>,
    ) -> ExecutorResult<bool> {
        let mut exists_results = FxHashMap::default();
        
        for sub in subqueries {
            let mut replacements = HashMap::new();
            for corr in &sub.correlated_columns {
                let col = batch.column_by_name(&corr.column).ok_or_else(|| Error::Internal(format!("Correlated column not found: {}", corr.column)))?;
                let lit = Self::get_scalar(col, row_index)?;
                replacements.insert(corr.placeholder.clone(), lit);
            }
            
            let sub_plan = Self::rewrite_select_plan_placeholders(&sub.plan, &replacements);
            
            let executor = QueryExecutor::new(provider.clone());
            let execution = executor.execute_select_with_filter(&sub_plan, &FxHashMap::default(), None)?;
            
            let batches = execution.collect()?;
            let mut has_rows = false;
            for b in batches {
                if b.num_rows() > 0 {
                    has_rows = true;
                    break;
                }
            }
            
            exists_results.insert(sub.id, has_rows);
        }
        
        let rewritten_predicate = Self::rewrite_expr_exists(predicate, &exists_results);
        let scalar_expr = Self::expr_to_scalar_expr(&rewritten_predicate);
        let remapped_expr = Self::remap_string_expr_to_indices(&scalar_expr, &batch.schema(), None)?;

        let slice = batch.slice(row_index, 1);
        
        let mut field_arrays = FxHashMap::default();
        for (i, col) in slice.columns().iter().enumerate() {
            field_arrays.insert((0, i as u32), col.clone());
        }
        let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, 1);
        
        let result = ScalarEvaluator::evaluate_batch_simplified(&remapped_expr, 1, &numeric_arrays)?;
        
        if result.is_null(0) {
            Ok(false)
        } else {
            let bool_arr = result.as_any().downcast_ref::<arrow::array::BooleanArray>().ok_or_else(|| Error::Internal("Predicate did not evaluate to boolean".into()))?;
            Ok(bool_arr.value(0))
        }
    }

    fn rewrite_expr_exists(
        expr: &Expr<'static, String>,
        results: &FxHashMap<llkv_expr::expr::SubqueryId, bool>,
    ) -> Expr<'static, String> {
        match expr {
            Expr::Exists(sub) => {
                let val = results.get(&sub.id).copied().unwrap_or(false);
                let val = if sub.negated { !val } else { val };
                Expr::Literal(val)
            }
            Expr::And(l) => Expr::And(l.iter().map(|e| Self::rewrite_expr_exists(e, results)).collect()),
            Expr::Or(l) => Expr::Or(l.iter().map(|e| Self::rewrite_expr_exists(e, results)).collect()),
            Expr::Not(e) => Expr::Not(Box::new(Self::rewrite_expr_exists(e, results))),
            _ => expr.clone(),
        }
    }

    fn normalize_not_isnull(expr: Expr<'static, String>) -> Expr<'static, String> {
        match expr {
            Expr::IsNull { expr, negated } => {
                if let ScalarExpr::Not(inner) = expr {
                    let rewritten = Expr::IsNull {
                        expr: *inner,
                        negated,
                    };
                    Expr::Not(Box::new(Self::normalize_not_isnull(rewritten)))
                } else {
                    Expr::IsNull { expr, negated }
                }
            }
            Expr::Not(inner) => Expr::Not(Box::new(Self::normalize_not_isnull(*inner))),
            Expr::And(list) => Expr::And(list.into_iter().map(Self::normalize_not_isnull).collect()),
            Expr::Or(list) => Expr::Or(list.into_iter().map(Self::normalize_not_isnull).collect()),
            other => other,
        }
    }

    fn apply_residual_filter(
        &self,
        input: BatchIter,
        predicate: Expr<'static, String>,
        subqueries: Vec<FilterSubquery>,
    ) -> BatchIter {
        let provider = self.provider.clone();
        let subqueries = subqueries;
        let predicate = Self::normalize_not_isnull(predicate);
        
        let iter = input.flat_map(move |batch_res| {
            let batch = match batch_res {
                Ok(b) => b,
                Err(e) => return vec![Err(e)].into_iter(),
            };

            let num_rows = batch.num_rows();
            let mut bool_builder = arrow::array::BooleanBuilder::new();
            
            for i in 0..num_rows {
                let matches = Self::evaluate_row_predicate(&batch, i, &predicate, &subqueries, &provider);
                match matches {
                    Ok(b) => bool_builder.append_value(b),
                    Err(e) => return vec![Err(e)].into_iter(),
                }
            }
            
            let bool_array = bool_builder.finish();
            match arrow::compute::filter_record_batch(&batch, &bool_array) {
                Ok(filtered) => vec![Ok(filtered)].into_iter(),
                Err(e) => vec![Err(Error::Internal(e.to_string()))].into_iter(),
            }
        });

        Box::new(iter)
    }

    fn execute_complex_aggregation(
        &self,
        base_stream: BatchIter,
        base_schema: Arc<Schema>,
        new_aggregates: Vec<AggregateExpr>,
        final_exprs: Vec<ScalarExpr<String>>,
        final_names: Vec<String>,
        plan: &SelectPlan,
        table_name: String,
        subquery_results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if !plan.group_by.is_empty() {
            return Err(Error::InvalidArgumentError(
                "GROUP BY aggregates are not supported yet".into(),
            ));
        }

        // 1. Pre-aggregation stream is just base_stream
        // The base_stream is expected to yield batches containing the arguments for aggregation
        // (e.g. _agg_arg_0, _agg_arg_1, etc.) produced by the Scan.
        
        // We need to ensure the schema has the field_ids we expect (10000+i) so AggregateStream can match them.
        let mut fields_with_metadata = Vec::new();
        for (i, field) in base_schema.fields().iter().enumerate() {
            let mut metadata = field.metadata().clone();
            metadata.insert("field_id".to_string(), format!("{}", 10000 + i));
            let new_field = field.as_ref().clone().with_metadata(metadata);
            fields_with_metadata.push(new_field);
        }
        let pre_agg_schema = Arc::new(Schema::new(fields_with_metadata));
        
        let pre_agg_schema_captured = pre_agg_schema.clone();
        let pre_agg_stream = base_stream.map(move |batch_res| {
            let batch = batch_res?;
            // Just replace schema, data is same
            RecordBatch::try_new(pre_agg_schema_captured.clone(), batch.columns().to_vec())
                .map_err(|e| Error::Internal(e.to_string()))
        });

        // 2. Aggregation
        let mut agg_plan = plan.clone();
        agg_plan.aggregates = new_aggregates;
        
        let mut plan_columns = Vec::new();
        let mut name_to_index = FxHashMap::default();
        for (i, field) in pre_agg_schema.fields().iter().enumerate() {
            plan_columns.push(PlanColumn {
                name: field.name().clone(),
                data_type: field.data_type().clone(),
                field_id: 10000 + i as u32,
                is_nullable: field.is_nullable(),
                is_primary_key: false,
                is_unique: false,
                default_value: None,
                check_expr: None,
            });
            name_to_index.insert(field.name().clone(), i);
        }
        let pre_agg_plan_schema = PlanSchema { columns: plan_columns, name_to_index };

        let agg_iter = AggregateStream::new(Box::new(pre_agg_stream), &agg_plan, &pre_agg_plan_schema, &pre_agg_schema)?;
        let agg_schema = agg_iter.schema();

        // 3. Final Projection
        let mut final_output_fields = Vec::new();
        let mut col_mapping = FxHashMap::default();
        for (i, _field) in agg_schema.fields().iter().enumerate() {
            col_mapping.insert((0, i as u32), i);
        }

        for (expr_str, name) in final_exprs.iter().zip(final_names.iter()) {
            let expr = resolve_scalar_expr_string(expr_str, &agg_schema, subquery_results)?;
            let dt = infer_type(&expr, &agg_schema, &col_mapping).unwrap_or(arrow::datatypes::DataType::Int64);
            final_output_fields.push(ArrowField::new(name, dt, true));
        }
        let final_output_schema = Arc::new(Schema::new(final_output_fields));
        let final_output_schema_captured = final_output_schema.clone();
        let agg_schema_captured = agg_schema.clone();
        let subquery_results_captured = subquery_results.clone();

        let final_stream = agg_iter.map(move |batch_res| {
            let batch = batch_res?;
            let mut columns = Vec::new();
            
            // Prepare arrays for evaluation
            let mut field_arrays = FxHashMap::default();
            for (i, col) in batch.columns().iter().enumerate() {
                field_arrays.insert((0, i as u32), col.clone());
            }
            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
            
            for expr_str in &final_exprs {
                let expr = resolve_scalar_expr_string(expr_str, &agg_schema_captured, &subquery_results_captured)?;
                let result = ScalarEvaluator::evaluate_batch_simplified(&expr, batch.num_rows(), &numeric_arrays)?;
                columns.push(result);
            }
            
            RecordBatch::try_new(final_output_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
        });

        let mut iter: BatchIter = Box::new(final_stream);
        if plan.distinct {
             iter = Box::new(DistinctStream::new(final_output_schema.clone(), iter)?);
        }
        
        let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);
        Ok(SelectExecution::from_stream(table_name, final_output_schema, trimmed))
    }

    fn contains_subquery<F: std::fmt::Debug>(expr: &ScalarExpr<F>) -> bool {
        match expr {
            ScalarExpr::ScalarSubquery(_) => true,
            ScalarExpr::Binary { left, right, .. } => Self::contains_subquery(left) || Self::contains_subquery(right),
            ScalarExpr::Not(e) => Self::contains_subquery(e),
            ScalarExpr::IsNull { expr, .. } => Self::contains_subquery(expr),
            ScalarExpr::Cast { expr, .. } => Self::contains_subquery(expr),
            ScalarExpr::Compare { left, right, .. } => Self::contains_subquery(left) || Self::contains_subquery(right),
            ScalarExpr::Case { operand, branches, else_expr } => {
                operand.as_ref().map_or(false, |o| Self::contains_subquery(o)) ||
                branches.iter().any(|(w, t)| Self::contains_subquery(w) || Self::contains_subquery(t)) ||
                else_expr.as_ref().map_or(false, |e| Self::contains_subquery(e))
            }
            ScalarExpr::Coalesce(list) => list.iter().any(Self::contains_subquery),
            ScalarExpr::GetField { base, .. } => Self::contains_subquery(base),
            ScalarExpr::Aggregate(call) => match call {
                AggregateCall::Count { expr, .. } | AggregateCall::Sum { expr, .. } |
                AggregateCall::Total { expr, .. } | AggregateCall::Avg { expr, .. } |
                AggregateCall::Min(expr) | AggregateCall::Max(expr) |
                AggregateCall::CountNulls(expr) | AggregateCall::GroupConcat { expr, .. } => Self::contains_subquery(expr),
                AggregateCall::CountStar => false,
            },
            _ => false,
        }
    }

    fn rewrite_expr_with_row_subqueries(
        &self,
        expr: &ScalarExpr<(usize, u32)>,
        batch: &RecordBatch,
        row_index: usize,
        provider: &Arc<dyn ExecutorTableProvider<P>>,
        scalar_subqueries: &[llkv_plan::plans::ScalarSubquery],
    ) -> ExecutorResult<ScalarExpr<(usize, u32)>> {
        match expr {
            ScalarExpr::ScalarSubquery(sub_expr) => {
                let sub_def = scalar_subqueries.iter().find(|s| s.id == sub_expr.id)
                    .ok_or_else(|| Error::Internal(format!("Scalar subquery definition not found for ID {}", sub_expr.id.0)))?;

                let mut replacements = HashMap::new();
                for corr in &sub_def.correlated_columns {
                    let col = batch.column_by_name(&corr.column).ok_or_else(|| Error::Internal(format!("Correlated column not found: {}", corr.column)))?;
                    let lit = Self::get_scalar(col, row_index)?;
                    replacements.insert(corr.placeholder.clone(), lit);
                }
                
                let sub_plan = Self::rewrite_select_plan_placeholders(&sub_def.plan, &replacements);
                let executor = QueryExecutor::new(provider.clone());
                let execution = executor.execute_select(sub_plan)?;
                let batches = execution.collect()?;
                
                let mut result_val = None;
                for b in batches {
                    if b.num_rows() > 0 {
                        if result_val.is_some() || b.num_rows() > 1 {
                             return Err(Error::Internal("Scalar subquery returned more than one row".into()));
                        }
                        if b.num_columns() != 1 {
                             return Err(Error::Internal("Scalar subquery returned more than one column".into()));
                        }
                        let col = b.column(0);
                        let scalar = Self::get_scalar(col, 0)?;
                        result_val = Some(scalar);
                    }
                }
                Ok(ScalarExpr::Literal(result_val.unwrap_or(Literal::Null)))
            }
            ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
                left: Box::new(self.rewrite_expr_with_row_subqueries(left, batch, row_index, provider, scalar_subqueries)?),
                op: *op,
                right: Box::new(self.rewrite_expr_with_row_subqueries(right, batch, row_index, provider, scalar_subqueries)?),
            }),
            ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(self.rewrite_expr_with_row_subqueries(e, batch, row_index, provider, scalar_subqueries)?))),
            ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
                expr: Box::new(self.rewrite_expr_with_row_subqueries(expr, batch, row_index, provider, scalar_subqueries)?),
                negated: *negated,
            }),
            ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
                expr: Box::new(self.rewrite_expr_with_row_subqueries(expr, batch, row_index, provider, scalar_subqueries)?),
                data_type: data_type.clone(),
            }),
            ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
                left: Box::new(self.rewrite_expr_with_row_subqueries(left, batch, row_index, provider, scalar_subqueries)?),
                op: *op,
                right: Box::new(self.rewrite_expr_with_row_subqueries(right, batch, row_index, provider, scalar_subqueries)?),
            }),
            ScalarExpr::Case { operand, branches, else_expr } => Ok(ScalarExpr::Case {
                operand: operand.as_ref().map(|o| self.rewrite_expr_with_row_subqueries(o, batch, row_index, provider, scalar_subqueries)).transpose()?.map(Box::new),
                branches: branches.iter().map(|(w, t)| {
                    Ok((
                        self.rewrite_expr_with_row_subqueries(w, batch, row_index, provider, scalar_subqueries)?,
                        self.rewrite_expr_with_row_subqueries(t, batch, row_index, provider, scalar_subqueries)?
                    ))
                }).collect::<ExecutorResult<Vec<_>>>()?,
                else_expr: else_expr.as_ref().map(|e| self.rewrite_expr_with_row_subqueries(e, batch, row_index, provider, scalar_subqueries)).transpose()?.map(Box::new),
            }),
            ScalarExpr::Coalesce(list) => Ok(ScalarExpr::Coalesce(
                list.iter().map(|e| self.rewrite_expr_with_row_subqueries(e, batch, row_index, provider, scalar_subqueries)).collect::<ExecutorResult<Vec<_>>>()?
            )),
            ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
                base: Box::new(self.rewrite_expr_with_row_subqueries(base, batch, row_index, provider, scalar_subqueries)?),
                field_name: field_name.clone(),
            }),
            _ => Ok(expr.clone()),
        }
    }

    fn project_stream(
        &self,
        input: BatchIter,
        input_schema: &Schema,
        projections: &[llkv_plan::plans::SelectProjection],
        table_name: String,
        scalar_subqueries: &[llkv_plan::plans::ScalarSubquery],
        distinct: bool,
        order_by: &[llkv_plan::plans::OrderByPlan],
        offset: Option<usize>,
        limit: Option<usize>,
        name_overrides: Option<FxHashMap<String, usize>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let name_overrides = name_overrides.as_ref();
        let mut output_fields = Vec::new();
        let mut exprs = Vec::new();
        
        let mut col_mapping = FxHashMap::default();
        for (i, _field) in input_schema.fields().iter().enumerate() {
            col_mapping.insert((0, i as u32), i);
        }
        
        let mut name_to_idx = FxHashMap::default();
        for (i, field) in input_schema.fields().iter().enumerate() {
            name_to_idx.insert(field.name().clone(), i);
        }
        
        for proj in projections {
            match proj {
                llkv_plan::plans::SelectProjection::AllColumns => {
                    for (i, field) in input_schema.fields().iter().enumerate() {
                        output_fields.push(field.clone());
                        exprs.push(ScalarExpr::Column((0, i as u32)));
                    }
                }
                llkv_plan::plans::SelectProjection::AllColumnsExcept { exclude } => {
                    for (i, field) in input_schema.fields().iter().enumerate() {
                        if !exclude.contains(field.name()) {
                            output_fields.push(field.clone());
                            exprs.push(ScalarExpr::Column((0, i as u32)));
                        }
                    }
                }
                llkv_plan::plans::SelectProjection::Column { name, alias } => {
                    let lowered = name.to_ascii_lowercase();
                    let idx = name_overrides
                        .and_then(|m| m.get(&lowered).copied())
                        .or_else(|| resolve_schema_index(name, input_schema))
                        .ok_or_else(|| Error::Internal(format!("Column not found: {}", name)))?;
                    let field = input_schema.field(idx);
                    let out_name = alias.clone().unwrap_or_else(|| name.clone());
                    output_fields.push(Arc::new(ArrowField::new(out_name, field.data_type().clone(), field.is_nullable())));
                    exprs.push(ScalarExpr::Column((0, idx as u32)));
                }
                llkv_plan::plans::SelectProjection::Computed { expr, alias } => {
                    let remapped = Self::remap_string_expr_to_indices(expr, input_schema, name_overrides)?;
                    let dt = infer_type(&remapped, input_schema, &col_mapping).unwrap_or(arrow::datatypes::DataType::Int64);
                    output_fields.push(Arc::new(ArrowField::new(alias.clone(), dt, true)));
                    exprs.push(remapped);
                }
            }
        }
        
        let output_schema = Arc::new(Schema::new(output_fields));
        let output_schema_captured = output_schema.clone();
        let provider = self.provider.clone();
        let self_clone = QueryExecutor::new(provider.clone());
        let scalar_subqueries_captured = scalar_subqueries.to_vec();
        
        let final_stream = input.map(move |batch_res| {
            let batch = batch_res?;
            let mut columns = Vec::new();
            
            let mut field_arrays = FxHashMap::default();
            for (i, col) in batch.columns().iter().enumerate() {
                field_arrays.insert((0, i as u32), col.clone());
            }
            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
            
            for expr in &exprs {
                if Self::contains_subquery(expr) {
                    let mut result_arrays = Vec::new();
                    for i in 0..batch.num_rows() {
                        let rewritten = self_clone.rewrite_expr_with_row_subqueries(expr, &batch, i, &provider, &scalar_subqueries_captured)?;
                        
                        let slice = batch.slice(i, 1);
                        let mut slice_arrays = FxHashMap::default();
                        for (j, col) in slice.columns().iter().enumerate() {
                            slice_arrays.insert((0, j as u32), col.clone());
                        }
                        let slice_numeric = ScalarEvaluator::prepare_numeric_arrays(&slice_arrays, 1);
                        
                        let res = ScalarEvaluator::evaluate_batch_simplified(&rewritten, 1, &slice_numeric)?;
                        result_arrays.push(res);
                    }
                    
                    let refs: Vec<&dyn Array> = result_arrays.iter().map(|a| a.as_ref()).collect();
                    let concatenated = arrow::compute::concat(&refs).map_err(|e| Error::Internal(e.to_string()))?;
                    columns.push(concatenated);
                } else {
                    let result = ScalarEvaluator::evaluate_batch_simplified(expr, batch.num_rows(), &numeric_arrays)?;
                    columns.push(result);
                }
            }
            
            RecordBatch::try_new(output_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
        });
        
        let mut stream: BatchIter = Box::new(final_stream);

        if distinct {
            stream = Box::new(DistinctStream::new(output_schema.clone(), stream)?);
        }
        
        // Apply sort
        if !order_by.is_empty() {
             let mut sort_columns = Vec::new();
             for ob in order_by {
                 match &ob.target {
                     llkv_plan::plans::OrderTarget::Index(idx) => {
                         if *idx < output_schema.fields().len() {
                             sort_columns.push((*idx, ob.ascending, ob.nulls_first));
                         }
                     }
                     llkv_plan::plans::OrderTarget::Column(name) => {
                         if let Ok(idx) = output_schema.index_of(name) {
                             sort_columns.push((idx, ob.ascending, ob.nulls_first));
                         }
                     }
                     _ => {}
                 }
             }
             
             if !sort_columns.is_empty() {
                 stream = Box::new(SortStream::new(output_schema.clone(), stream, sort_columns));
             }
        }
        
        // Apply offset/limit
        let trimmed = apply_offset_limit_stream(stream, offset, limit);
        
        Ok(SelectExecution::from_stream(table_name, output_schema, trimmed))
    }

    fn execute_compound(
        &self,
        compound: &CompoundSelectPlan,
        subquery_results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let mut current_exec = self.execute_select_with_filter(
            &compound.initial,
            subquery_results,
            row_filter.clone(),
        )?;

        for op in &compound.operations {
            let next_exec = self.execute_select_with_filter(
                &op.plan,
                subquery_results,
                row_filter.clone(),
            )?;

            // Ensure schemas match
            if current_exec.schema().fields().len() != next_exec.schema().fields().len() {
                return Err(Error::Internal(format!(
                    "Compound operator schema mismatch: left has {} columns, right has {}",
                    current_exec.schema().fields().len(),
                    next_exec.schema().fields().len()
                )));
            }

            match op.operator {
                CompoundOperator::Union => {
                    let schema = current_exec.schema();
                    let stream = Box::new(current_exec.into_stream()?.chain(next_exec.into_stream()?));
                    current_exec = SelectExecution::from_stream(
                        "union".to_string(),
                        schema,
                        stream,
                    );

                    if matches!(op.quantifier, CompoundQuantifier::Distinct) {
                        let schema = current_exec.schema();
                        let distinct_stream = Box::new(DistinctStream::new(
                            schema.clone(),
                            current_exec.into_stream()?,
                        )?);
                        current_exec = SelectExecution::from_stream(
                            "union_distinct".to_string(),
                            schema,
                            distinct_stream,
                        );
                    }
                }
                CompoundOperator::Except => {
                    // For EXCEPT, we need to materialize the right side to filter the left side.
                    // EXCEPT ALL is not supported by standard SQL usually (it's EXCEPT), 
                    // but if quantifier is All, we might need to handle duplicates differently.
                    // For now, let's implement standard EXCEPT (distinct).
                    
                    // Materialize right side
                    let right_batches = next_exec.collect()?;
                    let (right_rows, right_hashes) = if !right_batches.is_empty() {
                        let right_batch = concat_batches(&next_exec.schema(), &right_batches)?;
                        let converter = RowConverter::new(
                            right_batch
                                .schema()
                                .fields()
                                .iter()
                                .map(|f| SortField::new(f.data_type().clone()))
                                .collect(),
                        )?;
                        let rows = converter.convert_columns(right_batch.columns())?;
                        let mut set = FxHashSet::default();
                        let mut hashes = FxHashSet::default();
                        for row in rows.iter() {
                            let mut hasher = FxHasher::default();
                            row.hash(&mut hasher);
                            hashes.insert(hasher.finish());
                            set.insert(row.owned());
                        }
                        (set, hashes)
                    } else {
                        (FxHashSet::default(), FxHashSet::default())
                    };

                    let schema = current_exec.schema();
                    let converter = RowConverter::new(
                        schema
                            .fields()
                            .iter()
                            .map(|f| SortField::new(f.data_type().clone()))
                            .collect(),
                    )?;

                    let stream_iter = current_exec.into_stream()?;
                    let stream = stream_iter.map(move |batch_res| {
                        let batch = batch_res?;
                        let rows = converter.convert_columns(batch.columns())?;
                        let mut keep = Vec::with_capacity(batch.num_rows());
                        for row in rows.iter() {
                            let mut hasher = FxHasher::default();
                            row.hash(&mut hasher);
                            let h = hasher.finish();
                            if !right_hashes.contains(&h) {
                                keep.push(true);
                            } else {
                                keep.push(!right_rows.contains(&row.owned()));
                            }
                        }
                        let bool_array = arrow::array::BooleanArray::from(keep);
                        arrow::compute::filter_record_batch(&batch, &bool_array)
                            .map_err(|e| Error::Internal(e.to_string()))
                    });
                    
                    current_exec = SelectExecution::from_stream(
                        "except".to_string(),
                        schema,
                        Box::new(stream),
                    );
                    
                    // If distinct, apply distinct
                    if matches!(op.quantifier, CompoundQuantifier::Distinct) {
                         let schema = current_exec.schema();
                         let distinct_stream = Box::new(DistinctStream::new(
                            schema.clone(),
                            current_exec.into_stream()?,
                        )?);
                        current_exec = SelectExecution::from_stream(
                            "except_distinct".to_string(),
                            schema,
                            distinct_stream,
                        );
                    }
                }
                CompoundOperator::Intersect => {
                    // Materialize right side
                    let right_batches = next_exec.collect()?;
                    let (right_rows, right_hashes) = if !right_batches.is_empty() {
                        let right_batch = concat_batches(&next_exec.schema(), &right_batches)?;
                        let converter = RowConverter::new(
                            right_batch
                                .schema()
                                .fields()
                                .iter()
                                .map(|f| SortField::new(f.data_type().clone()))
                                .collect(),
                        )?;
                        let rows = converter.convert_columns(right_batch.columns())?;
                        let mut set = FxHashSet::default();
                        let mut hashes = FxHashSet::default();
                        for row in rows.iter() {
                            let mut hasher = FxHasher::default();
                            row.hash(&mut hasher);
                            hashes.insert(hasher.finish());
                            set.insert(row.owned());
                        }
                        (set, hashes)
                    } else {
                        (FxHashSet::default(), FxHashSet::default())
                    };

                    let schema = current_exec.schema();
                    let converter = RowConverter::new(
                        schema
                            .fields()
                            .iter()
                            .map(|f| SortField::new(f.data_type().clone()))
                            .collect(),
                    )?;

                    let stream_iter = current_exec.into_stream()?;
                    let stream = stream_iter.map(move |batch_res| {
                        let batch = batch_res?;
                        let rows = converter.convert_columns(batch.columns())?;
                        let mut keep = Vec::with_capacity(batch.num_rows());
                        for row in rows.iter() {
                            let mut hasher = FxHasher::default();
                            row.hash(&mut hasher);
                            let h = hasher.finish();
                            if !right_hashes.contains(&h) {
                                keep.push(false);
                            } else {
                                keep.push(right_rows.contains(&row.owned()));
                            }
                        }
                        let bool_array = arrow::array::BooleanArray::from(keep);
                        arrow::compute::filter_record_batch(&batch, &bool_array)
                            .map_err(|e| Error::Internal(e.to_string()))
                    });
                    
                    current_exec = SelectExecution::from_stream(
                        "intersect".to_string(),
                        schema,
                        Box::new(stream),
                    );
                    
                     // If distinct, apply distinct
                    if matches!(op.quantifier, CompoundQuantifier::Distinct) {
                         let schema = current_exec.schema();
                         let distinct_stream = Box::new(DistinctStream::new(
                            schema.clone(),
                            current_exec.into_stream()?,
                        )?);
                        current_exec = SelectExecution::from_stream(
                            "intersect_distinct".to_string(),
                            schema,
                            distinct_stream,
                        );
                    }
                }
            }
        }

        Ok(current_exec)
    }

    pub fn execute_select(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        let subquery_results = self.execute_scalar_subqueries(&plan.scalar_subqueries)?;
        self.execute_select_with_filter(&plan, &subquery_results, None)
    }

    pub fn execute_select_with_filter(
        &self,
        plan: &SelectPlan,
        subquery_results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if let Some(compound) = &plan.compound {
            return self.execute_compound(compound, subquery_results, row_filter);
        }

        let mut plan = plan.clone();

        // Rewrite projections
        let mut new_projections = Vec::new();
        for proj in &plan.projections {
            match proj {
                llkv_plan::plans::SelectProjection::Computed { expr, alias } => {
                    let rewritten = Self::rewrite_scalar_subqueries(expr, &subquery_results);
                    new_projections.push(llkv_plan::plans::SelectProjection::Computed {
                        expr: rewritten,
                        alias: alias.clone(),
                    });
                }
                other => new_projections.push(other.clone()),
            }
        }
        plan.projections = new_projections;

        // Rewrite filter
        if let Some(filter) = &mut plan.filter {
             filter.predicate = Self::rewrite_expr_subqueries(&filter.predicate, &subquery_results);
        }
        
        // Rewrite having
        if let Some(having) = &mut plan.having {
             *having = Self::rewrite_expr_subqueries(having, &subquery_results);
        }

        // Split filter
        let (pushable_filter, residual_filter) = if let Some(filter) = &plan.filter {
            Self::split_predicate(&filter.predicate)
        } else {
            (None, None)
        };
        
        let residual_subqueries = if residual_filter.is_some() {
            plan.filter.as_ref().map(|f| f.subqueries.clone()).unwrap_or_default()
        } else {
            Vec::new()
        };
        
        if let Some(pushable) = pushable_filter {
            if let Some(filter) = &mut plan.filter {
                filter.predicate = pushable;
            }
        } else {
            if plan.filter.is_some() {
                 plan.filter = None;
            }
        }

        // Check for complex aggregates in single-table queries BEFORE creating logical plan
        if plan.tables.len() == 1 {
            let table_ref = &plan.tables[0];
            let table_name = table_ref.qualified_name();
            
            // We need the schema to expand wildcards and check for aggregates
            if let Ok(schema) = self.logical_planner.get_table_schema(&table_name) {
                let mut string_projections = Vec::new();
                let mut final_names = Vec::new();
                
                for proj in &plan.projections {
                     match proj {
                        llkv_plan::plans::SelectProjection::AllColumns => {
                             for col in schema.columns.iter() {
                                 string_projections.push(ScalarExpr::Column(col.name.clone()));
                                 final_names.push(col.name.clone());
                             }
                        }
                        llkv_plan::plans::SelectProjection::AllColumnsExcept { exclude } => {
                             for col in schema.columns.iter() {
                                 if !exclude.contains(&col.name) {
                                     string_projections.push(ScalarExpr::Column(col.name.clone()));
                                     final_names.push(col.name.clone());
                                 }
                             }
                        }
                        llkv_plan::plans::SelectProjection::Column { name, alias } => {
                            string_projections.push(ScalarExpr::Column(name.clone()));
                            final_names.push(alias.clone().unwrap_or(name.clone()));
                        }
                        llkv_plan::plans::SelectProjection::Computed { expr, alias } => {
                            string_projections.push(expr.clone());
                            final_names.push(alias.clone());
                        }
                    }
                }

                let (new_aggregates, final_exprs, pre_agg_exprs) = extract_complex_aggregates(&string_projections);
                
                if !new_aggregates.is_empty() {
                     // Create pre-agg plan that projects the arguments for aggregation
                     let mut pre_agg_plan = plan.clone();
                     let mut projs: Vec<_> = pre_agg_exprs.iter().enumerate().map(|(i, expr)| {
                         llkv_plan::plans::SelectProjection::Computed {
                             expr: expr.clone(),
                             alias: format!("_agg_arg_{}", i),
                         }
                     }).collect();
                     
                     if residual_filter.is_some() {
                         projs.push(llkv_plan::plans::SelectProjection::AllColumns);
                     }
                     pre_agg_plan.projections = projs;
                     
                     // Create logical plan for pre-agg
                     // This should succeed because pre_agg_exprs do not contain aggregates
                     let logical_plan = self.logical_planner.create_logical_plan(&pre_agg_plan)?;
                     
                     match logical_plan {
                        LogicalPlan::Single(single) => {
                            let physical_plan = self
                                .physical_planner
                                .create_physical_plan(&single, row_filter)
                                .map_err(Error::Internal)?;
                            let schema = physical_plan.schema();

                            let mut base_iter: BatchIter = Box::new(
                                physical_plan
                                    .execute()
                                    .map_err(Error::Internal)?
                                    .map(|b| b.map_err(Error::Internal)),
                            );
                            
                            if let Some(residual) = &residual_filter {
                                base_iter = self.apply_residual_filter(base_iter, residual.clone(), residual_subqueries.clone());
                            }
                            
                            return self.execute_complex_aggregation(
                                base_iter,
                                schema,
                                new_aggregates,
                                final_exprs,
                                final_names,
                                &plan,
                                table_name,
                                &subquery_results,
                            );
                        }
                        _ => {
                            // Should not happen for single table plan
                            return Err(Error::Internal("Expected single table plan for pre-aggregation".into()));
                        }
                     }
                }
            }
        }



        // Check if we need to force manual projection
        let has_subqueries = plan.projections.iter().any(|p| {
            if let llkv_plan::plans::SelectProjection::Computed { expr, .. } = p {
                Self::contains_subquery(expr)
            } else {
                false
            }
        });
        let group_needs_full_row = plan.aggregates.is_empty() && !plan.group_by.is_empty();
        let force_manual_projection = residual_filter.is_some() || has_subqueries || group_needs_full_row;

        let plan_for_scan = if force_manual_projection {
            let mut p = plan.clone();
            p.projections = vec![llkv_plan::plans::SelectProjection::AllColumns];
            // Clear order by to avoid physical plan sorting on computed columns or dropping GROUP BY keys
            p.order_by = Vec::new();
            p
        } else {
            plan.clone()
        };

        let logical_plan = self.logical_planner.create_logical_plan(&plan_for_scan)?;

        match logical_plan {
            LogicalPlan::Single(single) => {
                let table_name = single.table_name.clone();
                
                // Check if we need to force manual projection (e.g. for computed columns or subqueries)
                let has_subqueries = plan.projections.iter().any(|p| {
                    if let llkv_plan::plans::SelectProjection::Computed { expr, .. } = p {
                        Self::contains_subquery(expr)
                    } else {
                        false
                    }
                });
                let group_needs_full_row = plan.aggregates.is_empty() && !plan.group_by.is_empty();
                let force_manual_projection = residual_filter.is_some() || has_subqueries || group_needs_full_row;

                let physical_plan = self
                    .physical_planner
                    .create_physical_plan(&single, row_filter)
                    .map_err(Error::Internal)?;

                let schema = physical_plan.schema();

                let mut base_iter: BatchIter = Box::new(
                    physical_plan
                        .execute()
                        .map_err(Error::Internal)?
                        .map(|b| b.map_err(Error::Internal)),
                );
                
                if let Some(residual) = &residual_filter {
                    base_iter = self.apply_residual_filter(base_iter, residual.clone(), residual_subqueries.clone());
                }

                // GROUP BY with no aggregates should collapse to distinct group keys.
                if plan.aggregates.is_empty() && !plan.group_by.is_empty() {
                    let mut key_indices = Vec::with_capacity(plan.group_by.len());
                    for name in &plan.group_by {
                        let idx = resolve_group_index(name, &schema, &single.schema)
                            .ok_or_else(|| Error::Internal(format!("GROUP BY column not found: {name}")))?;
                        key_indices.push(idx);
                    }

                    let mut grouped: BatchIter = Box::new(DistinctOnKeysStream::new(schema.clone(), key_indices, base_iter)?);

                    if let Some(having) = &plan.having {
                        grouped = self.apply_residual_filter(grouped, having.clone(), Vec::new());
                    }

                    return self.project_stream(
                        grouped,
                        &schema,
                        &plan.projections,
                        table_name,
                        &plan.scalar_subqueries,
                        plan.distinct,
                        &plan.order_by,
                        plan.offset,
                        plan.limit,
                        None,
                    );
                }

                if !plan.aggregates.is_empty() {
                    if !plan.group_by.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "GROUP BY aggregates are not supported yet".into(),
                        ));
                    }

                    let agg_iter = AggregateStream::new(base_iter, &plan, &single.schema, &schema)?;
                    let agg_schema = agg_iter.schema();
                    let trimmed =
                        apply_offset_limit_stream(Box::new(agg_iter), plan.offset, plan.limit);
                    return Ok(SelectExecution::from_stream(table_name, agg_schema, trimmed));
                }

                if force_manual_projection {
                 return self.project_stream(base_iter, &schema, &plan.projections, table_name, &plan.scalar_subqueries, plan.distinct, &plan.order_by, plan.offset, plan.limit, None);
                }

                let mut iter: BatchIter = base_iter;
                if plan.distinct {
                    iter = Box::new(DistinctStream::new(schema.clone(), iter)?);
                }

                let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);
                Ok(SelectExecution::from_stream(table_name, schema, trimmed))
            }
            LogicalPlan::Multi(multi) => {
                let table_count = multi.tables.len();
                if table_count == 0 {
                     return Err(Error::Internal("Multi-table plan with no tables".into()));
                }

                if multi.table_filters.len() != table_count {
                    return Err(Error::Internal("table filter metadata does not match table count".into()));
                }

                let table_filters = multi.table_filters.clone();
                let residual_filter = multi.filter.clone();
                
                // 1. Analyze required columns
                let mut required_fields: Vec<FxHashSet<FieldId>> = vec![FxHashSet::default(); table_count];
                
                for proj in &multi.projections {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, .. } => {
                            required_fields[*table_index].insert(logical_field_id.field_id());
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { expr, .. } => {
                             let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                             ScalarEvaluator::collect_fields(&remap_scalar_expr(expr), &mut fields);
                             for (tbl, fid) in fields {
                                 required_fields[tbl].insert(fid);
                             }
                        }
                    }
                }
                
                for join in &multi.joins {
                    let (keys, filters) = extract_join_keys_and_filters(join)?;
                    for key in keys {
                        required_fields[key.left_table].insert(key.left_field);
                        required_fields[key.right_table].insert(key.right_field);
                    }
                    for filter in filters {
                         let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                         ScalarEvaluator::collect_fields(&remap_filter_expr(&filter)?, &mut fields);
                         for (tbl, fid) in fields {
                             required_fields[tbl].insert(fid);
                         }
                    }
                }
                
                for (tbl, filter) in table_filters.iter().enumerate() {
                    if let Some(filter) = filter {
                        let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                        ScalarEvaluator::collect_fields(&remap_filter_expr(filter)?, &mut fields);
                        for (table_idx, fid) in fields {
                            if table_idx == tbl {
                                required_fields[table_idx].insert(fid);
                            }
                        }
                    }
                }

                if let Some(filter) = &residual_filter {
                     let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                     ScalarEvaluator::collect_fields(&remap_filter_expr(filter)?, &mut fields);
                     for (tbl, fid) in fields {
                         required_fields[tbl].insert(fid);
                     }
                }

                // 2. Create Streams
                let mut streams: Vec<BatchIter> = Vec::with_capacity(table_count);
                let mut schemas: Vec<SchemaRef> = Vec::with_capacity(table_count);
                let mut table_field_map: Vec<Vec<FieldId>> = Vec::with_capacity(table_count);

                for (i, table) in multi.tables.iter().enumerate() {
                    let adapter = table.table.as_any().downcast_ref::<ExecutionTableAdapter<P>>()
                        .ok_or_else(|| Error::Internal("unexpected table adapter type".into()))?;
                    
                    let mut fields: Vec<FieldId> = required_fields[i].iter().copied().collect();
                    if fields.is_empty() {
                        if let Some(col) = table.schema.columns.first() {
                            fields.push(col.field_id);
                        }
                    }
                    fields.sort_unstable();
                    table_field_map.push(fields.clone());
                    
                    let projections: Vec<ScanProjection> = fields.iter().map(|&fid| {
                        let col = table.schema.columns.iter().find(|c| c.field_id == fid).unwrap();
                        let lfid = LogicalFieldId::for_user(adapter.executor_table().table_id(), fid);
                        ScanProjection::Column(Projection::with_alias(lfid, col.name.clone()))
                    }).collect();
                    
                    let arrow_fields: Vec<ArrowField> = fields.iter().map(|&fid| {
                        let col = table.schema.columns.iter().find(|c| c.field_id == fid).unwrap();
                        ArrowField::new(col.name.clone(), col.data_type.clone(), col.is_nullable)
                    }).collect();
                    schemas.push(Arc::new(Schema::new(arrow_fields)));

                    let (tx, rx) = mpsc::sync_channel(16);
                    let storage = adapter.executor_table().storage().clone();
                    let row_filter = row_filter.clone();
                    
                    std::thread::spawn(move || {
                        let res = storage.scan_stream(
                            &projections,
                            &Expr::Pred(Filter { field_id: 0, op: Operator::Range { lower: std::ops::Bound::Unbounded, upper: std::ops::Bound::Unbounded } }),
                            llkv_scan::ScanStreamOptions { row_id_filter: row_filter.clone(), ..Default::default() },
                            &mut |batch| {
                                tx.send(Ok(batch)).ok();
                            },
                        );
                        if let Err(e) = res {
                            tx.send(Err(e)).ok();
                        }
                    });
                    
                    let mut stream: BatchIter = Box::new(rx.into_iter());

                    if let Some(filter_expr) = &table_filters[i] {
                        let mapping: FxHashMap<(usize, FieldId), usize> = table_field_map[i]
                            .iter()
                            .enumerate()
                            .map(|(idx, fid)| ((i, *fid), idx))
                            .collect();

                        stream = apply_filter_to_stream(stream, filter_expr, mapping)?;
                    }

                    streams.push(stream);
                }

                // 3. Build Pipeline
                let mut current_stream = streams.remove(0);
                let mut current_schema = schemas[0].clone();
                let mut col_mapping: FxHashMap<(usize, FieldId), usize> = FxHashMap::default();
                let mut pending_filters: Vec<LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>> =
                    match residual_filter.clone() {
                        Some(LlkvExpr::And(list)) => list.clone(),
                        Some(expr) => vec![expr.clone()],
                        None => Vec::new(),
                    };
                
                for (idx, fid) in table_field_map[0].iter().enumerate() {
                    col_mapping.insert((0, *fid), idx);
                }

                for i in 0..table_count - 1 {
                    let right_stream = streams.remove(0);
                    let right_schema = schemas[i+1].clone();
                    
                    let right_batches: Vec<RecordBatch> = right_stream.collect::<Result<Vec<_>, _>>()?;
                    let right_batch = if right_batches.is_empty() {
                        RecordBatch::new_empty(right_schema.clone())
                    } else {
                        concat_batches(&right_schema, &right_batches)?
                    };
                    
                    let join = multi
                        .joins
                        .iter()
                        .find(|j| j.left_table_index == i)
                        .ok_or_else(|| Error::Internal("missing join metadata for multi-table query".into()))?;

                    let (keys, residuals) = extract_join_keys_and_filters(join)?;
                    pending_filters.extend(residuals);
                    let mut left_indices = Vec::new();
                    let mut right_indices = Vec::new();

                    for key in keys {
                        if key.right_table != i + 1 {
                            return Err(Error::Internal("join key points to unexpected right table".into()));
                        }

                        if key.left_table > i {
                            return Err(Error::Internal("join key references future left table".into()));
                        }

                        let left_idx = *col_mapping
                            .get(&(key.left_table, key.left_field))
                            .ok_or_else(|| Error::Internal("left join key missing".into()))?;
                        left_indices.push(left_idx);

                        let right_idx = table_field_map[key.right_table]
                            .iter()
                            .position(|&f| f == key.right_field)
                            .ok_or_else(|| Error::Internal("right join key missing".into()))?;
                        right_indices.push(right_idx);
                    }

                    let join_type = map_join_type(join.join_type)?;
                        
                    let mut new_fields = current_schema.fields().to_vec();
                    new_fields.extend_from_slice(right_schema.fields());
                    let new_schema = Arc::new(Schema::new(new_fields));

                    current_stream = Box::new(VectorizedHashJoinStream::try_new(
                        new_schema.clone(),
                        current_stream,
                        right_batch,
                        join_type,
                        left_indices,
                        right_indices,
                    )?);
                    
                    let old_len = current_schema.fields().len();
                    current_schema = new_schema;
                    
                    for (idx, fid) in table_field_map[i+1].iter().enumerate() {
                        col_mapping.insert((i+1, *fid), old_len + idx);
                    }

                    let prefix_limit = i + 1;
                    let mut applicable = Vec::new();
                    let mut remaining = Vec::new();
                    for clause in pending_filters.into_iter() {
                        let mapped = remap_filter_expr(&clause)?;
                        let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                        ScalarEvaluator::collect_fields(&mapped, &mut fields);
                        let tables: FxHashSet<usize> = fields.into_iter().map(|(tbl, _)| tbl).collect();

                        if tables.iter().all(|t| *t <= prefix_limit) {
                            applicable.push(clause);
                        } else {
                            remaining.push(clause);
                        }
                    }
                    pending_filters = remaining;

                    if let Some(apply_expr) = combine_clauses(applicable) {
                        current_stream = apply_filter_to_stream(current_stream, &apply_expr, col_mapping.clone())?;
                    }
                }
                
                // 3. Apply any remaining filters not yet pushed earlier in the join chain.
                if let Some(final_expr) = combine_clauses(pending_filters) {
                    current_stream = apply_filter_to_stream(current_stream, &final_expr, col_mapping.clone())?;
                }
                
                // 4. Final Projection
                let mut plan_columns = Vec::new();
                let mut name_to_index = FxHashMap::default();

                let mut name_overrides: FxHashMap<String, usize> = FxHashMap::default();
                for ((tbl, fid), idx) in &col_mapping {
                    if let Some(table_ref) = multi.table_order.get(*tbl) {
                        if let Some(col) = multi.tables.get(*tbl).and_then(|t| t.schema.columns.iter().find(|c| c.field_id == *fid)) {
                            let qualifier = table_ref.display_name().to_ascii_lowercase();
                            let key = format!("{}.{}", qualifier, col.name.to_ascii_lowercase());
                            name_overrides.insert(key, *idx);
                        }
                    }
                }

                let output_fields: Vec<ArrowField> = multi.projections.iter().enumerate().map(|(i, proj)| {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, alias } => {
                            let idx = col_mapping.get(&(*table_index, logical_field_id.field_id())).unwrap();
                            let field = current_schema.field(*idx);
                            
                            let name = alias.clone().unwrap_or_else(|| field.name().clone());
                            let mut metadata = HashMap::new();
                            metadata.insert("field_id".to_string(), logical_field_id.field_id().to_string());
                            
                            plan_columns.push(PlanColumn {
                                name: name.clone(),
                                data_type: field.data_type().clone(),
                                field_id: logical_field_id.field_id(),
                                is_nullable: field.is_nullable(),
                                is_primary_key: false,
                                is_unique: false,
                                default_value: None,
                                check_expr: None,
                            });
                            name_to_index.insert(name.to_ascii_lowercase(), i);

                            ArrowField::new(name, field.data_type().clone(), field.is_nullable())
                                .with_metadata(metadata)
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { alias, expr } => {
                            let name = alias.clone();
                            let dummy_fid = 999999 + i as u32;
                            
                            let remapped = remap_scalar_expr(expr);
                            let inferred_type = infer_type(&remapped, &current_schema, &col_mapping).unwrap_or(arrow::datatypes::DataType::Int64);
                            
                            let mut metadata = HashMap::new();
                            metadata.insert("field_id".to_string(), dummy_fid.to_string());

                            plan_columns.push(PlanColumn {
                                name: name.clone(),
                                data_type: inferred_type.clone(),
                                field_id: dummy_fid,
                                is_nullable: true,
                                is_primary_key: false,
                                is_unique: false,
                                default_value: None,
                                check_expr: None,
                            });
                            name_to_index.insert(name.to_ascii_lowercase(), i);

                            ArrowField::new(name, inferred_type, true)
                                .with_metadata(metadata)
                        }
                    }
                }).collect();
                let output_schema = Arc::new(Schema::new(output_fields));
                let _logical_schema = PlanSchema { columns: plan_columns, name_to_index };
                
                // Check for aggregates
                let mut string_projections = Vec::new();
                for proj in &multi.projections {
                    let expr = match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, .. } => {
                            ScalarExpr::Column((*table_index, logical_field_id.field_id()))
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { expr, .. } => convert_resolved_expr(expr),
                    };
                    string_projections.push(map_expr_to_names(&expr, &col_mapping, &current_schema)?);
                }

                let (new_aggregates, final_exprs, pre_agg_exprs) = extract_complex_aggregates(&string_projections);
                let has_aggregates = !new_aggregates.is_empty();

                if !has_aggregates && !multi.group_by.is_empty() {
                    let mut key_indices = Vec::with_capacity(multi.group_by.len());
                    for key in &multi.group_by {
                        let field_id = key.logical_field_id.field_id();
                        let idx = col_mapping
                            .get(&(key.table_index, field_id))
                            .copied()
                            .ok_or_else(|| Error::Internal("GROUP BY column not found in join output".into()))?;
                        key_indices.push(idx);
                    }

                    let mut grouped: BatchIter = Box::new(DistinctOnKeysStream::new(current_schema.clone(), key_indices, current_stream)?);

                    if let Some(having) = &plan.having {
                        grouped = self.apply_residual_filter(grouped, having.clone(), Vec::new());
                    }

                    let table_name = multi.table_order.first().map(|t| t.qualified_name()).unwrap_or_default();

                    return self.project_stream(
                        grouped,
                        &current_schema,
                        &plan.projections,
                        table_name,
                        &plan.scalar_subqueries,
                        plan.distinct,
                        &plan.order_by,
                        plan.offset,
                        plan.limit,
                        Some(name_overrides.clone()),
                    );
                }

                let mut iter: BatchIter = if has_aggregates {
                    if !plan.group_by.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "GROUP BY aggregates are not supported yet".into(),
                        ));
                    }

                    // 1. Pre-aggregation stream
                    let pre_agg_schema_fields: Vec<ArrowField> = pre_agg_exprs.iter().enumerate().map(|(i, _)| {
                        ArrowField::new(format!("_agg_arg_{}", i), arrow::datatypes::DataType::Int64, true)
                            .with_metadata(HashMap::from([("field_id".to_string(), format!("{}", 10000+i))]))
                    }).collect();
                    let pre_agg_schema = Arc::new(Schema::new(pre_agg_schema_fields));
                    
                    let pre_agg_schema_captured = pre_agg_schema.clone();
                    let current_schema_captured = current_schema.clone();
                    let subquery_results_captured = subquery_results.clone();
                    
                    let pre_agg_stream = current_stream.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();
                        
                        for expr_str in &pre_agg_exprs {
                            let expr = resolve_scalar_expr_string(expr_str, &current_schema_captured, &subquery_results_captured)?;
                            
                            let mut required_fields = FxHashSet::default();
                            ScalarEvaluator::collect_fields(&expr, &mut required_fields);
                            
                            let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
                            for (tbl, fid) in required_fields {
                                field_arrays.insert((tbl, fid), batch.column(fid as usize).clone());
                            }
                            
                            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
                            let result = ScalarEvaluator::evaluate_batch_simplified(&expr, batch.num_rows(), &numeric_arrays)?;
                            columns.push(result);
                        }
                        
                        if columns.is_empty() {
                            let options = arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                            RecordBatch::try_new_with_options(pre_agg_schema_captured.clone(), columns, &options).map_err(|e| Error::Internal(e.to_string()))
                        } else {
                            RecordBatch::try_new(pre_agg_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
                        }
                    });

                    // 2. Aggregate Stream
                    let mut dummy_plan = plan.clone();
                    dummy_plan.aggregates = new_aggregates;
                    
                    let mut dummy_cols = Vec::new();
                    let mut dummy_name_to_idx = FxHashMap::default();
                    for (i, field) in pre_agg_schema.fields().iter().enumerate() {
                        let name = field.name().clone();
                        let fid = field.metadata().get("field_id").unwrap().parse::<u32>().unwrap();
                        dummy_cols.push(PlanColumn {
                            name: name.clone(),
                            data_type: field.data_type().clone(),
                            field_id: fid,
                            is_nullable: true,
                            is_primary_key: false,
                            is_unique: false,
                            default_value: None,
                            check_expr: None,
                        });
                        dummy_name_to_idx.insert(name, i);
                    }
                    let dummy_logical_schema = PlanSchema { columns: dummy_cols, name_to_index: dummy_name_to_idx };

                    let agg_iter = AggregateStream::new(Box::new(pre_agg_stream), &dummy_plan, &dummy_logical_schema, &pre_agg_schema)?;
                    let agg_schema = agg_iter.schema();

                    // 3. Final Projection Stream
                    let output_schema_captured = output_schema.clone();
                    let subquery_results_captured = subquery_results.clone();
                    let final_stream = agg_iter.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();
                        
                        for expr_str in &final_exprs {
                            let expr = resolve_scalar_expr_string(expr_str, &agg_schema, &subquery_results_captured)?;
                            
                            let mut required_fields = FxHashSet::default();
                            ScalarEvaluator::collect_fields(&expr, &mut required_fields);
                            
                            let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
                            for (tbl, fid) in required_fields {
                                field_arrays.insert((tbl, fid), batch.column(fid as usize).clone());
                            }
                            
                            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
                            let result = ScalarEvaluator::evaluate_batch_simplified(&expr, batch.num_rows(), &numeric_arrays)?;
                            columns.push(result);
                        }
                        RecordBatch::try_new(output_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
                    });
                    
                    Box::new(final_stream)

                } else {
                    let output_schema_captured = output_schema.clone();
                    let final_stream = current_stream.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();
                        
                        for proj in &multi.projections {
                            match proj {
                                llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, .. } => {
                                    let idx = col_mapping.get(&(*table_index, logical_field_id.field_id()))
                                        .ok_or_else(|| Error::Internal("projection column missing".into()))?;
                                    columns.push(batch.column(*idx).clone());
                                }
                                llkv_plan::logical_planner::ResolvedProjection::Computed { expr, .. } => {
                                    let remapped = remap_scalar_expr(expr);
                                    
                                    let mut required_fields = FxHashSet::default();
                                    ScalarEvaluator::collect_fields(&remapped, &mut required_fields);
                                    
                                    let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
                                    for (tbl, fid) in required_fields {
                                        if let Some(idx) = col_mapping.get(&(tbl, fid)) {
                                            field_arrays.insert((tbl, fid), batch.column(*idx).clone());
                                        } else {
                                            return Err(Error::Internal(format!("Missing field {:?} in batch for computed column", (tbl, fid))));
                                        }
                                    }
                                    
                                    let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
                                    let result = ScalarEvaluator::evaluate_batch_simplified(&remapped, batch.num_rows(), &numeric_arrays)?;
                                    columns.push(result);
                                }
                            }
                        }
                        RecordBatch::try_new(output_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
                    });
                    Box::new(final_stream)
                };

                if plan.distinct {
                    iter = Box::new(DistinctStream::new(output_schema.clone(), iter)?);
                }

                let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);
                let table_name = multi.table_order.first().map(|t| t.qualified_name()).unwrap_or_default();
                Ok(SelectExecution::from_stream(table_name, output_schema, trimmed))
        }
    }
}
}



struct OffsetLimitStream {
    input: BatchIter,
    remaining_offset: usize,
    remaining_limit: Option<usize>,
}

impl Iterator for OffsetLimitStream {
    type Item = ExecutorResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let batch = match self.input.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => return None,
            };

            if let Some(limit) = self.remaining_limit {
                if limit == 0 {
                    return None;
                }
            }

            let mut start = 0usize;
            let mut len = batch.num_rows();

            if self.remaining_offset > 0 {
                if self.remaining_offset >= len {
                    self.remaining_offset -= len;
                    continue;
                }
                start = self.remaining_offset;
                len -= self.remaining_offset;
                self.remaining_offset = 0;
            }

            if let Some(limit) = self.remaining_limit {
                if len > limit {
                    len = limit;
                }
                self.remaining_limit = Some(limit - len);
            }

            return Some(Ok(batch.slice(start, len)));
        }
    }
}

fn apply_offset_limit_stream(
    iter: BatchIter,
    offset: Option<usize>,
    limit: Option<usize>,
) -> BatchIter {
    Box::new(OffsetLimitStream {
        input: iter,
        remaining_offset: offset.unwrap_or(0),
        remaining_limit: limit,
    })
}

enum SelectSource {
    Stream(BatchIter),
    Materialized(Vec<RecordBatch>),
}

/// Streaming-friendly SELECT execution handle.
pub struct SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table_name: String,
    schema: SchemaRef,
    source: Arc<Mutex<SelectSource>>,
    _marker: PhantomData<P>,
}

impl<P> Clone for SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            table_name: self.table_name.clone(),
            schema: Arc::clone(&self.schema),
            source: Arc::clone(&self.source),
            _marker: PhantomData,
        }
    }
}

impl<P> fmt::Debug for SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectExecution")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .finish()
    }
}

impl<P> SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(table_name: String, schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self::from_materialized(table_name, schema, batches)
    }

    pub fn new_single_batch(table_name: String, schema: SchemaRef, batch: RecordBatch) -> Self {
        Self::from_materialized(table_name, schema, vec![batch])
    }

    pub fn from_batch(table_name: String, schema: SchemaRef, batch: RecordBatch) -> Self {
        Self::new_single_batch(table_name, schema, batch)
    }

    pub fn from_stream(table_name: String, schema: SchemaRef, iter: BatchIter) -> Self {
        Self {
            table_name,
            schema,
            source: Arc::new(Mutex::new(SelectSource::Stream(iter))),
            _marker: PhantomData,
        }
    }

    fn from_materialized(table_name: String, schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self {
            table_name,
            schema,
            source: Arc::new(Mutex::new(SelectSource::Materialized(batches))),
            _marker: PhantomData,
        }
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn into_stream(self) -> ExecutorResult<BatchIter> {
        let mut guard = self
            .source
            .lock()
            .map_err(|_| Error::Internal("Failed to lock select source".to_string()))?;

        let source = std::mem::replace(&mut *guard, SelectSource::Materialized(vec![]));
        match source {
            SelectSource::Stream(iter) => Ok(iter),
            SelectSource::Materialized(batches) => Ok(Box::new(batches.into_iter().map(Ok))),
        }
    }

    pub fn collect(&self) -> ExecutorResult<Vec<RecordBatch>> {
        let mut guard = self
            .source
            .lock()
            .map_err(|_| Error::Internal("select stream poisoned".into()))?;
        match &mut *guard {
            SelectSource::Materialized(batches) => Ok(batches.clone()),
            SelectSource::Stream(iter) => {
                let mut collected = Vec::new();
                for batch in iter {
                    collected.push(batch?);
                }
                *guard = SelectSource::Materialized(collected.clone());
                Ok(collected)
            }
        }
    }

    pub fn stream<F>(&self, mut on_batch: F) -> ExecutorResult<()>
    where
        F: FnMut(RecordBatch) -> ExecutorResult<()>,
    {
        let mut guard = self
            .source
            .lock()
            .map_err(|_| Error::Internal("select stream poisoned".into()))?;
        match &mut *guard {
            SelectSource::Materialized(batches) => {
                for batch in batches.iter() {
                    on_batch(batch.clone())?;
                }
            }
            SelectSource::Stream(iter) => {
                for batch in iter {
                    on_batch(batch?)?;
                }
                *guard = SelectSource::Materialized(Vec::new());
            }
        }
        Ok(())
    }
}

/// Adapter to expose executor tables to the planner as `ExecutionTable`s.
struct PlannerTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    inner: Arc<dyn ExecutorTableProvider<P>>,
}

impl<P> TableProvider<P> for PlannerTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, name: &str) -> llkv_result::Result<Arc<dyn ExecutionTable<P>>> {
        let table = self.inner.get_table(name)?;
        Ok(Arc::new(ExecutionTableAdapter::new(table)))
    }
}

struct ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: Arc<ExecutorTable<P>>,
    plan_schema: Arc<llkv_plan::schema::PlanSchema>,
}



impl<P> fmt::Debug for ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionTableAdapter")
            .field("table_id", &self.table.table_id())
            .finish()
    }
}

impl<P> ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new(table: Arc<ExecutorTable<P>>) -> Self {
        let mut name_to_index = FxHashMap::default();
        let columns: Vec<llkv_plan::schema::PlanColumn> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                name_to_index.insert(c.name.to_ascii_lowercase(), idx);
                llkv_plan::schema::PlanColumn {
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    field_id: c.field_id,
                    is_nullable: c.is_nullable,
                    is_primary_key: c.is_primary_key,
                    is_unique: c.is_unique,
                    default_value: c.default_value.clone(),
                    check_expr: c.check_expr.clone(),
                }
            })
            .collect();
        let plan_schema = Arc::new(llkv_plan::schema::PlanSchema {
            columns,
            name_to_index,
        });
        Self { table, plan_schema }
    }

    fn executor_table(&self) -> Arc<ExecutorTable<P>> {
        Arc::clone(&self.table)
    }
}



struct DistinctStream {
    schema: SchemaRef,
    converter: RowConverter,
    seen: FxHashSet<OwnedRow>,
    input: BatchIter,
}

struct DistinctOnKeysStream {
    schema: SchemaRef,
    converter: RowConverter,
    key_indices: Vec<usize>,
    seen: FxHashSet<OwnedRow>,
    input: BatchIter,
}

impl DistinctStream {
    fn new(schema: SchemaRef, input: BatchIter) -> ExecutorResult<Self> {
        let sort_fields: Vec<SortField> = schema
            .fields()
            .iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();
        let converter =
            RowConverter::new(sort_fields).map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            schema,
            converter,
            seen: FxHashSet::default(),
            input,
        })
    }
}

impl DistinctOnKeysStream {
    fn new(schema: SchemaRef, key_indices: Vec<usize>, input: BatchIter) -> ExecutorResult<Self> {
        let key_fields: Vec<ArrowField> = key_indices
            .iter()
            .map(|&idx| schema.field(idx).clone())
            .collect();
        let sort_fields: Vec<SortField> = key_fields
            .iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();
        let converter = RowConverter::new(sort_fields).map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            schema,
            converter,
            key_indices,
            seen: FxHashSet::default(),
            input,
        })
    }
}

impl Iterator for DistinctStream {
    type Item = ExecutorResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(batch) = self.input.next() {
            let batch = match batch {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            let rows = match self.converter.convert_columns(batch.columns()) {
                Ok(r) => r,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };

            let mut unique_rows: Vec<Option<(usize, usize)>> = Vec::new();
            for row_idx in 0..batch.num_rows() {
                let owned = rows.row(row_idx).owned();
                if self.seen.insert(owned) {
                    unique_rows.push(Some((0, row_idx)));
                }
            }

            if unique_rows.is_empty() {
                continue;
            }

            let projection: Vec<usize> = (0..self.schema.fields().len()).collect();
            let arrays = match gather_optional_projected_indices_from_batches(
                &[batch],
                &unique_rows,
                &projection,
            ) {
                Ok(a) => a,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };

            let out = match RecordBatch::try_new(Arc::clone(&self.schema), arrays) {
                Ok(b) => b,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };
            return Some(Ok(out));
        }
        None
    }
}

impl Iterator for DistinctOnKeysStream {
    type Item = ExecutorResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(batch) = self.input.next() {
            let batch = match batch {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            let key_columns: Vec<ArrayRef> = self
                .key_indices
                .iter()
                .map(|&idx| batch.column(idx).clone())
                .collect();

            let rows = match self.converter.convert_columns(&key_columns) {
                Ok(r) => r,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };

            let mut unique_rows: Vec<Option<(usize, usize)>> = Vec::new();
            for row_idx in 0..batch.num_rows() {
                let owned = rows.row(row_idx).owned();
                if self.seen.insert(owned) {
                    unique_rows.push(Some((0, row_idx)));
                }
            }

            if unique_rows.is_empty() {
                continue;
            }

            let projection: Vec<usize> = (0..self.schema.fields().len()).collect();
            let arrays = match gather_optional_projected_indices_from_batches(
                &[batch],
                &unique_rows,
                &projection,
            ) {
                Ok(a) => a,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };

            let out = match RecordBatch::try_new(Arc::clone(&self.schema), arrays) {
                Ok(b) => b,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };
            return Some(Ok(out));
        }

        None
    }
}

fn remap_scalar_expr(
    expr: &ScalarExpr<llkv_plan::logical_planner::ResolvedFieldRef>,
) -> ScalarExpr<(usize, FieldId)> {
    match expr {
        ScalarExpr::Column(resolved) => {
            ScalarExpr::Column((resolved.table_index, resolved.logical_field_id.field_id()))
        }
        ScalarExpr::Literal(lit) => ScalarExpr::Literal(lit.clone()),
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(remap_scalar_expr(left)),
            op: *op,
            right: Box::new(remap_scalar_expr(right)),
        },
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(remap_scalar_expr(left)),
            op: *op,
            right: Box::new(remap_scalar_expr(right)),
        },
        ScalarExpr::Not(inner) => ScalarExpr::Not(Box::new(remap_scalar_expr(inner))),
        ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
            expr: Box::new(remap_scalar_expr(expr)),
            negated: *negated,
        },
        ScalarExpr::Aggregate(call) => ScalarExpr::Aggregate(match call {
            AggregateCall::CountStar => AggregateCall::CountStar,
            AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Min(expr) => AggregateCall::Min(Box::new(remap_scalar_expr(expr))),
            AggregateCall::Max(expr) => AggregateCall::Max(Box::new(remap_scalar_expr(expr))),
            AggregateCall::CountNulls(expr) => {
                AggregateCall::CountNulls(Box::new(remap_scalar_expr(expr)))
            }
            AggregateCall::GroupConcat {
                expr,
                distinct,
                separator,
            } => AggregateCall::GroupConcat {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
                separator: separator.clone(),
            },
        }),
        ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
            base: Box::new(remap_scalar_expr(base)),
            field_name: field_name.clone(),
        },
        ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
            expr: Box::new(remap_scalar_expr(expr)),
            data_type: data_type.clone(),
        },
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => ScalarExpr::Case {
            operand: operand.as_deref().map(remap_scalar_expr).map(Box::new),
            branches: branches
                .iter()
                .map(|(when_expr, then_expr)| {
                    (remap_scalar_expr(when_expr), remap_scalar_expr(then_expr))
                })
                .collect(),
            else_expr: else_expr.as_deref().map(remap_scalar_expr).map(Box::new),
        },
        ScalarExpr::Coalesce(items) => {
            ScalarExpr::Coalesce(items.iter().map(remap_scalar_expr).collect())
        }
        ScalarExpr::Random => ScalarExpr::Random,
        ScalarExpr::ScalarSubquery(subquery) => ScalarExpr::ScalarSubquery(subquery.clone()),
    }
}

fn simplify_predicate(expr: &ScalarExpr<(usize, FieldId)>) -> ScalarExpr<(usize, FieldId)> {
    match expr {
        ScalarExpr::IsNull { expr: inner, negated } => {
            let simplified_inner = simplify_predicate(inner);
            if let ScalarExpr::Literal(Literal::Null) = simplified_inner {
                ScalarExpr::Literal(Literal::Boolean(!*negated))
            } else {
                ScalarExpr::IsNull {
                    expr: Box::new(simplified_inner),
                    negated: *negated,
                }
            }
        }
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(simplify_predicate(left)),
            op: *op,
            right: Box::new(simplify_predicate(right)),
        },
        ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(simplify_predicate(e))),
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(simplify_predicate(left)),
            op: *op,
            right: Box::new(simplify_predicate(right)),
        },
        _ => expr.clone(),
    }
}

fn remap_filter_expr(
    expr: &LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
) -> ExecutorResult<ScalarExpr<(usize, FieldId)>> {
    fn combine_with_op(
        mut exprs: impl Iterator<Item = ExecutorResult<ScalarExpr<(usize, FieldId)>>>,
        op: llkv_expr::expr::BinaryOp,
    ) -> ExecutorResult<ScalarExpr<(usize, FieldId)>> {
        let first = exprs
            .next()
            .transpose()?
            .unwrap_or_else(|| ScalarExpr::Literal(Literal::Boolean(true)));
        exprs.try_fold(first, |acc, next| {
            let rhs = next?;
            Ok(ScalarExpr::Binary {
                left: Box::new(acc),
                op,
                right: Box::new(rhs),
            })
        })
    }

    match expr {
        LlkvExpr::And(list) => combine_with_op(list.iter().map(remap_filter_expr), llkv_expr::expr::BinaryOp::And),
        LlkvExpr::Or(list) => combine_with_op(list.iter().map(remap_filter_expr), llkv_expr::expr::BinaryOp::Or),
        LlkvExpr::Not(inner) => Ok(ScalarExpr::Not(Box::new(remap_filter_expr(inner)?))),
        LlkvExpr::Pred(filter) => predicate_to_scalar(filter),
        LlkvExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(remap_scalar_expr(left)),
            op: *op,
            right: Box::new(remap_scalar_expr(right)),
        }),
        LlkvExpr::InList {
            expr,
            list,
            negated,
        } => {
            let col = remap_scalar_expr(expr);
            let mut combined = ScalarExpr::Literal(Literal::Boolean(false));
            for lit in list {
                let eq = ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Eq,
                    right: Box::new(remap_scalar_expr(lit)),
                };
                combined = ScalarExpr::Binary {
                    left: Box::new(combined),
                    op: llkv_expr::expr::BinaryOp::Or,
                    right: Box::new(eq),
                };
            }
            if *negated {
                Ok(ScalarExpr::Not(Box::new(combined)))
            } else {
                Ok(combined)
            }
        }
        LlkvExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(remap_scalar_expr(expr)),
            negated: *negated,
        }),
        LlkvExpr::Literal(b) => Ok(ScalarExpr::Literal(Literal::Boolean(*b))),
        _ => Err(Error::InvalidArgumentError(
            "unsupported predicate in multi-table filter".into(),
        )),
    }
}

fn predicate_to_scalar(
    filter: &Filter<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
) -> ExecutorResult<ScalarExpr<(usize, FieldId)>> {
    let col = ScalarExpr::Column((
        filter.field_id.table_index,
        filter.field_id.logical_field_id.field_id(),
    ));

    let expr = match &filter.op {
        Operator::Equals(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::Eq,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::GreaterThan(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::Gt,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::GreaterThanOrEquals(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::GtEq,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::LessThan(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::Lt,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::LessThanOrEquals(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::LtEq,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::Range { lower, upper } => {
            let lower_expr = match lower {
                std::ops::Bound::Included(l) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::GtEq,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                }),
                std::ops::Bound::Excluded(l) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Gt,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                }),
                std::ops::Bound::Unbounded => None,
            };
            let upper_expr = match upper {
                std::ops::Bound::Included(u) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::LtEq,
                    right: Box::new(ScalarExpr::Literal(u.clone())),
                }),
                std::ops::Bound::Excluded(u) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Lt,
                    right: Box::new(ScalarExpr::Literal(u.clone())),
                }),
                std::ops::Bound::Unbounded => None,
            };

            match (lower_expr, upper_expr) {
                (Some(l), Some(u)) => ScalarExpr::Binary {
                    left: Box::new(l),
                    op: llkv_expr::expr::BinaryOp::And,
                    right: Box::new(u),
                },
                (Some(l), None) => l,
                (None, Some(u)) => u,
                (None, None) => ScalarExpr::Literal(Literal::Boolean(true)),
            }
        }
        Operator::In(list) => {
            let mut combined = ScalarExpr::Literal(Literal::Boolean(false));
            for lit in list.iter() {
                let eq = ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Eq,
                    right: Box::new(ScalarExpr::Literal(lit.clone())),
                };
                combined = ScalarExpr::Binary {
                    left: Box::new(combined),
                    op: llkv_expr::expr::BinaryOp::Or,
                    right: Box::new(eq),
                };
            }
            combined
        }
        Operator::IsNull => ScalarExpr::IsNull {
            expr: Box::new(col),
            negated: false,
        },
        Operator::IsNotNull => ScalarExpr::IsNull {
            expr: Box::new(col),
            negated: true,
        },
        Operator::StartsWith { .. }
        | Operator::EndsWith { .. }
        | Operator::Contains { .. } => {
            return Err(Error::InvalidArgumentError(
                "string pattern predicates are not supported in multi-table execution".into(),
            ))
        }
    };

    Ok(expr)
}



#[derive(Debug, Clone)]
struct PhysicalJoinKey {
    left_table: usize,
    left_field: FieldId,
    right_table: usize,
    right_field: FieldId,
}

fn map_join_type(join_plan: JoinPlan) -> ExecutorResult<JoinType> {
    match join_plan {
        JoinPlan::Inner => Ok(JoinType::Inner),
        JoinPlan::Left => Ok(JoinType::Left),
        JoinPlan::Right => Err(Error::InvalidArgumentError(
            "RIGHT JOIN is not supported yet".into(),
        )),
        JoinPlan::Full => Err(Error::InvalidArgumentError(
            "FULL JOIN is not supported yet".into(),
        )),
    }
}

fn resolve_schema_index(name: &str, schema: &Schema) -> Option<usize> {
    // Try exact match first
    if let Ok(idx) = schema.index_of(name) {
        return Some(idx);
    }
    // Fallback: compare lowercased and strip qualifiers.
    let needle = name.rsplit('.').next().unwrap_or(name).to_ascii_lowercase();
    schema
        .fields()
        .iter()
        .position(|f| f.name().to_ascii_lowercase() == needle)
}

fn resolve_group_index(name: &str, schema: &Schema, plan_schema: &llkv_plan::schema::PlanSchema) -> Option<usize> {
    if let Some(idx) = resolve_schema_index(name, schema) {
        return Some(idx);
    }

    let field_id = plan_schema
        .column_by_name(name)
        .map(|c| c.field_id)?;

    schema
        .fields()
        .iter()
        .enumerate()
        .find_map(|(idx, f)| {
            f.metadata()
                .get("field_id")
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|fid| *fid == field_id)
                .map(|_| idx)
        })
}

fn extract_join_keys_and_filters(
    join: &ResolvedJoin,
) -> ExecutorResult<(
    Vec<PhysicalJoinKey>,
    Vec<LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>>,
)> {
    let mut keys = Vec::new();
    let mut residuals = Vec::new();

    if let Some(on) = &join.on {
        collect_join_predicates(
            on,
            join.left_table_index,
            join.left_table_index + 1,
            &mut keys,
            &mut residuals,
        )?;
    }

    Ok((keys, residuals))
}

fn collect_join_predicates(
    expr: &LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
    left_table_index: usize,
    right_table_index: usize,
    keys: &mut Vec<PhysicalJoinKey>,
    residuals: &mut Vec<LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>>,
) -> ExecutorResult<()> {
    match expr {
        LlkvExpr::And(list) => {
            for e in list {
                collect_join_predicates(e, left_table_index, right_table_index, keys, residuals)?;
            }
            Ok(())
        }
        LlkvExpr::Compare { left, op, right } => {
            if matches!(op, CompareOp::Eq) {
                if let (ScalarExpr::Column(l), ScalarExpr::Column(r)) = (left, right) {
                    let (left_col, right_col) = if r.table_index == right_table_index {
                        (l, r)
                    } else if l.table_index == right_table_index {
                        (r, l)
                    } else {
                        residuals.push(expr.clone());
                        return Ok(());
                    };

                    if left_col.table_index <= left_table_index
                        && right_col.table_index == right_table_index
                    {
                        keys.push(PhysicalJoinKey {
                            left_table: left_col.table_index,
                            left_field: left_col.logical_field_id.field_id(),
                            right_table: right_col.table_index,
                            right_field: right_col.logical_field_id.field_id(),
                        });
                        return Ok(());
                    }
                }
            }

            residuals.push(expr.clone());
            Ok(())
        }
        _ => {
            residuals.push(expr.clone());
            Ok(())
        }
    }
}

fn combine_clauses(
    mut clauses: Vec<LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>>,
) -> Option<LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>> {
    if clauses.is_empty() {
        None
    } else if clauses.len() == 1 {
        clauses.pop()
    } else {
        Some(LlkvExpr::And(clauses))
    }
}

fn apply_filter_to_stream(
    stream: BatchIter,
    filter_expr: &LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
    col_mapping: FxHashMap<(usize, FieldId), usize>,
) -> ExecutorResult<BatchIter> {
    let predicate = simplify_predicate(&remap_filter_expr(filter_expr)?);
    let col_mapping_captured = col_mapping.clone();

    let filtered = stream.map(move |batch_res| {
        let batch = batch_res?;

        let mut required_fields = FxHashSet::default();
        ScalarEvaluator::collect_fields(&predicate, &mut required_fields);

        let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
        for (tbl, fid) in &required_fields {
            if let Some(idx) = col_mapping_captured.get(&(*tbl, *fid)) {
                field_arrays.insert((*tbl, *fid), batch.column(*idx).clone());
            }
        }

        let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
        let result = ScalarEvaluator::evaluate_batch_simplified(&predicate, batch.num_rows(), &numeric_arrays)?;

        let bool_array = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| Error::Internal("Filter predicate must evaluate to boolean".into()))?;

        arrow::compute::filter_record_batch(&batch, bool_array)
            .map_err(|e| Error::Internal(e.to_string()))
    });

    Ok(Box::new(filtered))
}

impl<P> ExecutionTable<P> for ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> Arc<llkv_plan::schema::PlanSchema> {
        Arc::clone(&self.plan_schema)
    }

    fn table_id(&self) -> llkv_types::ids::TableId {
        self.table.table_id()
    }

    fn scan_stream(
        &self,
        projections: &[llkv_scan::ScanProjection],
        predicate: &llkv_expr::Expr<'static, llkv_types::FieldId>,
        options: llkv_scan::ScanStreamOptions<P>,
        callback: &mut dyn FnMut(RecordBatch),
    ) -> llkv_result::Result<()> {
        self.table
            .storage()
            .scan_stream(projections, predicate, options, callback)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct AggVisitor {
    aggregates: Vec<AggregateExpr>,
    pre_agg_projections: Vec<ScalarExpr<String>>,
}

impl AggVisitor {
    fn new() -> Self {
        Self {
            aggregates: Vec::new(),
            pre_agg_projections: Vec::new(),
        }
    }

    fn visit(&mut self, expr: &ScalarExpr<String>) -> ScalarExpr<String> {
        match expr {
            ScalarExpr::Column(c) => ScalarExpr::Column(c.clone()),
            ScalarExpr::Literal(l) => ScalarExpr::Literal(l.clone()),
            ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
                left: Box::new(self.visit(left)),
                op: *op,
                right: Box::new(self.visit(right)),
            },
            ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(self.visit(e))),
            ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
                expr: Box::new(self.visit(expr)),
                negated: *negated,
            },
            ScalarExpr::Aggregate(call) => {
                let (arg_expr, distinct, func) = match call {
                    AggregateCall::CountStar => {
                         let alias = format!("_agg_res_{}", self.aggregates.len());
                         self.aggregates.push(AggregateExpr::CountStar {
                             alias: alias.clone(),
                             distinct: false,
                         });
                         return ScalarExpr::Column(alias);
                    }
                    AggregateCall::Count { expr, distinct } => (expr, *distinct, AggregateFunction::Count),
                    AggregateCall::Sum { expr, distinct } => (expr, *distinct, AggregateFunction::SumInt64),
                    AggregateCall::Total { expr, distinct } => (expr, *distinct, AggregateFunction::TotalInt64),
                    AggregateCall::Min(expr) => (expr, false, AggregateFunction::MinInt64),
                    AggregateCall::Max(expr) => (expr, false, AggregateFunction::MaxInt64),
                    AggregateCall::CountNulls(expr) => (expr, false, AggregateFunction::CountNulls),
                    AggregateCall::GroupConcat { expr, distinct, separator: _ } => (expr, *distinct, AggregateFunction::GroupConcat),
                    AggregateCall::Avg { expr, distinct } => {
                        let arg_idx = self.pre_agg_projections.len();
                        self.pre_agg_projections.push(*expr.clone());
                        let arg_col_name = format!("_agg_arg_{}", arg_idx);
                        
                        let sum_alias = format!("_agg_res_{}", self.aggregates.len());
                        self.aggregates.push(AggregateExpr::Column {
                            column: arg_col_name.clone(),
                            alias: sum_alias.clone(),
                            function: AggregateFunction::SumInt64,
                            distinct: *distinct,
                        });
                        
                        let count_alias = format!("_agg_res_{}", self.aggregates.len());
                        self.aggregates.push(AggregateExpr::Column {
                            column: arg_col_name,
                            alias: count_alias.clone(),
                            function: AggregateFunction::Count,
                            distinct: *distinct,
                        });
                        
                        return ScalarExpr::Binary {
                            left: Box::new(ScalarExpr::Column(sum_alias)),
                            op: llkv_expr::expr::BinaryOp::Divide,
                            right: Box::new(ScalarExpr::Column(count_alias)),
                        };
                    }
                };

                let arg_idx = self.pre_agg_projections.len();
                self.pre_agg_projections.push(*arg_expr.clone());
                let arg_col_name = format!("_agg_arg_{}", arg_idx);
                
                let alias = format!("_agg_res_{}", self.aggregates.len());
                self.aggregates.push(AggregateExpr::Column {
                    column: arg_col_name,
                    alias: alias.clone(),
                    function: func,
                    distinct,
                });
                
                ScalarExpr::Column(alias)
            }
            ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
                base: Box::new(self.visit(base)),
                field_name: field_name.clone(),
            },
            ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
                expr: Box::new(self.visit(expr)),
                data_type: data_type.clone(),
            },
            ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
                left: Box::new(self.visit(left)),
                op: *op,
                right: Box::new(self.visit(right)),
            },
            ScalarExpr::Coalesce(exprs) => ScalarExpr::Coalesce(
                exprs.iter().map(|e| self.visit(e)).collect()
            ),
            ScalarExpr::ScalarSubquery(s) => ScalarExpr::ScalarSubquery(s.clone()),
            ScalarExpr::Case { operand, branches, else_expr } => ScalarExpr::Case {
                operand: operand.as_ref().map(|e| Box::new(self.visit(e))),
                branches: branches.iter().map(|(w, t)| (self.visit(w), self.visit(t))).collect(),
                else_expr: else_expr.as_ref().map(|e| Box::new(self.visit(e))),
            },
            ScalarExpr::Random => ScalarExpr::Random,
        }
    }
}

fn extract_complex_aggregates(
    projections: &[ScalarExpr<String>],
) -> (Vec<AggregateExpr>, Vec<ScalarExpr<String>>, Vec<ScalarExpr<String>>) {
    let mut visitor = AggVisitor::new();
    let rewritten = projections.iter().map(|p| visitor.visit(p)).collect();
    (visitor.aggregates, rewritten, visitor.pre_agg_projections)
}

fn resolve_scalar_expr_string(
    expr: &ScalarExpr<String>,
    schema: &Schema,
    subquery_results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
) -> Result<ScalarExpr<(usize, FieldId)>, Error> {
    match expr {
        ScalarExpr::Column(name) => {
            let (idx, _field) = schema.column_with_name(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("Column not found: {}", name))
            })?;
            // FieldId is usually not available in Arrow Schema directly as metadata unless we put it there.
            // But ScalarEvaluator uses (table_idx, field_idx) where field_idx is the index in the array list.
            // So we can just use  as FieldId.
            Ok(ScalarExpr::Column((0, idx as u32)))
        }
        ScalarExpr::Literal(l) => Ok(ScalarExpr::Literal(l.clone())),
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(resolve_scalar_expr_string(left, schema, subquery_results)?),
            op: *op,
            right: Box::new(resolve_scalar_expr_string(right, schema, subquery_results)?),
        }),
        ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(resolve_scalar_expr_string(e, schema, subquery_results)?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(resolve_scalar_expr_string(expr, schema, subquery_results)?),
            negated: *negated,
        }),
        ScalarExpr::Aggregate(_) => Err(Error::Internal("Nested aggregates not supported in resolution".into())),
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(resolve_scalar_expr_string(base, schema, subquery_results)?),
            field_name: field_name.clone(),
        }),
        ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
            expr: Box::new(resolve_scalar_expr_string(expr, schema, subquery_results)?),
            data_type: data_type.clone(),
        }),
        ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(resolve_scalar_expr_string(left, schema, subquery_results)?),
            op: *op,
            right: Box::new(resolve_scalar_expr_string(right, schema, subquery_results)?),
        }),
        ScalarExpr::Coalesce(exprs) => {
            let mut resolved = Vec::new();
            for e in exprs {
                resolved.push(resolve_scalar_expr_string(e, schema, subquery_results)?);
            }
            Ok(ScalarExpr::Coalesce(resolved))
        }
        ScalarExpr::ScalarSubquery(s) => {
            if let Some(lit) = subquery_results.get(&s.id) {
                Ok(ScalarExpr::Literal(lit.clone()))
            } else {
                Ok(ScalarExpr::ScalarSubquery(s.clone()))
            }
        }
        ScalarExpr::Case { operand, branches, else_expr } => {
            let op = if let Some(o) = operand {
                Some(Box::new(resolve_scalar_expr_string(o, schema, subquery_results)?))
            } else {
                None
            };
            let mut br = Vec::new();
            for (w, t) in branches {
                br.push((resolve_scalar_expr_string(w, schema, subquery_results)?, resolve_scalar_expr_string(t, schema, subquery_results)?));
            }
            let el = if let Some(e) = else_expr {
                Some(Box::new(resolve_scalar_expr_string(e, schema, subquery_results)?))
            } else {
                None
            };
            Ok(ScalarExpr::Case { operand: op, branches: br, else_expr: el })
        }
        ScalarExpr::Random => Ok(ScalarExpr::Random),
    }
}


use arrow::datatypes::DataType;

fn infer_type(
    expr: &ScalarExpr<(usize, FieldId)>, 
    schema: &Schema,
    col_mapping: &FxHashMap<(usize, FieldId), usize>
) -> Result<DataType, Error> {
    let res = match expr {
        ScalarExpr::Column(id) => {
            let idx = col_mapping.get(id).ok_or_else(|| Error::Internal("Column not found in mapping".into()))?;
            Ok(schema.field(*idx).data_type().clone())
        },
        ScalarExpr::Literal(l) => {
             match l {
                 Literal::Int128(_) => Ok(DataType::Int64),
                 Literal::Float64(_) => Ok(DataType::Float64),
                 Literal::Boolean(_) => Ok(DataType::Boolean),
                 Literal::String(_) => Ok(DataType::Utf8),
                 Literal::Null => Ok(DataType::Null),
                 Literal::Decimal128(d) => Ok(DataType::Decimal128(d.precision(), d.scale())),
                 _ => Ok(DataType::Int64),
             }
        },
        ScalarExpr::Binary { left, op: _, right } => {
            let l = infer_type(left, schema, col_mapping)?;
            let r = infer_type(right, schema, col_mapping)?;
            if matches!(l, DataType::Float64 | DataType::Float32) || matches!(r, DataType::Float64 | DataType::Float32) {
                Ok(DataType::Float64)
            } else if matches!(l, DataType::Decimal128(..)) || matches!(r, DataType::Decimal128(..)) {
                 // Simplified decimal inference
                 Ok(DataType::Decimal128(38, 10)) 
            } else {
                Ok(DataType::Int64)
            }
        },
        ScalarExpr::Cast { data_type, .. } => Ok(data_type.clone()),
        ScalarExpr::Case { else_expr, branches, .. } => {
            let mut types = Vec::new();
            for (_, t) in branches {
                types.push(infer_type(t, schema, col_mapping)?);
            }
            if let Some(e) = else_expr {
                types.push(infer_type(e, schema, col_mapping)?);
            }

            if let Some(non_null) = types.iter().find(|t| **t != DataType::Null) {
                Ok(non_null.clone())
            } else {
                Ok(DataType::Null)
            }
        },
        ScalarExpr::Coalesce(items) => {
            if let Some(first) = items.first() {
                infer_type(first, schema, col_mapping)
            } else {
                Ok(DataType::Null)
            }
        },
        ScalarExpr::Not(_) | ScalarExpr::IsNull { .. } | ScalarExpr::Compare { .. } => Ok(DataType::Boolean),
        _ => Ok(DataType::Int64),
    };
    res
}

fn map_expr_to_names(
    expr: &ScalarExpr<(usize, FieldId)>,
    col_mapping: &FxHashMap<(usize, FieldId), usize>,
    schema: &Schema,
) -> Result<ScalarExpr<String>, Error> {
    match expr {
        ScalarExpr::Column(id) => {
            let idx = col_mapping.get(id).ok_or_else(|| {
                Error::Internal(format!("Column {:?} not found in mapping", id))
            })?;
            let field = schema.field(*idx);
            Ok(ScalarExpr::Column(field.name().clone()))
        }
        ScalarExpr::Literal(l) => Ok(ScalarExpr::Literal(l.clone())),
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(map_expr_to_names(left, col_mapping, schema)?),
            op: *op,
            right: Box::new(map_expr_to_names(right, col_mapping, schema)?),
        }),
        ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(map_expr_to_names(e, col_mapping, schema)?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
            negated: *negated,
        }),
        ScalarExpr::Aggregate(call) => {
             // Recurse into aggregate arguments
             let new_call = match call {
                 AggregateCall::CountStar => AggregateCall::CountStar,
                 AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
                 AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
                 AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
                 AggregateCall::Min(expr) => AggregateCall::Min(Box::new(map_expr_to_names(expr, col_mapping, schema)?)),
                 AggregateCall::Max(expr) => AggregateCall::Max(Box::new(map_expr_to_names(expr, col_mapping, schema)?)),
                 AggregateCall::CountNulls(expr) => AggregateCall::CountNulls(Box::new(map_expr_to_names(expr, col_mapping, schema)?)),
                 AggregateCall::GroupConcat { expr, distinct, separator } => AggregateCall::GroupConcat {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                     separator: separator.clone(),
                 },
                 AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
             };
             Ok(ScalarExpr::Aggregate(new_call))
        },
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(map_expr_to_names(base, col_mapping, schema)?),
            field_name: field_name.clone(),
        }),
        ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
            expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
            data_type: data_type.clone(),
        }),
        ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(map_expr_to_names(left, col_mapping, schema)?),
            op: *op,
            right: Box::new(map_expr_to_names(right, col_mapping, schema)?),
        }),
        ScalarExpr::Coalesce(exprs) => {
            let mut mapped = Vec::new();
            for e in exprs {
                mapped.push(map_expr_to_names(e, col_mapping, schema)?);
            }
            Ok(ScalarExpr::Coalesce(mapped))
        },
        ScalarExpr::ScalarSubquery(s) => Ok(ScalarExpr::ScalarSubquery(s.clone())),
        ScalarExpr::Case { operand, branches, else_expr } => {
            let op = if let Some(o) = operand {
                Some(Box::new(map_expr_to_names(o, col_mapping, schema)?))
            } else {
                None
            };
            let mut br = Vec::new();
            for (w, t) in branches {
                br.push((map_expr_to_names(w, col_mapping, schema)?, map_expr_to_names(t, col_mapping, schema)?));
            }
            let el = if let Some(e) = else_expr {
                Some(Box::new(map_expr_to_names(e, col_mapping, schema)?))
            } else {
                None
            };
            Ok(ScalarExpr::Case { operand: op, branches: br, else_expr: el })
        },
        ScalarExpr::Random => Ok(ScalarExpr::Random),
    }
}

use llkv_plan::logical_planner::ResolvedFieldRef;

fn convert_resolved_expr(expr: &ScalarExpr<ResolvedFieldRef>) -> ScalarExpr<(usize, FieldId)> {
    match expr {
        ScalarExpr::Column(r) => ScalarExpr::Column((r.table_index, r.logical_field_id.field_id())),
        ScalarExpr::Literal(l) => ScalarExpr::Literal(l.clone()),
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(convert_resolved_expr(left)),
            op: *op,
            right: Box::new(convert_resolved_expr(right)),
        },
        ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(convert_resolved_expr(e))),
        ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
            expr: Box::new(convert_resolved_expr(expr)),
            negated: *negated,
        },
        ScalarExpr::Aggregate(call) => {
             let new_call = match call {
                 AggregateCall::CountStar => AggregateCall::CountStar,
                 AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
                 AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
                 AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
                 AggregateCall::Min(expr) => AggregateCall::Min(Box::new(convert_resolved_expr(expr))),
                 AggregateCall::Max(expr) => AggregateCall::Max(Box::new(convert_resolved_expr(expr))),
                 AggregateCall::CountNulls(expr) => AggregateCall::CountNulls(Box::new(convert_resolved_expr(expr))),
                 AggregateCall::GroupConcat { expr, distinct, separator } => AggregateCall::GroupConcat {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                     separator: separator.clone(),
                 },
                 AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
             };
             ScalarExpr::Aggregate(new_call)
        },
        ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
            base: Box::new(convert_resolved_expr(base)),
            field_name: field_name.clone(),
        },
        ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
            expr: Box::new(convert_resolved_expr(expr)),
            data_type: data_type.clone(),
        },
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(convert_resolved_expr(left)),
            op: *op,
            right: Box::new(convert_resolved_expr(right)),
        },
        ScalarExpr::Coalesce(exprs) => ScalarExpr::Coalesce(
            exprs.iter().map(|e| convert_resolved_expr(e)).collect()
        ),
        ScalarExpr::ScalarSubquery(s) => ScalarExpr::ScalarSubquery(s.clone()),
        ScalarExpr::Case { operand, branches, else_expr } => ScalarExpr::Case {
            operand: operand.as_ref().map(|e| Box::new(convert_resolved_expr(e))),
            branches: branches.iter().map(|(w, t)| (convert_resolved_expr(w), convert_resolved_expr(t))).collect(),
            else_expr: else_expr.as_ref().map(|e| Box::new(convert_resolved_expr(e))),
        },
        ScalarExpr::Random => ScalarExpr::Random,
    }
}

struct SortStream {
    schema: SchemaRef,
    input: BatchIter,
    sort_columns: Vec<(usize, bool, bool)>, // index, asc, nulls_first
    executed: bool,
}

impl SortStream {
    fn new(schema: SchemaRef, input: BatchIter, sort_columns: Vec<(usize, bool, bool)>) -> Self {
        Self { schema, input, sort_columns, executed: false }
    }
}

impl Iterator for SortStream {
    type Item = ExecutorResult<RecordBatch>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.executed {
            return None;
        }
        self.executed = true;
        
        let mut batches = Vec::new();
        while let Some(res) = self.input.next() {
            match res {
                Ok(batch) => batches.push(batch),
                Err(e) => return Some(Err(e)),
            }
        }
        
        if batches.is_empty() {
            return None;
        }
        
        let batch = match arrow::compute::concat_batches(&self.schema, &batches) {
            Ok(b) => b,
            Err(e) => return Some(Err(Error::Internal(e.to_string()))),
        };
        
        let mut sort_reqs = Vec::new();
        for (idx, asc, nulls_first) in &self.sort_columns {
            if *idx >= batch.num_columns() {
                return Some(Err(Error::Internal(format!("Sort column index {} out of bounds (num columns: {})", idx, batch.num_columns()))));
            }
            sort_reqs.push(arrow::compute::SortColumn {
                values: batch.column(*idx).clone(),
                options: Some(arrow::compute::SortOptions {
                    descending: !*asc,
                    nulls_first: *nulls_first,
                }),
            });
        }
        
        let indices = match arrow::compute::lexsort_to_indices(&sort_reqs, None) {
            Ok(i) => i,
            Err(e) => return Some(Err(Error::Internal(e.to_string()))),
        };
        
        let columns = batch.columns().iter().map(|col| {
            arrow::compute::take(col, &indices, None).map_err(|e| Error::Internal(e.to_string()))
        }).collect::<ExecutorResult<Vec<_>>>();
        
        match columns {
            Ok(cols) => match RecordBatch::try_new(self.schema.clone(), cols) {
                Ok(b) => Some(Ok(b)),
                Err(e) => Some(Err(Error::Internal(e.to_string()))),
            },
            Err(e) => Some(Err(e)),
        }
    }
}
