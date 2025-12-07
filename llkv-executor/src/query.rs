use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, mpsc};

use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::compute::{concat_batches, filter_record_batch};
use arrow::datatypes::DataType;

impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn execute_complex_aggregation(
        &self,
        pre_agg_iter: BatchIter,
        pre_agg_schema: SchemaRef,
        aggregates: Vec<AggregateExpr>,
        final_exprs: Vec<ScalarExpr<String>>,
        final_names: Vec<String>,
        plan: &SelectPlan,
        table_name: String,
        subquery_results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
        key_indices: Vec<usize>,
        having_expr: Option<ScalarExpr<String>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let mut plan_columns = Vec::new();
        let mut name_to_index = FxHashMap::default();
        for (i, field) in pre_agg_schema.fields().iter().enumerate() {
            let fid = field
                .metadata()
                .get("field_id")
                .and_then(|s| s.parse::<FieldId>().ok())
                .unwrap_or(i as FieldId);

            plan_columns.push(PlanColumn {
                name: field.name().clone(),
                data_type: field.data_type().clone(),
                field_id: fid,
                is_nullable: field.is_nullable(),
                is_primary_key: false,
                is_unique: false,
                default_value: None,
                check_expr: None,
            });
            name_to_index.insert(field.name().to_ascii_lowercase(), i);
        }
        let logical_schema = PlanSchema {
            columns: plan_columns,
            name_to_index,
        };

        let mut agg_plan = plan.clone();
        agg_plan.aggregates = aggregates;

        let (mut agg_iter, agg_schema): (BatchIter, SchemaRef) = if key_indices.is_empty() {
            let stream =
                AggregateStream::new(pre_agg_iter, &agg_plan, &logical_schema, &pre_agg_schema)?;
            let schema = stream.schema();
            (Box::new(stream), schema)
        } else {
            let stream = GroupedAggregateStream::new(
                pre_agg_iter,
                key_indices,
                &agg_plan,
                &logical_schema,
                &pre_agg_schema,
            )?;
            let schema = stream.schema();
            (Box::new(stream.map(|b| b)), schema)
        };

        let agg_count = agg_plan.aggregates.len();
        let agg_offset = agg_schema
            .fields()
            .len()
            .saturating_sub(agg_count);

        if let Some(having_expr) = having_expr {
            let having_expr_captured =
                resolve_scalar_expr_string(&having_expr, &agg_schema, subquery_results)?;
            let mut next_agg_idx = 0;
            let having_expr_rewritten = Self::rewrite_aggregate_refs(
                having_expr_captured,
                agg_offset,
                agg_count,
                &mut next_agg_idx,
            )?;
            agg_iter = Box::new(agg_iter.map(move |batch_res| {
                let batch = batch_res?;

                let mut field_arrays = FxHashMap::default();
                for (i, col) in batch.columns().iter().enumerate() {
                    field_arrays.insert((0, i as u32), col.clone());
                }
                let numeric_arrays =
                    ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());

                let filter_value = ScalarEvaluator::evaluate_batch_simplified(
                    &having_expr_rewritten,
                    batch.num_rows(),
                    &numeric_arrays,
                )?;

                let bool_arr = filter_value
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        Error::Internal(
                            "HAVING expression must evaluate to a boolean array".to_string(),
                        )
                    })?;

                filter_record_batch(&batch, bool_arr)
                    .map_err(|e| Error::Internal(format!("HAVING filter failed: {e}")))
            }));
        }

        let mut final_output_fields = Vec::new();
        let mut col_mapping = FxHashMap::default();
        for (i, _field) in agg_schema.fields().iter().enumerate() {
            col_mapping.insert((0, i as u32), i);
        }

        let mut resolved_final_exprs = Vec::new();

        for (expr_str, name) in final_exprs.iter().zip(final_names.iter()) {
            let expr = resolve_scalar_expr_string(expr_str, &agg_schema, subquery_results)?;
            let mut next_agg_idx = 0;
            let rewritten = Self::rewrite_aggregate_refs(
                expr,
                agg_offset,
                agg_count,
                &mut next_agg_idx,
            )?;
            let dt = infer_type(&rewritten, &agg_schema, &col_mapping)
                .unwrap_or(arrow::datatypes::DataType::Int64);
            final_output_fields.push(ArrowField::new(name, dt, true));
            resolved_final_exprs.push(rewritten);
        }
        let final_output_schema = Arc::new(Schema::new(final_output_fields));
        let final_output_schema_captured = final_output_schema.clone();
        let resolved_final_exprs_captured = resolved_final_exprs.clone();

        let final_stream = agg_iter.map(move |batch_res| {
            let batch = batch_res?;
            let mut columns = Vec::new();

            let mut field_arrays = FxHashMap::default();
            for (i, col) in batch.columns().iter().enumerate() {
                field_arrays.insert((0, i as u32), col.clone());
            }
            let numeric_arrays =
                ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());

            for expr in &resolved_final_exprs_captured {
                let result = ScalarEvaluator::evaluate_batch_simplified(
                    &expr,
                    batch.num_rows(),
                    &numeric_arrays,
                )?;
                columns.push(result);
            }

            RecordBatch::try_new(final_output_schema_captured.clone(), columns)
                .map_err(|e| Error::Internal(e.to_string()))
        });

        let mut iter: BatchIter = Box::new(final_stream);
        if plan.distinct {
            iter = Box::new(DistinctStream::new(final_output_schema.clone(), iter)?);
        }

        let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);

        Ok(SelectExecution::from_stream(
            table_name,
            final_output_schema,
            trimmed,
        ))
    }
}

use arrow::datatypes::{Field as ArrowField, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{OwnedRow, RowConverter, SortField};
use llkv_aggregate::{AggregateStream, GroupedAggregateStream};
use llkv_column_map::gather::gather_optional_projected_indices_from_batches;
use llkv_column_map::store::Projection;
use llkv_compute::eval::ScalarEvaluator;
use llkv_expr::AggregateCall;
use llkv_expr::expr::{CompareOp, Expr as LlkvExpr, ScalarExpr, SubqueryId};
use llkv_expr::literal::Literal;
use llkv_expr::{BinaryOp, Expr, Filter, InList, Operator};
use llkv_join::JoinType;
use llkv_plan::logical_planner::{LogicalPlan, ResolvedJoin};
use llkv_plan::plans::AggregateExpr;
use llkv_plan::plans::FilterSubquery;
use llkv_plan::plans::JoinPlan;
use llkv_plan::plans::{CompoundOperator, CompoundQuantifier};
use llkv_plan::prepared::{
    DefaultPreparedSelectPlanner, PreparedCompoundSelect, PreparedScalarSubquery,
    PreparedSelectPlan, PreparedSelectPlanner,
};
use llkv_plan::physical::table::{ExecutionTable, TableProvider};
use llkv_plan::schema::{PlanColumn, PlanSchema};
use llkv_plan::SelectPlan;
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
    provider: Arc<dyn ExecutorTableProvider<P>>,
    planner: Arc<dyn PreparedSelectPlanner<P>>,
}

impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(
        provider: Arc<dyn ExecutorTableProvider<P>>,
        planner: Arc<dyn PreparedSelectPlanner<P>>,
    ) -> Self {
        Self { provider, planner }
    }

    /// Convenience constructor that wires the default planner to the executor provider.
    pub fn with_default_planner(provider: Arc<dyn ExecutorTableProvider<P>>) -> Self {
        let planner_provider = Arc::new(PlannerTableProvider { inner: provider.clone() });
        let planner: Arc<dyn PreparedSelectPlanner<P>> =
            Arc::new(DefaultPreparedSelectPlanner::new(planner_provider));
        Self::new(provider, planner)
    }

    fn scalar_contains_aggregate<F>(expr: &ScalarExpr<F>) -> bool {
        match expr {
            ScalarExpr::Aggregate(_) => true,
            ScalarExpr::Binary { left, right, .. }
            | ScalarExpr::Compare { left, right, .. } => {
                Self::scalar_contains_aggregate(left) || Self::scalar_contains_aggregate(right)
            }
            ScalarExpr::Not(e)
            | ScalarExpr::Cast { expr: e, .. }
            | ScalarExpr::IsNull { expr: e, .. }
            | ScalarExpr::GetField { base: e, .. } => Self::scalar_contains_aggregate(e),
            ScalarExpr::Coalesce(items) => items.iter().any(Self::scalar_contains_aggregate),
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                operand
                    .as_ref()
                    .map_or(false, |o| Self::scalar_contains_aggregate(o))
                    || branches.iter().any(|(w, t)| {
                        Self::scalar_contains_aggregate(w) || Self::scalar_contains_aggregate(t)
                    })
                    || else_expr
                        .as_ref()
                        .map_or(false, |e| Self::scalar_contains_aggregate(e))
            }
            _ => false,
        }
    }

    fn contains_aggregate_in_expr(expr: &Expr<'_, String>) -> bool {
        match expr {
            Expr::Pred(_) | Expr::Literal(_) | Expr::Exists(_) => false,
            Expr::Compare { left, right, .. } => {
                Self::scalar_contains_aggregate(left) || Self::scalar_contains_aggregate(right)
            }
            Expr::InList { expr, list, .. } => {
                Self::scalar_contains_aggregate(expr)
                    || list.iter().any(Self::scalar_contains_aggregate)
            }
            Expr::IsNull { expr, .. } => Self::scalar_contains_aggregate(expr),
            Expr::And(list) | Expr::Or(list) => list.iter().any(Self::contains_aggregate_in_expr),
            Expr::Not(inner) => Self::contains_aggregate_in_expr(inner),
        }
    }

    fn rewrite_aggregate_refs(
        expr: ScalarExpr<(usize, FieldId)>,
        agg_offset: usize,
        agg_count: usize,
        next_agg_idx: &mut usize,
    ) -> ExecutorResult<ScalarExpr<(usize, FieldId)>> {
        let rewritten = match expr {
            ScalarExpr::Aggregate(_) => {
                let current = *next_agg_idx;
                if current >= agg_count {
                    return Err(Error::Internal(
                        "Aggregate reference exceeds computed aggregates".into(),
                    ));
                }
                *next_agg_idx += 1;
                ScalarExpr::Column((0, (agg_offset + current) as FieldId))
            }
            ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
                left: Box::new(Self::rewrite_aggregate_refs(
                    *left,
                    agg_offset,
                    agg_count,
                    next_agg_idx,
                )?),
                op,
                right: Box::new(Self::rewrite_aggregate_refs(
                    *right,
                    agg_offset,
                    agg_count,
                    next_agg_idx,
                )?),
            },
            ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(Self::rewrite_aggregate_refs(
                *e,
                agg_offset,
                agg_count,
                next_agg_idx,
            )?)),
            ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
                expr: Box::new(Self::rewrite_aggregate_refs(
                    *expr,
                    agg_offset,
                    agg_count,
                    next_agg_idx,
                )?),
                negated,
            },
            ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
                expr: Box::new(Self::rewrite_aggregate_refs(
                    *expr,
                    agg_offset,
                    agg_count,
                    next_agg_idx,
                )?),
                data_type,
            },
            ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
                left: Box::new(Self::rewrite_aggregate_refs(
                    *left,
                    agg_offset,
                    agg_count,
                    next_agg_idx,
                )?),
                op,
                right: Box::new(Self::rewrite_aggregate_refs(
                    *right,
                    agg_offset,
                    agg_count,
                    next_agg_idx,
                )?),
            },
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand_rewritten = if let Some(op) = operand {
                    Some(Box::new(Self::rewrite_aggregate_refs(
                        *op,
                        agg_offset,
                        agg_count,
                        next_agg_idx,
                    )?))
                } else {
                    None
                };

                let rewritten_branches = branches
                    .into_iter()
                    .map(|(w, t)| {
                        Ok((
                            Self::rewrite_aggregate_refs(w, agg_offset, agg_count, next_agg_idx)?,
                            Self::rewrite_aggregate_refs(t, agg_offset, agg_count, next_agg_idx)?,
                        ))
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?;

                let else_rewritten = if let Some(e) = else_expr {
                    Some(Box::new(Self::rewrite_aggregate_refs(
                        *e,
                        agg_offset,
                        agg_count,
                        next_agg_idx,
                    )?))
                } else {
                    None
                };

                ScalarExpr::Case {
                    operand: operand_rewritten,
                    branches: rewritten_branches,
                    else_expr: else_rewritten,
                }
            }
            ScalarExpr::Coalesce(items) => ScalarExpr::Coalesce(
                items
                    .into_iter()
                    .map(|e| Self::rewrite_aggregate_refs(e, agg_offset, agg_count, next_agg_idx))
                    .collect::<ExecutorResult<Vec<_>>>()?,
            ),
            other => other,
        };

        Ok(rewritten)
    }

    fn get_scalar(array: &ArrayRef, index: usize) -> ExecutorResult<Literal> {
        if array.is_null(index) {
            return Ok(Literal::Null);
        }
        match array.data_type() {
            arrow::datatypes::DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                Ok(Literal::Boolean(arr.value(index)))
            }
            arrow::datatypes::DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            arrow::datatypes::DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                Ok(Literal::Float64(arr.value(index)))
            }
            arrow::datatypes::DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap();
                Ok(Literal::String(arr.value(index).to_string()))
            }
            arrow::datatypes::DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::LargeStringArray>()
                    .unwrap();
                Ok(Literal::String(arr.value(index).to_string()))
            }
            arrow::datatypes::DataType::Null => Ok(Literal::Null),
            _ => Err(Error::Internal(format!(
                "Unsupported scalar subquery return type: {:?}",
                array.data_type()
            ))),
        }
    }

    fn rewrite_expr_subqueries_with_exec(
        &self,
        expr: &Expr<'_, String>,
        prepared_subqueries: &[PreparedScalarSubquery<P>],
        cache: &mut FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
    ) -> ExecutorResult<Expr<'static, String>> {
        match expr {
            Expr::And(list) => Ok(Expr::And(
                list.iter()
                    .map(|e| {
                        self.rewrite_expr_subqueries_with_exec(
                            e,
                            prepared_subqueries,
                            cache,
                        )
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
            )),
            Expr::Or(list) => Ok(Expr::Or(
                list.iter()
                    .map(|e| {
                        self.rewrite_expr_subqueries_with_exec(
                            e,
                            prepared_subqueries,
                            cache,
                        )
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
            )),
            Expr::Not(e) => Ok(Expr::Not(Box::new(
                self.rewrite_expr_subqueries_with_exec(e, prepared_subqueries, cache)?,
            ))),
            Expr::Pred(f) => {
                let op = match &f.op {
                    Operator::Equals(l) => Operator::Equals(l.clone()),
                    Operator::GreaterThan(l) => Operator::GreaterThan(l.clone()),
                    Operator::GreaterThanOrEquals(l) => Operator::GreaterThanOrEquals(l.clone()),
                    Operator::LessThan(l) => Operator::LessThan(l.clone()),
                    Operator::LessThanOrEquals(l) => Operator::LessThanOrEquals(l.clone()),
                    Operator::Range { lower, upper } => Operator::Range {
                        lower: lower.clone(),
                        upper: upper.clone(),
                    },
                    Operator::IsNull => Operator::IsNull,
                    Operator::IsNotNull => Operator::IsNotNull,
                    Operator::StartsWith {
                        pattern,
                        case_sensitive,
                    } => Operator::StartsWith {
                        pattern: pattern.clone(),
                        case_sensitive: *case_sensitive,
                    },
                    Operator::EndsWith {
                        pattern,
                        case_sensitive,
                    } => Operator::EndsWith {
                        pattern: pattern.clone(),
                        case_sensitive: *case_sensitive,
                    },
                    Operator::Contains {
                        pattern,
                        case_sensitive,
                    } => Operator::Contains {
                        pattern: pattern.clone(),
                        case_sensitive: *case_sensitive,
                    },
                    Operator::In(list) => {
                        let owned: Vec<Literal> = list.iter().cloned().collect();
                        Operator::In(InList::shared(owned))
                    }
                };
                Ok(Expr::Pred(llkv_expr::Filter {
                    field_id: f.field_id.clone(),
                    op,
                }))
            }
            Expr::Compare { left, op, right } => Ok(Expr::Compare {
                left: self.rewrite_scalar_subqueries_with_exec(
                    left,
                    prepared_subqueries,
                    cache,
                )?,
                op: *op,
                right: self.rewrite_scalar_subqueries_with_exec(
                    right,
                    prepared_subqueries,
                    cache,
                )?,
            }),
            Expr::InList {
                expr,
                list,
                negated,
            } => Ok(Expr::InList {
                expr: self.rewrite_scalar_subqueries_with_exec(
                    expr,
                    prepared_subqueries,
                    cache,
                )?,
                list: list
                    .iter()
                    .map(|e| {
                        self.rewrite_scalar_subqueries_with_exec(
                            e,
                            prepared_subqueries,
                            cache,
                        )
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
                negated: *negated,
            }),
            Expr::IsNull { expr, negated } => Ok(Expr::IsNull {
                expr: self.rewrite_scalar_subqueries_with_exec(
                    expr,
                    prepared_subqueries,
                    cache,
                )?,
                negated: *negated,
            }),
            Expr::Literal(b) => Ok(Expr::Literal(*b)),
            Expr::Exists(s) => Ok(Expr::Exists(s.clone())),
        }
    }

    /// Rewrite scalar subqueries to literals, executing any missing prepared subqueries on-demand.
    fn rewrite_scalar_subqueries_with_exec<F: Clone>(
        &self,
        expr: &ScalarExpr<F>,
        prepared_subqueries: &[PreparedScalarSubquery<P>],
        cache: &mut FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
    ) -> ExecutorResult<ScalarExpr<F>> {
        match expr {
            ScalarExpr::ScalarSubquery(s) => {
                if let Some(lit) = cache.get(&s.id) {
                    return Ok(ScalarExpr::Literal(lit.clone()));
                }

                if let Some(def) = prepared_subqueries.iter().find(|p| p.id == s.id) {
                    if let Some(plan) = &def.prepared_plan {
                        let exec = self.execute_prepared_select(plan)?;
                        let batches = exec.collect()?;

                        let mut result_val = None;
                        for b in batches {
                            if b.num_rows() > 0 {
                                if result_val.is_some() || b.num_rows() > 1 {
                                    return Err(Error::Internal(
                                        "Scalar subquery returned more than one row".into(),
                                    ));
                                }
                                if b.num_columns() != 1 {
                                    return Err(Error::Internal(
                                        "Scalar subquery returned more than one column".into(),
                                    ));
                                }
                                let col = b.column(0);
                                let scalar = Self::get_scalar(col, 0)?;
                                result_val = Some(scalar);
                            }
                        }

                        let lit = result_val.unwrap_or(Literal::Null);
                        cache.insert(s.id, lit.clone());
                        return Ok(ScalarExpr::Literal(lit));
                    }
                }

                Ok(ScalarExpr::ScalarSubquery(s.clone()))
            }
            ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
                left: Box::new(self.rewrite_scalar_subqueries_with_exec(
                    left,
                    prepared_subqueries,
                    cache,
                )?),
                op: *op,
                right: Box::new(self.rewrite_scalar_subqueries_with_exec(
                    right,
                    prepared_subqueries,
                    cache,
                )?),
            }),
            ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(
                self.rewrite_scalar_subqueries_with_exec(e, prepared_subqueries, cache)?,
            ))),
            ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
                expr: Box::new(self.rewrite_scalar_subqueries_with_exec(
                    expr,
                    prepared_subqueries,
                    cache,
                )?),
                negated: *negated,
            }),
            ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
                expr: Box::new(self.rewrite_scalar_subqueries_with_exec(
                    expr,
                    prepared_subqueries,
                    cache,
                )?),
                data_type: data_type.clone(),
            }),
            ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
                left: Box::new(self.rewrite_scalar_subqueries_with_exec(
                    left,
                    prepared_subqueries,
                    cache,
                )?),
                op: *op,
                right: Box::new(self.rewrite_scalar_subqueries_with_exec(
                    right,
                    prepared_subqueries,
                    cache,
                )?),
            }),
            ScalarExpr::Coalesce(exprs) => Ok(ScalarExpr::Coalesce(
                exprs
                    .iter()
                    .map(|e| {
                        self.rewrite_scalar_subqueries_with_exec(
                            e,
                            prepared_subqueries,
                            cache,
                        )
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
            )),
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => Ok(ScalarExpr::Case {
                operand: operand
                    .as_ref()
                    .map(|o| {
                        self.rewrite_scalar_subqueries_with_exec(
                            o,
                            prepared_subqueries,
                            cache,
                        )
                    })
                    .transpose()?
                    .map(Box::new),
                branches: branches
                    .iter()
                    .map(|(w, t)| {
                        Ok((
                            self.rewrite_scalar_subqueries_with_exec(
                                w,
                                prepared_subqueries,
                                cache,
                            )?,
                            self.rewrite_scalar_subqueries_with_exec(
                                t,
                                prepared_subqueries,
                                cache,
                            )?,
                        ))
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
                else_expr: else_expr
                    .as_ref()
                    .map(|e| {
                        self.rewrite_scalar_subqueries_with_exec(
                            e,
                            prepared_subqueries,
                            cache,
                        )
                    })
                    .transpose()?
                    .map(Box::new),
            }),
            ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
                base: Box::new(self.rewrite_scalar_subqueries_with_exec(
                    base,
                    prepared_subqueries,
                    cache,
                )?),
                field_name: field_name.clone(),
            }),
            _ => Ok(expr.clone()),
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
                        return Err(Error::Internal(
                            "Scalar subquery returned more than one row".into(),
                        ));
                    }
                    if batch.num_columns() != 1 {
                        return Err(Error::Internal(
                            "Scalar subquery returned more than one column".into(),
                        ));
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

    fn execute_prepared_scalar_subqueries(
        &self,
        subqueries: &[PreparedScalarSubquery<P>],
    ) -> ExecutorResult<FxHashMap<SubqueryId, Literal>> {
        let mut results = FxHashMap::default();
        for sub in subqueries {
            let Some(plan) = &sub.prepared_plan else { continue };
            let execution = self.execute_prepared_select(plan)?;
            let batches = execution.collect()?;

            let mut result_val = None;
            for batch in batches {
                if batch.num_rows() > 0 {
                    if result_val.is_some() || batch.num_rows() > 1 {
                        return Err(Error::Internal(
                            "Scalar subquery returned more than one row".into(),
                        ));
                    }
                    if batch.num_columns() != 1 {
                        return Err(Error::Internal(
                            "Scalar subquery returned more than one column".into(),
                        ));
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

    fn rewrite_expr_placeholders(
        expr: &Expr<'static, String>,
        replacements: &HashMap<String, Literal>,
    ) -> Expr<'static, String> {
        match expr {
            Expr::And(l) => Expr::And(
                l.iter()
                    .map(|e| Self::rewrite_expr_placeholders(e, replacements))
                    .collect(),
            ),
            Expr::Or(l) => Expr::Or(
                l.iter()
                    .map(|e| Self::rewrite_expr_placeholders(e, replacements))
                    .collect(),
            ),
            Expr::Not(e) => Expr::Not(Box::new(Self::rewrite_expr_placeholders(e, replacements))),
            Expr::Pred(f) => Expr::Pred(f.clone()),
            Expr::Compare { left, op, right } => Expr::Compare {
                left: Self::rewrite_scalar_expr_placeholders(left.clone(), replacements),
                op: *op,
                right: Self::rewrite_scalar_expr_placeholders(right.clone(), replacements),
            },
            Expr::InList {
                expr,
                list,
                negated,
            } => Expr::InList {
                expr: Self::rewrite_scalar_expr_placeholders(expr.clone(), replacements),
                list: list
                    .iter()
                    .map(|e| Self::rewrite_scalar_expr_placeholders(e.clone(), replacements))
                    .collect(),
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
            ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(
                Self::rewrite_scalar_expr_placeholders(*e, replacements),
            )),
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
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => ScalarExpr::Case {
                operand: operand
                    .map(|o| Box::new(Self::rewrite_scalar_expr_placeholders(*o, replacements))),
                branches: branches
                    .into_iter()
                    .map(|(w, t)| {
                        (
                            Self::rewrite_scalar_expr_placeholders(w, replacements),
                            Self::rewrite_scalar_expr_placeholders(t, replacements),
                        )
                    })
                    .collect(),
                else_expr: else_expr
                    .map(|e| Box::new(Self::rewrite_scalar_expr_placeholders(*e, replacements))),
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
                sub.plan = Box::new(Self::rewrite_select_plan_placeholders(
                    &sub.plan,
                    replacements,
                ));
            }
        }

        new_plan.projections = new_plan
            .projections
            .into_iter()
            .map(|proj| match proj {
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
            })
            .collect();

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
            Expr::InList {
                expr,
                list,
                negated,
            } => {
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
            ScalarExpr::Aggregate(call) => Ok(ScalarExpr::Aggregate(match call {
                AggregateCall::CountStar => AggregateCall::CountStar,
                AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                    expr: Box::new(Self::remap_string_expr_to_indices(
                        expr,
                        schema,
                        name_overrides,
                    )?),
                    distinct: *distinct,
                },
                AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                    expr: Box::new(Self::remap_string_expr_to_indices(
                        expr,
                        schema,
                        name_overrides,
                    )?),
                    distinct: *distinct,
                },
                AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                    expr: Box::new(Self::remap_string_expr_to_indices(
                        expr,
                        schema,
                        name_overrides,
                    )?),
                    distinct: *distinct,
                },
                AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                    expr: Box::new(Self::remap_string_expr_to_indices(
                        expr,
                        schema,
                        name_overrides,
                    )?),
                    distinct: *distinct,
                },
                AggregateCall::Min(expr) => AggregateCall::Min(Box::new(
                    Self::remap_string_expr_to_indices(expr, schema, name_overrides)?,
                )),
                AggregateCall::Max(expr) => AggregateCall::Max(Box::new(
                    Self::remap_string_expr_to_indices(expr, schema, name_overrides)?,
                )),
                AggregateCall::CountNulls(expr) => AggregateCall::CountNulls(Box::new(
                    Self::remap_string_expr_to_indices(expr, schema, name_overrides)?,
                )),
                AggregateCall::GroupConcat {
                    expr,
                    distinct,
                    separator,
                } => AggregateCall::GroupConcat {
                    expr: Box::new(Self::remap_string_expr_to_indices(
                        expr,
                        schema,
                        name_overrides,
                    )?),
                    distinct: *distinct,
                    separator: separator.clone(),
                },
            })),
            ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
                left: Box::new(Self::remap_string_expr_to_indices(
                    left,
                    schema,
                    name_overrides,
                )?),
                op: *op,
                right: Box::new(Self::remap_string_expr_to_indices(
                    right,
                    schema,
                    name_overrides,
                )?),
            }),
            ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(
                Self::remap_string_expr_to_indices(e, schema, name_overrides)?,
            ))),
            ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
                expr: Box::new(Self::remap_string_expr_to_indices(
                    expr,
                    schema,
                    name_overrides,
                )?),
                negated: *negated,
            }),
            ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
                left: Box::new(Self::remap_string_expr_to_indices(
                    left,
                    schema,
                    name_overrides,
                )?),
                op: *op,
                right: Box::new(Self::remap_string_expr_to_indices(
                    right,
                    schema,
                    name_overrides,
                )?),
            }),
            ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
                expr: Box::new(Self::remap_string_expr_to_indices(
                    expr,
                    schema,
                    name_overrides,
                )?),
                data_type: data_type.clone(),
            }),
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => Ok(ScalarExpr::Case {
                operand: operand
                    .as_ref()
                    .map(|o| Self::remap_string_expr_to_indices(o, schema, name_overrides))
                    .transpose()?
                    .map(Box::new),
                branches: branches
                    .iter()
                    .map(|(w, t)| {
                        Ok((
                            Self::remap_string_expr_to_indices(w, schema, name_overrides)?,
                            Self::remap_string_expr_to_indices(t, schema, name_overrides)?,
                        ))
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
                else_expr: else_expr
                    .as_ref()
                    .map(|e| Self::remap_string_expr_to_indices(e, schema, name_overrides))
                    .transpose()?
                    .map(Box::new),
            }),
            ScalarExpr::Coalesce(items) => Ok(ScalarExpr::Coalesce(
                items
                    .iter()
                    .map(|e| Self::remap_string_expr_to_indices(e, schema, name_overrides))
                    .collect::<ExecutorResult<Vec<_>>>()?,
            )),
            ScalarExpr::ScalarSubquery(s) => Ok(ScalarExpr::ScalarSubquery(s.clone())),
            _ => Err(Error::Internal(format!(
                "Unsupported expression for remapping: {:?}",
                expr
            ))),
        }
    }

    fn evaluate_row_predicate(
        batch: &RecordBatch,
        row_index: usize,
        predicate: &Expr<'static, String>,
        subqueries: &[FilterSubquery],
        provider: &Arc<dyn ExecutorTableProvider<P>>,
        planner: &Arc<dyn PreparedSelectPlanner<P>>,
    ) -> ExecutorResult<bool> {
        let mut exists_results = FxHashMap::default();

        for sub in subqueries {
            let mut replacements = HashMap::new();
            for corr in &sub.correlated_columns {
                let col = batch.column_by_name(&corr.column).ok_or_else(|| {
                    Error::Internal(format!("Correlated column not found: {}", corr.column))
                })?;
                let lit = Self::get_scalar(col, row_index)?;
                replacements.insert(corr.placeholder.clone(), lit);
            }

            let sub_plan = Self::rewrite_select_plan_placeholders(&sub.plan, &replacements);

            let executor = QueryExecutor::new(provider.clone(), Arc::clone(planner));
            let execution = executor.execute_select(sub_plan)?;

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
        let remapped_expr =
            Self::remap_string_expr_to_indices(&scalar_expr, &batch.schema(), None)?;

        let slice = batch.slice(row_index, 1);

        let mut field_arrays = FxHashMap::default();
        for (i, col) in slice.columns().iter().enumerate() {
            field_arrays.insert((0, i as u32), col.clone());
        }
        let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, 1);

        let result =
            ScalarEvaluator::evaluate_batch_simplified(&remapped_expr, 1, &numeric_arrays)?;

        if result.is_null(0) {
            Ok(false)
        } else {
            let bool_arr = result
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .ok_or_else(|| Error::Internal("Predicate did not evaluate to boolean".into()))?;
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
            Expr::And(l) => Expr::And(
                l.iter()
                    .map(|e| Self::rewrite_expr_exists(e, results))
                    .collect(),
            ),
            Expr::Or(l) => Expr::Or(
                l.iter()
                    .map(|e| Self::rewrite_expr_exists(e, results))
                    .collect(),
            ),
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
            Expr::And(list) => {
                Expr::And(list.into_iter().map(Self::normalize_not_isnull).collect())
            }
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
        let planner = self.planner.clone();
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
                let matches = Self::evaluate_row_predicate(
                    &batch,
                    i,
                    &predicate,
                    &subqueries,
                    &provider,
                    &planner,
                );
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

    fn contains_subquery<F: std::fmt::Debug>(expr: &ScalarExpr<F>) -> bool {
        match expr {
            ScalarExpr::ScalarSubquery(_) => true,
            ScalarExpr::Binary { left, right, .. } => {
                Self::contains_subquery(left) || Self::contains_subquery(right)
            }
            ScalarExpr::Not(e) => Self::contains_subquery(e),
            ScalarExpr::IsNull { expr, .. } => Self::contains_subquery(expr),
            ScalarExpr::Cast { expr, .. } => Self::contains_subquery(expr),
            ScalarExpr::Compare { left, right, .. } => {
                Self::contains_subquery(left) || Self::contains_subquery(right)
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                operand
                    .as_ref()
                    .map_or(false, |o| Self::contains_subquery(o))
                    || branches
                        .iter()
                        .any(|(w, t)| Self::contains_subquery(w) || Self::contains_subquery(t))
                    || else_expr
                        .as_ref()
                        .map_or(false, |e| Self::contains_subquery(e))
            }
            ScalarExpr::Coalesce(list) => list.iter().any(Self::contains_subquery),
            ScalarExpr::GetField { base, .. } => Self::contains_subquery(base),
            ScalarExpr::Aggregate(call) => match call {
                AggregateCall::Count { expr, .. }
                | AggregateCall::Sum { expr, .. }
                | AggregateCall::Total { expr, .. }
                | AggregateCall::Avg { expr, .. }
                | AggregateCall::Min(expr)
                | AggregateCall::Max(expr)
                | AggregateCall::CountNulls(expr)
                | AggregateCall::GroupConcat { expr, .. } => Self::contains_subquery(expr),
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
                let sub_def = scalar_subqueries
                    .iter()
                    .find(|s| s.id == sub_expr.id)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Scalar subquery definition not found for ID {}",
                            sub_expr.id.0
                        ))
                    })?;

                let mut replacements = HashMap::new();
                for corr in &sub_def.correlated_columns {
                    let col = batch.column_by_name(&corr.column).ok_or_else(|| {
                        Error::Internal(format!("Correlated column not found: {}", corr.column))
                    })?;
                    let lit = Self::get_scalar(col, row_index)?;
                    replacements.insert(corr.placeholder.clone(), lit);
                }

                let sub_plan = Self::rewrite_select_plan_placeholders(&sub_def.plan, &replacements);
                let executor = QueryExecutor::new(provider.clone(), self.planner.clone());
                let execution = executor.execute_select(sub_plan)?;
                let batches = execution.collect()?;

                let mut result_val = None;
                for b in batches {
                    if b.num_rows() > 0 {
                        if result_val.is_some() || b.num_rows() > 1 {
                            return Err(Error::Internal(
                                "Scalar subquery returned more than one row".into(),
                            ));
                        }
                        if b.num_columns() != 1 {
                            return Err(Error::Internal(
                                "Scalar subquery returned more than one column".into(),
                            ));
                        }
                        let col = b.column(0);
                        let scalar = Self::get_scalar(col, 0)?;
                        result_val = Some(scalar);
                    }
                }
                Ok(ScalarExpr::Literal(result_val.unwrap_or(Literal::Null)))
            }
            ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
                left: Box::new(self.rewrite_expr_with_row_subqueries(
                    left,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?),
                op: *op,
                right: Box::new(self.rewrite_expr_with_row_subqueries(
                    right,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?),
            }),
            ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(
                self.rewrite_expr_with_row_subqueries(
                    e,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?,
            ))),
            ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
                expr: Box::new(self.rewrite_expr_with_row_subqueries(
                    expr,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?),
                negated: *negated,
            }),
            ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
                expr: Box::new(self.rewrite_expr_with_row_subqueries(
                    expr,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?),
                data_type: data_type.clone(),
            }),
            ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
                left: Box::new(self.rewrite_expr_with_row_subqueries(
                    left,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?),
                op: *op,
                right: Box::new(self.rewrite_expr_with_row_subqueries(
                    right,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?),
            }),
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => Ok(ScalarExpr::Case {
                operand: operand
                    .as_ref()
                    .map(|o| {
                        self.rewrite_expr_with_row_subqueries(
                            o,
                            batch,
                            row_index,
                            provider,
                            scalar_subqueries,
                        )
                    })
                    .transpose()?
                    .map(Box::new),
                branches: branches
                    .iter()
                    .map(|(w, t)| {
                        Ok((
                            self.rewrite_expr_with_row_subqueries(
                                w,
                                batch,
                                row_index,
                                provider,
                                scalar_subqueries,
                            )?,
                            self.rewrite_expr_with_row_subqueries(
                                t,
                                batch,
                                row_index,
                                provider,
                                scalar_subqueries,
                            )?,
                        ))
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
                else_expr: else_expr
                    .as_ref()
                    .map(|e| {
                        self.rewrite_expr_with_row_subqueries(
                            e,
                            batch,
                            row_index,
                            provider,
                            scalar_subqueries,
                        )
                    })
                    .transpose()?
                    .map(Box::new),
            }),
            ScalarExpr::Coalesce(list) => Ok(ScalarExpr::Coalesce(
                list.iter()
                    .map(|e| {
                        self.rewrite_expr_with_row_subqueries(
                            e,
                            batch,
                            row_index,
                            provider,
                            scalar_subqueries,
                        )
                    })
                    .collect::<ExecutorResult<Vec<_>>>()?,
            )),
            ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
                base: Box::new(self.rewrite_expr_with_row_subqueries(
                    base,
                    batch,
                    row_index,
                    provider,
                    scalar_subqueries,
                )?),
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
        agg_metadata: Option<(usize, usize)>,
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
                    output_fields.push(Arc::new(ArrowField::new(
                        out_name,
                        field.data_type().clone(),
                        field.is_nullable(),
                    )));
                    exprs.push(ScalarExpr::Column((0, idx as u32)));
                }
                llkv_plan::plans::SelectProjection::Computed { expr, alias } => {
                    let remapped =
                        Self::remap_string_expr_to_indices(expr, input_schema, name_overrides)?;
                    let rewritten = if let Some((agg_offset, agg_count)) = agg_metadata {
                        let mut next_agg_idx = 0;
                        Self::rewrite_aggregate_refs(
                            remapped,
                            agg_offset,
                            agg_count,
                            &mut next_agg_idx,
                        )?
                    } else {
                        remapped
                    };
                    let dt = infer_type(&rewritten, input_schema, &col_mapping)
                        .unwrap_or(arrow::datatypes::DataType::Int64);
                    output_fields.push(Arc::new(ArrowField::new(alias.clone(), dt, true)));
                    exprs.push(rewritten);
                }
            }
        }

        let output_schema = Arc::new(Schema::new(output_fields));
        let output_schema_captured = output_schema.clone();
        let provider = self.provider.clone();
        let planner = self.planner.clone();
        let self_clone = QueryExecutor::new(provider.clone(), planner);
        let scalar_subqueries_captured = scalar_subqueries.to_vec();

        let final_stream = input.map(move |batch_res| {
            let batch = batch_res?;
            let mut columns = Vec::new();

            let mut field_arrays = FxHashMap::default();
            for (i, col) in batch.columns().iter().enumerate() {
                field_arrays.insert((0, i as u32), col.clone());
            }
            let numeric_arrays =
                ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());

            for expr in &exprs {
                if Self::contains_subquery(expr) {
                    let mut result_arrays = Vec::new();
                    for i in 0..batch.num_rows() {
                        let rewritten = self_clone.rewrite_expr_with_row_subqueries(
                            expr,
                            &batch,
                            i,
                            &provider,
                            &scalar_subqueries_captured,
                        )?;

                        let slice = batch.slice(i, 1);
                        let mut slice_arrays = FxHashMap::default();
                        for (j, col) in slice.columns().iter().enumerate() {
                            slice_arrays.insert((0, j as u32), col.clone());
                        }
                        let slice_numeric =
                            ScalarEvaluator::prepare_numeric_arrays(&slice_arrays, 1);

                        let res = ScalarEvaluator::evaluate_batch_simplified(
                            &rewritten,
                            1,
                            &slice_numeric,
                        )?;
                        result_arrays.push(res);
                    }

                    let refs: Vec<&dyn Array> = result_arrays.iter().map(|a| a.as_ref()).collect();
                    let concatenated = arrow::compute::concat(&refs)
                        .map_err(|e| Error::Internal(e.to_string()))?;
                    columns.push(concatenated);
                } else {
                    let result = ScalarEvaluator::evaluate_batch_simplified(
                        expr,
                        batch.num_rows(),
                        &numeric_arrays,
                    )?;
                    columns.push(result);
                }
            }

            RecordBatch::try_new(output_schema_captured.clone(), columns)
                .map_err(|e| Error::Internal(e.to_string()))
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

        Ok(SelectExecution::from_stream(
            table_name,
            output_schema,
            trimmed,
        ))
    }

    fn execute_compound_prepared(
        &self,
        compound: &PreparedCompoundSelect<P>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let mut current_exec = self.execute_prepared_select(&compound.initial)?;

        for op in &compound.operations {
            let next_exec = self.execute_prepared_select(&op.plan)?;

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
                    let stream =
                        Box::new(current_exec.into_stream()?.chain(next_exec.into_stream()?));
                    current_exec =
                        SelectExecution::from_stream("union".to_string(), schema, stream);

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
        self.execute_select_with_row_filter(plan, None)
    }

    pub fn execute_select_with_row_filter(
        &self,
        plan: SelectPlan,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let prepared = self
            .planner
            .prepare_select(plan, row_filter)
            .map_err(Error::from)?;
        self.execute_prepared_select(&prepared)
    }

    pub fn execute_prepared_select(
        &self,
        prepared: &PreparedSelectPlan<P>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if std::env::var("LLKV_DEBUG_SUBQS").is_ok() {
            eprintln!(
                "[executor] scalar subqueries: {}",
                prepared.scalar_subqueries.len()
            );
        }
        let subquery_results =
            self.execute_prepared_scalar_subqueries(&prepared.scalar_subqueries)?;
        self.execute_prepared_with_filter(prepared, &subquery_results)
    }

    fn execute_prepared_with_filter(
        &self,
        prepared: &PreparedSelectPlan<P>,
        subquery_results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
    ) -> ExecutorResult<SelectExecution<P>> {
        if let Some(compound) = &prepared.compound {
            return self.execute_compound_prepared(compound);
        }

        let mut plan = prepared.plan.clone();
        let row_filter = prepared.row_filter.clone();

        // Rewrite projections
        let mut scalar_subquery_cache = subquery_results.clone();
        let mut new_projections = Vec::new();
        for proj in &plan.projections {
            match proj {
                llkv_plan::plans::SelectProjection::Computed { expr, alias } => {
                    let rewritten = self.rewrite_scalar_subqueries_with_exec(
                        expr,
                        &prepared.scalar_subqueries,
                        &mut scalar_subquery_cache,
                    )?;
                    new_projections.push(llkv_plan::plans::SelectProjection::Computed {
                        expr: rewritten,
                        alias: alias.clone(),
                    });
                }
                other => new_projections.push(other.clone()),
            }
        }
        plan.projections = new_projections;

        if std::env::var("LLKV_DEBUG_SUBQS").is_ok() {
            eprintln!("[executor] projections after rewrite: {:?}", plan.projections);
            eprintln!("[executor] aggregates: {:?}", plan.aggregates);
        }

        // Rewrite filter
        if let Some(filter) = &mut plan.filter {
            filter.predicate = self.rewrite_expr_subqueries_with_exec(
                &filter.predicate,
                &prepared.scalar_subqueries,
                &mut scalar_subquery_cache,
            )?;
        }

        // Rewrite having
        if let Some(having) = &mut plan.having {
            *having = self.rewrite_expr_subqueries_with_exec(
                having,
                &prepared.scalar_subqueries,
                &mut scalar_subquery_cache,
            )?;
        }

        if std::env::var("LLKV_DEBUG_SUBQS").is_ok() {
            eprintln!("[executor] filter: {:?}", plan.filter);
            eprintln!("[executor] having: {:?}", plan.having);
        }

        let residual_filter = prepared.residual_filter.clone();
        let residual_subqueries = if residual_filter.is_some() {
            prepared.residual_filter_subqueries.clone()
        } else {
            Vec::new()
        };

        // Check if we need to force manual projection (cached from planner, but re-evaluate if
        // subqueries produced aggregates after rewrite).
        let has_subqueries = plan.projections.iter().any(|p| {
            if let llkv_plan::plans::SelectProjection::Computed { expr, .. } = p {
                Self::contains_subquery(expr)
            } else {
                false
            }
        });
        let has_aggregates = !plan.aggregates.is_empty()
            || plan.projections.iter().any(|p| {
                if let llkv_plan::plans::SelectProjection::Computed { expr, .. } = p {
                    Self::scalar_contains_aggregate(expr)
                } else {
                    false
                }
            })
            || plan
                .having
                .as_ref()
                .map(|h| Self::contains_aggregate_in_expr(h))
                .unwrap_or(false);
        let group_needs_full_row =
            plan.aggregates.is_empty() && !plan.group_by.is_empty() && !has_aggregates;
        let force_manual_projection = prepared.force_manual_projection
            || residual_filter.is_some()
            || has_subqueries
            || group_needs_full_row
            || has_aggregates;

        if std::env::var("LLKV_DEBUG_SUBQS").is_ok() {
            eprintln!(
                "[executor] force_manual_projection? {}, residual_filter: {}",
                force_manual_projection,
                residual_filter.is_some()
            );
        }

        match &prepared.logical_plan {
            LogicalPlan::Single(single) => {
                let table_name = single.table_name.clone();

                // Preserve planner-provided aggregate rewrite even when forcing
                // manual projection; the rewrite drives the aggregation path.
                let aggregate_rewrite = single.aggregate_rewrite.clone();

                if std::env::var("LLKV_DEBUG_SUBQS").is_ok() {
                    eprintln!("[executor] aggregate_rewrite: {:?}", aggregate_rewrite);
                }

                let physical_plan = prepared
                    .physical_plan
                    .clone()
                    .ok_or_else(|| {
                        Error::Internal(
                            "Physical plan missing for single-table SELECT".to_string(),
                        )
                    })?;

                let schema = physical_plan.schema();

                let mut base_iter: BatchIter = Box::new(
                    physical_plan
                        .execute()
                        .map_err(Error::Internal)?
                        .map(|b| b.map_err(Error::Internal)),
                );

                if let Some(residual) = &residual_filter {
                    base_iter = self.apply_residual_filter(
                        base_iter,
                        residual.clone(),
                        residual_subqueries.clone(),
                    );
                }

                if let Some(rewrite) = aggregate_rewrite.clone() {
                    let mut col_mapping = FxHashMap::default();
                    for (i, _field) in schema.fields().iter().enumerate() {
                        col_mapping.insert((0, i as u32), i);
                    }

                    let mut pre_agg_schema_fields: Vec<ArrowField> = Vec::new();
                    for (i, expr_str) in rewrite.pre_aggregate_expressions.iter().enumerate() {
                        let resolved = resolve_scalar_expr_string(
                            expr_str,
                            &schema,
                            subquery_results,
                        )?;
                        let dt = infer_type(&resolved, &schema, &col_mapping)
                            .unwrap_or(arrow::datatypes::DataType::Int64);
                        pre_agg_schema_fields.push(
                            ArrowField::new(format!("_agg_arg_{}", i), dt, true).with_metadata(
                                HashMap::from([(
                                    "field_id".to_string(),
                                    format!("{}", 10000 + i),
                                )]),
                            ),
                        );
                    }

                    let mut group_key_batch_indices = Vec::new();
                    if !plan.group_by.is_empty() {
                        for name in &plan.group_by {
                            let idx = resolve_group_index(name, &schema, &single.schema).ok_or_else(
                                || Error::Internal(format!("GROUP BY column not found: {name}")),
                            )?;
                            group_key_batch_indices.push(idx);
                            pre_agg_schema_fields.push(schema.field(idx).clone());
                        }
                    }

                    let group_key_indices: Vec<usize> = (0..group_key_batch_indices.len())
                        .map(|i| rewrite.pre_aggregate_expressions.len() + i)
                        .collect();

                    let pre_agg_schema = Arc::new(Schema::new(pre_agg_schema_fields));
                    let pre_agg_schema_captured = pre_agg_schema.clone();
                    let schema_captured = schema.clone();
                    let subquery_results_captured = subquery_results.clone();
                    let pre_agg_exprs = rewrite.pre_aggregate_expressions.clone();
                    let group_key_batch_indices_captured = group_key_batch_indices.clone();

                    let pre_agg_stream = base_iter.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();

                        let mut field_arrays = FxHashMap::default();
                        for (i, col) in batch.columns().iter().enumerate() {
                            field_arrays.insert((0, i as u32), col.clone());
                        }
                        let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(
                            &field_arrays,
                            batch.num_rows(),
                        );

                        for expr_str in &pre_agg_exprs {
                            let expr = resolve_scalar_expr_string(
                                expr_str,
                                &schema_captured,
                                &subquery_results_captured,
                            )?;
                            let result = ScalarEvaluator::evaluate_batch_simplified(
                                &expr,
                                batch.num_rows(),
                                &numeric_arrays,
                            )?;
                            columns.push(result);
                        }

                        for idx in &group_key_batch_indices_captured {
                            columns.push(batch.column(*idx).clone());
                        }

                        if columns.is_empty() {
                            let options = arrow::record_batch::RecordBatchOptions::new()
                                .with_row_count(Some(batch.num_rows()));
                            RecordBatch::try_new_with_options(
                                pre_agg_schema_captured.clone(),
                                columns,
                                &options,
                            )
                            .map_err(|e| Error::Internal(e.to_string()))
                        } else {
                            RecordBatch::try_new(pre_agg_schema_captured.clone(), columns)
                                .map_err(|e| Error::Internal(e.to_string()))
                        }
                    });

                    return self.execute_complex_aggregation(
                        Box::new(pre_agg_stream),
                        pre_agg_schema,
                        rewrite.aggregates,
                        rewrite.final_expressions,
                        rewrite.final_names,
                        &plan,
                        table_name,
                        subquery_results,
                        group_key_indices,
                        rewrite.rewritten_having,
                    );
                }

                // GROUP BY with no aggregates should collapse to distinct group keys.
                if plan.aggregates.is_empty() && !plan.group_by.is_empty() {
                    let mut key_indices = Vec::with_capacity(plan.group_by.len());
                    for name in &plan.group_by {
                        let idx = resolve_group_index(name, &schema, &single.schema).ok_or_else(
                            || Error::Internal(format!("GROUP BY column not found: {name}")),
                        )?;
                        key_indices.push(idx);
                    }

                    let mut grouped: BatchIter = Box::new(DistinctOnKeysStream::new(
                        schema.clone(),
                        key_indices,
                        base_iter,
                    )?);

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
                        None,
                    );
                }

                if !plan.aggregates.is_empty() {
                    if !plan.group_by.is_empty() {
                        let mut key_indices = Vec::with_capacity(plan.group_by.len());
                        for name in &plan.group_by {
                            let idx = resolve_group_index(name, &schema, &single.schema)
                                .ok_or_else(|| {
                                    Error::Internal(format!("GROUP BY column not found: {name}"))
                                })?;
                            key_indices.push(idx);
                        }

                        let grouped = GroupedAggregateStream::new(
                            base_iter,
                            key_indices,
                            &plan,
                            &single.schema,
                            &schema,
                        )?;
                        let agg_schema = grouped.schema();
                        let agg_count = plan.aggregates.len();
                        let agg_offset =
                            agg_schema.fields().len().saturating_sub(agg_count);
                        let mut agg_iter: BatchIter = Box::new(grouped.map(|b| b));

                        if let Some(having) = &plan.having {
                            let having_scalar = Self::expr_to_scalar_expr(having);
                            let having_remapped = Self::remap_string_expr_to_indices(
                                &having_scalar,
                                &agg_schema,
                                None,
                            )?;
                            let mut next_agg_idx = 0;
                            let having_rewritten = Self::rewrite_aggregate_refs(
                                having_remapped,
                                agg_offset,
                                agg_count,
                                &mut next_agg_idx,
                            )?;
                            let having_schema = agg_schema.clone();

                            agg_iter = Box::new(agg_iter.map(move |batch_res| {
                                let batch = batch_res?;

                                let mut field_arrays = FxHashMap::default();
                                for (i, col) in batch.columns().iter().enumerate() {
                                    field_arrays.insert((0, i as u32), col.clone());
                                }
                                let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(
                                    &field_arrays,
                                    batch.num_rows(),
                                );

                                let filter_value = ScalarEvaluator::evaluate_batch_simplified(
                                    &having_rewritten,
                                    batch.num_rows(),
                                    &numeric_arrays,
                                )?;

                                let bool_arr = filter_value
                                    .as_any()
                                    .downcast_ref::<BooleanArray>()
                                    .ok_or_else(|| {
                                        Error::Internal(
                                            "HAVING expression must evaluate to a boolean array"
                                                .to_string(),
                                        )
                                    })?;

                                filter_record_batch(&batch, bool_arr).map_err(|e| {
                                    Error::Internal(format!(
                                        "HAVING filter failed: {e} (schema: {:?})",
                                        having_schema
                                    ))
                                })
                            }));
                        }

                        return self.project_stream(
                            agg_iter,
                            &agg_schema,
                            &plan.projections,
                            table_name,
                            &plan.scalar_subqueries,
                            plan.distinct,
                            &plan.order_by,
                            plan.offset,
                            plan.limit,
                            None,
                            Some((agg_offset, agg_count)),
                        );
                    }

                    let agg_iter =
                        AggregateStream::new(base_iter, &plan, &single.schema, &schema)?;
                    let agg_schema = agg_iter.schema();
                    let agg_count = plan.aggregates.len();
                    let agg_offset = agg_schema.fields().len().saturating_sub(agg_count);
                    let mut filtered_iter: BatchIter = Box::new(agg_iter);

                    if let Some(having) = &plan.having {
                        let having_scalar = Self::expr_to_scalar_expr(having);
                        let having_remapped = Self::remap_string_expr_to_indices(
                            &having_scalar,
                            &agg_schema,
                            None,
                        )?;
                        let mut next_agg_idx = 0;
                        let having_rewritten = Self::rewrite_aggregate_refs(
                            having_remapped,
                            agg_offset,
                            agg_count,
                            &mut next_agg_idx,
                        )?;
                        let having_schema = agg_schema.clone();

                        filtered_iter = Box::new(filtered_iter.map(move |batch_res| {
                            let batch = batch_res?;

                            let mut field_arrays = FxHashMap::default();
                            for (i, col) in batch.columns().iter().enumerate() {
                                field_arrays.insert((0, i as u32), col.clone());
                            }
                            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(
                                &field_arrays,
                                batch.num_rows(),
                            );

                            let filter_value = ScalarEvaluator::evaluate_batch_simplified(
                                &having_rewritten,
                                batch.num_rows(),
                                &numeric_arrays,
                            )?;

                            let bool_arr = filter_value
                                .as_any()
                                .downcast_ref::<BooleanArray>()
                                .ok_or_else(|| {
                                    Error::Internal(
                                        "HAVING expression must evaluate to a boolean array"
                                            .to_string(),
                                    )
                                })?;

                            filter_record_batch(&batch, bool_arr).map_err(|e| {
                                Error::Internal(format!(
                                    "HAVING filter failed: {e} (schema: {:?})",
                                    having_schema
                                ))
                            })
                        }));
                    }

                    let trimmed =
                        apply_offset_limit_stream(filtered_iter, plan.offset, plan.limit);
                    return Ok(SelectExecution::from_stream(
                        table_name, agg_schema, trimmed,
                    ));
                }

                if force_manual_projection {
                    return self.project_stream(
                        base_iter,
                        &schema,
                        &plan.projections,
                        table_name,
                        &plan.scalar_subqueries,
                        plan.distinct,
                        &plan.order_by,
                        plan.offset,
                        plan.limit,
                        None,
                        None,
                    );
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
                    return Err(Error::Internal(
                        "table filter metadata does not match table count".into(),
                    ));
                }

                let table_filters = multi.table_filters.clone();
                let residual_filter = multi.filter.clone();

                // 1. Analyze required columns
                let mut required_fields: Vec<FxHashSet<FieldId>> =
                    vec![FxHashSet::default(); table_count];

                for proj in &multi.projections {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column {
                            table_index,
                            logical_field_id,
                            ..
                        } => {
                            required_fields[*table_index].insert(logical_field_id.field_id());
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed {
                            expr, ..
                        } => {
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
                    let adapter = table
                        .table
                        .as_any()
                        .downcast_ref::<ExecutionTableAdapter<P>>()
                        .ok_or_else(|| Error::Internal("unexpected table adapter type".into()))?;

                    let mut fields: Vec<FieldId> = required_fields[i].iter().copied().collect();
                    if fields.is_empty() {
                        if let Some(col) = table.schema.columns.first() {
                            fields.push(col.field_id);
                        }
                    }
                    fields.sort_unstable();
                    table_field_map.push(fields.clone());

                    let projections: Vec<ScanProjection> = fields
                        .iter()
                        .map(|&fid| {
                            let col = table
                                .schema
                                .columns
                                .iter()
                                .find(|c| c.field_id == fid)
                                .unwrap();
                            let lfid =
                                LogicalFieldId::for_user(adapter.executor_table().table_id(), fid);
                            ScanProjection::Column(Projection::with_alias(lfid, col.name.clone()))
                        })
                        .collect();

                    let arrow_fields: Vec<ArrowField> = fields
                        .iter()
                        .map(|&fid| {
                            let col = table
                                .schema
                                .columns
                                .iter()
                                .find(|c| c.field_id == fid)
                                .unwrap();
                            ArrowField::new(
                                col.name.clone(),
                                col.data_type.clone(),
                                col.is_nullable,
                            )
                        })
                        .collect();
                    schemas.push(Arc::new(Schema::new(arrow_fields)));

                    let (tx, rx) = mpsc::sync_channel(16);
                    let storage = adapter.executor_table().storage().clone();
                    let row_filter = row_filter.clone();

                    std::thread::spawn(move || {
                        let res = storage.scan_stream(
                            &projections,
                            &Expr::Pred(Filter {
                                field_id: 0,
                                op: Operator::Range {
                                    lower: std::ops::Bound::Unbounded,
                                    upper: std::ops::Bound::Unbounded,
                                },
                            }),
                            llkv_scan::ScanStreamOptions {
                                row_id_filter: row_filter.clone(),
                                ..Default::default()
                            },
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
                let mut pending_filters: Vec<
                    LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
                > = match residual_filter.clone() {
                    Some(LlkvExpr::And(list)) => list.clone(),
                    Some(expr) => vec![expr.clone()],
                    None => Vec::new(),
                };

                for (idx, fid) in table_field_map[0].iter().enumerate() {
                    col_mapping.insert((0, *fid), idx);
                }

                for i in 0..table_count - 1 {
                    let right_stream = streams.remove(0);
                    let right_schema = schemas[i + 1].clone();

                    let right_batches: Vec<RecordBatch> =
                        right_stream.collect::<Result<Vec<_>, _>>()?;
                    let right_batch = if right_batches.is_empty() {
                        RecordBatch::new_empty(right_schema.clone())
                    } else {
                        concat_batches(&right_schema, &right_batches)?
                    };

                    let join = multi
                        .joins
                        .iter()
                        .find(|j| j.left_table_index == i)
                        .ok_or_else(|| {
                            Error::Internal("missing join metadata for multi-table query".into())
                        })?;

                    let (keys, residuals) = extract_join_keys_and_filters(join)?;
                    pending_filters.extend(residuals);
                    let mut left_indices = Vec::new();
                    let mut right_indices = Vec::new();

                    for key in keys {
                        if key.right_table != i + 1 {
                            return Err(Error::Internal(
                                "join key points to unexpected right table".into(),
                            ));
                        }

                        if key.left_table > i {
                            return Err(Error::Internal(
                                "join key references future left table".into(),
                            ));
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

                    for (idx, fid) in table_field_map[i + 1].iter().enumerate() {
                        col_mapping.insert((i + 1, *fid), old_len + idx);
                    }

                    let prefix_limit = i + 1;
                    let mut applicable = Vec::new();
                    let mut remaining = Vec::new();
                    for clause in pending_filters.into_iter() {
                        let mapped = remap_filter_expr(&clause)?;
                        let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                        ScalarEvaluator::collect_fields(&mapped, &mut fields);
                        let tables: FxHashSet<usize> =
                            fields.into_iter().map(|(tbl, _)| tbl).collect();

                        if tables.iter().all(|t| *t <= prefix_limit) {
                            applicable.push(clause);
                        } else {
                            remaining.push(clause);
                        }
                    }
                    pending_filters = remaining;

                    if let Some(apply_expr) = combine_clauses(applicable) {
                        current_stream = apply_filter_to_stream(
                            current_stream,
                            &apply_expr,
                            col_mapping.clone(),
                        )?;
                    }
                }

                // 3. Apply any remaining filters not yet pushed earlier in the join chain.
                if let Some(final_expr) = combine_clauses(pending_filters) {
                    current_stream =
                        apply_filter_to_stream(current_stream, &final_expr, col_mapping.clone())?;
                }

                // 4. Final Projection
                let mut plan_columns = Vec::new();
                let mut name_to_index = FxHashMap::default();

                let mut name_overrides: FxHashMap<String, usize> = FxHashMap::default();
                for ((tbl, fid), idx) in &col_mapping {
                    if let Some(table_ref) = multi.table_order.get(*tbl) {
                        if let Some(col) = multi
                            .tables
                            .get(*tbl)
                            .and_then(|t| t.schema.columns.iter().find(|c| c.field_id == *fid))
                        {
                            let qualifier = table_ref.display_name().to_ascii_lowercase();
                            let key = format!("{}.{}", qualifier, col.name.to_ascii_lowercase());
                            name_overrides.insert(key, *idx);
                        }
                    }
                }

                let output_fields: Vec<ArrowField> = multi
                    .projections
                    .iter()
                    .enumerate()
                    .map(|(i, proj)| match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column {
                            table_index,
                            logical_field_id,
                            alias,
                        } => {
                            let idx = col_mapping
                                .get(&(*table_index, logical_field_id.field_id()))
                                .unwrap();
                            let field = current_schema.field(*idx);

                            let name = alias.clone().unwrap_or_else(|| field.name().clone());
                            let mut metadata = HashMap::new();
                            metadata.insert(
                                "field_id".to_string(),
                                logical_field_id.field_id().to_string(),
                            );

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
                        llkv_plan::logical_planner::ResolvedProjection::Computed {
                            alias,
                            expr,
                        } => {
                            let name = alias.clone();
                            let dummy_fid = 999999 + i as u32;

                            let remapped = remap_scalar_expr(expr);
                            let inferred_type =
                                infer_type(&remapped, &current_schema, &col_mapping)
                                    .unwrap_or(arrow::datatypes::DataType::Int64);

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

                            ArrowField::new(name, inferred_type, true).with_metadata(metadata)
                        }
                    })
                    .collect();
                let output_schema = Arc::new(Schema::new(output_fields));
                let _logical_schema = PlanSchema {
                    columns: plan_columns,
                    name_to_index,
                };

                if let Some(rewrite) = multi.aggregate_rewrite.clone() {
                    let mut simple_mapping = FxHashMap::default();
                    for (i, _field) in current_schema.fields().iter().enumerate() {
                        simple_mapping.insert((0, i as u32), i);
                    }

                    let mut pre_agg_schema_fields: Vec<ArrowField> = Vec::new();
                    for (i, expr_str) in rewrite.pre_aggregate_expressions.iter().enumerate() {
                        let resolved = resolve_scalar_expr_string(
                            expr_str,
                            &current_schema,
                            &subquery_results,
                        )?;
                        let dt = infer_type(&resolved, &current_schema, &simple_mapping)
                            .unwrap_or(arrow::datatypes::DataType::Int64);
                        pre_agg_schema_fields.push(
                            ArrowField::new(format!("_agg_arg_{}", i), dt, true).with_metadata(
                                HashMap::from([(
                                    "field_id".to_string(),
                                    format!("{}", 10000 + i),
                                )]),
                            ),
                        );
                    }

                    let mut group_key_batch_indices = Vec::new();
                    for key in &multi.group_by {
                        let idx = col_mapping
                            .get(&(key.table_index, key.logical_field_id.field_id()))
                            .copied()
                            .ok_or_else(|| {
                                Error::Internal("GROUP BY column not found in join output".into())
                            })?;
                        group_key_batch_indices.push(idx);
                        pre_agg_schema_fields.push(current_schema.field(idx).clone());
                    }

                    let group_key_indices: Vec<usize> = (0..group_key_batch_indices.len())
                        .map(|i| rewrite.pre_aggregate_expressions.len() + i)
                        .collect();

                    let pre_agg_schema = Arc::new(Schema::new(pre_agg_schema_fields));
                    let pre_agg_schema_captured = pre_agg_schema.clone();
                    let current_schema_captured = current_schema.clone();
                    let subquery_results_captured = subquery_results.clone();
                    let pre_agg_exprs = rewrite.pre_aggregate_expressions.clone();
                    let group_key_batch_indices_captured = group_key_batch_indices.clone();

                    let pre_agg_stream = current_stream.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();

                        let mut field_arrays = FxHashMap::default();
                        for (i, col) in batch.columns().iter().enumerate() {
                            field_arrays.insert((0, i as u32), col.clone());
                        }
                        let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(
                            &field_arrays,
                            batch.num_rows(),
                        );

                        for expr_str in &pre_agg_exprs {
                            let expr = resolve_scalar_expr_string(
                                expr_str,
                                &current_schema_captured,
                                &subquery_results_captured,
                            )?;
                            let result = ScalarEvaluator::evaluate_batch_simplified(
                                &expr,
                                batch.num_rows(),
                                &numeric_arrays,
                            )?;
                            columns.push(result);
                        }

                        for idx in &group_key_batch_indices_captured {
                            columns.push(batch.column(*idx).clone());
                        }

                        if columns.is_empty() {
                            let options = arrow::record_batch::RecordBatchOptions::new()
                                .with_row_count(Some(batch.num_rows()));
                            RecordBatch::try_new_with_options(
                                pre_agg_schema_captured.clone(),
                                columns,
                                &options,
                            )
                            .map_err(|e| Error::Internal(e.to_string()))
                        } else {
                            RecordBatch::try_new(pre_agg_schema_captured.clone(), columns)
                                .map_err(|e| Error::Internal(e.to_string()))
                        }
                    });

                    let group_key_indices_cloned = group_key_indices.clone();
                    let table_name = multi
                        .table_order
                        .first()
                        .map(|t| t.qualified_name())
                        .unwrap_or_default();

                    return self.execute_complex_aggregation(
                        Box::new(pre_agg_stream),
                        pre_agg_schema,
                        rewrite.aggregates,
                        rewrite.final_expressions,
                        rewrite.final_names,
                        &plan,
                        table_name,
                        &subquery_results,
                        group_key_indices_cloned,
                        rewrite.rewritten_having,
                    );
                }

                if !multi.group_by.is_empty() {
                    let mut key_indices = Vec::with_capacity(multi.group_by.len());
                    for key in &multi.group_by {
                        let field_id = key.logical_field_id.field_id();
                        let idx = col_mapping
                            .get(&(key.table_index, field_id))
                            .copied()
                            .ok_or_else(|| {
                                Error::Internal("GROUP BY column not found in join output".into())
                            })?;
                        key_indices.push(idx);
                    }

                    let mut grouped: BatchIter = Box::new(DistinctOnKeysStream::new(
                        current_schema.clone(),
                        key_indices,
                        current_stream,
                    )?);

                    if let Some(having) = &plan.having {
                        grouped = self.apply_residual_filter(grouped, having.clone(), Vec::new());
                    }

                    let table_name = multi
                        .table_order
                        .first()
                        .map(|t| t.qualified_name())
                        .unwrap_or_default();

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
                        None,
                    );
                }

                let projections = multi.projections.clone();
                let col_mapping = col_mapping.clone();

                let mut iter: BatchIter = {
                    let output_schema_captured = output_schema.clone();
                    let final_stream = current_stream.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();

                        for proj in &projections {
                            match proj {
                                llkv_plan::logical_planner::ResolvedProjection::Column {
                                    table_index,
                                    logical_field_id,
                                    ..
                                } => {
                                    let idx = col_mapping
                                        .get(&(*table_index, logical_field_id.field_id()))
                                        .ok_or_else(|| {
                                            Error::Internal("projection column missing".into())
                                        })?;
                                    columns.push(batch.column(*idx).clone());
                                }
                                llkv_plan::logical_planner::ResolvedProjection::Computed {
                                    expr,
                                    ..
                                } => {
                                    let remapped = remap_scalar_expr(expr);

                                    let mut required_fields = FxHashSet::default();
                                    ScalarEvaluator::collect_fields(
                                        &remapped,
                                        &mut required_fields,
                                    );

                                    let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> =
                                        FxHashMap::default();
                                    for (tbl, fid) in required_fields {
                                        if let Some(idx) = col_mapping.get(&(tbl, fid)) {
                                            field_arrays
                                                .insert((tbl, fid), batch.column(*idx).clone());
                                        } else {
                                            return Err(Error::Internal(format!(
                                                "Missing field {:?} in batch for computed column",
                                                (tbl, fid)
                                            )));
                                        }
                                    }

                                    let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(
                                        &field_arrays,
                                        batch.num_rows(),
                                    );
                                    let result = ScalarEvaluator::evaluate_batch_simplified(
                                        &remapped,
                                        batch.num_rows(),
                                        &numeric_arrays,
                                    )?;
                                    columns.push(result);
                                }
                            }
                        }
                        RecordBatch::try_new(output_schema_captured.clone(), columns)
                            .map_err(|e| Error::Internal(e.to_string()))
                    });
                    Box::new(final_stream)
                };

                if plan.distinct {
                    iter = Box::new(DistinctStream::new(output_schema.clone(), iter)?);
                }

                let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);
                let table_name = multi
                    .table_order
                    .first()
                    .map(|t| t.qualified_name())
                    .unwrap_or_default();
                Ok(SelectExecution::from_stream(
                    table_name,
                    output_schema,
                    trimmed,
                ))
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
        let converter =
            RowConverter::new(sort_fields).map_err(|e| Error::Internal(e.to_string()))?;
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
        ScalarExpr::IsNull {
            expr: inner,
            negated,
        } => {
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
        LlkvExpr::And(list) => combine_with_op(
            list.iter().map(remap_filter_expr),
            llkv_expr::expr::BinaryOp::And,
        ),
        LlkvExpr::Or(list) => combine_with_op(
            list.iter().map(remap_filter_expr),
            llkv_expr::expr::BinaryOp::Or,
        ),
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
        Operator::StartsWith { .. } | Operator::EndsWith { .. } | Operator::Contains { .. } => {
            return Err(Error::InvalidArgumentError(
                "string pattern predicates are not supported in multi-table execution".into(),
            ));
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

fn resolve_group_index(
    name: &str,
    schema: &Schema,
    plan_schema: &llkv_plan::schema::PlanSchema,
) -> Option<usize> {
    if let Some(idx) = resolve_schema_index(name, schema) {
        return Some(idx);
    }

    let field_id = plan_schema.column_by_name(name).map(|c| c.field_id)?;

    schema.fields().iter().enumerate().find_map(|(idx, f)| {
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

        let numeric_arrays =
            ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
        let result = ScalarEvaluator::evaluate_batch_simplified(
            &predicate,
            batch.num_rows(),
            &numeric_arrays,
        )?;

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

fn resolve_scalar_expr_string(
    expr: &ScalarExpr<String>,
    schema: &Schema,
    subquery_results: &FxHashMap<llkv_expr::expr::SubqueryId, Literal>,
) -> Result<ScalarExpr<(usize, FieldId)>, Error> {
    match expr {
        ScalarExpr::Column(name) => {
            let (idx, _field) = schema.column_with_name(name).ok_or_else(|| {
                let available: Vec<&str> = schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect();
                Error::InvalidArgumentError(format!(
                    "Column not found: {name}. Available columns: {}",
                    available.join(", ")
                ))
            })?;
            Ok(ScalarExpr::Column((0, idx as u32)))
        }
        ScalarExpr::Literal(l) => Ok(ScalarExpr::Literal(l.clone())),
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(resolve_scalar_expr_string(left, schema, subquery_results)?),
            op: *op,
            right: Box::new(resolve_scalar_expr_string(right, schema, subquery_results)?),
        }),
        ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(resolve_scalar_expr_string(
            e,
            schema,
            subquery_results,
        )?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(resolve_scalar_expr_string(expr, schema, subquery_results)?),
            negated: *negated,
        }),
        ScalarExpr::Aggregate(_) => Err(Error::Internal(
            "Nested aggregates not supported in resolution".into(),
        )),
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
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            let op = if let Some(o) = operand {
                Some(Box::new(resolve_scalar_expr_string(
                    o,
                    schema,
                    subquery_results,
                )?))
            } else {
                None
            };
            let mut br = Vec::new();
            for (w, t) in branches {
                br.push((
                    resolve_scalar_expr_string(w, schema, subquery_results)?,
                    resolve_scalar_expr_string(t, schema, subquery_results)?,
                ));
            }
            let el = if let Some(e) = else_expr {
                Some(Box::new(resolve_scalar_expr_string(
                    e,
                    schema,
                    subquery_results,
                )?))
            } else {
                None
            };
            Ok(ScalarExpr::Case {
                operand: op,
                branches: br,
                else_expr: el,
            })
        }
        ScalarExpr::Random => Ok(ScalarExpr::Random),
    }
}

fn infer_type(
    expr: &ScalarExpr<(usize, FieldId)>,
    schema: &Schema,
    col_mapping: &FxHashMap<(usize, FieldId), usize>,
) -> Result<DataType, Error> {
    let res = match expr {
        ScalarExpr::Column(id) => {
            let idx = col_mapping
                .get(id)
                .ok_or_else(|| Error::Internal("Column not found in mapping".into()))?;
            Ok(schema.field(*idx).data_type().clone())
        }
        ScalarExpr::Literal(l) => match l {
            Literal::Int128(_) => Ok(DataType::Int64),
            Literal::Float64(_) => Ok(DataType::Float64),
            Literal::Boolean(_) => Ok(DataType::Boolean),
            Literal::String(_) => Ok(DataType::Utf8),
            Literal::Null => Ok(DataType::Null),
            Literal::Decimal128(d) => Ok(DataType::Decimal128(d.precision(), d.scale())),
            _ => Ok(DataType::Int64),
        },
        ScalarExpr::Binary { left, op: _, right } => {
            let l = infer_type(left, schema, col_mapping)?;
            let r = infer_type(right, schema, col_mapping)?;
            if matches!(l, DataType::Float64 | DataType::Float32)
                || matches!(r, DataType::Float64 | DataType::Float32)
            {
                Ok(DataType::Float64)
            } else if matches!(l, DataType::Decimal128(..)) || matches!(r, DataType::Decimal128(..))
            {
                Ok(DataType::Decimal128(38, 10))
            } else {
                Ok(DataType::Int64)
            }
        }
        ScalarExpr::Cast { data_type, .. } => Ok(data_type.clone()),
        ScalarExpr::Case {
            else_expr,
            branches,
            ..
        } => {
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
        }
        ScalarExpr::Coalesce(items) => {
            if let Some(first) = items.first() {
                infer_type(first, schema, col_mapping)
            } else {
                Ok(DataType::Null)
            }
        }
        ScalarExpr::Aggregate(call) => match call {
            AggregateCall::CountStar
            | AggregateCall::Count { .. }
            | AggregateCall::CountNulls(_) => Ok(DataType::Int64),
            AggregateCall::Sum { expr, .. } | AggregateCall::Total { expr, .. } => {
                infer_type(expr, schema, col_mapping)
            }
            AggregateCall::Avg { .. } => Ok(DataType::Float64),
            AggregateCall::Min(expr) | AggregateCall::Max(expr) => {
                infer_type(expr, schema, col_mapping)
            }
            AggregateCall::GroupConcat { .. } => Ok(DataType::Utf8),
        },
        ScalarExpr::Not(_) | ScalarExpr::IsNull { .. } | ScalarExpr::Compare { .. } => {
            Ok(DataType::Boolean)
        }
        _ => Ok(DataType::Int64),
    };
    res
}

struct SortStream {
    schema: SchemaRef,
    input: BatchIter,
    sort_columns: Vec<(usize, bool, bool)>, // index, asc, nulls_first
    executed: bool,
}

impl SortStream {
    fn new(schema: SchemaRef, input: BatchIter, sort_columns: Vec<(usize, bool, bool)>) -> Self {
        Self {
            schema,
            input,
            sort_columns,
            executed: false,
        }
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
                return Some(Err(Error::Internal(format!(
                    "Sort column index {} out of bounds (num columns: {})",
                    idx,
                    batch.num_columns()
                ))));
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

        let columns = batch
            .columns()
            .iter()
            .map(|col| {
                arrow::compute::take(col, &indices, None)
                    .map_err(|e| Error::Internal(e.to_string()))
            })
            .collect::<ExecutorResult<Vec<_>>>();

        match columns {
            Ok(cols) => match RecordBatch::try_new(self.schema.clone(), cols) {
                Ok(b) => Some(Ok(b)),
                Err(e) => Some(Err(Error::Internal(e.to_string()))),
            },
            Err(e) => Some(Err(e)),
        }
    }
}
