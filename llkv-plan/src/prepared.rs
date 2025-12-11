use std::sync::Arc;

use llkv_expr::expr::{Expr, ScalarExpr, SubqueryId};
use llkv_expr::{AggregateCall, Expr as LlkvExpr};
use llkv_result::Result;
use llkv_scan::RowIdFilter;
use llkv_storage::pager::Pager;
use llkv_types::QueryContext;
use simd_r_drive_entry_handle::EntryHandle;

use crate::aggregate_rewrite::build_single_aggregate_rewrite;
use crate::logical_planner::{
    LogicalPlan, LogicalPlanner, ResolutionContext, TableResolutionInfo,
    build_multi_aggregate_rewrite,
};
// use crate::planner::PhysicalPlanner;
use crate::plans::{
    CompoundOperator, CompoundQuantifier, CompoundSelectPlan, CorrelatedColumn, FilterSubquery,
    ScalarSubquery, SelectFilter, SelectPlan, SelectProjection,
};
use crate::table_provider::TableProvider;
// use crate::physical::PhysicalPlan;

/// A fully prepared SELECT plan ready for execution.
pub struct PreparedSelectPlan<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub plan: Arc<SelectPlan>,
    pub logical_plan: Arc<LogicalPlan<P>>,
    // pub physical_plan: Option<Arc<dyn PhysicalPlan>>, // Only populated for single-table plans.
    pub scalar_subqueries: Arc<[PreparedScalarSubquery<P>]>, // Prepared subqueries for ScalarExpr.
    pub filter_subqueries: Arc<[PreparedFilterSubquery<P>]>, // Prepared subqueries referenced from filters.
    pub compound: Option<Arc<PreparedCompoundSelect<P>>>,    // Prepared compound operations.
    pub residual_filter: Option<Arc<Expr<'static, String>>>, // Residual predicate kept for executor-side filtering.
    pub residual_filter_subqueries: Arc<[FilterSubquery]>, // Correlated EXISTS subqueries tied to residual filters.
    pub force_manual_projection: bool, // Whether execution must project manually.
    pub row_filter: Option<Arc<dyn RowIdFilter<P>>>, // MVCC row filter to thread into scans.
}

impl<P> Clone for PreparedSelectPlan<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            plan: self.plan.clone(),
            logical_plan: self.logical_plan.clone(),
            scalar_subqueries: self.scalar_subqueries.clone(),
            filter_subqueries: self.filter_subqueries.clone(),
            compound: self.compound.clone(),
            residual_filter: self.residual_filter.clone(),
            residual_filter_subqueries: self.residual_filter_subqueries.clone(),
            force_manual_projection: self.force_manual_projection,
            row_filter: self.row_filter.clone(),
        }
    }
}

pub struct PreparedScalarSubquery<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub id: SubqueryId,
    pub correlated_columns: Arc<Vec<CorrelatedColumn>>,
    pub prepared_plan: Option<Arc<PreparedSelectPlan<P>>>,
    pub template: Arc<SelectPlan>,
}

impl<P> Clone for PreparedScalarSubquery<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            correlated_columns: self.correlated_columns.clone(),
            prepared_plan: self.prepared_plan.clone(),
            template: self.template.clone(),
        }
    }
}

pub struct PreparedFilterSubquery<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub id: SubqueryId,
    pub correlated_columns: Arc<Vec<CorrelatedColumn>>,
    pub prepared_plan: Option<Arc<PreparedSelectPlan<P>>>,
    pub template: Arc<SelectPlan>,
}

impl<P> Clone for PreparedFilterSubquery<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            correlated_columns: self.correlated_columns.clone(),
            prepared_plan: self.prepared_plan.clone(),
            template: self.template.clone(),
        }
    }
}

pub struct PreparedCompoundSelect<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub initial: Arc<PreparedSelectPlan<P>>,
    pub operations: Arc<Vec<PreparedCompoundOp<P>>>,
}

impl<P> Clone for PreparedCompoundSelect<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            initial: self.initial.clone(),
            operations: self.operations.clone(),
        }
    }
}

pub struct PreparedCompoundOp<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub operator: CompoundOperator,
    pub quantifier: CompoundQuantifier,
    pub plan: Arc<PreparedSelectPlan<P>>,
}

impl<P> Clone for PreparedCompoundOp<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            operator: self.operator.clone(),
            quantifier: self.quantifier.clone(),
            plan: self.plan.clone(),
        }
    }
}

/// Planner interface that produces executor-ready plans.
pub trait PreparedSelectPlanner<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn prepare_select(
        &self,
        plan: SelectPlan,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
        _context: Option<&str>,
        ctx: &QueryContext,
    ) -> Result<PreparedSelectPlan<P>>;
}

/// Default implementation backed by logical and physical planners.
pub struct DefaultPreparedSelectPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    logical_planner: LogicalPlanner<P>,
    // physical_planner: PhysicalPlanner<P>,
}

impl<P> DefaultPreparedSelectPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn TableProvider<P>>) -> Self {
        Self {
            logical_planner: LogicalPlanner::new(provider),
            // physical_planner: PhysicalPlanner::new(),
        }
    }

    fn prepare_scalar_subqueries(
        &self,
        subs: &[ScalarSubquery],
        ctx: &QueryContext,
    ) -> Result<Vec<PreparedScalarSubquery<P>>> {
        let mut prepared = Vec::with_capacity(subs.len());
        for sub in subs {
            let prepared_plan = if sub.correlated_columns.is_empty() {
                Some(Arc::new(self.prepare_select(
                    (*sub.plan).clone(),
                    None,
                    Some(&format!("scalar subquery {:?}", sub.id)),
                    ctx,
                )?))
            } else {
                None
            };
            prepared.push(PreparedScalarSubquery {
                id: sub.id,
                correlated_columns: Arc::new(sub.correlated_columns.clone()),
                prepared_plan,
                template: Arc::new(*sub.plan.clone()),
            });
        }
        Ok(prepared)
    }

    fn prepare_filter_subqueries(
        &self,
        subs: &[FilterSubquery],
        ctx: &QueryContext,
    ) -> Result<Vec<PreparedFilterSubquery<P>>> {
        let mut prepared = Vec::with_capacity(subs.len());
        for sub in subs {
            let prepared_plan = if sub.correlated_columns.is_empty() {
                Some(Arc::new(self.prepare_select(
                    (*sub.plan).clone(),
                    None,
                    Some(&format!("filter subquery {:?}", sub.id)),
                    ctx,
                )?))
            } else {
                None
            };
            prepared.push(PreparedFilterSubquery {
                id: sub.id,
                correlated_columns: Arc::new(sub.correlated_columns.clone()),
                prepared_plan,
                template: Arc::new(*sub.plan.clone()),
            });
        }
        Ok(prepared)
    }

    fn prepare_compound(
        &self,
        compound: &CompoundSelectPlan,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
        ctx: &QueryContext,
    ) -> Result<PreparedCompoundSelect<P>> {
        let initial = Arc::new(self.prepare_select(
            (*compound.initial).clone(),
            row_filter.clone(),
            Some("compound initial"),
            ctx,
        )?);
        let mut operations = Vec::with_capacity(compound.operations.len());
        for op in &compound.operations {
            operations.push(PreparedCompoundOp {
                operator: op.operator.clone(),
                quantifier: op.quantifier.clone(),
                plan: Arc::new(self.prepare_select(
                    op.plan.clone(),
                    row_filter.clone(),
                    Some(&format!("compound {:?}", op.operator)),
                    ctx,
                )?),
            });
        }
        Ok(PreparedCompoundSelect {
            initial,
            operations: Arc::new(operations),
        })
    }

    fn split_predicate(
        expr: &Expr<'static, String>,
    ) -> (Option<Expr<'static, String>>, Option<Expr<'static, String>>) {
        if !contains_exists(expr) {
            return (Some(expr.clone()), None);
        }

        match expr {
            Expr::And(list) => {
                let mut pushable = Vec::new();
                let mut residual = Vec::new();
                for e in list {
                    let (p, r) = Self::split_predicate(e);
                    if let Some(p) = p {
                        pushable.push(p);
                    }
                    if let Some(r) = r {
                        residual.push(r);
                    }
                }
                let p = if pushable.is_empty() {
                    None
                } else {
                    Some(Expr::And(pushable))
                };
                let r = if residual.is_empty() {
                    None
                } else {
                    Some(Expr::And(residual))
                };
                (p, r)
            }
            _ => (None, Some(expr.clone())),
        }
    }
}

impl<P> PreparedSelectPlanner<P> for DefaultPreparedSelectPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn prepare_select(
        &self,
        mut plan: SelectPlan,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
        _context: Option<&str>,
        ctx: &QueryContext,
    ) -> Result<PreparedSelectPlan<P>> {
        llkv_perf_monitor::maybe_record!(["perf-mon"], ctx, "simplify", {
            for proj in &mut plan.projections {
                if let crate::plans::SelectProjection::Computed { expr, .. } = proj {
                    *expr = llkv_compute::eval::ScalarEvaluator::simplify(expr);
                }
            }

            if let Some(having) = &mut plan.having {
                *having = simplify_expr(having.clone());
            }
        });

        let res = llkv_perf_monitor::maybe_record!(["perf-mon"], ctx, "subqueries", {
            let prepared_scalar_subqueries =
                self.prepare_scalar_subqueries(&plan.scalar_subqueries, ctx)?;

            let filter_subqueries = plan
                .filter
                .as_ref()
                .map(|f| self.prepare_filter_subqueries(&f.subqueries, ctx))
                .transpose()?
                .unwrap_or_default();

            let compound = plan
                .compound
                .as_ref()
                .map(|c| self.prepare_compound(c, row_filter.clone(), ctx))
                .transpose()?;

            Ok::<_, llkv_result::Error>((prepared_scalar_subqueries, filter_subqueries, compound))
        });
        let (prepared_scalar_subqueries, filter_subqueries, compound) = res?;

        let (
            plan_for_execution,
            residual_filter,
            residual_filter_subqueries,
            force_manual_projection,
            plan_for_scan,
            has_aggregates,
        ) = llkv_perf_monitor::maybe_record!(["perf-mon"], ctx, "analysis", {
            if let Some(filter) = &mut plan.filter {
                filter.predicate = simplify_expr(filter.predicate.clone());
            }

            let (pushable_filter, residual_filter) = if let Some(filter) = &plan.filter {
                Self::split_predicate(&filter.predicate)
            } else {
                (None, None)
            };

            let residual_filter_subqueries = if residual_filter.is_some() {
                plan.filter
                    .as_ref()
                    .map(|f| f.subqueries.clone())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let mut plan_for_execution = plan.clone();
            plan_for_execution.filter = pushable_filter.map(|predicate| SelectFilter {
                predicate,
                subqueries: plan
                    .filter
                    .as_ref()
                    .map(|f| f.subqueries.clone())
                    .unwrap_or_default(),
            });

            let has_subqueries = plan_for_execution.projections.iter().any(|p| match p {
                SelectProjection::Computed { expr, .. } => contains_subquery(expr),
                _ => false,
            });

            let has_aggregates = !plan_for_execution.aggregates.is_empty()
                || plan_for_execution.projections.iter().any(|p| match p {
                    SelectProjection::Computed { expr, .. } => scalar_contains_aggregate(expr),
                    _ => false,
                })
                || plan_for_execution
                    .having
                    .as_ref()
                    .map(|h| contains_aggregate_in_expr(h))
                    .unwrap_or(false);

            let group_needs_full_row = plan_for_execution.aggregates.is_empty()
                && !plan_for_execution.group_by.is_empty()
                && !has_aggregates;

            let force_manual_projection = (residual_filter.is_some()
                || has_subqueries
                || group_needs_full_row
                || has_aggregates)
                && !plan_for_execution.tables.is_empty();

            let plan_for_scan = if force_manual_projection {
                let mut p = plan_for_execution.clone();
                p.projections = vec![SelectProjection::AllColumns];
                p.order_by = Vec::new();
                p
            } else {
                plan_for_execution.clone()
            };

            (
                plan_for_execution,
                residual_filter,
                residual_filter_subqueries,
                force_manual_projection,
                plan_for_scan,
                has_aggregates,
            )
        });

        let res = llkv_perf_monitor::maybe_record!(
            ["perf-mon"],
            ctx,
            "logical_plan",
            self.logical_planner
                .create_logical_plan(&plan_for_scan, ctx)
        );
        let mut logical_plan = res?;

        llkv_perf_monitor::maybe_record!(["perf-mon"], ctx, "inference", {
            // Ensure the expression output type matches the inferred schema type.
            // This is necessary because some expressions (like CASE) might evaluate to an untyped NullArray
            // even if the schema inference determined a specific type (e.g. Int64).
            // Wrapping in Cast ensures the Executor produces the correct array type.
            if let LogicalPlan::Single(single_plan) = &mut logical_plan {
                let single_plan = single_plan.as_mut();
                for proj in &mut single_plan.requested_projections {
                    if let llkv_scan::ScanProjection::Computed { expr, .. } = proj
                        && let Ok(inferred_type) =
                            crate::translation::schema::infer_computed_data_type(
                                &single_plan.schema,
                                expr,
                            )
                        && inferred_type != arrow::datatypes::DataType::Null
                    {
                        match expr {
                            ScalarExpr::Literal(_)
                            | ScalarExpr::Column(_)
                            | ScalarExpr::Cast { .. } => {}
                            _ => {
                                *expr = ScalarExpr::Cast {
                                    expr: Box::new(expr.clone()),
                                    data_type: inferred_type,
                                };
                            }
                        }
                    }
                }
            }
        });

        let res = llkv_perf_monitor::maybe_record!(["perf-mon"], ctx, "rewrite", {
            if force_manual_projection && has_aggregates {
                match &mut logical_plan {
                    LogicalPlan::Single(single) => {
                        let single = single.as_mut();
                        // When forcing manual projection, the logical plan was created with AllColumns,
                        // which causes the aggregate rewrite to include all columns in the final output.
                        // We need to re-compute the rewrite using the original projections.
                        single.aggregate_rewrite =
                            build_single_aggregate_rewrite(&plan_for_execution, &single.schema)?;
                    }
                    LogicalPlan::Multi(multi) => {
                        let multi = multi.as_mut();
                        let mut infos = Vec::with_capacity(multi.tables.len());
                        for table_ref in &plan_for_execution.tables {
                            let table_lower = table_ref.table.to_ascii_lowercase();
                            let schema_lower = table_ref.schema.to_ascii_lowercase();
                            let qualified_lower = if schema_lower.is_empty() {
                                table_lower.clone()
                            } else {
                                format!("{}.{}", schema_lower, table_lower)
                            };
                            let alias_lower =
                                table_ref.alias.as_ref().map(|a| a.to_ascii_lowercase());
                            infos.push(TableResolutionInfo {
                                table_lower,
                                schema_lower,
                                qualified_lower,
                                alias_lower,
                            });
                        }

                        let ctx = ResolutionContext {
                            tables: &multi.tables,
                            infos: &infos,
                        };
                        multi.aggregate_rewrite =
                            build_multi_aggregate_rewrite(&plan_for_execution, &ctx)?;
                    }
                }
            }
            Ok::<_, llkv_result::Error>(())
        });
        res?;

        // let physical_plan = Some(
        //     self.physical_planner
        //         .create_physical_plan(&logical_plan, row_filter.clone())
        //         .map_err(Error::Internal)?,
        // );

        Ok(PreparedSelectPlan {
            plan: Arc::new(plan_for_execution),
            logical_plan: Arc::new(logical_plan),
            // physical_plan,
            scalar_subqueries: Arc::from(prepared_scalar_subqueries),
            filter_subqueries: Arc::from(filter_subqueries),
            compound: compound.map(Arc::new),
            residual_filter: residual_filter.map(Arc::new),
            residual_filter_subqueries: Arc::from(residual_filter_subqueries),
            force_manual_projection,
            row_filter,
        })
    }
}

fn contains_exists(expr: &Expr<'static, String>) -> bool {
    match expr {
        Expr::Exists(_) => true,
        Expr::And(l) | Expr::Or(l) => l.iter().any(contains_exists),
        Expr::Not(e) => contains_exists(e),
        _ => false,
    }
}

fn contains_subquery<F: std::fmt::Debug>(expr: &ScalarExpr<F>) -> bool {
    match expr {
        ScalarExpr::ScalarSubquery(_) => true,
        ScalarExpr::Binary { left, right, .. } => {
            contains_subquery(left) || contains_subquery(right)
        }
        ScalarExpr::Not(e) => contains_subquery(e),
        ScalarExpr::IsNull { expr, .. } => contains_subquery(expr),
        ScalarExpr::Cast { expr, .. } => contains_subquery(expr),
        ScalarExpr::Compare { left, right, .. } => {
            contains_subquery(left) || contains_subquery(right)
        }
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            operand.as_ref().is_some_and(|o| contains_subquery::<F>(o))
                || branches
                    .iter()
                    .any(|(w, t)| contains_subquery(w) || contains_subquery(t))
                || else_expr
                    .as_ref()
                    .is_some_and(|e| contains_subquery::<F>(e))
        }
        ScalarExpr::Coalesce(list) => list.iter().any(contains_subquery),
        ScalarExpr::GetField { base, .. } => contains_subquery(base),
        ScalarExpr::Aggregate(call) => match call {
            AggregateCall::Count { expr, .. }
            | AggregateCall::Sum { expr, .. }
            | AggregateCall::Total { expr, .. }
            | AggregateCall::Avg { expr, .. }
            | AggregateCall::Min(expr)
            | AggregateCall::Max(expr)
            | AggregateCall::CountNulls(expr)
            | AggregateCall::GroupConcat { expr, .. } => contains_subquery(expr),
            AggregateCall::CountStar => false,
        },
        _ => false,
    }
}

fn scalar_contains_aggregate<F>(expr: &ScalarExpr<F>) -> bool {
    match expr {
        ScalarExpr::Aggregate(_) => true,
        ScalarExpr::Binary { left, right, .. } | ScalarExpr::Compare { left, right, .. } => {
            scalar_contains_aggregate(left) || scalar_contains_aggregate(right)
        }
        ScalarExpr::Not(e)
        | ScalarExpr::Cast { expr: e, .. }
        | ScalarExpr::IsNull { expr: e, .. }
        | ScalarExpr::GetField { base: e, .. } => scalar_contains_aggregate(e),
        ScalarExpr::Coalesce(items) => items.iter().any(scalar_contains_aggregate),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            operand
                .as_ref()
                .is_some_and(|o| scalar_contains_aggregate(o))
                || branches
                    .iter()
                    .any(|(w, t)| scalar_contains_aggregate(w) || scalar_contains_aggregate(t))
                || else_expr
                    .as_ref()
                    .is_some_and(|e| scalar_contains_aggregate(e))
        }
        _ => false,
    }
}

fn contains_aggregate_in_expr(expr: &LlkvExpr<'_, String>) -> bool {
    match expr {
        LlkvExpr::Pred(_) | LlkvExpr::Literal(_) | LlkvExpr::Exists(_) => false,
        LlkvExpr::Compare { left, right, .. } => {
            scalar_contains_aggregate(left) || scalar_contains_aggregate(right)
        }
        LlkvExpr::InList { expr, list, .. } => {
            scalar_contains_aggregate(expr) || list.iter().any(scalar_contains_aggregate)
        }
        LlkvExpr::IsNull { expr, .. } => scalar_contains_aggregate(expr),
        LlkvExpr::And(list) | LlkvExpr::Or(list) => list.iter().any(contains_aggregate_in_expr),
        LlkvExpr::Not(inner) => contains_aggregate_in_expr(inner),
    }
}

fn simplify_expr<F: std::hash::Hash + Eq + Clone>(expr: Expr<'static, F>) -> Expr<'static, F> {
    match expr {
        Expr::And(exprs) => Expr::And(exprs.into_iter().map(simplify_expr).collect()),
        Expr::Or(exprs) => Expr::Or(exprs.into_iter().map(simplify_expr).collect()),
        Expr::Not(expr) => Expr::Not(Box::new(simplify_expr(*expr))),
        Expr::Pred(filter) => Expr::Pred(filter),
        Expr::Compare { left, op, right } => Expr::Compare {
            left: llkv_compute::eval::ScalarEvaluator::simplify(&left),
            op,
            right: llkv_compute::eval::ScalarEvaluator::simplify(&right),
        },
        Expr::InList {
            expr,
            list,
            negated,
        } => Expr::InList {
            expr: llkv_compute::eval::ScalarEvaluator::simplify(&expr),
            list: list
                .into_iter()
                .map(|e| llkv_compute::eval::ScalarEvaluator::simplify(&e))
                .collect(),
            negated,
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: llkv_compute::eval::ScalarEvaluator::simplify(&expr),
            negated,
        },
        Expr::Literal(b) => Expr::Literal(b),
        Expr::Exists(sub) => Expr::Exists(sub),
    }
}
