//! Logical planning for translating [`SelectPlan`]s into storage-aware requests.
//!
//! The logical planner resolves projections, ORDER BY targets, and filter
//! expressions against table schemas supplied by a [`TableProvider`]. The
//! resulting plans carry resolved schemas so physical planning can focus on
//! execution path selection (index usage, pruning, sorted scans, etc.).

use std::sync::Arc;

use arrow::datatypes::Schema;
use llkv_column_map::store::Projection as StoreProjection;
use llkv_expr::expr::Expr;
use llkv_result::Error;
use llkv_result::Result;
use llkv_scan::ScanProjection;
use llkv_storage::pager::Pager;
use llkv_types::{FieldId, LogicalFieldId, TableId};
use simd_r_drive_entry_handle::EntryHandle;

use crate::physical::table::{ExecutionTable, TableProvider};
use crate::plans::{OrderByPlan, OrderTarget, SelectPlan, TableRef};
use crate::schema::PlanSchema;
use crate::translation::{
    build_projected_columns, build_wildcard_projections, schema_for_projections,
    translate_predicate,
};

/// Logical plan with projections and predicates resolved against table metadata.
pub enum LogicalPlan<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    Single(SingleTableLogicalPlan<P>),
    Multi(MultiTableLogicalPlan<P>),
}

/// Logical plan metadata for a single-table query.
pub struct SingleTableLogicalPlan<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub table_name: String,
    pub table_id: TableId,
    pub table: Arc<dyn ExecutionTable<P>>,
    pub schema: Arc<PlanSchema>,
    pub requested_projections: Vec<ScanProjection>,
    pub scan_projections: Vec<ScanProjection>,
    pub final_schema: Arc<Schema>,
    pub scan_schema: Arc<Schema>,
    pub resolved_order_by: Vec<OrderByPlan>,
    pub filter: Option<Expr<'static, FieldId>>,
    pub original_filter: Option<Expr<'static, String>>,
    pub extra_columns: Vec<String>,
}

impl<P> SingleTableLogicalPlan<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn has_extra_columns(&self) -> bool {
        !self.extra_columns.is_empty()
    }
}

/// Logical metadata for multi-table queries. Projections remain expressed in
/// terms of column names to allow join/cross-product planning layers to apply
/// their own resolution strategies while benefiting from catalog-backed table
/// validation.
pub struct MultiTableLogicalPlan<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub tables: Vec<PlannedTable<P>>,
    pub table_order: Vec<TableRef>,
    pub filter: Option<Expr<'static, ResolvedFieldRef>>,
    pub original_filter: Option<Expr<'static, String>>,
    pub having: Option<Expr<'static, ResolvedFieldRef>>,
    pub original_having: Option<Expr<'static, String>>,
    /// Joins with ON clauses resolved to concrete tables/fields.
    pub joins: Vec<ResolvedJoin>,
    /// Projections resolved to concrete columns or computed expressions.
    pub projections: Vec<ResolvedProjection>,
    /// Aggregates resolved to concrete input columns.
    pub aggregates: Vec<ResolvedAggregate>,
    /// GROUP BY keys resolved to concrete columns.
    pub group_by: Vec<ResolvedGroupKey>,
    /// ORDER BY targets resolved to concrete columns when possible.
    pub order_by: Vec<ResolvedOrderBy>,
    pub distinct: bool,
    pub compound: Option<crate::plans::CompoundSelectPlan>,
    /// Columns resolved to concrete tables/fields for pushdown.
    pub resolved_required: Vec<ResolvedColumn>,
    /// Names we could not resolve uniquely (ambiguous or missing).
    pub unresolved_required: Vec<String>,
}

pub struct PlannedTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub name: String,
    pub table_id: TableId,
    pub table: Arc<dyn ExecutionTable<P>>,
    pub schema: Arc<PlanSchema>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedColumn {
    pub original: String,
    pub table_index: usize,
    pub logical_field_id: LogicalFieldId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedFieldRef {
    pub table_index: usize,
    pub logical_field_id: LogicalFieldId,
}

#[derive(Clone, Debug)]
pub struct ResolvedJoin {
    pub left_table_index: usize,
    pub join_type: crate::plans::JoinPlan,
    pub on: Option<Expr<'static, ResolvedFieldRef>>, // predicate with resolved columns
    pub original_on: Option<Expr<'static, String>>,
}

#[derive(Clone, Debug)]
pub enum ResolvedProjection {
    Column {
        table_index: usize,
        logical_field_id: LogicalFieldId,
        alias: Option<String>,
    },
    Computed {
        expr: llkv_expr::expr::ScalarExpr<ResolvedFieldRef>,
        alias: String,
    },
}

#[derive(Clone, Debug)]
pub struct ResolvedAggregate {
    pub input: Option<(usize, LogicalFieldId)>,
    pub alias: String,
    pub function: crate::plans::AggregateFunction,
    pub distinct: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedGroupKey {
    pub table_index: usize,
    pub logical_field_id: LogicalFieldId,
}

#[derive(Clone, Debug)]
pub struct ResolvedOrderBy {
    pub target: crate::plans::OrderTarget,
    pub sort_type: crate::plans::OrderSortType,
    pub ascending: bool,
    pub nulls_first: bool,
    /// If the target is a column, record the resolved location.
    pub resolved_column: Option<(usize, LogicalFieldId)>,
}

/// Planner that resolves column references and ORDER BY targets before physical planning.
pub struct LogicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    provider: Arc<dyn TableProvider<P>>,
}

impl<P> LogicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn TableProvider<P>>) -> Self {
        Self { provider }
    }

    pub fn get_table_schema(&self, table_name: &str) -> Result<Arc<PlanSchema>> {
        let table = self.provider.get_table(table_name)?;
        Ok(table.schema())
    }

    pub fn create_logical_plan(&self, plan: &SelectPlan) -> Result<LogicalPlan<P>> {
        if plan.tables.len() != 1 {
            return self.plan_multi_table(plan);
        }

        let table_ref = plan.tables.first().expect("validated length above");
        let table_name = table_ref.qualified_name();
        let table = self.provider.get_table(&table_name)?;
        let table_id = table.table_id();
        let schema = table.schema();

        let requested_projections = if plan.projections.is_empty() {
            build_wildcard_projections(&schema, table_id)
        } else {
            build_projected_columns(&schema, table_id, &plan.projections)?
        };

        let mut scan_projections = requested_projections.clone();
        let mut extra_columns: Vec<String> = Vec::new();

        // Ensure filter columns are available for scan-level evaluation.
        if let Some(filter) = &plan.filter {
            for name in collect_filter_columns(&filter.predicate) {
                push_column_if_known(
                    table_id,
                    &schema,
                    &name,
                    &mut scan_projections,
                    &mut extra_columns,
                );
            }
        }

        // Ensure GROUP BY and aggregate inputs are present for scan-level computation.
        for name in &plan.group_by {
            push_column_if_known(
                table_id,
                &schema,
                name,
                &mut scan_projections,
                &mut extra_columns,
            );
        }
        for agg in &plan.aggregates {
            for name in collect_aggregate_columns(agg) {
                push_column_if_known(
                    table_id,
                    &schema,
                    &name,
                    &mut scan_projections,
                    &mut extra_columns,
                );
            }
        }

        let OrderResolution {
            scan_projections: order_scan_projections,
            extra_columns: order_extra_columns,
            resolved_order_by,
        } = resolve_order_by_targets(&schema, table_id, &scan_projections, &plan.order_by)?;

        scan_projections = order_scan_projections;
        for col in order_extra_columns {
            if !extra_columns
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&col))
            {
                extra_columns.push(col);
            }
        }

        let scan_schema = schema_for_projections(&schema, &scan_projections)?;
        let final_schema = schema_for_projections(&schema, &requested_projections)?;

        let filter = match &plan.filter {
            Some(filter) => Some(translate_predicate(
                filter.predicate.clone(),
                &schema,
                |_| Error::Internal("Unknown column".to_string()),
            )?),
            None => None,
        };

        Ok(LogicalPlan::Single(SingleTableLogicalPlan {
            table_name,
            table,
            table_id,
            schema,
            requested_projections,
            scan_projections,
            final_schema,
            scan_schema,
            resolved_order_by,
            filter,
            original_filter: plan.filter.as_ref().map(|f| f.predicate.clone()),
            extra_columns,
        }))
    }

    fn plan_multi_table(&self, plan: &SelectPlan) -> Result<LogicalPlan<P>> {
        let mut planned_tables = Vec::with_capacity(plan.tables.len());
        for table_ref in &plan.tables {
            let table_name = table_ref.qualified_name();
            let table = self.provider.get_table(&table_name)?;
            let table_id = table.table_id();
            let schema = table.schema();

            planned_tables.push(PlannedTable {
                name: table_name,
                table_id,
                table,
                schema,
            });
        }

        let (resolved_columns, unresolved_required) =
            resolve_required_columns(plan, &planned_tables);

        let ctx = ResolutionContext {
            tables: &planned_tables,
            table_refs: &plan.tables,
        };

        let filter = match &plan.filter {
            Some(filter) => Some(resolve_predicate(&ctx, &filter.predicate)?),
            None => None,
        };

        let having = match &plan.having {
            Some(having) => Some(resolve_predicate(&ctx, having)?),
            None => None,
        };

        let projections = resolve_projections(plan, &ctx)?;
        let aggregates = resolve_aggregates(plan, &ctx)?;
        let group_by = resolve_group_by(plan, &ctx)?;
        let order_by = resolve_order_by(plan, &ctx)?;
        let joins = resolve_joins(plan, &ctx)?;

        Ok(LogicalPlan::Multi(MultiTableLogicalPlan {
            tables: planned_tables,
            table_order: plan.tables.clone(),
            filter,
            original_filter: plan.filter.as_ref().map(|f| f.predicate.clone()),
            having,
            original_having: plan.having.clone(),
            joins,
            projections,
            aggregates,
            group_by,
            order_by,
            distinct: plan.distinct,
            compound: plan.compound.clone(),
            resolved_required: resolved_columns,
            unresolved_required,
        }))
    }
}

struct OrderResolution {
    scan_projections: Vec<ScanProjection>,
    extra_columns: Vec<String>,
    resolved_order_by: Vec<OrderByPlan>,
}

fn resolve_order_by_targets(
    schema: &PlanSchema,
    table_id: TableId,
    requested_projections: &[ScanProjection],
    order_by: &[OrderByPlan],
) -> Result<OrderResolution> {
    let mut scan_projections = requested_projections.to_vec();
    let mut extra_columns = Vec::new();
    let mut resolved_order_by = Vec::new();

    for order in order_by {
        match &order.target {
            OrderTarget::All => {
                for proj in requested_projections {
                    let name = projection_name(proj, schema);
                    let mut new_order = order.clone();
                    new_order.target = OrderTarget::Column(name);
                    resolved_order_by.push(new_order);
                }
            }
            OrderTarget::Index(idx) => {
                if *idx < requested_projections.len() {
                    let proj = &requested_projections[*idx];
                    let name = projection_name(proj, schema);
                    let mut new_order = order.clone();
                    new_order.target = OrderTarget::Column(name);
                    resolved_order_by.push(new_order);
                } else {
                    return Err(Error::InvalidArgumentError(format!(
                        "ORDER BY index {} out of bounds",
                        idx
                    )));
                }
            }
            OrderTarget::Column(name) => {
                let already_projected = scan_projections
                    .iter()
                    .any(|p| projection_name(p, schema).eq_ignore_ascii_case(name.as_str()));

                if !already_projected {
                    if let Some(col) = schema
                        .columns
                        .iter()
                        .find(|c| c.name.eq_ignore_ascii_case(name))
                    {
                        let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
                        let projection = ScanProjection::Column(StoreProjection {
                            logical_field_id,
                            alias: None,
                        });
                        scan_projections.push(projection);
                        extra_columns.push(name.clone());
                    } else {
                        return Err(Error::InvalidArgumentError(format!(
                            "ORDER BY references unknown column '{name}'"
                        )));
                    }
                }

                let mut new_order = order.clone();
                new_order.target = OrderTarget::Column(name.clone());
                resolved_order_by.push(new_order);
            }
        }
    }

    Ok(OrderResolution {
        scan_projections,
        extra_columns,
        resolved_order_by,
    })
}

/// Resolve the display name for a projection (alias or canonical column name).
pub fn projection_name(proj: &ScanProjection, schema: &PlanSchema) -> String {
    match proj {
        ScanProjection::Column(p) => p.alias.clone().unwrap_or_else(|| {
            schema
                .columns
                .iter()
                .find(|c| c.field_id == p.logical_field_id.field_id())
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("col_{}", p.logical_field_id.field_id()))
        }),
        ScanProjection::Computed { alias, .. } => alias.clone(),
    }
}

fn push_column_if_known(
    table_id: TableId,
    schema: &PlanSchema,
    name: &str,
    scan_projections: &mut Vec<ScanProjection>,
    extra_columns: &mut Vec<String>,
) {
    if scan_projections
        .iter()
        .any(|p| projection_name(p, schema).eq_ignore_ascii_case(name))
    {
        return;
    }
    if let Some(col) = schema
        .columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(name))
    {
        let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
        scan_projections.push(ScanProjection::Column(StoreProjection {
            logical_field_id,
            alias: None,
        }));
        if !extra_columns
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(name))
        {
            extra_columns.push(name.to_string());
        }
    }
}

fn collect_filter_columns(expr: &Expr<'static, String>) -> Vec<String> {
    let mut cols = Vec::new();
    collect_filter_columns_inner(expr, &mut cols);
    cols
}

fn collect_filter_columns_inner(expr: &Expr<'static, String>, out: &mut Vec<String>) {
    match expr {
        Expr::Pred(filter) => {
            out.push(filter.field_id.clone());
        }
        Expr::And(list) | Expr::Or(list) => {
            for e in list {
                collect_filter_columns_inner(e, out);
            }
        }
        Expr::Not(inner) => collect_filter_columns_inner(inner, out),
        Expr::Compare { left, right, .. } => {
            collect_scalar_columns(left, out);
            collect_scalar_columns(right, out);
        }
        Expr::InList { expr, list, .. } => {
            collect_scalar_columns(expr, out);
            for item in list {
                collect_scalar_columns(item, out);
            }
        }
        Expr::IsNull { expr, .. } => collect_scalar_columns(expr, out),
        Expr::Literal(_) => {}
        Expr::Exists(_) => {}
    }
}

fn collect_scalar_columns(expr: &llkv_expr::expr::ScalarExpr<String>, out: &mut Vec<String>) {
    use llkv_expr::expr::ScalarExpr;
    match expr {
        ScalarExpr::Column(name) => out.push(name.clone()),
        ScalarExpr::Binary { left, right, .. } => {
            collect_scalar_columns(left, out);
            collect_scalar_columns(right, out);
        }
        ScalarExpr::Not(inner) => collect_scalar_columns(inner, out),
        ScalarExpr::Compare { left, right, .. } => {
            collect_scalar_columns(left, out);
            collect_scalar_columns(right, out);
        }
        ScalarExpr::IsNull { expr, .. } => collect_scalar_columns(expr, out),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(op) = operand {
                collect_scalar_columns(op, out);
            }
            for (when, then) in branches {
                collect_scalar_columns(when, out);
                collect_scalar_columns(then, out);
            }
            if let Some(else_expr) = else_expr {
                collect_scalar_columns(else_expr, out);
            }
        }
        ScalarExpr::Coalesce(list) => {
            for item in list {
                collect_scalar_columns(item, out);
            }
        }
        ScalarExpr::Cast { expr, .. } => collect_scalar_columns(expr, out),
        ScalarExpr::GetField { base, .. } => collect_scalar_columns(base, out),
        ScalarExpr::Aggregate(_) => {}
        ScalarExpr::Literal(_) => {}
        ScalarExpr::Random => {}
        ScalarExpr::ScalarSubquery(_) => {}
    }
}

fn collect_aggregate_columns(agg: &crate::plans::AggregateExpr) -> Vec<String> {
    match agg {
        crate::plans::AggregateExpr::CountStar { .. } => Vec::new(),
        crate::plans::AggregateExpr::Column { column, .. } => vec![column.clone()],
    }
}

fn collect_required_columns(plan: &SelectPlan) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    // Projections
    for proj in &plan.projections {
        match proj {
            crate::plans::SelectProjection::Column { name, .. } => out.push(name.clone()),
            crate::plans::SelectProjection::Computed { expr, .. } => {
                collect_scalar_columns(expr, &mut out);
            }
            crate::plans::SelectProjection::AllColumns => {}
            crate::plans::SelectProjection::AllColumnsExcept { .. } => {}
        }
    }

    // Filters
    if let Some(filter) = &plan.filter {
        out.extend(collect_filter_columns(&filter.predicate));
    }

    // Group by and aggregates
    out.extend(plan.group_by.iter().cloned());
    for agg in &plan.aggregates {
        out.extend(collect_aggregate_columns(agg));
    }

    // Join ON predicates
    for join in &plan.joins {
        if let Some(on) = &join.on_condition {
            out.extend(collect_filter_columns(on));
        }
    }

    // ORDER BY
    for order in &plan.order_by {
        if let OrderTarget::Column(col) = &order.target {
            out.push(col.clone());
        }
    }

    // Dedup preserving order, case-insensitive
    let mut deduped: Vec<String> = Vec::new();
    for name in out {
        if !deduped
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(&name))
        {
            deduped.push(name);
        }
    }
    deduped
}

fn resolve_required_columns<P>(
    plan: &SelectPlan,
    tables: &[PlannedTable<P>],
) -> (Vec<ResolvedColumn>, Vec<String>)
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let required = collect_required_columns(plan);
    let mut resolved = Vec::new();
    let mut unresolved = Vec::new();

    for name in required {
        let parts: Vec<&str> = name.split('.').collect();
        let mut candidates = Vec::new();

        for (idx, table) in tables.iter().enumerate() {
            let Some(table_ref) = plan.tables.get(idx) else {
                continue;
            };
            let table_lower = table_ref.table.to_ascii_lowercase();
            let schema_lower = table_ref.schema.to_ascii_lowercase();
            let qualified_lower = if schema_lower.is_empty() {
                table_lower.clone()
            } else {
                format!("{}.{}", schema_lower, table_lower)
            };
            let alias_lower = table_ref.alias.as_ref().map(|a| a.to_ascii_lowercase());

            // If a qualifier exists, check it.
            if parts.len() > 1 {
                let qualifier = parts[0].to_ascii_lowercase();

                // Support schema.table.column by matching the first two components against the
                // fully-qualified table name and advancing the column start accordingly.
                let qualifier_matches_alias = alias_lower
                    .as_ref()
                    .is_some_and(|alias| qualifier == *alias);
                let qualifier_matches_table = alias_lower.is_none()
                    && (qualifier == qualified_lower
                        || qualifier == table_lower
                        || (!schema_lower.is_empty() && qualifier == schema_lower));
                let col_start = if alias_lower.is_none()
                    && parts.len() >= 3
                    && format!("{}.{}", qualifier, parts[1].to_ascii_lowercase()) == qualified_lower
                {
                    Some(2)
                } else if qualifier_matches_alias || qualifier_matches_table {
                    Some(1)
                } else {
                    None
                };

                if let Some(start_idx) = col_start {
                    let col_name = parts[start_idx..].join("."); // nested paths collapse; schema matching is on base column
                    if table.schema.column_by_name(&col_name).is_some() {
                        candidates.push((idx, col_name));
                    }
                }
            } else if table.schema.column_by_name(&name).is_some() {
                candidates.push((idx, name.clone()));
            }
        }

        if candidates.len() == 1 {
            let (table_idx, col_name) = candidates.pop().unwrap();
            if let Some(col) = tables[table_idx].schema.column_by_name(&col_name) {
                let lfid = LogicalFieldId::for_user(tables[table_idx].table_id, col.field_id);
                resolved.push(ResolvedColumn {
                    original: name.clone(),
                    table_index: table_idx,
                    logical_field_id: lfid,
                });
                continue;
            }
        }

        unresolved.push(name);
    }

    (resolved, unresolved)
}

struct ResolutionContext<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    tables: &'a [PlannedTable<P>],
    table_refs: &'a [TableRef],
}

fn resolve_column_ref<P>(ctx: &ResolutionContext<P>, name: &str) -> Result<(usize, LogicalFieldId)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let parts: Vec<&str> = name.split('.').collect();
    let mut candidates = Vec::new();

    for (idx, table) in ctx.tables.iter().enumerate() {
        let Some(table_ref) = ctx.table_refs.get(idx) else {
            continue;
        };
        let table_lower = table_ref.table.to_ascii_lowercase();
        let schema_lower = table_ref.schema.to_ascii_lowercase();
        let qualified_lower = if schema_lower.is_empty() {
            table_lower.clone()
        } else {
            format!("{}.{}", schema_lower, table_lower)
        };
        let alias_lower = table_ref.alias.as_ref().map(|a| a.to_ascii_lowercase());

        if parts.len() > 1 {
            let qualifier = parts[0].to_ascii_lowercase();

            let qualifier_matches_alias = alias_lower
                .as_ref()
                .is_some_and(|alias| qualifier == *alias);
            let qualifier_matches_table = alias_lower.is_none()
                && (qualifier == qualified_lower
                    || qualifier == table_lower
                    || (!schema_lower.is_empty() && qualifier == schema_lower));
            let col_start = if alias_lower.is_none()
                && parts.len() >= 3
                && format!("{}.{}", qualifier, parts[1].to_ascii_lowercase()) == qualified_lower
            {
                Some(2)
            } else if qualifier_matches_alias || qualifier_matches_table {
                Some(1)
            } else {
                None
            };

            if let Some(start_idx) = col_start {
                let col_name = parts[start_idx..].join(".");
                if let Some(col) = table.schema.column_by_name(&col_name) {
                    let lfid = LogicalFieldId::for_user(table.table_id, col.field_id);
                    candidates.push((idx, lfid));
                }
            }
        } else if let Some(col) = table.schema.column_by_name(name) {
            let lfid = LogicalFieldId::for_user(table.table_id, col.field_id);
            candidates.push((idx, lfid));
        }
    }

    match candidates.len() {
        0 => Err(Error::InvalidArgumentError(format!(
            "Unknown column '{name}' in multi-table query"
        ))),
        1 => Ok(candidates.pop().expect("matched len 1")),
        _ => Err(Error::InvalidArgumentError(format!(
            "Ambiguous column reference '{name}'"
        ))),
    }
}

fn resolve_scalar_expr<P>(
    ctx: &ResolutionContext<P>,
    expr: &llkv_expr::expr::ScalarExpr<String>,
) -> Result<llkv_expr::expr::ScalarExpr<ResolvedFieldRef>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    use llkv_expr::expr::{AggregateCall, ScalarExpr};

    Ok(match expr {
        ScalarExpr::Column(name) => {
            let (table_idx, lfid) = resolve_column_ref(ctx, name)?;
            ScalarExpr::Column(ResolvedFieldRef {
                table_index: table_idx,
                logical_field_id: lfid,
            })
        }
        ScalarExpr::Literal(lit) => ScalarExpr::Literal(lit.clone()),
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(resolve_scalar_expr(ctx, left)?),
            op: *op,
            right: Box::new(resolve_scalar_expr(ctx, right)?),
        },
        ScalarExpr::Not(inner) => ScalarExpr::Not(Box::new(resolve_scalar_expr(ctx, inner)?)),
        ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
            expr: Box::new(resolve_scalar_expr(ctx, expr)?),
            negated: *negated,
        },
        ScalarExpr::Aggregate(call) => ScalarExpr::Aggregate(match call {
            AggregateCall::CountStar => AggregateCall::CountStar,
            AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                expr: Box::new(resolve_scalar_expr(ctx, expr)?),
                distinct: *distinct,
            },
            AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                expr: Box::new(resolve_scalar_expr(ctx, expr)?),
                distinct: *distinct,
            },
            AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                expr: Box::new(resolve_scalar_expr(ctx, expr)?),
                distinct: *distinct,
            },
            AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                expr: Box::new(resolve_scalar_expr(ctx, expr)?),
                distinct: *distinct,
            },
            AggregateCall::Min(expr) => {
                AggregateCall::Min(Box::new(resolve_scalar_expr(ctx, expr)?))
            }
            AggregateCall::Max(expr) => {
                AggregateCall::Max(Box::new(resolve_scalar_expr(ctx, expr)?))
            }
            AggregateCall::CountNulls(expr) => {
                AggregateCall::CountNulls(Box::new(resolve_scalar_expr(ctx, expr)?))
            }
            AggregateCall::GroupConcat {
                expr,
                distinct,
                separator,
            } => AggregateCall::GroupConcat {
                expr: Box::new(resolve_scalar_expr(ctx, expr)?),
                distinct: *distinct,
                separator: separator.clone(),
            },
        }),
        ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
            base: Box::new(resolve_scalar_expr(ctx, base)?),
            field_name: field_name.clone(),
        },
        ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
            expr: Box::new(resolve_scalar_expr(ctx, expr)?),
            data_type: data_type.clone(),
        },
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(resolve_scalar_expr(ctx, left)?),
            op: *op,
            right: Box::new(resolve_scalar_expr(ctx, right)?),
        },
        ScalarExpr::Coalesce(list) => ScalarExpr::Coalesce(
            list.iter()
                .map(|item| resolve_scalar_expr(ctx, item))
                .collect::<Result<Vec<_>>>()?,
        ),
        ScalarExpr::ScalarSubquery(subq) => ScalarExpr::ScalarSubquery(subq.clone()),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => ScalarExpr::Case {
            operand: operand
                .as_deref()
                .map(|op| resolve_scalar_expr(ctx, op))
                .transpose()?
                .map(Box::new),
            branches: {
                let mut out = Vec::with_capacity(branches.len());
                for (when, then) in branches {
                    out.push((
                        resolve_scalar_expr(ctx, when)?,
                        resolve_scalar_expr(ctx, then)?,
                    ));
                }
                out
            },
            else_expr: else_expr
                .as_deref()
                .map(|e| resolve_scalar_expr(ctx, e))
                .transpose()?
                .map(Box::new),
        },
        ScalarExpr::Random => ScalarExpr::Random,
    })
}

fn resolve_predicate<P>(
    ctx: &ResolutionContext<P>,
    expr: &Expr<'static, String>,
) -> Result<Expr<'static, ResolvedFieldRef>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    Ok(match expr {
        Expr::And(list) => Expr::And(
            list.iter()
                .map(|inner| resolve_predicate(ctx, inner))
                .collect::<Result<Vec<_>>>()?,
        ),
        Expr::Or(list) => Expr::Or(
            list.iter()
                .map(|inner| resolve_predicate(ctx, inner))
                .collect::<Result<Vec<_>>>()?,
        ),
        Expr::Not(inner) => Expr::Not(Box::new(resolve_predicate(ctx, inner)?)),
        Expr::Pred(filter) => {
            let (table_idx, lfid) = resolve_column_ref(ctx, &filter.field_id)?;
            Expr::Pred(llkv_expr::expr::Filter {
                field_id: ResolvedFieldRef {
                    table_index: table_idx,
                    logical_field_id: lfid,
                },
                op: filter.op.clone(),
            })
        }
        Expr::Compare { left, op, right } => Expr::Compare {
            left: resolve_scalar_expr(ctx, left)?,
            op: *op,
            right: resolve_scalar_expr(ctx, right)?,
        },
        Expr::InList {
            expr,
            list,
            negated,
        } => Expr::InList {
            expr: resolve_scalar_expr(ctx, expr)?,
            list: list
                .iter()
                .map(|item| resolve_scalar_expr(ctx, item))
                .collect::<Result<Vec<_>>>()?,
            negated: *negated,
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: resolve_scalar_expr(ctx, expr)?,
            negated: *negated,
        },
        Expr::Literal(v) => Expr::Literal(*v),
        Expr::Exists(subq) => Expr::Exists(subq.clone()),
    })
}

fn resolve_projections<P>(
    plan: &SelectPlan,
    ctx: &ResolutionContext<P>,
) -> Result<Vec<ResolvedProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut out = Vec::new();
    let projections = if plan.projections.is_empty() {
        vec![crate::plans::SelectProjection::AllColumns]
    } else {
        plan.projections.clone()
    };

    for proj in projections {
        match proj {
            crate::plans::SelectProjection::AllColumns => {
                for (table_idx, table) in ctx.tables.iter().enumerate() {
                    for col in &table.schema.columns {
                        out.push(ResolvedProjection::Column {
                            table_index: table_idx,
                            logical_field_id: LogicalFieldId::for_user(
                                table.table_id,
                                col.field_id,
                            ),
                            alias: None,
                        });
                    }
                }
            }
            crate::plans::SelectProjection::AllColumnsExcept { exclude } => {
                for (table_idx, table) in ctx.tables.iter().enumerate() {
                    for col in &table.schema.columns {
                        if exclude.iter().any(|ex| ex.eq_ignore_ascii_case(&col.name)) {
                            continue;
                        }
                        out.push(ResolvedProjection::Column {
                            table_index: table_idx,
                            logical_field_id: LogicalFieldId::for_user(
                                table.table_id,
                                col.field_id,
                            ),
                            alias: None,
                        });
                    }
                }
            }
            crate::plans::SelectProjection::Column { name, alias } => {
                let (table_idx, lfid) = resolve_column_ref(ctx, &name)?;
                out.push(ResolvedProjection::Column {
                    table_index: table_idx,
                    logical_field_id: lfid,
                    alias,
                });
            }
            crate::plans::SelectProjection::Computed { expr, alias } => {
                out.push(ResolvedProjection::Computed {
                    expr: resolve_scalar_expr(ctx, &expr)?,
                    alias,
                });
            }
        }
    }

    Ok(out)
}

fn resolve_aggregates<P>(
    plan: &SelectPlan,
    ctx: &ResolutionContext<P>,
) -> Result<Vec<ResolvedAggregate>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut out = Vec::with_capacity(plan.aggregates.len());
    for agg in &plan.aggregates {
        match agg {
            crate::plans::AggregateExpr::CountStar { alias, distinct } => {
                out.push(ResolvedAggregate {
                    input: None,
                    alias: alias.clone(),
                    function: crate::plans::AggregateFunction::Count,
                    distinct: *distinct,
                })
            }
            crate::plans::AggregateExpr::Column {
                column,
                alias,
                function,
                distinct,
            } => {
                let (table_idx, lfid) = resolve_column_ref(ctx, column)?;
                out.push(ResolvedAggregate {
                    input: Some((table_idx, lfid)),
                    alias: alias.clone(),
                    function: function.clone(),
                    distinct: *distinct,
                });
            }
        }
    }
    Ok(out)
}

fn resolve_group_by<P>(
    plan: &SelectPlan,
    ctx: &ResolutionContext<P>,
) -> Result<Vec<ResolvedGroupKey>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut out = Vec::with_capacity(plan.group_by.len());
    for key in &plan.group_by {
        let (table_idx, lfid) = resolve_column_ref(ctx, key)?;
        out.push(ResolvedGroupKey {
            table_index: table_idx,
            logical_field_id: lfid,
        });
    }
    Ok(out)
}

fn resolve_order_by<P>(
    plan: &SelectPlan,
    ctx: &ResolutionContext<P>,
) -> Result<Vec<ResolvedOrderBy>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut out = Vec::with_capacity(plan.order_by.len());
    for entry in &plan.order_by {
        let resolved_column = match &entry.target {
            OrderTarget::Column(name) => Some(resolve_column_ref(ctx, name)?),
            _ => None,
        };

        out.push(ResolvedOrderBy {
            target: entry.target.clone(),
            sort_type: entry.sort_type.clone(),
            ascending: entry.ascending,
            nulls_first: entry.nulls_first,
            resolved_column,
        });
    }
    Ok(out)
}

fn resolve_joins<P>(plan: &SelectPlan, ctx: &ResolutionContext<P>) -> Result<Vec<ResolvedJoin>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut out = Vec::with_capacity(plan.joins.len());
    for join in &plan.joins {
        let on = match &join.on_condition {
            Some(expr) => Some(resolve_predicate(ctx, expr)?),
            None => None,
        };

        out.push(ResolvedJoin {
            left_table_index: join.left_table_index,
            join_type: join.join_type,
            on,
            original_on: join.on_condition.clone(),
        });
    }
    Ok(out)
}
