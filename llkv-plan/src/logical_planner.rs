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
    pub filter: Option<Expr<'static, String>>,
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

        Ok(LogicalPlan::Multi(MultiTableLogicalPlan {
            tables: planned_tables,
            table_order: plan.tables.clone(),
            filter: plan.filter.as_ref().map(|f| f.predicate.clone()),
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
            let table_name_lower = table.name.to_ascii_lowercase();
            let alias_lower = plan
                .tables
                .get(idx)
                .and_then(|t| t.alias.as_ref())
                .map(|a| a.to_ascii_lowercase());

            // If a qualifier exists, check it.
            if parts.len() > 1 {
                let prefix = parts[0].to_ascii_lowercase();
                if prefix != table_name_lower
                    && alias_lower.as_ref().is_some_and(|alias| prefix != *alias)
                {
                    continue;
                }
                let col_name = parts[1..].join("."); // nested paths collapse; schema matching is on base column
                if table.schema.column_by_name(&col_name).is_some() {
                    candidates.push((idx, col_name));
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
