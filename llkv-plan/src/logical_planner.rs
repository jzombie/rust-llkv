//! Logical planning for translating [`SelectPlan`]s into storage-aware requests.
//!
//! The logical planner resolves projections, ORDER BY targets, and filter
//! expressions against table schemas supplied by a [`TableProvider`]. The
//! resulting plans carry resolved schemas so physical planning can focus on
//! execution path selection (index usage, pruning, sorted scans, etc.).

use std::sync::Arc;

use arrow::datatypes::Schema;
use llkv_column_map::store::Projection as StoreProjection;
use llkv_expr::expr::{CompareOp, Expr, ScalarExpr};
use llkv_result::Error;
use llkv_result::Result;
use llkv_scan::ScanProjection;
use llkv_storage::pager::Pager;
use llkv_types::{FieldId, LogicalFieldId, TableId};
use rustc_hash::FxHashSet;
use simd_r_drive_entry_handle::EntryHandle;
use tracing::debug;

use crate::aggregate_rewrite::{
    AggregateRewrite, build_single_aggregate_rewrite, expr_to_scalar_expr,
    extract_complex_aggregates,
};

use crate::table_provider::{ExecutionTable, TableProvider};
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
    pub scalar_subqueries: Vec<crate::plans::ScalarSubquery>,
    pub filter_subqueries: Vec<crate::plans::FilterSubquery>,
    pub aggregate_rewrite: Option<AggregateRewrite>,
    pub group_by: Vec<String>,
    pub distinct: bool,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
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
    /// Filter predicates that reference a single table, partitioned by table index.
    pub table_filters: Vec<Option<Expr<'static, ResolvedFieldRef>>>,
    /// Residual filter predicates that reference multiple tables.
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
    pub scalar_subqueries: Vec<crate::plans::ScalarSubquery>,
    pub filter_subqueries: Vec<crate::plans::FilterSubquery>,
    pub aggregate_rewrite: Option<AggregateRewrite>,
}

pub struct PlannedTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub name: String,
    pub table_id: TableId,
    pub table: Arc<dyn ExecutionTable<P>>,
    pub schema: Arc<PlanSchema>,
    pub original_index: usize,
}

impl<P> Clone for PlannedTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            table_id: self.table_id,
            table: self.table.clone(),
            schema: self.schema.clone(),
            original_index: self.original_index,
        }
    }
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

    pub fn provider(&self) -> Arc<dyn TableProvider<P>> {
        self.provider.clone()
    }

    pub fn get_table_schema(&self, table_name: &str) -> Result<Arc<PlanSchema>> {
        let table = self.provider.get_table(table_name).ok_or_else(|| Error::Internal(format!("Catalog Error: Table with name {} does not exist", table_name)))?;
        Ok(Arc::new(table.schema().clone()))
    }

    pub fn create_logical_plan(&self, plan: &SelectPlan) -> Result<LogicalPlan<P>> {
        debug!("create_logical_plan tables len: {}", plan.tables.len());
        debug!("create_logical_plan projections len: {}", plan.projections.len());
        if !plan.projections.is_empty() {
            debug!("create_logical_plan first projection: {:?}", plan.projections[0]);
        }
        if plan.tables.len() != 1 {
            return self.plan_multi_table(plan);
        }

        let table_ref = plan.tables.first().expect("validated length above");
        let table_name = table_ref.qualified_name();
        let table = self.provider.get_table(&table_name).ok_or_else(|| Error::Internal(format!("Catalog Error: Table with name {} does not exist", table_name)))?;
        let table_id = table.table_id();
        let schema = table.schema();

        let requested_projections = if plan.projections.is_empty() {
            build_wildcard_projections(schema, table_id)
        } else {
            build_projected_columns(schema, table_id, &plan.projections)?
        };

        let mut scan_projections = requested_projections.clone();
        let mut extra_columns: Vec<String> = Vec::new();

        // Ensure filter columns are available for scan-level evaluation.
        // Note: We do NOT add filter columns to scan_projections because ScanExec handles
        // filtering internally using the storage engine, so these columns don't need to be
        // materialized in the output batch unless they are also requested projections.
        /*
        if let Some(filter) = &plan.filter {
            for name in collect_filter_columns(&filter.predicate) {
                push_column_if_known(
                    table_id,
                    schema,
                    &name,
                    &mut scan_projections,
                    &mut extra_columns,
                );
            }
        }
        */


        // Ensure GROUP BY and aggregate inputs are present for scan-level computation.
        for name in &plan.group_by {
            push_column_if_known(
                table_id,
                schema,
                name,
                &mut scan_projections,
                &mut extra_columns,
            );
        }
        for agg in &plan.aggregates {
            for name in collect_aggregate_columns(agg) {
                push_column_if_known(
                    table_id,
                    schema,
                    &name,
                    &mut scan_projections,
                    &mut extra_columns,
                );
            }
        }

        // If scan_projections is still empty (e.g. SELECT COUNT(*) with no filter columns projected),
        // we must project at least one column for the scan to work.
        // Prefer the primary key, otherwise the first column.

        let OrderResolution {
            scan_projections: order_scan_projections,
            extra_columns: order_extra_columns,
            resolved_order_by,
        } = resolve_order_by_targets(schema, table_id, &scan_projections, &plan.order_by)?;

        scan_projections = order_scan_projections;
        for col in order_extra_columns {
            if !extra_columns
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&col))
            {
                extra_columns.push(col);
            }
        }

        let aggregate_rewrite = build_single_aggregate_rewrite(plan, schema)?;
        if let Some(rewrite) = &aggregate_rewrite {
            let mut rewrite_columns = Vec::new();
            for expr in &rewrite.pre_aggregate_expressions {
                collect_scalar_columns(expr, &mut rewrite_columns);
            }
            if let Some(having) = &rewrite.rewritten_having {
                collect_scalar_columns(having, &mut rewrite_columns);
            }
            for name in rewrite_columns {
                if let Some(col) = schema
                    .columns
                    .iter()
                    .find(|c| c.name.eq_ignore_ascii_case(&name))
                {
                    let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
                    let already_present = scan_projections.iter().any(|p| match p {
                        ScanProjection::Column(sp) => {
                            sp.logical_field_id.field_id() == logical_field_id.field_id()
                        }
                        _ => false,
                    });
                    if !already_present {
                        scan_projections.insert(
                            0,
                            ScanProjection::Column(StoreProjection {
                                logical_field_id,
                                alias: None,
                            }),
                        );
                        if !extra_columns
                            .iter()
                            .any(|existing| existing.eq_ignore_ascii_case(&col.name))
                        {
                            extra_columns.push(col.name.clone());
                        }
                    }
                }
            }
        }

        let scan_schema = schema_for_projections(schema, &scan_projections)?;
        let final_schema = schema_for_projections(schema, &requested_projections)?;

        let filter = match &plan.filter {
            Some(filter) => {
                debug!("LogicalPlan filter input: {:?}", filter.predicate);
                let translated = translate_predicate(
                    filter.predicate.clone(),
                    schema,
                    |_| Error::Internal("Unknown column".to_string()),
                )?;
                debug!("LogicalPlan filter translated: {:?}", translated);
                Some(translated)
            },
            None => None,
        };

        let plan_schema = Arc::new(schema.clone());

        Ok(LogicalPlan::Single(SingleTableLogicalPlan {
            table_name,
            table,
            table_id,
            schema: plan_schema,
            requested_projections,
            scan_projections,
            final_schema,
            scan_schema,
            resolved_order_by,
            filter,
            original_filter: plan.filter.as_ref().map(|f| f.predicate.clone()),
            extra_columns,
            scalar_subqueries: plan.scalar_subqueries.clone(),
            filter_subqueries: plan
                .filter
                .as_ref()
                .map(|f| f.subqueries.clone())
                .unwrap_or_default(),
            aggregate_rewrite,
            group_by: plan.group_by.clone(),
            distinct: plan.distinct,
            limit: plan.limit,
            offset: plan.offset,
        }))
    }

    fn plan_multi_table(&self, plan: &SelectPlan) -> Result<LogicalPlan<P>> {
        let mut table_pairs = Vec::with_capacity(plan.tables.len());
        for (original_index, table_ref) in plan.tables.iter().enumerate() {
            let table_name = table_ref.qualified_name();
            let table = self.provider.get_table(&table_name).ok_or_else(|| Error::Internal(format!("Catalog Error: Table with name {} does not exist", table_name)))?;
            let table_id = table.table_id();
            let schema = table.schema();
            let plan_schema = Arc::new(schema.clone());

            let planned = PlannedTable {
                name: table_name,
                table_id,
                table,
                schema: plan_schema,
                original_index,
            };
            table_pairs.push((planned, table_ref.clone()));
        }

        // Reorder tables using greedy algorithm to avoid cross joins
        // Only reorder if there are no explicit joins, as explicit joins enforce a specific order
        // (especially for outer joins) and our join resolution logic assumes the table order matches the join metadata.
        let table_pairs = if plan.joins.is_empty() {
            reorder_tables_greedy(table_pairs, plan.filter.as_ref().map(|f| &f.predicate))
        } else {
            table_pairs
        };

        debug!("Planned tables order:");
        for (i, (table, table_ref)) in table_pairs.iter().enumerate() {
            debug!("  {}: {} (rows: {:?})", i, table_ref.table, table.table.approximate_row_count());
        }

        let (planned_tables, ordered_table_refs): (Vec<_>, Vec<_>) = table_pairs.into_iter().unzip();

        let (resolved_columns, unresolved_required) =
            resolve_required_columns(plan, &planned_tables, &ordered_table_refs);

        let ctx = ResolutionContext {
            tables: &planned_tables,
            table_refs: &ordered_table_refs,
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
        let mut joins = resolve_joins(plan, &ctx)?;

        attach_implicit_joins(&mut joins, planned_tables.len(), filter.as_ref());

        let table_count = planned_tables.len();
        let (table_filters, residual_filter) =
            partition_table_filters(filter.as_ref(), table_count);

        let aggregate_rewrite = build_multi_aggregate_rewrite(plan, &ctx)?;

        Ok(LogicalPlan::Multi(MultiTableLogicalPlan {
            tables: planned_tables,
            table_order: ordered_table_refs,
            table_filters,
            filter: residual_filter.or_else(|| filter.clone()),
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
            scalar_subqueries: plan.scalar_subqueries.clone(),
            filter_subqueries: plan
                .filter
                .as_ref()
                .map(|f| f.subqueries.clone())
                .unwrap_or_default(),
            aggregate_rewrite,
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


fn build_multi_projection_exprs<P>(
    plan: &SelectPlan,
    ctx: &ResolutionContext<P>,
) -> Result<Vec<(ScalarExpr<String>, String)>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let projections = if plan.projections.is_empty() {
        vec![crate::plans::SelectProjection::AllColumns]
    } else {
        plan.projections.clone()
    };

    let mut out = Vec::new();
    for proj in projections {
        match proj {
            crate::plans::SelectProjection::AllColumns => {
                debug!("build_multi_projection_exprs: processing AllColumns, tables len: {}", ctx.tables.len());
                for table in ctx.tables.iter() {
                    for col in &table.schema.columns {
                        out.push((ScalarExpr::Column(col.name.clone()), col.name.clone()));
                    }
                }
            }
            crate::plans::SelectProjection::AllColumnsExcept { exclude } => {
                let mut excluded_fields = FxHashSet::default();
                for ex in exclude {
                    let (table_idx, lfid) = resolve_column_ref(ctx, ex.as_str())?;
                    excluded_fields.insert((table_idx, lfid.field_id()));
                }

                for (table_idx, table) in ctx.tables.iter().enumerate() {
                    for col in &table.schema.columns {
                        if excluded_fields.contains(&(table_idx, col.field_id)) {
                            continue;
                        }
                        out.push((ScalarExpr::Column(col.name.clone()), col.name.clone()));
                    }
                }
            }
            crate::plans::SelectProjection::Column { name, alias } => {
                let (table_idx, lfid) = resolve_column_ref(ctx, &name)?;
                let canonical_name = ctx
                    .tables
                    .get(table_idx)
                    .and_then(|t| {
                        t.schema
                            .columns
                            .iter()
                            .find(|c| c.field_id == lfid.field_id())
                    })
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| name.clone());

                let final_name = alias.clone().unwrap_or(canonical_name);
                out.push((ScalarExpr::Column(name), final_name));
            }
            crate::plans::SelectProjection::Computed { expr, alias } => {
                out.push((expr, alias));
            }
        }
    }

    Ok(out)
}

pub(crate) fn build_multi_aggregate_rewrite<P>(
    plan: &SelectPlan,
    ctx: &ResolutionContext<P>,
) -> Result<Option<AggregateRewrite>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let projections = build_multi_projection_exprs(plan, ctx)?;
    let (projection_exprs, final_names): (Vec<_>, Vec<_>) = projections.into_iter().unzip();

    let having_exprs: Vec<ScalarExpr<String>> = plan
        .having
        .as_ref()
        .map(|h| vec![expr_to_scalar_expr(h)])
        .unwrap_or_default();

    let (aggregates, final_exprs, pre_agg_exprs, rewritten_having) =
        extract_complex_aggregates(&projection_exprs, &having_exprs);

    if aggregates.is_empty() {
        return Ok(None);
    }

    Ok(Some(AggregateRewrite {
        aggregates,
        final_expressions: final_exprs,
        pre_aggregate_expressions: pre_agg_exprs,
        final_names,
        rewritten_having: rewritten_having.into_iter().next(),
    }))
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
    table_refs: &[TableRef],
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
            let Some(table_ref) = table_refs.get(idx) else {
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

pub(crate) struct ResolutionContext<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub tables: &'a [PlannedTable<P>],
    pub table_refs: &'a [TableRef],
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

    debug!("resolve_projections: input len: {}", projections.len());

    for proj in projections {
        debug!("resolve_projections: processing proj variant: {:?}", proj);
        match proj {
            crate::plans::SelectProjection::AllColumns => {
                // Sort tables by original_index to ensure projection order matches FROM clause
                let mut sorted_tables: Vec<(usize, &PlannedTable<P>)> = ctx.tables.iter().enumerate().collect();
                sorted_tables.sort_by_key(|(_, t)| t.original_index);

                for (table_idx, table) in sorted_tables {
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
                let mut excluded_fields = FxHashSet::default();
                for ex in exclude {
                    let (table_idx, lfid) = resolve_column_ref(ctx, ex.as_str())?;
                    excluded_fields.insert((table_idx, lfid.field_id()));
                }

                // Sort tables by original_index to ensure projection order matches FROM clause
                let mut sorted_tables: Vec<(usize, &PlannedTable<P>)> = ctx.tables.iter().enumerate().collect();
                sorted_tables.sort_by_key(|(_, t)| t.original_index);

                for (table_idx, table) in sorted_tables {
                    for col in &table.schema.columns {
                        if excluded_fields.contains(&(table_idx, col.field_id)) {
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
                debug!("resolve_projections: pushing Computed alias={}", alias);
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

fn attach_implicit_joins(
    joins: &mut Vec<ResolvedJoin>,
    table_count: usize,
    filter: Option<&Expr<'static, ResolvedFieldRef>>,
) {
    if table_count <= 1 {
        return;
    }

    for left_table_index in 0..(table_count - 1) {
        let left_tables: FxHashSet<usize> = (0..=left_table_index).collect();
        let derived_keys = derive_filter_join_keys(filter, &left_tables, left_table_index + 1);
        
        debug!("attach_implicit_joins: left_table_index={}, right_table={}, keys={:?}", left_table_index, left_table_index + 1, derived_keys);

        let derived_on = build_join_predicate(&derived_keys);

        if let Some(existing_join) = joins
            .iter_mut()
            .find(|j| j.left_table_index == left_table_index)
        {
            let is_cross = match &existing_join.on {
                None => true,
                Some(Expr::Literal(true)) => true,
                _ => false,
            };

            if is_cross {
                if let Some(new_on) = derived_on {
                    existing_join.on = Some(new_on);
                    existing_join.join_type = crate::plans::JoinPlan::Inner;
                }
            }
        } else {
            joins.push(ResolvedJoin {
                left_table_index,
                join_type: crate::plans::JoinPlan::Inner,
                on: derived_on,
                original_on: None,
            });
        }
    }

    joins.sort_by_key(|j| j.left_table_index);
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DerivedJoinKey {
    left_table_index: usize,
    right_table_index: usize,
    left_field: LogicalFieldId,
    right_field: LogicalFieldId,
}

fn derive_filter_join_keys(
    filter: Option<&Expr<'static, ResolvedFieldRef>>,
    left_tables: &FxHashSet<usize>,
    right_table: usize,
) -> Vec<DerivedJoinKey> {
    fn collect_keys(
        expr: &Expr<'static, ResolvedFieldRef>,
        left_tables: &FxHashSet<usize>,
        right_table: usize,
        out: &mut Vec<DerivedJoinKey>,
    ) {
        match expr {
            Expr::And(list) => {
                for e in list {
                    collect_keys(e, left_tables, right_table, out);
                }
            }
            Expr::Compare { left, op, right } if matches!(op, CompareOp::Eq) => {
                if let (ScalarExpr::Column(l), ScalarExpr::Column(r)) = (left, right) {
                    if left_tables.contains(&l.table_index) && r.table_index == right_table {
                        out.push(DerivedJoinKey {
                            left_table_index: l.table_index,
                            right_table_index: right_table,
                            left_field: l.logical_field_id,
                            right_field: r.logical_field_id,
                        });
                        return;
                    }

                    if left_tables.contains(&r.table_index) && l.table_index == right_table {
                        out.push(DerivedJoinKey {
                            left_table_index: r.table_index,
                            right_table_index: right_table,
                            left_field: r.logical_field_id,
                            right_field: l.logical_field_id,
                        });
                        return;
                    }
                }
            }
            _ => {}
        }
    }

    let mut keys = Vec::new();
    if let Some(filter) = filter {
        collect_keys(filter, left_tables, right_table, &mut keys);
    }
    keys
}

fn build_join_predicate(keys: &[DerivedJoinKey]) -> Option<Expr<'static, ResolvedFieldRef>> {
    if keys.is_empty() {
        return None;
    }

    let mut clauses = Vec::with_capacity(keys.len());
    for key in keys {
        let left = ScalarExpr::Column(ResolvedFieldRef {
            table_index: key.left_table_index,
            logical_field_id: key.left_field,
        });
        let right = ScalarExpr::Column(ResolvedFieldRef {
            table_index: key.right_table_index,
            logical_field_id: key.right_field,
        });

        clauses.push(Expr::Compare {
            left,
            op: CompareOp::Eq,
            right,
        });
    }

    if clauses.len() == 1 {
        clauses.pop()
    } else {
        Some(Expr::And(clauses))
    }
}

fn partition_table_filters(
    filter: Option<&Expr<'static, ResolvedFieldRef>>,
    table_count: usize,
) -> (
    Vec<Option<Expr<'static, ResolvedFieldRef>>>,
    Option<Expr<'static, ResolvedFieldRef>>,
) {
    let mut per_table: Vec<Vec<Expr<'static, ResolvedFieldRef>>> = vec![Vec::new(); table_count];
    let mut residuals = Vec::new();

    let clauses: Vec<Expr<'static, ResolvedFieldRef>> = match filter {
        Some(Expr::And(list)) => list.clone(),
        Some(expr) => vec![expr.clone()],
        None => Vec::new(),
    };

    for clause in clauses {
        let mut fields = FxHashSet::default();
        collect_filter_tables(&clause, &mut fields);
        if fields.len() == 1 {
            let tbl = *fields.iter().next().unwrap();
            if tbl < table_count {
                per_table[tbl].push(clause);
                continue;
            }
        }
        residuals.push(clause);
    }

    let per_table = per_table
        .into_iter()
        .map(|mut clauses| {
            if clauses.is_empty() {
                None
            } else if clauses.len() == 1 {
                Some(clauses.pop().unwrap())
            } else {
                Some(Expr::And(clauses))
            }
        })
        .collect();

    let residual = if residuals.is_empty() {
        None
    } else if residuals.len() == 1 {
        Some(residuals.pop().unwrap())
    } else {
        Some(Expr::And(residuals))
    };

    (per_table, residual)
}

fn collect_filter_tables(expr: &Expr<'static, ResolvedFieldRef>, out: &mut FxHashSet<usize>) {
    match expr {
        Expr::And(list) | Expr::Or(list) => {
            for e in list {
                collect_filter_tables(e, out);
            }
        }
        Expr::Not(e) => collect_filter_tables(e, out),
        Expr::Compare { left, right, .. } => {
            collect_scalar_tables(left, out);
            collect_scalar_tables(right, out);
        }
        Expr::Pred(filter) => {
            out.insert(filter.field_id.table_index);
        }
        Expr::InList { expr, list, .. } => {
            collect_scalar_tables(expr, out);
            for item in list {
                collect_scalar_tables(item, out);
            }
        }
        Expr::IsNull { expr, .. } => collect_scalar_tables(expr, out),
        Expr::Literal(_) => {}
        Expr::Exists(_) => {}
    }
}

fn collect_scalar_tables(expr: &ScalarExpr<ResolvedFieldRef>, out: &mut FxHashSet<usize>) {
    match expr {
        ScalarExpr::Column(c) => {
            out.insert(c.table_index);
        }
        ScalarExpr::Binary { left, right, .. } | ScalarExpr::Compare { left, right, .. } => {
            collect_scalar_tables(left, out);
            collect_scalar_tables(right, out);
        }
        ScalarExpr::Not(e) | ScalarExpr::IsNull { expr: e, .. } => collect_scalar_tables(e, out),
        ScalarExpr::Cast { expr, .. } => collect_scalar_tables(expr, out),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(o) = operand {
                collect_scalar_tables(o, out);
            }
            for (w, t) in branches {
                collect_scalar_tables(w, out);
                collect_scalar_tables(t, out);
            }
            if let Some(e) = else_expr {
                collect_scalar_tables(e, out);
            }
        }
        ScalarExpr::Coalesce(list) => {
            for e in list {
                collect_scalar_tables(e, out);
            }
        }
        ScalarExpr::Random => {}
        ScalarExpr::Aggregate(_) => {}
        ScalarExpr::ScalarSubquery(_) => {}
        ScalarExpr::Literal(_) => {}
        ScalarExpr::GetField { base, .. } => collect_scalar_tables(base, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(table_index: usize, field_id: FieldId) -> ScalarExpr<ResolvedFieldRef> {
        ScalarExpr::Column(ResolvedFieldRef {
            table_index,
            logical_field_id: LogicalFieldId::for_user(42, field_id),
        })
    }

    #[test]
    fn derives_join_keys_from_filter_conjunction() {
        let filter = Expr::And(vec![Expr::Compare {
            left: col(0, 1),
            op: CompareOp::Eq,
            right: col(1, 2),
        }]);

        let left_tables: FxHashSet<usize> = [0].into_iter().collect();
        let keys = derive_filter_join_keys(Some(&filter), &left_tables, 1);

        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].left_table_index, 0);
        assert_eq!(keys[0].left_field.field_id(), 1);
        assert_eq!(keys[0].right_field.field_id(), 2);
    }

    #[test]
    fn ignores_non_conjunctive_shapes() {
        let filter = Expr::Or(vec![Expr::Compare {
            left: col(0, 1),
            op: CompareOp::Eq,
            right: col(1, 2),
        }]);

        let left_tables: FxHashSet<usize> = [0].into_iter().collect();
        let keys = derive_filter_join_keys(Some(&filter), &left_tables, 1);

        assert!(keys.is_empty());
    }

    #[test]
    fn attaches_implicit_join_when_missing() {
        let filter = Expr::Compare {
            left: col(0, 1),
            op: CompareOp::Eq,
            right: col(1, 2),
        };

        let mut joins = Vec::new();
        attach_implicit_joins(&mut joins, 2, Some(&filter));

        assert_eq!(joins.len(), 1);
        assert_eq!(joins[0].left_table_index, 0);
        assert!(matches!(joins[0].join_type, crate::plans::JoinPlan::Inner));
        assert!(matches!(joins[0].on, Some(Expr::Compare { .. })));
    }

    #[test]
    fn attaches_cross_join_when_no_keys() {
        let mut joins = Vec::new();
        attach_implicit_joins(&mut joins, 2, None);

        assert_eq!(joins.len(), 1);
        assert!(joins[0].on.is_none());
    }
}

fn reorder_tables_greedy<P>(
    tables: Vec<(PlannedTable<P>, TableRef)>,
    filter: Option<&Expr<'static, String>>,
) -> Vec<(PlannedTable<P>, TableRef)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if tables.is_empty() {
        return tables;
    }

    // Pre-calculate row counts to avoid repeated calls in the loop
    let row_counts: Vec<usize> = tables.iter()
        .map(|(t, _)| t.table.approximate_row_count().unwrap_or(0))
        .collect();

    // 1. Build Adjacency Graph
    let mut adj = vec![FxHashSet::default(); tables.len()];
    if let Some(expr) = filter {
        build_join_graph(expr, &tables, &mut adj);
    }

    debug!("Join Graph Adjacency:");
    for (i, neighbors) in adj.iter().enumerate() {
        debug!("  Table {}: {:?}", i, neighbors);
    }

    // 2. Greedy Selection
    let mut ordered_indices = Vec::with_capacity(tables.len());
    let mut remaining: FxHashSet<usize> = (0..tables.len()).collect();
    let mut connected_to_ordered: FxHashSet<usize> = FxHashSet::default();

    // Start with the largest table
    let first_idx = find_largest_table(&row_counts, &remaining);
    move_table(first_idx, &mut remaining, &mut ordered_indices, &mut connected_to_ordered, &adj);

    while !remaining.is_empty() {
        // Find candidates that are connected to the already ordered tables
        let candidates: Vec<usize> = remaining.iter().cloned()
            .filter(|idx| connected_to_ordered.contains(idx))
            .collect();

        let next_idx = if !candidates.is_empty() {
            // Pick the largest connected table
            find_largest_table_in_candidates(&row_counts, &candidates)
        } else {
            // No connected table (Cross Join or Disjoint). Pick largest remaining.
            find_largest_table(&row_counts, &remaining)
        };

        move_table(next_idx, &mut remaining, &mut ordered_indices, &mut connected_to_ordered, &adj);
    }

    // Reconstruct the vector in the new order
    let mut result = Vec::with_capacity(tables.len());
    for idx in ordered_indices {
        result.push(tables[idx].clone());
    }
    
    result
}

fn move_table(
    idx: usize,
    remaining: &mut FxHashSet<usize>,
    ordered: &mut Vec<usize>,
    connected_to_ordered: &mut FxHashSet<usize>,
    adj: &[FxHashSet<usize>],
) {
    remaining.remove(&idx);
    ordered.push(idx);
    for &neighbor in &adj[idx] {
        if remaining.contains(&neighbor) {
            connected_to_ordered.insert(neighbor);
        }
    }
}

fn find_largest_table(
    row_counts: &[usize],
    candidates: &FxHashSet<usize>,
) -> usize 
{
    candidates.iter().cloned().max_by(|&a, &b| {
        row_counts[a].cmp(&row_counts[b])
    }).unwrap()
}

fn find_largest_table_in_candidates(
    row_counts: &[usize],
    candidates: &[usize],
) -> usize 
{
    candidates.iter().cloned().max_by(|&a, &b| {
        row_counts[a].cmp(&row_counts[b])
    }).unwrap()
}

fn build_join_graph<P>(
    expr: &Expr<'static, String>,
    tables: &[(PlannedTable<P>, TableRef)],
    adj: &mut Vec<FxHashSet<usize>>,
) 
where P: Pager<Blob = EntryHandle> + Send + Sync
{
    match expr {
        Expr::Compare { left, op, right } => {
            if matches!(op, CompareOp::Eq) {
                if let (Some(t1), Some(t2)) = (resolve_table_idx(left, tables), resolve_table_idx(right, tables)) {
                    if t1 != t2 {
                        adj[t1].insert(t2);
                        adj[t2].insert(t1);
                    }
                }
            }
        }
        Expr::And(list) => {
            for e in list {
                build_join_graph(e, tables, adj);
            }
        }
        _ => {}
    }
}

fn resolve_table_idx<P>(
    expr: &ScalarExpr<String>,
    tables: &[(PlannedTable<P>, TableRef)],
) -> Option<usize>
where P: Pager<Blob = EntryHandle> + Send + Sync
{
    match expr {
        ScalarExpr::Column(name) => {
            let parts: Vec<&str> = name.split('.').collect();
            let mut candidates = Vec::new();
            
            for (idx, (table, table_ref)) in tables.iter().enumerate() {
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
                    
                    if qualifier_matches_alias || qualifier_matches_table {
                         // Check if column exists in table schema
                         let col_name = parts[1..].join(".");
                         if table.schema.column_by_name(&col_name).is_some() {
                             candidates.push(idx);
                         }
                    }
                } else {
                    // Unqualified column name
                    if table.schema.column_by_name(name).is_some() {
                        candidates.push(idx);
                    }
                }
            }
            
            if candidates.len() == 1 {
                Some(candidates[0])
            } else {
                None
            }
        }
        _ => None
    }
}
