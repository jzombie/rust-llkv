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

        let OrderResolution {
            scan_projections,
            extra_columns,
            resolved_order_by,
        } = resolve_order_by_targets(&schema, table_id, &requested_projections, &plan.order_by)?;

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

        Ok(LogicalPlan::Multi(MultiTableLogicalPlan {
            tables: planned_tables,
            table_order: plan.tables.clone(),
            filter: plan.filter.as_ref().map(|f| f.predicate.clone()),
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
