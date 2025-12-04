use crate::physical::PhysicalPlan;
use crate::physical::projection::ProjectionExec;
use crate::physical::ranges::extract_ranges;
use crate::physical::scan::{ScanExec, ScanExecParams};
use crate::physical::sort::SortExec;
use crate::physical::table::TableProvider;
use crate::plans::OrderTarget;
use crate::plans::SelectPlan;
use crate::translation::{
    build_projected_columns, build_wildcard_projections, schema_for_projections,
    translate_predicate,
};
use llkv_result::Error;
use llkv_scan::RowIdFilter;
use llkv_scan::ScanProjection;
use llkv_storage::pager::Pager;
use llkv_types::LogicalFieldId;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

pub struct PhysicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    provider: Arc<dyn TableProvider<P>>,
}

impl<P> PhysicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn TableProvider<P>>) -> Self {
        Self { provider }
    }

    pub fn create_physical_plan(
        &self,
        plan: &SelectPlan,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    ) -> Result<Arc<dyn PhysicalPlan>, String> {
        if let Some(table_ref) = plan.tables.first() {
            let table_name = table_ref.qualified_name();
            let table = self
                .provider
                .get_table(&table_name)
                .map_err(|e| e.to_string())?;
            let table_id = table.table_id();

            // 1. Determine requested projections
            let requested_projections = if plan.projections.is_empty() {
                build_wildcard_projections(&table.schema(), table_id)
            } else {
                build_projected_columns(&table.schema(), table_id, &plan.projections)
                    .map_err(|e| e.to_string())?
            };

            // 2. Determine extra columns needed for ORDER BY and resolve targets
            let mut scan_projections = requested_projections.clone();
            let mut extra_columns = Vec::new();
            let mut resolved_order_by = Vec::new();

            for order in &plan.order_by {
                match &order.target {
                    OrderTarget::All => {
                        // Expand to all requested projections
                        for proj in &requested_projections {
                            let name = get_projection_name(proj, &table.schema());
                            let mut new_order = order.clone();
                            new_order.target = OrderTarget::Column(name);
                            resolved_order_by.push(new_order);
                        }
                    }
                    OrderTarget::Index(idx) => {
                        if *idx < requested_projections.len() {
                            let proj = &requested_projections[*idx];
                            let name = get_projection_name(proj, &table.schema());
                            let mut new_order = order.clone();
                            new_order.target = OrderTarget::Column(name);
                            resolved_order_by.push(new_order);
                        } else {
                            return Err(format!("ORDER BY index {} out of bounds", idx));
                        }
                    }
                    OrderTarget::Column(name) => {
                        // Check if column is already in projections
                        let is_projected = scan_projections.iter().any(|p| {
                            get_projection_name(p, &table.schema()).eq_ignore_ascii_case(name)
                        });

                        if !is_projected {
                            // Find the column in the table schema
                            if let Some(col) = table
                                .schema()
                                .columns
                                .iter()
                                .find(|c| c.name.eq_ignore_ascii_case(name))
                            {
                                let logical_field_id =
                                    LogicalFieldId::for_user(table_id, col.field_id);
                                let projection =
                                    ScanProjection::Column(llkv_column_map::store::Projection {
                                        logical_field_id,
                                        alias: None,
                                    });
                                scan_projections.push(projection);
                                extra_columns.push(name.clone());
                            } else {
                                return Err(format!(
                                    "ORDER BY references unknown column '{}'",
                                    name
                                ));
                            }
                        }

                        // Add to resolved order by
                        let mut new_order = order.clone();
                        new_order.target = OrderTarget::Column(name.clone());
                        resolved_order_by.push(new_order);
                    }
                }
            }

            // 3. Build schema for the scan (includes extra columns)
            let scan_schema = schema_for_projections(&table.schema(), &scan_projections)
                .map_err(|e| e.to_string())?;

            // Try to extract ranges from filter
            let mut ranges = None;
            let mut driving_column = None;
            let mut physical_filter = None;

            if let Some(filter) = &plan.filter {
                // Translate the filter to physical expression
                // We need to clone the predicate because translate_predicate takes ownership
                let predicate = filter.predicate.clone();
                match translate_predicate(predicate, &table.schema(), |_| {
                    Error::Internal("Unknown column".to_string())
                }) {
                    Ok(expr) => physical_filter = Some(expr),
                    Err(e) => return Err(format!("Failed to translate filter: {}", e)),
                }

                // Iterate over columns to find a driving column
                for col in &table.schema().columns {
                    if let Some(r) = extract_ranges(&filter.predicate, &col.name, &col.data_type) {
                        ranges = Some(r);
                        let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
                        driving_column = Some(logical_field_id);
                        // TODO: Support combining multiple range columns, choosing the most selective column
                        break;
                    }
                }
            }

            let scan_exec = Arc::new(ScanExec::new(ScanExecParams {
                table_name: table_name.clone(),
                schema: scan_schema.clone(),
                table: table.clone(),
                ranges,
                driving_column,
                filter: physical_filter,
                projections: scan_projections,
                row_filter,
                order: None,
            }));

            let mut current_plan: Arc<dyn PhysicalPlan> = scan_exec;

            // 5. Add SortExec if needed
            if !resolved_order_by.is_empty() {
                current_plan = Arc::new(SortExec::new(current_plan, resolved_order_by));
            }

            // 4. If we added extra columns, wrap in ProjectionExec
            if !extra_columns.is_empty() {
                // Build the final schema (without extra columns)
                let final_schema = schema_for_projections(&table.schema(), &requested_projections)
                    .map_err(|e| e.to_string())?;

                // Map requested projections to indices in scan_schema
                let mut expr = Vec::new();
                for (i, _) in requested_projections.iter().enumerate() {
                    let field = final_schema.field(i);
                    // Find index in scan_schema
                    if let Ok(idx) = scan_schema.index_of(field.name()) {
                        expr.push((idx, field.name().clone()));
                    } else {
                        return Err(format!(
                            "Failed to find column '{}' in scan schema",
                            field.name()
                        ));
                    }
                }

                Ok(Arc::new(ProjectionExec::new(
                    current_plan,
                    final_schema,
                    expr,
                )))
            } else {
                Ok(current_plan)
            }
        } else {
            Err("No tables in select plan".to_string())
        }
    }
}

fn get_projection_name(proj: &ScanProjection, schema: &crate::schema::PlanSchema) -> String {
    match proj {
        ScanProjection::Column(p) => {
            if let Some(alias) = &p.alias {
                alias.clone()
            } else {
                // Look up column name
                schema
                    .columns
                    .iter()
                    .find(|c| c.field_id == p.logical_field_id.field_id())
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| format!("col_{}", p.logical_field_id.field_id()))
            }
        }
        ScanProjection::Computed { alias, .. } => alias.clone(),
    }
}
