use std::sync::Arc;
use crate::plans::SelectPlan;
use crate::physical::PhysicalPlan;
use crate::physical::scan::ScanExec;
use crate::physical::table::TableProvider;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use llkv_types::LogicalFieldId;
use crate::translation::{translate_predicate, build_projected_columns, build_wildcard_projections, schema_for_projections};
use llkv_result::Error;
use crate::physical::ranges::extract_ranges;
use llkv_scan::RowIdFilter;

// I need to move ranges module too?
// llkv-executor/src/physical_plan/ranges.rs

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
             let table = self.provider.get_table(&table_name).map_err(|e| e.to_string())?;
             
             // Build projections
             // table.schema() returns Arc<PlanSchema>
             // table.table_id() ? ExecutionTable needs table_id() method?
             // Or PlanSchema has table_id? No.
             // We need table_id to build LogicalFieldId.
             
             // I need to add table_id() to ExecutionTable trait.
             
             let table_id = table.table_id(); // Assuming I add this
             
             let projections = if plan.projections.is_empty() {
                 build_wildcard_projections(&table.schema(), table_id)
             } else {
                 build_projected_columns(&table.schema(), table_id, &plan.projections).map_err(|e| e.to_string())?
             };

             // Build schema from projections
             let schema = schema_for_projections(&table.schema(), &projections).map_err(|e| e.to_string())?;
             
             // Try to extract ranges from filter
             let mut ranges = None;
             let mut driving_column = None;
             let mut physical_filter = None;

             if let Some(filter) = &plan.filter {
                 // Translate the filter to physical expression
                 // We need to clone the predicate because translate_predicate takes ownership
                 let predicate = filter.predicate.clone();
                 match translate_predicate(predicate, &table.schema(), |_| Error::Internal("Unknown column".to_string())) {
                     Ok(expr) => physical_filter = Some(expr),
                     Err(e) => return Err(format!("Failed to translate filter: {}", e)),
                 }

                 // Iterate over columns to find a driving column
                 for col in &table.schema().columns {
                     if let Some(r) = extract_ranges(&filter.predicate, &col.name, &col.data_type) {
                         ranges = Some(r);
                         let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
                         driving_column = Some(logical_field_id);
                          // TODO: Support multiple driving columns? It's currently first-match wins.
                         break;
                     }
                 }
             }

             Ok(Arc::new(ScanExec::new(
                 table_name.clone(), 
                 schema, 
                 table.clone(), 
                 ranges, 
                 driving_column, 
                 physical_filter, 
                 projections,
                 row_filter,
                 None, // TODO: Implement ORDER BY support in PhysicalPlanner
             )))
        } else {
            Err("No tables in select plan".to_string())
        }
    }
}
