use crate::logical_planner::SingleTableLogicalPlan;
use crate::physical::PhysicalPlan;
use crate::physical::projection::ProjectionExec;
use crate::physical::ranges::extract_ranges;
use crate::physical::scan::{ScanExec, ScanExecParams};
use crate::physical::sort::SortExec;
use llkv_scan::RowIdFilter;
use llkv_storage::pager::Pager;
use llkv_types::LogicalFieldId;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

pub struct PhysicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    _marker: std::marker::PhantomData<P>,
}

impl<P> PhysicalPlanner<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    pub fn create_physical_plan(
        &self,
        logical_plan: &SingleTableLogicalPlan<P>,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    ) -> Result<Arc<dyn PhysicalPlan>, String> {
        let table = logical_plan.table.clone();
        let table_id = logical_plan.table_id;

        let mut ranges = None;
        let mut driving_column = None;

        if let Some(filter) = &logical_plan.original_filter {
            for col in &logical_plan.schema.columns {
                if let Some(r) = extract_ranges(filter, &col.name, &col.data_type) {
                    ranges = Some(r);
                    let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
                    driving_column = Some(logical_field_id);
                    break;
                }
            }
        }

        let scan_exec = Arc::new(ScanExec::new(ScanExecParams {
            table_name: logical_plan.table_name.clone(),
            schema: logical_plan.scan_schema.clone(),
            table: table.clone(),
            ranges,
            driving_column,
            filter: logical_plan.filter.clone(),
            projections: logical_plan.scan_projections.clone(),
            row_filter,
            order: None,
        }));

        let mut current_plan: Arc<dyn PhysicalPlan> = scan_exec;

        if !logical_plan.resolved_order_by.is_empty() {
            current_plan = Arc::new(SortExec::new(
                current_plan,
                logical_plan.resolved_order_by.clone(),
            ));
        }

        if logical_plan.has_extra_columns() {
            let mut expr = Vec::new();
            for (i, _) in logical_plan.requested_projections.iter().enumerate() {
                let field = logical_plan.final_schema.field(i);
                if let Ok(idx) = logical_plan.scan_schema.index_of(field.name()) {
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
                logical_plan.final_schema.clone(),
                expr,
            )))
        } else {
            Ok(current_plan)
        }
    }
}
