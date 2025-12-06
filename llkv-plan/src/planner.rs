use crate::logical_planner::SingleTableLogicalPlan;
use crate::physical::PhysicalPlan;
use crate::physical::projection::ProjectionExec;
use crate::physical::ranges::extract_ranges;
use crate::physical::scan::{ScanExec, ScanExecParams};
use crate::physical::sort::SortExec;
use llkv_scan::RowIdFilter;
use llkv_scan::ScanOrderSpec;
use llkv_storage::pager::Pager;
use llkv_types::LogicalFieldId;
use simd_r_drive_entry_handle::EntryHandle;
use std::ops::Not;
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

        fn resolve_scan_order<P>(logical_plan: &SingleTableLogicalPlan<P>) -> Option<ScanOrderSpec>
        where
            P: Pager<Blob = EntryHandle> + Send + Sync,
        {
            if logical_plan.resolved_order_by.len() != 1 {
                return None;
            }

            let order = &logical_plan.resolved_order_by[0];
            let column_name = match &order.target {
                crate::plans::OrderTarget::Column(name) => name,
                _ => return None,
            };

            // Only push scan-order hints when the ORDER BY references a column projection
            // (not a computed expression). This avoids skipping SortExec for computed ORDER BY
            // targets where storage cannot provide ordering guarantees.
            let matching_projection =
                logical_plan
                    .scan_projections
                    .iter()
                    .find_map(|proj| match proj {
                        llkv_scan::ScanProjection::Column(p)
                            if crate::logical_planner::projection_name(
                                proj,
                                &logical_plan.schema,
                            )
                            .eq_ignore_ascii_case(column_name) =>
                        {
                            Some(p)
                        }
                        _ => None,
                    })?;

            let column = logical_plan
                .schema
                .column_by_field_id(matching_projection.logical_field_id.field_id())?;

            let transform = match order.sort_type {
                crate::plans::OrderSortType::Native => match column.data_type {
                    arrow::datatypes::DataType::Int64 => {
                        llkv_scan::ScanOrderTransform::IdentityInt64
                    }
                    arrow::datatypes::DataType::Int32 => {
                        llkv_scan::ScanOrderTransform::IdentityInt32
                    }
                    arrow::datatypes::DataType::Utf8 => llkv_scan::ScanOrderTransform::IdentityUtf8,
                    _ => return None,
                },
                crate::plans::OrderSortType::CastTextToInteger => {
                    if column.data_type == arrow::datatypes::DataType::Utf8 {
                        llkv_scan::ScanOrderTransform::CastUtf8ToInteger
                    } else {
                        return None;
                    }
                }
            };

            let direction = if order.ascending {
                llkv_scan::ScanOrderDirection::Ascending
            } else {
                llkv_scan::ScanOrderDirection::Descending
            };

            Some(ScanOrderSpec {
                field_id: column.field_id,
                direction,
                nulls_first: order.nulls_first,
                transform,
            })
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
            order: resolve_scan_order(&logical_plan),
        }));

        let mut current_plan: Arc<dyn PhysicalPlan> = scan_exec;

        let need_sort = logical_plan.resolved_order_by.is_empty().not()
            && resolve_scan_order(&logical_plan).is_none();

        if need_sort {
            current_plan = Arc::new(SortExec::new(
                current_plan,
                logical_plan.resolved_order_by.clone(),
            ));
        }

        if logical_plan.has_extra_columns() {
            let mut expr = Vec::new();
            let requested_len = logical_plan.requested_projections.len();
            if requested_len > logical_plan.scan_schema.fields().len() {
                return Err("Scan schema shorter than requested projections".to_string());
            }

            // Map requested projections positionally to preserve duplicate names.
            for i in 0..requested_len {
                let field = logical_plan.final_schema.field(i);
                expr.push((i, field.name().clone()));
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
