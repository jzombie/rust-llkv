use crate::ExecutorResult;
use crate::types::ExecutorTable;
use llkv_column_map::store::Projection as StoreProjection;
use llkv_plan::SelectProjection;
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::ScanProjection;
use llkv_types::LogicalFieldId;
use simd_r_drive_entry_handle::EntryHandle;

pub fn build_wildcard_projections<P>(table: &ExecutorTable<P>) -> Vec<ScanProjection>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table
        .schema
        .columns
        .iter()
        .map(|column| {
            ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table_id(), column.field_id),
                column.name.clone(),
            ))
        })
        .collect()
}

pub fn build_projected_columns<P>(
    table: &ExecutorTable<P>,
    projections: &[SelectProjection],
) -> ExecutorResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut result = Vec::with_capacity(projections.len());
    for projection in projections.iter() {
        match projection {
            SelectProjection::AllColumns => {
                result.extend(build_wildcard_projections(table));
            }
            SelectProjection::AllColumnsExcept { exclude } => {
                // Get all columns except the excluded ones
                let exclude_lower: Vec<String> =
                    exclude.iter().map(|e| e.to_ascii_lowercase()).collect();

                for col in &table.schema.columns {
                    let col_name_lower = col.name.to_ascii_lowercase();
                    if !exclude_lower.contains(&col_name_lower) {
                        result.push(ScanProjection::from(StoreProjection::with_alias(
                            LogicalFieldId::for_user(table.table_id(), col.field_id),
                            col.name.clone(),
                        )));
                    }
                }
            }
            SelectProjection::Column { name, alias } => {
                let column = table.schema.resolve(name).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "Binder Error: does not have a column named '{}'",
                        name
                    ))
                })?;
                let alias = alias.clone().unwrap_or_else(|| column.name.clone());
                result.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table_id(), column.field_id),
                    alias,
                )));
            }
            SelectProjection::Computed { expr, alias } => {
                let scalar =
                    super::expression::translate_scalar(expr, table.schema.as_ref(), |name| {
                        Error::InvalidArgumentError(format!(
                            "Binder Error: does not have a column named '{}'",
                            name
                        ))
                    })?;
                result.push(ScanProjection::computed(scalar, alias.clone()));
            }
        }
    }
    if result.is_empty() {
        return Err(Error::InvalidArgumentError(
            "projection must include at least one column".into(),
        ));
    }
    Ok(result)
}
