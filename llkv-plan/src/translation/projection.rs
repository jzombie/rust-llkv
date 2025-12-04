use crate::plans::PlanResult;
use crate::plans::SelectProjection;
use crate::schema::PlanSchema;
use crate::translation::expression::translate_scalar;
use llkv_column_map::store::Projection as StoreProjection;
use llkv_scan::ScanProjection;
use llkv_types::LogicalFieldId;
use llkv_types::ids::TableId;

pub fn build_wildcard_projections(schema: &PlanSchema, table_id: TableId) -> Vec<ScanProjection> {
    schema
        .columns
        .iter()
        .map(|column| {
            ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table_id, column.field_id),
                column.name.clone(),
            ))
        })
        .collect()
}

pub fn build_projected_columns(
    schema: &PlanSchema,
    table_id: TableId,
    projections: &[SelectProjection],
) -> PlanResult<Vec<ScanProjection>> {
    let mut result = Vec::with_capacity(projections.len());
    for projection in projections.iter() {
        match projection {
            SelectProjection::AllColumns => {
                result.extend(build_wildcard_projections(schema, table_id));
            }
            SelectProjection::AllColumnsExcept { exclude } => {
                // Get all columns except the excluded ones
                let exclude_lower: Vec<String> =
                    exclude.iter().map(|e| e.to_ascii_lowercase()).collect();

                for col in &schema.columns {
                    let col_name_lower = col.name.to_ascii_lowercase();
                    if !exclude_lower.contains(&col_name_lower) {
                        result.push(ScanProjection::from(StoreProjection::with_alias(
                            LogicalFieldId::for_user(table_id, col.field_id),
                            col.name.clone(),
                        )));
                    }
                }
            }
            SelectProjection::Column { name, alias } => {
                let mut col = schema.column_by_name(name);

                if col.is_none() {
                    // Try stripping qualifiers
                    let mut current_name = name.as_str();
                    while let Some(idx) = current_name.find('.') {
                        current_name = &current_name[idx + 1..];
                        col = schema.column_by_name(current_name);
                        if col.is_some() {
                            break;
                        }
                    }
                }

                let col = col.ok_or_else(|| {
                    llkv_result::Error::Internal(format!("Unknown column: {}", name))
                })?;

                let projection = StoreProjection::with_alias(
                    LogicalFieldId::for_user(table_id, col.field_id),
                    alias.clone().unwrap_or_else(|| col.name.clone()),
                );
                result.push(ScanProjection::from(projection));
            }
            SelectProjection::Computed { expr, alias } => {
                let translated_expr = translate_scalar(expr, schema, |name| {
                    llkv_result::Error::Internal(format!("Unknown column in expression: {}", name))
                })?;
                result.push(ScanProjection::computed(translated_expr, alias.clone()));
            }
        }
    }
    Ok(result)
}
