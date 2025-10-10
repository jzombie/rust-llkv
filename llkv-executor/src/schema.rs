use super::{ExecutorResult, ExecutorTable, ScanProjection};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_expr::expr::ScalarExpr;
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::collections::HashMap;
use std::sync::Arc;

pub fn schema_for_projections<P>(
    table: &ExecutorTable<P>,
    projections: &[ScanProjection],
) -> ExecutorResult<Arc<Schema>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut fields: Vec<Field> = Vec::with_capacity(projections.len());
    for projection in projections {
        match projection {
            ScanProjection::Column(proj) => {
                let field_id = proj.logical_field_id.field_id();
                let column = table.schema.column_by_field_id(field_id).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column with field id {} in projection",
                        field_id
                    ))
                })?;
                let name = proj.alias.clone().unwrap_or_else(|| column.name.clone());
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    column.field_id.to_string(),
                );
                let field = Field::new(&name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata);
                fields.push(field);
            }
            ScanProjection::Computed { alias, expr } => {
                let dtype = match expr {
                    ScalarExpr::Literal(Literal::Integer(_)) => DataType::Int64,
                    ScalarExpr::Literal(Literal::Float(_)) => DataType::Float64,
                    ScalarExpr::Literal(Literal::String(_)) => DataType::Utf8,
                    ScalarExpr::Column(field_id) => {
                        let column =
                            table.schema.column_by_field_id(*field_id).ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "unknown column with field id {} in computed projection",
                                    field_id
                                ))
                            })?;
                        column.data_type.clone()
                    }
                    ScalarExpr::Binary { .. } => DataType::Float64,
                };
                let field = Field::new(alias, dtype, true);
                fields.push(field);
            }
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}
