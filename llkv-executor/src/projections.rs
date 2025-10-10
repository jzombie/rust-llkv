use super::{DslResult, DslSchema, DslTable, FieldId, ScanProjection, LogicalFieldId, StoreProjection};
use llkv_plan::SelectProjection;
use llkv_result::Error;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

pub fn build_wildcard_projections<P>(table: &DslTable<P>) -> Vec<ScanProjection>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table
        .schema
        .columns
        .iter()
        .map(|column| {
            ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                column.name.clone(),
            ))
        })
        .collect()
}

pub fn build_projected_columns<P>(
    table: &DslTable<P>,
    projections: &[SelectProjection],
) -> DslResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut result = Vec::with_capacity(projections.len());
    for projection in projections.iter() {
        match projection {
            SelectProjection::AllColumns => {
                result.extend(build_wildcard_projections(table));
            }
            SelectProjection::Column { name, alias } => {
                let column = table.schema.resolve(name).ok_or_else(|| {
                    Error::InvalidArgumentError(format!("unknown column '{}' in projection", name))
                })?;
                let alias = alias.clone().unwrap_or_else(|| column.name.clone());
                result.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                    alias,
                )));
            }
            SelectProjection::Computed { expr, alias } => {
                let scalar = translate_scalar(expr, table.schema.as_ref())?;
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

fn translate_scalar(
    expr: &llkv_expr::expr::ScalarExpr<String>,
    schema: &DslSchema,
) -> DslResult<llkv_expr::expr::ScalarExpr<FieldId>> {
    use llkv_expr::expr::ScalarExpr;
    match expr {
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Column(name) => {
            let column = schema.resolve(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{}'", name))
            })?;
            Ok(ScalarExpr::Column(column.field_id))
        }
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(translate_scalar(left, schema)?),
            op: *op,
            right: Box::new(translate_scalar(right, schema)?),
        }),
    }
}
