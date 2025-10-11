use super::{
    ExecutorResult, ExecutorSchema, ExecutorTable, FieldId, LogicalFieldId, ScanProjection,
    StoreProjection,
};
use llkv_plan::SelectProjection;
use llkv_result::Error;
use llkv_storage::pager::Pager;
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
            let mut p = StoreProjection::from(LogicalFieldId::for_user(
                table.table.table_id(),
                column.field_id,
            ));
            // preserve human-visible name for downstream results
            p.alias = Some(column.name.clone());
            ScanProjection::from(p)
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
            SelectProjection::Column { name, alias } => {
                let column = table.schema.resolve(name).ok_or_else(|| {
                    Error::InvalidArgumentError(format!("unknown column '{}' in projection", name))
                })?;
                // We keep user-visible alias names at the SQL/executor layer
                // for final result schema, but internal projections are numeric.
                let mut p = StoreProjection::from(LogicalFieldId::for_user(
                    table.table.table_id(),
                    column.field_id,
                ));
                // If the SQL projection provided an alias, use it; otherwise use the
                // original column name so downstream batches are labeled correctly.
                p.alias = alias.clone().or_else(|| Some(column.name.clone()));
                result.push(ScanProjection::from(p));
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
    schema: &ExecutorSchema,
) -> ExecutorResult<llkv_expr::expr::ScalarExpr<FieldId>> {
    use llkv_expr::expr::ScalarExpr;
    match expr {
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Column(name) => {
            let column = schema
                .resolve(name)
                .ok_or_else(|| Error::InvalidArgumentError(format!("unknown column '{}'", name)))?;
            Ok(ScalarExpr::Column(column.field_id))
        }
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(translate_scalar(left, schema)?),
            op: *op,
            right: Box::new(translate_scalar(right, schema)?),
        }),
    }
}
