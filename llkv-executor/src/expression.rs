use llkv_expr::expr::{AggregateCall, ScalarExpr};
use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_table::ROW_ID_FIELD_ID;
use crate::{ExecutorResult, ExecutorSchema, FieldId};
use llkv_result::{Error, Result as LlkvResult};

pub(crate) fn translate_scalar<F>(
    expr: &ScalarExpr<String>,
    schema: &ExecutorSchema,
    unknown_column: F,
) -> ExecutorResult<ScalarExpr<FieldId>>
where
    F: Fn(&str) -> Error + Copy,
{
    match expr {
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Column(name) => {
            let field_id = resolve_field_id(schema, name, unknown_column)?;
            Ok(ScalarExpr::Column(field_id))
        }
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(translate_scalar(left, schema, unknown_column)?),
            op: *op,
            right: Box::new(translate_scalar(right, schema, unknown_column)?),
        }),
        ScalarExpr::Aggregate(agg) => {
            let translated = match agg {
                AggregateCall::CountStar => AggregateCall::CountStar,
                AggregateCall::Count(name) => {
                    AggregateCall::Count(resolve_field_id(schema, name, unknown_column)?)
                }
                AggregateCall::Sum(name) => {
                    AggregateCall::Sum(resolve_field_id(schema, name, unknown_column)?)
                }
                AggregateCall::Min(name) => {
                    AggregateCall::Min(resolve_field_id(schema, name, unknown_column)?)
                }
                AggregateCall::Max(name) => {
                    AggregateCall::Max(resolve_field_id(schema, name, unknown_column)?)
                }
                AggregateCall::CountNulls(name) => {
                    AggregateCall::CountNulls(resolve_field_id(schema, name, unknown_column)?)
                }
            };
            Ok(ScalarExpr::Aggregate(translated))
        }
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(translate_scalar(base, schema, unknown_column)?),
            field_name: field_name.clone(),
        }),
    }
}

// TODO: Move to `resolvers.rs`
fn resolve_field_id<F>(
    schema: &ExecutorSchema,
    name: &str,
    unknown_column: F,
) -> ExecutorResult<FieldId>
where
    F: Fn(&str) -> Error,
{
    schema
        .resolve(name)
        .map(|column| column.field_id)
        .ok_or_else(|| unknown_column(name))
}

// TODO: Move to `resolvers.rs`
pub fn resolve_field_id_from_schema(schema: &ExecutorSchema, name: &str) -> LlkvResult<FieldId> {
    if name.eq_ignore_ascii_case(ROW_ID_COLUMN_NAME) {
        return Ok(ROW_ID_FIELD_ID);
    }

    schema
        .resolve(name)
        .map(|column| column.field_id)
        .ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{name}'"
            ))
        })
}