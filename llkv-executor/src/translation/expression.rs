use crate::{ExecutorResult, ExecutorSchema, FieldId};
use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_expr::expr::{AggregateCall, Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use llkv_table::ROW_ID_FIELD_ID;
use std::ops::Bound;

pub fn full_table_scan_filter(field_id: FieldId) -> LlkvExpr<'static, FieldId> {
    LlkvExpr::Pred(Filter {
        field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    })
}

pub fn translate_predicate<F>(
    expr: LlkvExpr<'static, String>,
    schema: &ExecutorSchema,
    unknown_column: F,
) -> ExecutorResult<LlkvExpr<'static, FieldId>>
where
    F: Fn(&str) -> Error + Copy,
{
    translate_predicate_with(expr, schema, unknown_column, unknown_column)
}

pub fn translate_predicate_with<F, G>(
    expr: LlkvExpr<'static, String>,
    schema: &ExecutorSchema,
    unknown_column: F,
    unknown_aggregate: G,
) -> ExecutorResult<LlkvExpr<'static, FieldId>>
where
    F: Fn(&str) -> Error + Copy,
    G: Fn(&str) -> Error + Copy,
{
    match expr {
        LlkvExpr::And(exprs) => {
            let mut translated = Vec::with_capacity(exprs.len());
            for expr in exprs {
                translated.push(translate_predicate_with(
                    expr,
                    schema,
                    unknown_column,
                    unknown_aggregate,
                )?);
            }
            Ok(LlkvExpr::And(translated))
        }
        LlkvExpr::Or(exprs) => {
            let mut translated = Vec::with_capacity(exprs.len());
            for expr in exprs {
                translated.push(translate_predicate_with(
                    expr,
                    schema,
                    unknown_column,
                    unknown_aggregate,
                )?);
            }
            Ok(LlkvExpr::Or(translated))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(translate_predicate_with(
            *inner,
            schema,
            unknown_column,
            unknown_aggregate,
        )?))),
        LlkvExpr::Pred(filter) => {
            let field_id = resolve_field_id(schema, &filter.field_id, unknown_column)?;
            Ok(LlkvExpr::Pred(Filter {
                field_id,
                op: filter.op,
            }))
        }
        LlkvExpr::Compare { left, op, right } => {
            let left_expr =
                translate_scalar_with(&left, schema, unknown_column, unknown_aggregate)?;
            let right_expr =
                translate_scalar_with(&right, schema, unknown_column, unknown_aggregate)?;
            Ok(LlkvExpr::Compare {
                left: left_expr,
                op,
                right: right_expr,
            })
        }
    }
}

pub fn translate_scalar<F>(
    expr: &ScalarExpr<String>,
    schema: &ExecutorSchema,
    unknown_column: F,
) -> ExecutorResult<ScalarExpr<FieldId>>
where
    F: Fn(&str) -> Error + Copy,
{
    translate_scalar_with(expr, schema, unknown_column, unknown_column)
}

pub fn translate_scalar_with<F, G>(
    expr: &ScalarExpr<String>,
    schema: &ExecutorSchema,
    unknown_column: F,
    unknown_aggregate: G,
) -> ExecutorResult<ScalarExpr<FieldId>>
where
    F: Fn(&str) -> Error + Copy,
    G: Fn(&str) -> Error + Copy,
{
    match expr {
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Column(name) => {
            let field_id = resolve_field_id(schema, name, unknown_column)?;
            Ok(ScalarExpr::Column(field_id))
        }
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(translate_scalar_with(
                left,
                schema,
                unknown_column,
                unknown_aggregate,
            )?),
            op: *op,
            right: Box::new(translate_scalar_with(
                right,
                schema,
                unknown_column,
                unknown_aggregate,
            )?),
        }),
        ScalarExpr::Aggregate(agg) => {
            let translated = match agg {
                AggregateCall::CountStar => AggregateCall::CountStar,
                AggregateCall::Count(name) => {
                    AggregateCall::Count(resolve_field_id(schema, name, unknown_aggregate)?)
                }
                AggregateCall::Sum(name) => {
                    AggregateCall::Sum(resolve_field_id(schema, name, unknown_aggregate)?)
                }
                AggregateCall::Min(name) => {
                    AggregateCall::Min(resolve_field_id(schema, name, unknown_aggregate)?)
                }
                AggregateCall::Max(name) => {
                    AggregateCall::Max(resolve_field_id(schema, name, unknown_aggregate)?)
                }
                AggregateCall::CountNulls(name) => {
                    AggregateCall::CountNulls(resolve_field_id(schema, name, unknown_aggregate)?)
                }
            };
            Ok(ScalarExpr::Aggregate(translated))
        }
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(translate_scalar_with(
                base,
                schema,
                unknown_column,
                unknown_aggregate,
            )?),
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
    // Check for special rowid column
    if name.eq_ignore_ascii_case(ROW_ID_COLUMN_NAME) {
        return Ok(ROW_ID_FIELD_ID);
    }

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
