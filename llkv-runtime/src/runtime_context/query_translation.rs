//! Query translation helpers - converts string-based expressions to field-ID based expressions

use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_executor::ExecutorSchema;
use llkv_expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_result::{Error, Result};
use llkv_table::{FieldId, ROW_ID_FIELD_ID};
use std::ops::Bound;

pub(crate) fn full_table_scan_filter(field_id: FieldId) -> LlkvExpr<'static, FieldId> {
    LlkvExpr::Pred(Filter {
        field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    })
}

pub(crate) fn resolve_field_id_from_schema(schema: &ExecutorSchema, name: &str) -> Result<FieldId> {
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

pub(crate) fn translate_predicate(
    expr: LlkvExpr<'static, String>,
    schema: &ExecutorSchema,
) -> Result<LlkvExpr<'static, FieldId>> {
    match expr {
        LlkvExpr::And(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::And(converted))
        }
        LlkvExpr::Or(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::Or(converted))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(translate_predicate(
            *inner, schema,
        )?))),
        LlkvExpr::Pred(Filter { field_id, op }) => {
            let resolved = resolve_field_id_from_schema(schema, &field_id)?;
            Ok(LlkvExpr::Pred(Filter {
                field_id: resolved,
                op,
            }))
        }
        LlkvExpr::Compare { left, op, right } => {
            let left = translate_scalar(&left, schema)?;
            let right = translate_scalar(&right, schema)?;
            Ok(LlkvExpr::Compare { left, op, right })
        }
    }
}

pub(crate) fn translate_scalar(
    expr: &ScalarExpr<String>,
    schema: &ExecutorSchema,
) -> Result<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::Column(name) => {
            let field_id = resolve_field_id_from_schema(schema, name)?;
            Ok(ScalarExpr::column(field_id))
        }
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Binary { left, op, right } => {
            let left_expr = translate_scalar(left, schema)?;
            let right_expr = translate_scalar(right, schema)?;
            Ok(ScalarExpr::Binary {
                left: Box::new(left_expr),
                op: *op,
                right: Box::new(right_expr),
            })
        }
        ScalarExpr::Aggregate(agg) => {
            // Translate column names in aggregate calls to field IDs
            use llkv_expr::expr::AggregateCall;
            let translated_agg = match agg {
                AggregateCall::CountStar => AggregateCall::CountStar,
                AggregateCall::Count(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Count(field_id)
                }
                AggregateCall::Sum(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Sum(field_id)
                }
                AggregateCall::Min(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Min(field_id)
                }
                AggregateCall::Max(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Max(field_id)
                }
                AggregateCall::CountNulls(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::CountNulls(field_id)
                }
            };
            Ok(ScalarExpr::Aggregate(translated_agg))
        }
        ScalarExpr::GetField { base, field_name } => {
            let base_expr = translate_scalar(base, schema)?;
            Ok(ScalarExpr::GetField {
                base: Box::new(base_expr),
                field_name: field_name.clone(),
            })
        }
    }
}
