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
    // Iterative postorder traversal using a Frame-based pattern.
    // See llkv-plan::traversal module documentation for pattern details.
    //
    // This avoids stack overflow on deeply nested expressions (50k+ nodes) by using
    // explicit work_stack and result_stack instead of recursion.
    //
    // Note: We use a manual OwnedFrame enum here instead of TransformFrame because
    // this function takes ownership of the input expr, requiring owned values on the stack.

    /// Context passed through Exit frames during predicate translation
    enum PredicateExitContext {
        And(usize), // child count
        Or(usize),  // child count
        Not,
    }

    /// Frame enum for owned value traversal
    enum OwnedFrame {
        Enter(LlkvExpr<'static, String>),
        Exit(PredicateExitContext),
        Leaf(LlkvExpr<'static, FieldId>),
    }

    let mut owned_stack: Vec<OwnedFrame> = vec![OwnedFrame::Enter(expr)];
    let mut result_stack: Vec<LlkvExpr<'static, FieldId>> = Vec::new();

    while let Some(frame) = owned_stack.pop() {
        match frame {
            OwnedFrame::Enter(node) => match node {
                LlkvExpr::And(children) => {
                    let count = children.len();
                    owned_stack.push(OwnedFrame::Exit(PredicateExitContext::And(count)));
                    for child in children.into_iter().rev() {
                        owned_stack.push(OwnedFrame::Enter(child));
                    }
                }
                LlkvExpr::Or(children) => {
                    let count = children.len();
                    owned_stack.push(OwnedFrame::Exit(PredicateExitContext::Or(count)));
                    for child in children.into_iter().rev() {
                        owned_stack.push(OwnedFrame::Enter(child));
                    }
                }
                LlkvExpr::Not(inner) => {
                    owned_stack.push(OwnedFrame::Exit(PredicateExitContext::Not));
                    owned_stack.push(OwnedFrame::Enter(*inner));
                }
                LlkvExpr::Pred(filter) => {
                    let field_id = resolve_field_id(schema, &filter.field_id, unknown_column)?;
                    owned_stack.push(OwnedFrame::Leaf(LlkvExpr::Pred(Filter {
                        field_id,
                        op: filter.op,
                    })));
                }
                LlkvExpr::Compare { left, op, right } => {
                    let left_expr =
                        translate_scalar_with(&left, schema, unknown_column, unknown_aggregate)?;
                    let right_expr =
                        translate_scalar_with(&right, schema, unknown_column, unknown_aggregate)?;
                    owned_stack.push(OwnedFrame::Leaf(LlkvExpr::Compare {
                        left: left_expr,
                        op,
                        right: right_expr,
                    }));
                }
                LlkvExpr::InList {
                    expr: target,
                    list,
                    negated,
                } => {
                    let translated_target =
                        translate_scalar_with(&target, schema, unknown_column, unknown_aggregate)?;
                    let mut translated_list: Vec<ScalarExpr<FieldId>> =
                        Vec::with_capacity(list.len());
                    for value in list {
                        translated_list.push(translate_scalar_with(
                            &value,
                            schema,
                            unknown_column,
                            unknown_aggregate,
                        )?);
                    }
                    owned_stack.push(OwnedFrame::Leaf(LlkvExpr::InList {
                        expr: translated_target,
                        list: translated_list,
                        negated,
                    }));
                }
                LlkvExpr::IsNull {
                    expr: target,
                    negated,
                } => {
                    let translated_target =
                        translate_scalar_with(&target, schema, unknown_column, unknown_aggregate)?;
                    owned_stack.push(OwnedFrame::Leaf(LlkvExpr::IsNull {
                        expr: translated_target,
                        negated,
                    }));
                }
                LlkvExpr::Literal(value) => {
                    owned_stack.push(OwnedFrame::Leaf(LlkvExpr::Literal(value)));
                }
                LlkvExpr::Exists(subquery) => {
                    owned_stack.push(OwnedFrame::Leaf(LlkvExpr::Exists(subquery)));
                }
            },
            OwnedFrame::Leaf(translated) => {
                result_stack.push(translated);
            }
            OwnedFrame::Exit(exit_context) => match exit_context {
                PredicateExitContext::And(count) => {
                    let translated: Vec<_> =
                        result_stack.drain(result_stack.len() - count..).collect();
                    result_stack.push(LlkvExpr::And(translated));
                }
                PredicateExitContext::Or(count) => {
                    let translated: Vec<_> =
                        result_stack.drain(result_stack.len() - count..).collect();
                    result_stack.push(LlkvExpr::Or(translated));
                }
                PredicateExitContext::Not => {
                    let inner = result_stack.pop().ok_or_else(|| {
                        Error::Internal(
                            "translate_predicate_with: result stack underflow for Not".into(),
                        )
                    })?;
                    result_stack.push(LlkvExpr::Not(Box::new(inner)));
                }
            },
        }
    }

    result_stack
        .pop()
        .ok_or_else(|| Error::Internal("translate_predicate_with: empty result stack".into()))
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
    _unknown_aggregate: G,
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
                _unknown_aggregate,
            )?),
            op: *op,
            right: Box::new(translate_scalar_with(
                right,
                schema,
                unknown_column,
                _unknown_aggregate,
            )?),
        }),
        ScalarExpr::Not(inner) => Ok(ScalarExpr::Not(Box::new(translate_scalar_with(
            inner,
            schema,
            unknown_column,
            _unknown_aggregate,
        )?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(translate_scalar_with(
                expr,
                schema,
                unknown_column,
                _unknown_aggregate,
            )?),
            negated: *negated,
        }),
        ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(translate_scalar_with(
                left,
                schema,
                unknown_column,
                _unknown_aggregate,
            )?),
            op: *op,
            right: Box::new(translate_scalar_with(
                right,
                schema,
                unknown_column,
                _unknown_aggregate,
            )?),
        }),
        ScalarExpr::Aggregate(agg) => {
            let translated =
                match agg {
                    AggregateCall::CountStar => AggregateCall::CountStar,
                    AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                        expr: Box::new(translate_scalar_with(
                            expr,
                            schema,
                            unknown_column,
                            _unknown_aggregate,
                        )?),
                        distinct: *distinct,
                    },
                    AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                        expr: Box::new(translate_scalar_with(
                            expr,
                            schema,
                            unknown_column,
                            _unknown_aggregate,
                        )?),
                        distinct: *distinct,
                    },
                    AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                        expr: Box::new(translate_scalar_with(
                            expr,
                            schema,
                            unknown_column,
                            _unknown_aggregate,
                        )?),
                        distinct: *distinct,
                    },
                    AggregateCall::Min(expr) => AggregateCall::Min(Box::new(
                        translate_scalar_with(expr, schema, unknown_column, _unknown_aggregate)?,
                    )),
                    AggregateCall::Max(expr) => AggregateCall::Max(Box::new(
                        translate_scalar_with(expr, schema, unknown_column, _unknown_aggregate)?,
                    )),
                    AggregateCall::CountNulls(expr) => AggregateCall::CountNulls(Box::new(
                        translate_scalar_with(expr, schema, unknown_column, _unknown_aggregate)?,
                    )),
                };
            Ok(ScalarExpr::Aggregate(translated))
        }
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(translate_scalar_with(
                base,
                schema,
                unknown_column,
                _unknown_aggregate,
            )?),
            field_name: field_name.clone(),
        }),
        ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
            expr: Box::new(translate_scalar_with(
                expr,
                schema,
                unknown_column,
                _unknown_aggregate,
            )?),
            data_type: data_type.clone(),
        }),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            let translated_operand = match operand.as_deref() {
                Some(inner) => Some(translate_scalar_with(
                    inner,
                    schema,
                    unknown_column,
                    _unknown_aggregate,
                )?),
                None => None,
            };

            let mut translated_branches = Vec::with_capacity(branches.len());
            for (when_expr, then_expr) in branches {
                let translated_when =
                    translate_scalar_with(when_expr, schema, unknown_column, _unknown_aggregate)?;
                let translated_then =
                    translate_scalar_with(then_expr, schema, unknown_column, _unknown_aggregate)?;
                translated_branches.push((translated_when, translated_then));
            }

            let translated_else = match else_expr.as_deref() {
                Some(inner) => Some(translate_scalar_with(
                    inner,
                    schema,
                    unknown_column,
                    _unknown_aggregate,
                )?),
                None => None,
            };

            Ok(ScalarExpr::Case {
                operand: translated_operand.map(Box::new),
                branches: translated_branches,
                else_expr: translated_else.map(Box::new),
            })
        }
        ScalarExpr::Coalesce(items) => {
            let mut translated_items = Vec::with_capacity(items.len());
            for item in items {
                translated_items.push(translate_scalar_with(
                    item,
                    schema,
                    unknown_column,
                    _unknown_aggregate,
                )?);
            }
            Ok(ScalarExpr::Coalesce(translated_items))
        }
        ScalarExpr::ScalarSubquery(subquery) => Ok(ScalarExpr::ScalarSubquery(subquery.clone())),
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
