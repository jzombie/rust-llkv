use crate::{SqlResult, sql_engine::placeholder_marker, sql_engine::register_placeholder};
use llkv_executor::utils::parse_date32_literal;
use llkv_plan::plans::PlanValue;
use llkv_result::Error;
use rustc_hash::FxHashMap;
use sqlparser::ast::{BinaryOperator, DataType, Expr as SqlExpr, TypedString, UnaryOperator, Value, ValueWithSpan};

#[derive(Clone, Debug)]
pub(crate) enum SqlValue {
    Null,
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Date32(i32),
    Struct(FxHashMap<String, SqlValue>),
}

impl SqlValue {
    pub(crate) fn try_from_expr(expr: &SqlExpr) -> SqlResult<Self> {
        match expr {
            SqlExpr::Value(value) => SqlValue::from_value(value),
            SqlExpr::TypedString(typed) => SqlValue::from_typed_string(typed),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => match SqlValue::try_from_expr(expr)? {
                SqlValue::Integer(v) => Ok(SqlValue::Integer(-v)),
                SqlValue::Float(v) => Ok(SqlValue::Float(-v)),
                SqlValue::Null
                | SqlValue::Boolean(_)
                | SqlValue::String(_)
                | SqlValue::Date32(_)
                | SqlValue::Struct(_) => Err(Error::InvalidArgumentError(
                    "cannot negate non-numeric literal".into(),
                )),
            },
            SqlExpr::UnaryOp {
                op: UnaryOperator::Plus,
                expr,
            } => SqlValue::try_from_expr(expr),
            SqlExpr::Cast { expr, .. } => match SqlValue::try_from_expr(expr)? {
                SqlValue::Null => Ok(SqlValue::Null),
                SqlValue::Struct(_) => Err(Error::InvalidArgumentError(
                    "cannot CAST struct literals".into(),
                )),
                SqlValue::Date32(_) => Err(Error::InvalidArgumentError(
                    "cannot CAST DATE literals".into(),
                )),
                other => Err(Error::InvalidArgumentError(format!(
                    "unsupported literal CAST expression: {other:?}"
                ))),
            },
            SqlExpr::Nested(inner) => SqlValue::try_from_expr(inner),
            SqlExpr::Dictionary(fields) => SqlValue::from_dictionary(fields),
            SqlExpr::BinaryOp { left, op, right } => {
                // Support bitwise shifts for INSERT VALUES (e.g., 1<<63)
                match op {
                    BinaryOperator::PGBitwiseShiftLeft | BinaryOperator::PGBitwiseShiftRight => {
                        let lhs = SqlValue::try_from_expr(left)?;
                        let rhs = SqlValue::try_from_expr(right)?;

                        let lhs_i64 = match lhs {
                            SqlValue::Integer(i) => i,
                            SqlValue::Float(f) => f as i64,
                            SqlValue::Date32(days) => days as i64,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "bitwise shift requires numeric operands".into(),
                                ));
                            }
                        };

                        let rhs_i64 = match rhs {
                            SqlValue::Integer(i) => i,
                            SqlValue::Float(f) => f as i64,
                            SqlValue::Date32(days) => days as i64,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "bitwise shift requires numeric operands".into(),
                                ));
                            }
                        };

                        let result = match op {
                            BinaryOperator::PGBitwiseShiftLeft => {
                                lhs_i64.wrapping_shl(rhs_i64 as u32)
                            }
                            BinaryOperator::PGBitwiseShiftRight => {
                                lhs_i64.wrapping_shr(rhs_i64 as u32)
                            }
                            _ => unreachable!(),
                        };

                        Ok(SqlValue::Integer(result))
                    }
                    _ => Err(Error::InvalidArgumentError(
                        "unsupported literal expression: binary operation".into(),
                    )),
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported literal expression: {other:?}"
            ))),
        }
    }

    fn from_dictionary(fields: &[sqlparser::ast::DictionaryField]) -> SqlResult<Self> {
        let mut map = FxHashMap::with_capacity_and_hasher(fields.len(), Default::default());
        for field in fields {
            let key = field.key.value.clone();
            let value = match field.value.as_ref() {
                SqlExpr::Value(v) => SqlValue::from_value(v)?,
                other => SqlValue::try_from_expr(other)?,
            };
            map.insert(key, value);
        }
        Ok(SqlValue::Struct(map))
    }

    fn from_value(value: &ValueWithSpan) -> SqlResult<Self> {
        match &value.value {
            Value::Null => Ok(SqlValue::Null),
            Value::Number(text, _) => parse_number_literal(text),
            Value::Boolean(value) => Ok(SqlValue::Boolean(*value)),
            Value::Placeholder(raw) => {
                let index = register_placeholder(raw)?;
                Ok(SqlValue::String(placeholder_marker(index)))
            }
            other => {
                if let Some(text) = other.clone().into_string() {
                    Ok(SqlValue::String(text))
                } else {
                    Err(Error::InvalidArgumentError(format!(
                        "unsupported literal: {other:?}"
                    )))
                }
            }
        }
    }

    fn from_typed_string(typed: &TypedString) -> SqlResult<Self> {
        let text = match typed.value.value.clone().into_string() {
            Some(text) => text,
            None => {
                return Err(Error::InvalidArgumentError(
                    "typed string literal must be a quoted string".into(),
                ));
            }
        };

        match typed.data_type {
            DataType::Date => {
                let days = parse_date32_literal(&text)?;
                Ok(SqlValue::Date32(days))
            }
            _ => Ok(SqlValue::String(text)),
        }
    }
}

fn parse_number_literal(text: &str) -> SqlResult<SqlValue> {
    if text.contains(['.', 'e', 'E']) {
        let value = text
            .parse::<f64>()
            .map_err(|err| Error::InvalidArgumentError(format!("invalid float literal: {err}")))?;
        Ok(SqlValue::Float(value))
    } else {
        let value = text.parse::<i64>().map_err(|err| {
            Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
        })?;
        Ok(SqlValue::Integer(value))
    }
}

impl From<SqlValue> for PlanValue {
    fn from(value: SqlValue) -> Self {
        match value {
            SqlValue::Null => PlanValue::Null,
            SqlValue::Integer(v) => PlanValue::Integer(v),
            SqlValue::Float(v) => PlanValue::Float(v),
            SqlValue::Boolean(v) => PlanValue::Integer(if v { 1 } else { 0 }),
            SqlValue::String(s) => PlanValue::String(s),
            SqlValue::Date32(days) => PlanValue::Date32(days),
            SqlValue::Struct(fields) => {
                let converted: FxHashMap<String, PlanValue> = fields
                    .into_iter()
                    .map(|(k, v)| (k, PlanValue::from(v)))
                    .collect();
                PlanValue::Struct(converted)
            }
        }
    }
}
