use crate::SqlResult;
use llkv_plan::plans::PlanValue;
use llkv_result::Error;
use sqlparser::ast::{Expr as SqlExpr, UnaryOperator, Value, ValueWithSpan};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) enum SqlValue {
    Null,
    Integer(i64),
    Float(f64),
    String(String),
    Struct(HashMap<String, SqlValue>),
}

impl SqlValue {
    pub(crate) fn try_from_expr(expr: &SqlExpr) -> SqlResult<Self> {
        match expr {
            SqlExpr::Value(value) => SqlValue::from_value(value),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => match SqlValue::try_from_expr(expr)? {
                SqlValue::Integer(v) => Ok(SqlValue::Integer(-v)),
                SqlValue::Float(v) => Ok(SqlValue::Float(-v)),
                SqlValue::Null | SqlValue::String(_) | SqlValue::Struct(_) => {
                    Err(Error::InvalidArgumentError(
                        "cannot negate non-numeric literal".into(),
                    ))
                }
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
                other => Err(Error::InvalidArgumentError(format!(
                    "unsupported literal CAST expression: {other:?}"
                ))),
            },
            SqlExpr::Nested(inner) => SqlValue::try_from_expr(inner),
            SqlExpr::Dictionary(fields) => SqlValue::from_dictionary(fields),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported literal expression: {other:?}"
            ))),
        }
    }

    fn from_dictionary(fields: &[sqlparser::ast::DictionaryField]) -> SqlResult<Self> {
        let mut map = HashMap::new();
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
            Value::Boolean(_) => Err(Error::InvalidArgumentError(
                "BOOLEAN literals are not supported yet".into(),
            )),
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
            SqlValue::String(s) => PlanValue::String(s),
            SqlValue::Struct(fields) => {
                let converted: HashMap<String, PlanValue> = fields
                    .into_iter()
                    .map(|(k, v)| (k, PlanValue::from(v)))
                    .collect();
                PlanValue::Struct(converted)
            }
        }
    }
}
