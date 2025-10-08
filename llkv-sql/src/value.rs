use crate::SqlResult;
use llkv_dsl::DslValue;
use llkv_result::Error;
use sqlparser::ast::{Expr as SqlExpr, UnaryOperator, Value, ValueWithSpan};

#[derive(Clone, Debug)]
pub(crate) enum SqlValue {
    Null,
    Integer(i64),
    Float(f64),
    String(String),
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
                SqlValue::Null | SqlValue::String(_) => Err(Error::InvalidArgumentError(
                    "cannot negate non-numeric literal".into(),
                )),
            },
            SqlExpr::UnaryOp {
                op: UnaryOperator::Plus,
                expr,
            } => SqlValue::try_from_expr(expr),
            SqlExpr::Nested(inner) => SqlValue::try_from_expr(inner),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported literal expression: {other:?}"
            ))),
        }
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

impl From<SqlValue> for DslValue {
    fn from(value: SqlValue) -> Self {
        match value {
            SqlValue::Null => DslValue::Null,
            SqlValue::Integer(v) => DslValue::Integer(v),
            SqlValue::Float(v) => DslValue::Float(v),
            SqlValue::String(s) => DslValue::String(s),
        }
    }
}
