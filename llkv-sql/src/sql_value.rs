use crate::{
    SqlResult, interval::parse_interval_literal, sql_engine::placeholder_marker,
    sql_engine::register_placeholder,
};
use llkv_compute::date::parse_date32_literal;
use llkv_plan::plans::PlanValue;
use llkv_result::Error;
use llkv_types::decimal::DecimalValue;
use llkv_types::{IntervalValue, Literal};
use rustc_hash::FxHashMap;
use sqlparser::ast::{
    BinaryOperator, DataType, Expr as SqlExpr, TypedString, UnaryOperator, Value, ValueWithSpan,
};

#[derive(Clone, Debug)]
pub(crate) enum SqlValue {
    Null,
    Int64(i64),
    Float64(f64),
    Decimal128(DecimalValue),
    Boolean(bool),
    String(String),
    Date32(i32),
    Interval(IntervalValue),
    Struct(FxHashMap<String, SqlValue>),
}

impl SqlValue {
    fn into_literal(self) -> SqlResult<Literal> {
        match self {
            SqlValue::Null => Ok(Literal::Null),
            SqlValue::Int64(i) => Ok(Literal::Int128(i.into())),
            SqlValue::Float64(f) => Ok(Literal::Float64(f)),
            SqlValue::Decimal128(d) => Ok(Literal::Decimal128(d)),
            SqlValue::Date32(days) => Ok(Literal::Date32(days)),
            SqlValue::Interval(interval) => Ok(Literal::Interval(interval)),
            SqlValue::Boolean(_) | SqlValue::String(_) | SqlValue::Struct(_) => {
                Err(Error::InvalidArgumentError(
                    "unsupported literal expression: binary operation".into(),
                ))
            }
        }
    }

    pub fn from_number_literal(text: &str) -> SqlResult<SqlValue> {
        // Scientific notation (e/E) requires float
        if text.contains(['e', 'E']) {
            let value = text.parse::<f64>().map_err(|err| {
                Error::InvalidArgumentError(format!("invalid float literal: {err}"))
            })?;
            return Ok(SqlValue::Float64(value));
        }

        // Decimal point → parse as Decimal with exact precision
        if let Some(dot_pos) = text.find('.') {
            let integer_part = &text[..dot_pos];
            let fractional_part = &text[dot_pos + 1..];

            // Parse the number as i128 by removing the decimal point
            let combined = format!("{}{}", integer_part, fractional_part);
            let raw_value = combined.parse::<i128>().map_err(|err| {
                Error::InvalidArgumentError(format!("invalid decimal literal: {err}"))
            })?;

            // Scale is the number of digits after the decimal point
            let scale = fractional_part.len() as i8;

            // Create DecimalValue
            let decimal = DecimalValue::new(raw_value, scale).map_err(|err| {
                Error::InvalidArgumentError(format!("invalid decimal literal: {}", err))
            })?;

            return Ok(SqlValue::Decimal128(decimal));
        }

        // No decimal point → integer
        let value = text.parse::<i64>().map_err(|err| {
            Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
        })?;
        Ok(SqlValue::Int64(value))
    }

    #[allow(dead_code)]
    fn from_literal(lit: &Literal) -> SqlResult<SqlValue> {
        Ok(match lit {
            Literal::Null => SqlValue::Null,
            Literal::Int128(i) => {
                SqlValue::Int64((*i).try_into().map_err(|_| {
                    Error::InvalidArgumentError("integer literal out of range".into())
                })?)
            }
            Literal::Float64(f) => SqlValue::Float64(*f),
            Literal::Decimal128(d) => SqlValue::Decimal128(*d),
            Literal::String(s) => SqlValue::String(s.clone()),
            Literal::Boolean(b) => SqlValue::Boolean(*b),
            Literal::Date32(days) => SqlValue::Date32(*days),
            Literal::Interval(interval) => SqlValue::Interval(*interval),
            Literal::Struct(fields) => {
                let mut map = FxHashMap::with_capacity_and_hasher(fields.len(), Default::default());
                for (k, v) in fields {
                    map.insert(k.clone(), SqlValue::from_literal(v)?);
                }
                SqlValue::Struct(map)
            }
        })
    }

    pub(crate) fn try_from_expr(expr: &SqlExpr) -> SqlResult<Self> {
        match expr {
            SqlExpr::Value(value) => SqlValue::from_value(value),
            SqlExpr::TypedString(typed) => SqlValue::from_typed_string(typed),
            SqlExpr::Interval(interval) => {
                let parsed = parse_interval_literal(interval)?;
                Ok(SqlValue::Interval(parsed))
            }
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => match SqlValue::try_from_expr(expr)? {
                SqlValue::Int64(v) => Ok(SqlValue::Int64(-v)),
                SqlValue::Float64(v) => Ok(SqlValue::Float64(-v)),
                SqlValue::Decimal128(dec) => {
                    // Negate the raw i128 value while preserving scale
                    DecimalValue::new(-dec.raw_value(), dec.scale())
                        .map(SqlValue::Decimal128)
                        .map_err(|err| {
                            Error::InvalidArgumentError(format!(
                                "decimal negation overflow: {}",
                                err
                            ))
                        })
                }
                SqlValue::Interval(interval) => interval
                    .checked_neg()
                    .map(SqlValue::Interval)
                    .ok_or_else(|| Error::InvalidArgumentError("interval overflow".into())),
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
            SqlExpr::Cast {
                expr, data_type, ..
            } => {
                let inner = SqlValue::try_from_expr(expr)?;
                match data_type {
                    DataType::Date => match inner {
                        SqlValue::Null => Ok(SqlValue::Null),
                        SqlValue::String(text) => {
                            let days = parse_date32_literal(&text)?;
                            Ok(SqlValue::Date32(days))
                        }
                        SqlValue::Date32(days) => Ok(SqlValue::Date32(days)),
                        other => Err(Error::InvalidArgumentError(format!(
                            "cannot CAST literal {:?} to DATE",
                            other
                        ))),
                    },
                    _ => match inner {
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
                }
            }
            SqlExpr::Nested(inner) => SqlValue::try_from_expr(inner),
            SqlExpr::Dictionary(fields) => SqlValue::from_dictionary(fields),
            SqlExpr::BinaryOp { left, op, right } => {
                let lhs = SqlValue::try_from_expr(left)?;
                let rhs = SqlValue::try_from_expr(right)?;
                match op {
                    BinaryOperator::Plus => add_literals(lhs, rhs),
                    BinaryOperator::Minus => subtract_literals(lhs, rhs),
                    BinaryOperator::PGBitwiseShiftLeft | BinaryOperator::PGBitwiseShiftRight => {
                        bitshift_literals(op.clone(), lhs, rhs)
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
            Value::Number(text, _) => SqlValue::from_number_literal(text),
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

impl From<SqlValue> for PlanValue {
    fn from(value: SqlValue) -> Self {
        match value {
            SqlValue::Null => PlanValue::Null,
            SqlValue::Int64(v) => PlanValue::Integer(v),
            SqlValue::Float64(v) => PlanValue::Float(v),
            SqlValue::Decimal128(d) => PlanValue::Decimal(d),
            SqlValue::Boolean(v) => PlanValue::Integer(if v { 1 } else { 0 }),
            SqlValue::String(s) => PlanValue::String(s),
            SqlValue::Date32(days) => PlanValue::Date32(days),
            SqlValue::Interval(interval) => PlanValue::Interval(interval),
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

fn add_literals(lhs: SqlValue, rhs: SqlValue) -> SqlResult<SqlValue> {
    let lhs_conv = lhs.into_literal()?;
    let rhs_conv = rhs.into_literal()?;
    let result = llkv_compute::literal::add_literals(&lhs_conv, &rhs_conv)?;
    SqlValue::from_literal(&result)
}

fn subtract_literals(lhs: SqlValue, rhs: SqlValue) -> SqlResult<SqlValue> {
    let lhs_conv = lhs.into_literal()?;
    let rhs_conv = rhs.into_literal()?;
    let result = llkv_compute::literal::subtract_literals(&lhs_conv, &rhs_conv)?;
    SqlValue::from_literal(&result)
}

fn bitshift_literals(op: BinaryOperator, lhs: SqlValue, rhs: SqlValue) -> SqlResult<SqlValue> {
    let lhs_conv = lhs.into_literal()?;
    let rhs_conv = rhs.into_literal()?;
    let result = llkv_compute::literal::bitshift_literals(op, &lhs_conv, &rhs_conv)?;
    SqlValue::from_literal(&result)
}
