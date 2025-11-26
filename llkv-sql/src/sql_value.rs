use crate::{
    SqlResult, interval::parse_interval_literal, sql_engine::placeholder_marker,
    sql_engine::register_placeholder,
};
use llkv_compute::date::parse_date32_literal;
use llkv_types::{IntervalValue, Literal};
use llkv_plan::plans::PlanValue;
use llkv_result::Error;
use llkv_types::decimal::DecimalValue;
use rustc_hash::FxHashMap;
use sqlparser::ast::{
    BinaryOperator, DataType, Expr as SqlExpr, TypedString, UnaryOperator, Value, ValueWithSpan,
};

#[derive(Clone, Debug)]
pub(crate) enum SqlValue {
    Null,
    Integer(i64),
    Float(f64),
    Decimal(DecimalValue),
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
            SqlValue::Integer(i) => Ok(Literal::Integer(i.into())),
            SqlValue::Float(f) => Ok(Literal::Float(f)),
            SqlValue::Decimal(d) => Ok(Literal::Decimal(d)),
            SqlValue::Date32(days) => Ok(Literal::Date32(days)),
            SqlValue::Interval(interval) => Ok(Literal::Interval(interval)),
            SqlValue::Boolean(_) | SqlValue::String(_) | SqlValue::Struct(_) => Err(
                Error::InvalidArgumentError(
                    "unsupported literal expression: binary operation".into(),
                ),
            ),
        }
    }

    fn from_literal(lit: &Literal) -> SqlResult<SqlValue> {
        Ok(match lit {
            Literal::Null => SqlValue::Null,
            Literal::Integer(i) => SqlValue::Integer((*i).try_into().map_err(|_| {
                Error::InvalidArgumentError("integer literal out of range".into())
            })?),
            Literal::Float(f) => SqlValue::Float(*f),
            Literal::Decimal(d) => SqlValue::Decimal(*d),
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
                SqlValue::Integer(v) => Ok(SqlValue::Integer(-v)),
                SqlValue::Float(v) => Ok(SqlValue::Float(-v)),
                SqlValue::Decimal(dec) => {
                    // Negate the raw i128 value while preserving scale
                    DecimalValue::new(-dec.raw_value(), dec.scale())
                        .map(SqlValue::Decimal)
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
    // Scientific notation (e/E) requires float
    if text.contains(['e', 'E']) {
        let value = text
            .parse::<f64>()
            .map_err(|err| Error::InvalidArgumentError(format!("invalid float literal: {err}")))?;
        return Ok(SqlValue::Float(value));
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

        return Ok(SqlValue::Decimal(decimal));
    }

    // No decimal point → integer
    let value = text
        .parse::<i64>()
        .map_err(|err| Error::InvalidArgumentError(format!("invalid integer literal: {err}")))?;
    Ok(SqlValue::Integer(value))
}

impl From<SqlValue> for PlanValue {
    fn from(value: SqlValue) -> Self {
        match value {
            SqlValue::Null => PlanValue::Null,
            SqlValue::Integer(v) => PlanValue::Integer(v),
            SqlValue::Float(v) => PlanValue::Float(v),
            SqlValue::Decimal(d) => PlanValue::Decimal(d),
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
    use llkv_compute::date::add_interval_to_date32;

    let lhs_conv = lhs.into_literal()?;
    let rhs_conv = rhs.into_literal()?;
    match (lhs_conv, rhs_conv) {
        (Literal::Null, _) | (_, Literal::Null) => Ok(SqlValue::Null),
        (Literal::Date32(days), Literal::Interval(interval))
        | (Literal::Interval(interval), Literal::Date32(days)) => {
            let adjusted = add_interval_to_date32(days, interval)?;
            Ok(SqlValue::Date32(adjusted))
        }
        (Literal::Interval(left), Literal::Interval(right)) => left
            .checked_add(right)
            .map(SqlValue::Interval)
            .ok_or_else(|| Error::InvalidArgumentError("interval addition overflow".into())),
        _ => Err(Error::InvalidArgumentError(
            "unsupported literal expression: binary operation".into(),
        )),
    }
}

fn subtract_literals(lhs: SqlValue, rhs: SqlValue) -> SqlResult<SqlValue> {
    use llkv_compute::date::subtract_interval_from_date32;

    let lhs_conv = lhs.into_literal()?;
    let rhs_conv = rhs.into_literal()?;
    match (lhs_conv, rhs_conv) {
        (Literal::Null, _) | (_, Literal::Null) => Ok(SqlValue::Null),
        (Literal::Date32(days), Literal::Interval(interval)) => {
            let adjusted = subtract_interval_from_date32(days, interval)?;
            Ok(SqlValue::Date32(adjusted))
        }
        (Literal::Date32(lhs_days), Literal::Date32(rhs_days)) => {
            let delta = i64::from(lhs_days) - i64::from(rhs_days);
            if delta < i64::from(i32::MIN) || delta > i64::from(i32::MAX) {
                return Err(Error::InvalidArgumentError(
                    "DATE subtraction overflowed day precision".into(),
                ));
            }
            Ok(SqlValue::Interval(IntervalValue::new(0, delta as i32, 0)))
        }
        (Literal::Interval(left), Literal::Interval(right)) => left
            .checked_sub(right)
            .map(SqlValue::Interval)
            .ok_or_else(|| Error::InvalidArgumentError("interval subtraction overflow".into())),
        _ => Err(Error::InvalidArgumentError(
            "unsupported literal expression: binary operation".into(),
        )),
    }
}

fn bitshift_literals(op: BinaryOperator, lhs: SqlValue, rhs: SqlValue) -> SqlResult<SqlValue> {
    let lhs_conv = lhs.into_literal()?;
    let rhs_conv = rhs.into_literal()?;
    if matches!(lhs_conv, Literal::Null) || matches!(rhs_conv, Literal::Null) {
        return Ok(SqlValue::Null);
    }

    let lhs_i64 = match lhs_conv {
        Literal::Integer(i) => i
            .try_into()
            .map_err(|_| Error::InvalidArgumentError("bitshift lhs out of range".into()))?,
        Literal::Float(f) => f as i64,
        Literal::Date32(days) => days as i64,
        _ => {
            return Err(Error::InvalidArgumentError(
                "bitwise shift requires numeric operands".into(),
            ));
        }
    };

    let rhs_i64 = match rhs_conv {
        Literal::Integer(i) => i
            .try_into()
            .map_err(|_| Error::InvalidArgumentError("bitshift rhs out of range".into()))?,
        Literal::Float(f) => f as i64,
        Literal::Date32(days) => days as i64,
        _ => {
            return Err(Error::InvalidArgumentError(
                "bitwise shift requires numeric operands".into(),
            ));
        }
    };

    let result = match op {
        BinaryOperator::PGBitwiseShiftLeft => lhs_i64.wrapping_shl(rhs_i64 as u32),
        BinaryOperator::PGBitwiseShiftRight => lhs_i64.wrapping_shr(rhs_i64 as u32),
        _ => unreachable!(),
    };

    Ok(SqlValue::Integer(result))
}
