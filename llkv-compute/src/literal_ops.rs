//! Literal arithmetic helpers shared across layers.

use llkv_result::Error;
use llkv_types::decimal::DecimalValue;
use llkv_types::IntervalValue;

use crate::date::{add_interval_to_date32, subtract_interval_from_date32};

#[derive(Clone, Debug)]
pub enum LiteralValue {
    Null,
    Integer(i64),
    Float(f64),
    Decimal(DecimalValue),
    Date32(i32),
    Interval(IntervalValue),
}

impl From<DecimalValue> for LiteralValue {
    fn from(value: DecimalValue) -> Self {
        LiteralValue::Decimal(value)
    }
}

pub fn add(lhs: LiteralValue, rhs: LiteralValue) -> Result<LiteralValue, Error> {
    match (lhs, rhs) {
        (LiteralValue::Null, _) | (_, LiteralValue::Null) => Ok(LiteralValue::Null),
        (LiteralValue::Date32(days), LiteralValue::Interval(interval))
        | (LiteralValue::Interval(interval), LiteralValue::Date32(days)) => {
            let adjusted = add_interval_to_date32(days, interval)?;
            Ok(LiteralValue::Date32(adjusted))
        }
        (LiteralValue::Interval(left), LiteralValue::Interval(right)) => left
            .checked_add(right)
            .map(LiteralValue::Interval)
            .ok_or_else(|| Error::InvalidArgumentError("interval addition overflow".into())),
        _ => Err(Error::InvalidArgumentError(
            "unsupported literal expression: binary operation".into(),
        )),
    }
}

pub fn subtract(lhs: LiteralValue, rhs: LiteralValue) -> Result<LiteralValue, Error> {
    match (lhs, rhs) {
        (LiteralValue::Null, _) | (_, LiteralValue::Null) => Ok(LiteralValue::Null),
        (LiteralValue::Date32(days), LiteralValue::Interval(interval)) => {
            let adjusted = subtract_interval_from_date32(days, interval)?;
            Ok(LiteralValue::Date32(adjusted))
        }
        (LiteralValue::Date32(lhs_days), LiteralValue::Date32(rhs_days)) => {
            let delta = i64::from(lhs_days) - i64::from(rhs_days);
            if delta < i64::from(i32::MIN) || delta > i64::from(i32::MAX) {
                return Err(Error::InvalidArgumentError(
                    "DATE subtraction overflowed day precision".into(),
                ));
            }
            Ok(LiteralValue::Interval(IntervalValue::new(0, delta as i32, 0)))
        }
        (LiteralValue::Interval(left), LiteralValue::Interval(right)) => left
            .checked_sub(right)
            .map(LiteralValue::Interval)
            .ok_or_else(|| Error::InvalidArgumentError("interval subtraction overflow".into())),
        _ => Err(Error::InvalidArgumentError(
            "unsupported literal expression: binary operation".into(),
        )),
    }
}

pub fn bitshift(op: sqlparser::ast::BinaryOperator, lhs: LiteralValue, rhs: LiteralValue) -> Result<LiteralValue, Error> {
    use sqlparser::ast::BinaryOperator::{PGBitwiseShiftLeft, PGBitwiseShiftRight};

    if matches!(lhs, LiteralValue::Null) || matches!(rhs, LiteralValue::Null) {
        return Ok(LiteralValue::Null);
    }

    let lhs_i64 = match lhs {
        LiteralValue::Integer(i) => i,
        LiteralValue::Float(f) => f as i64,
        LiteralValue::Date32(days) => days as i64,
        _ => {
            return Err(Error::InvalidArgumentError(
                "bitwise shift requires numeric operands".into(),
            ));
        }
    };

    let rhs_i64 = match rhs {
        LiteralValue::Integer(i) => i,
        LiteralValue::Float(f) => f as i64,
        LiteralValue::Date32(days) => days as i64,
        _ => {
            return Err(Error::InvalidArgumentError(
                "bitwise shift requires numeric operands".into(),
            ));
        }
    };

    let result = match op {
        PGBitwiseShiftLeft => lhs_i64.wrapping_shl(rhs_i64 as u32),
        PGBitwiseShiftRight => lhs_i64.wrapping_shr(rhs_i64 as u32),
        _ => unreachable!(),
    };

    Ok(LiteralValue::Integer(result))
}
