//! Literal-level arithmetic helpers shared across layers.

use llkv_result::Error;
use llkv_types::{IntervalValue, Literal};

use sqlparser::ast::BinaryOperator;

use crate::date::{add_interval_to_date32, subtract_interval_from_date32};

/// Add two literals, supporting date/interval arithmetic.
pub fn add_literals(lhs: &Literal, rhs: &Literal) -> Result<Literal, Error> {
    match (lhs, rhs) {
        (Literal::Null, _) | (_, Literal::Null) => Ok(Literal::Null),
        (Literal::Date32(days), Literal::Interval(interval))
        | (Literal::Interval(interval), Literal::Date32(days)) => {
            let adjusted = add_interval_to_date32(*days, *interval)?;
            Ok(Literal::Date32(adjusted))
        }
        (Literal::Interval(left), Literal::Interval(right)) => left
            .checked_add(*right)
            .map(Literal::Interval)
            .ok_or_else(|| Error::InvalidArgumentError("interval addition overflow".into())),
        _ => Err(Error::InvalidArgumentError(
            "unsupported literal expression: binary operation".into(),
        )),
    }
}

/// Subtract two literals, supporting date/interval arithmetic.
pub fn subtract_literals(lhs: &Literal, rhs: &Literal) -> Result<Literal, Error> {
    match (lhs, rhs) {
        (Literal::Null, _) | (_, Literal::Null) => Ok(Literal::Null),
        (Literal::Date32(days), Literal::Interval(interval)) => {
            let adjusted = subtract_interval_from_date32(*days, *interval)?;
            Ok(Literal::Date32(adjusted))
        }
        (Literal::Date32(lhs_days), Literal::Date32(rhs_days)) => {
            let delta = i64::from(*lhs_days) - i64::from(*rhs_days);
            if delta < i64::from(i32::MIN) || delta > i64::from(i32::MAX) {
                return Err(Error::InvalidArgumentError(
                    "DATE subtraction overflowed day precision".into(),
                ));
            }
            Ok(Literal::Interval(IntervalValue::new(0, delta as i32, 0)))
        }
        (Literal::Interval(left), Literal::Interval(right)) => left
            .checked_sub(*right)
            .map(Literal::Interval)
            .ok_or_else(|| Error::InvalidArgumentError("interval subtraction overflow".into())),
        _ => Err(Error::InvalidArgumentError(
            "unsupported literal expression: binary operation".into(),
        )),
    }
}

/// Bitshift two literals, returning an integer literal.
pub fn bitshift_literals(
    op: BinaryOperator,
    lhs: &Literal,
    rhs: &Literal,
) -> Result<Literal, Error> {
    use sqlparser::ast::BinaryOperator::{PGBitwiseShiftLeft, PGBitwiseShiftRight};

    if matches!(lhs, Literal::Null) || matches!(rhs, Literal::Null) {
        return Ok(Literal::Null);
    }

    let lhs_i64 = match lhs {
        Literal::Int128(i) => (*i)
            .try_into()
            .map_err(|_| Error::InvalidArgumentError("bitwise shift requires i64 range".into()))?,
        Literal::Float64(f) => *f as i64,
        Literal::Date32(days) => *days as i64,
        _ => {
            return Err(Error::InvalidArgumentError(
                "bitwise shift requires numeric operands".into(),
            ));
        }
    };

    let rhs_i64 = match rhs {
        Literal::Int128(i) => (*i)
            .try_into()
            .map_err(|_| Error::InvalidArgumentError("bitwise shift requires i64 range".into()))?,
        Literal::Float64(f) => *f as i64,
        Literal::Date32(days) => *days as i64,
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

    Ok(Literal::Int128(result.into()))
}
