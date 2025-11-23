use arrow::datatypes::DECIMAL128_MAX_PRECISION;
use arrow_buffer::i256;
use llkv_expr::decimal::{DecimalError, DecimalValue, MAX_DECIMAL_PRECISION};

const POW10_BASE: i256 = i256::from_i128(10);

fn pow10(exp: u32) -> Result<i256, DecimalError> {
    let max = u32::from(MAX_DECIMAL_PRECISION) * 2;
    if exp > max {
        return Err(DecimalError::ScaleOutOfRange {
            scale: i8::try_from(exp).unwrap_or(i8::MAX),
        });
    }
    Ok(POW10_BASE.wrapping_pow(exp))
}

fn scale_within_bounds(scale: i16) -> bool {
    let max = MAX_DECIMAL_PRECISION as i16;
    (-max..=max).contains(&scale)
}

/// Convert the decimal into an `f64` (lossy for high precision inputs).
pub fn to_f64(value: DecimalValue) -> f64 {
    if value.raw_value() == 0 {
        return 0.0;
    }
    let denominator = 10_f64.powi(value.scale() as i32);
    (value.raw_value() as f64) / denominator
}

/// Rescale to a different exponent, preserving the numeric value when possible.
pub fn rescale(value: DecimalValue, target_scale: i8) -> Result<DecimalValue, DecimalError> {
    if !scale_within_bounds(target_scale as i16) {
        return Err(DecimalError::ScaleOutOfRange {
            scale: target_scale,
        });
    }
    if target_scale == value.scale() {
        return Ok(value);
    }

    if target_scale > value.scale() {
        let diff = (target_scale - value.scale()) as u32;
        let factor = pow10(diff)?;
        let scaled = i256::from_i128(value.raw_value())
            .checked_mul(factor)
            .ok_or(DecimalError::Overflow)?;
        let new_value = scaled.to_i128().ok_or(DecimalError::Overflow)?;
        return DecimalValue::new(new_value, target_scale);
    }

    // target_scale < value.scale() -> need exact division by 10^(value.scale() - target_scale)
    let diff = (value.scale() - target_scale) as u32;
    let factor = pow10(diff)?;
    let val_i256 = i256::from_i128(value.raw_value());
    let quotient = val_i256.checked_div(factor).ok_or(DecimalError::Overflow)?;
    let remainder = val_i256.checked_rem(factor).ok_or(DecimalError::Overflow)?;
    if remainder != i256::ZERO {
        return Err(DecimalError::InexactRescale {
            from: value.scale(),
            to: target_scale,
        });
    }
    let new_value = quotient.to_i128().ok_or(DecimalError::Overflow)?;
    DecimalValue::new(new_value, target_scale)
}

/// Rescale to a different exponent, rounding half-up if necessary.
pub fn rescale_with_rounding(value: DecimalValue, target_scale: i8) -> Result<DecimalValue, DecimalError> {
    if !scale_within_bounds(target_scale as i16) {
        return Err(DecimalError::ScaleOutOfRange {
            scale: target_scale,
        });
    }
    if target_scale == value.scale() {
        return Ok(value);
    }

    if target_scale > value.scale() {
        return rescale(value, target_scale);
    }

    // target_scale < value.scale() -> divide and round
    let diff = (value.scale() - target_scale) as u32;
    let factor = pow10(diff)?;
    let val_i256 = i256::from_i128(value.raw_value());

    let quotient = val_i256.checked_div(factor).ok_or(DecimalError::Overflow)?;
    let remainder = val_i256.checked_rem(factor).ok_or(DecimalError::Overflow)?;

    let mut new_value = quotient;

    if remainder != i256::ZERO {
        let abs_rem = if remainder < i256::ZERO {
            remainder.wrapping_neg()
        } else {
            remainder
        };
        let abs_factor = if factor < i256::ZERO {
            factor.wrapping_neg()
        } else {
            factor
        };

        // Round half up: if abs(remainder) * 2 >= abs(factor)
        let double_rem = abs_rem
            .checked_mul(i256::from_i128(2))
            .ok_or(DecimalError::Overflow)?;
        if double_rem >= abs_factor {
            if val_i256 > i256::ZERO {
                new_value = new_value
                    .checked_add(i256::ONE)
                    .ok_or(DecimalError::Overflow)?;
            } else {
                new_value = new_value
                    .checked_sub(i256::ONE)
                    .ok_or(DecimalError::Overflow)?;
            }
        }
    }

    let final_value = new_value.to_i128().ok_or(DecimalError::Overflow)?;
    DecimalValue::new(final_value, target_scale)
}

/// Add two decimals, aligning scales as needed.
pub fn add(lhs: DecimalValue, rhs: DecimalValue) -> Result<DecimalValue, DecimalError> {
    let target_scale = lhs.scale().max(rhs.scale());
    let l = rescale(lhs, target_scale)?;
    let r = rescale(rhs, target_scale)?;
    let sum = i256::from_i128(l.raw_value())
        .checked_add(i256::from_i128(r.raw_value()))
        .ok_or(DecimalError::Overflow)?;
    let value = sum.to_i128().ok_or(DecimalError::Overflow)?;
    DecimalValue::new(value, target_scale)
}

/// Subtract two decimals, aligning scales as needed.
pub fn sub(lhs: DecimalValue, rhs: DecimalValue) -> Result<DecimalValue, DecimalError> {
    let target_scale = lhs.scale().max(rhs.scale());
    let l = rescale(lhs, target_scale)?;
    let r = rescale(rhs, target_scale)?;
    let diff = i256::from_i128(l.raw_value())
        .checked_sub(i256::from_i128(r.raw_value()))
        .ok_or(DecimalError::Overflow)?;
    let value = diff.to_i128().ok_or(DecimalError::Overflow)?;
    DecimalValue::new(value, target_scale)
}

/// Multiply two decimals. The resulting scale is the sum of operand scales.
pub fn mul(lhs: DecimalValue, rhs: DecimalValue) -> Result<DecimalValue, DecimalError> {
    let sum = lhs.scale() as i16 + rhs.scale() as i16;
    if !scale_within_bounds(sum) {
        return Err(DecimalError::ScaleOutOfRange { scale: sum as i8 });
    }
    let scale = sum as i8;
    let product = i256::from_i128(lhs.raw_value())
        .checked_mul(i256::from_i128(rhs.raw_value()))
        .ok_or(DecimalError::Overflow)?;
    let value = product.to_i128().ok_or(DecimalError::Overflow)?;
    DecimalValue::new(value, scale)
}

/// Divide `lhs` by `rhs`, producing a value with the requested scale.
pub fn div(lhs: DecimalValue, rhs: DecimalValue, target_scale: i8) -> Result<DecimalValue, DecimalError> {
    if rhs.raw_value() == 0 {
        return Err(DecimalError::DivisionByZero);
    }
    if !scale_within_bounds(target_scale as i16) {
        return Err(DecimalError::ScaleOutOfRange {
            scale: target_scale,
        });
    }
    let numerator = i256::from_i128(lhs.raw_value());
    let denominator = i256::from_i128(rhs.raw_value());

    // Adjust numerator to reach the requested target scale.
    let scale_adjust = (target_scale as i32 + rhs.scale() as i32) - lhs.scale() as i32;
    let adjusted_num = if scale_adjust > 0 {
        let factor = pow10(scale_adjust as u32)?;
        numerator
            .checked_mul(factor)
            .ok_or(DecimalError::Overflow)?
    } else if scale_adjust < 0 {
        let factor = pow10((-scale_adjust) as u32)?;
        let quot = numerator
            .checked_div(factor)
            .ok_or(DecimalError::Overflow)?;
        let rem = numerator
            .checked_rem(factor)
            .ok_or(DecimalError::Overflow)?;
        if rem != i256::ZERO {
            return Err(DecimalError::InexactRescale {
                from: lhs.scale(),
                to: lhs.scale() + scale_adjust as i8,
            });
        }
        quot
    } else {
        numerator
    };

    let quotient = adjusted_num
        .checked_div(denominator)
        .ok_or(DecimalError::Overflow)?;
    let remainder = adjusted_num
        .checked_rem(denominator)
        .ok_or(DecimalError::Overflow)?;
    // Round half away from zero when remainder >= |denominator| / 2.
    let rounded = if remainder == i256::ZERO {
        quotient
    } else {
        let half = denominator.wrapping_div(i256::from_i128(2));
        let abs_rem = remainder.wrapping_abs();
        let abs_half = half.wrapping_abs();
        if abs_rem >= abs_half {
            if (quotient >= i256::ZERO) == (denominator >= i256::ZERO) {
                quotient.wrapping_add(i256::ONE)
            } else {
                quotient.wrapping_sub(i256::ONE)
            }
        } else {
            quotient
        }
    };

    let value = rounded.to_i128().ok_or(DecimalError::Overflow)?;
    DecimalValue::new(value, target_scale)
}

/// Compare two decimals by aligning their scales.
pub fn compare(lhs: DecimalValue, rhs: DecimalValue) -> Result<std::cmp::Ordering, DecimalError> {
    let target_scale = lhs.scale().max(rhs.scale());
    let l = rescale(lhs, target_scale)?;
    let r = rescale(rhs, target_scale)?;
    Ok(l.raw_value().cmp(&r.raw_value()))
}
