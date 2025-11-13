use llkv_expr::decimal::{DecimalError, DecimalValue};

fn pow10_i128(exp: u32) -> Result<i128, DecimalError> {
    let mut value: i128 = 1;
    for _ in 0..exp {
        value = value.checked_mul(10).ok_or(DecimalError::Overflow)?;
    }
    Ok(value)
}

pub fn truncate_decimal_to_i128(value: DecimalValue) -> Result<i128, DecimalError> {
    let scale = value.scale();
    if scale >= 0 {
        let divisor = pow10_i128(scale as u32)?;
        Ok(value.raw_value() / divisor)
    } else {
        let multiplier = pow10_i128((-scale) as u32)?;
        value
            .raw_value()
            .checked_mul(multiplier)
            .ok_or(DecimalError::Overflow)
    }
}

pub fn truncate_decimal_to_i64(value: DecimalValue) -> Result<i64, DecimalError> {
    let truncated = truncate_decimal_to_i128(value)?;
    if truncated < i128::from(i64::MIN) || truncated > i128::from(i64::MAX) {
        return Err(DecimalError::PrecisionOverflow {
            value: truncated,
            scale: 0,
        });
    }
    Ok(truncated as i64)
}

pub fn decimal_truthy(value: DecimalValue) -> bool {
    value.raw_value() != 0
}

pub fn align_decimal_to_scale(
    value: DecimalValue,
    precision: u8,
    scale: i8,
) -> Result<DecimalValue, DecimalError> {
    let rescaled = value.rescale(scale)?;
    if rescaled.precision() > precision {
        return Err(DecimalError::PrecisionOverflow {
            value: rescaled.raw_value(),
            scale: rescaled.scale(),
        });
    }
    Ok(rescaled)
}

pub fn decimal_from_i64(
    value: i64,
    precision: u8,
    scale: i8,
) -> Result<DecimalValue, DecimalError> {
    let base = DecimalValue::from_i64(value);
    align_decimal_to_scale(base, precision, scale)
}

pub fn decimal_from_f64(
    value: f64,
    precision: u8,
    scale: i8,
) -> Result<DecimalValue, DecimalError> {
    if !value.is_finite() {
        return Err(DecimalError::Overflow);
    }

    let scaled = value * 10f64.powi(i32::from(scale));
    if !scaled.is_finite() {
        return Err(DecimalError::Overflow);
    }

    let rounded = scaled.round();
    if !rounded.is_finite() {
        return Err(DecimalError::Overflow);
    }

    if rounded < i128::MIN as f64 || rounded > i128::MAX as f64 {
        return Err(DecimalError::Overflow);
    }

    let raw = rounded as i128;
    let decimal = DecimalValue::new(raw, scale)?;
    if decimal.precision() > precision {
        return Err(DecimalError::PrecisionOverflow {
            value: decimal.raw_value(),
            scale: decimal.scale(),
        });
    }

    Ok(decimal)
}
