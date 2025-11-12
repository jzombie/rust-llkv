//! Decimal utilities shared across LLKV crates.
//!
//! The runtime stores decimal values using Arrow's `Decimal128` semantics.
//! This module provides a lightweight helper type for manipulating those
//! values without pulling in heavier dependencies.

use std::fmt;

use arrow::datatypes::DECIMAL128_MAX_PRECISION;
use arrow_buffer::i256;

/// Maximum precision supported by `DecimalValue` (aligns with Arrow's Decimal128).
pub const MAX_DECIMAL_PRECISION: u8 = DECIMAL128_MAX_PRECISION;
const POW10_BASE: i256 = i256::from_i128(10);

/// Errors that can occur while manipulating decimal values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecimalError {
    /// Requested scale falls outside the supported range.
    ScaleOutOfRange { scale: i8 },
    /// Result exceeded the maximum representable precision.
    PrecisionOverflow { value: i128, scale: i8 },
    /// Arithmetic operation overflowed the Decimal128 range.
    Overflow,
    /// Attempted to divide by zero.
    DivisionByZero,
    /// Rescale operation attempted to lower scale without exact divisibility.
    InexactRescale { from: i8, to: i8 },
}

impl fmt::Display for DecimalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecimalError::ScaleOutOfRange { scale } => {
                write!(f, "decimal scale {scale} outside supported range")
            }
            DecimalError::PrecisionOverflow { value, scale } => {
                write!(
                    f,
                    "decimal value {value} with scale {scale} exceeds maximum precision"
                )
            }
            DecimalError::Overflow => write!(f, "decimal arithmetic overflow"),
            DecimalError::DivisionByZero => write!(f, "decimal division by zero"),
            DecimalError::InexactRescale { from, to } => {
                write!(
                    f,
                    "cannot rescale decimal from scale {from} to {to} without losing precision"
                )
            }
        }
    }
}

impl std::error::Error for DecimalError {}

/// Runtime representation of a Decimal128 value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecimalValue {
    value: i128,
    scale: i8,
}

impl DecimalValue {
    /// Create a decimal from its raw parts, validating precision bounds.
    pub fn new(value: i128, scale: i8) -> Result<Self, DecimalError> {
        if !scale_within_bounds(scale as i16) {
            return Err(DecimalError::ScaleOutOfRange { scale });
        }
        let precision = digit_count_i256(i256::from_i128(value));
        if precision > MAX_DECIMAL_PRECISION {
            return Err(DecimalError::PrecisionOverflow { value, scale });
        }
        Ok(Self { value, scale })
    }

    /// Construct a decimal from integer value with zero scale.
    pub fn from_i64(value: i64) -> Self {
        Self::new(value as i128, 0).expect("i64 fits within Decimal128 limits")
    }

    /// Return the scaled integer backing this decimal.
    #[inline]
    pub fn raw_value(self) -> i128 {
        self.value
    }

    /// Return the scale (number of fractional digits).
    #[inline]
    pub fn scale(self) -> i8 {
        self.scale
    }

    /// Return the decimal precision (total digit count).
    #[inline]
    pub fn precision(self) -> u8 {
        digit_count_i256(i256::from_i128(self.value))
    }

    /// Convert the decimal into an `f64` (lossy for high precision inputs).
    pub fn to_f64(self) -> f64 {
        if self.value == 0 {
            return 0.0;
        }
        let denominator = 10_f64.powi(self.scale as i32);
        (self.value as f64) / denominator
    }

    /// Rescale to a different exponent, preserving the numeric value when possible.
    pub fn rescale(self, target_scale: i8) -> Result<Self, DecimalError> {
        if !scale_within_bounds(target_scale as i16) {
            return Err(DecimalError::ScaleOutOfRange {
                scale: target_scale,
            });
        }
        if target_scale == self.scale {
            return Ok(self);
        }

        if target_scale > self.scale {
            let diff = (target_scale - self.scale) as u32;
            let factor = pow10(diff)?;
            let scaled = i256::from_i128(self.value)
                .checked_mul(factor)
                .ok_or(DecimalError::Overflow)?;
            let new_value = scaled.to_i128().ok_or(DecimalError::Overflow)?;
            return Self::new(new_value, target_scale);
        }

        // target_scale < self.scale -> need exact division by 10^(self.scale - target_scale)
        let diff = (self.scale - target_scale) as u32;
        let factor = pow10(diff)?;
        let value = i256::from_i128(self.value);
        let quotient = value.checked_div(factor).ok_or(DecimalError::Overflow)?;
        let remainder = value.checked_rem(factor).ok_or(DecimalError::Overflow)?;
        if remainder != i256::ZERO {
            return Err(DecimalError::InexactRescale {
                from: self.scale,
                to: target_scale,
            });
        }
        let new_value = quotient.to_i128().ok_or(DecimalError::Overflow)?;
        Self::new(new_value, target_scale)
    }

    /// Add two decimals, aligning scales as needed.
    pub fn checked_add(self, other: Self) -> Result<Self, DecimalError> {
        let target_scale = self.scale.max(other.scale);
        let lhs = self.rescale(target_scale)?;
        let rhs = other.rescale(target_scale)?;
        let sum = i256::from_i128(lhs.value)
            .checked_add(i256::from_i128(rhs.value))
            .ok_or(DecimalError::Overflow)?;
        let value = sum.to_i128().ok_or(DecimalError::Overflow)?;
        Self::new(value, target_scale)
    }

    /// Subtract two decimals, aligning scales as needed.
    pub fn checked_sub(self, other: Self) -> Result<Self, DecimalError> {
        let target_scale = self.scale.max(other.scale);
        let lhs = self.rescale(target_scale)?;
        let rhs = other.rescale(target_scale)?;
        let diff = i256::from_i128(lhs.value)
            .checked_sub(i256::from_i128(rhs.value))
            .ok_or(DecimalError::Overflow)?;
        let value = diff.to_i128().ok_or(DecimalError::Overflow)?;
        Self::new(value, target_scale)
    }

    /// Multiply two decimals. The resulting scale is the sum of operand scales.
    pub fn checked_mul(self, other: Self) -> Result<Self, DecimalError> {
        let sum = self.scale as i16 + other.scale as i16;
        if !scale_within_bounds(sum) {
            return Err(DecimalError::ScaleOutOfRange { scale: sum as i8 });
        }
        let scale = sum as i8;
        let product = i256::from_i128(self.value)
            .checked_mul(i256::from_i128(other.value))
            .ok_or(DecimalError::Overflow)?;
        let value = product.to_i128().ok_or(DecimalError::Overflow)?;
        Self::new(value, scale)
    }

    /// Divide `self` by `other`, producing a value with the requested scale.
    pub fn checked_div(self, other: Self, target_scale: i8) -> Result<Self, DecimalError> {
        if other.value == 0 {
            return Err(DecimalError::DivisionByZero);
        }
        if !scale_within_bounds(target_scale as i16) {
            return Err(DecimalError::ScaleOutOfRange {
                scale: target_scale,
            });
        }
        let numerator = i256::from_i128(self.value);
        let denominator = i256::from_i128(other.value);

        // Adjust numerator to reach the requested target scale.
        let scale_adjust = (target_scale as i32 + other.scale as i32) - self.scale as i32;
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
                    from: self.scale,
                    to: self.scale + scale_adjust as i8,
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
        Self::new(value, target_scale)
    }

    /// Compare two decimals by aligning their scales.
    ///
    /// Note: This method is fallible (returns Result) unlike std::cmp::Ord::cmp,
    /// so we use a distinct name to avoid confusion with the trait method.
    pub fn compare(self, other: Self) -> Result<std::cmp::Ordering, DecimalError> {
        let target_scale = self.scale.max(other.scale);
        let lhs = self.rescale(target_scale)?;
        let rhs = other.rescale(target_scale)?;
        Ok(lhs.value.cmp(&rhs.value))
    }
}

impl fmt::Display for DecimalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scale == 0 {
            return write!(f, "{}", self.value);
        }
        let negative = self.value < 0;
        let digits = digit_buffer(i256::from_i128(self.value));
        if digits.len() <= self.scale as usize {
            let mut result = String::with_capacity(self.scale as usize + 2);
            if negative {
                result.push('-');
            }
            result.push('0');
            result.push('.');
            for _ in digits.len()..self.scale as usize {
                result.push('0');
            }
            result.push_str(&digits);
            return f.write_str(&result);
        }
        let split = digits.len() - self.scale as usize;
        if negative {
            f.write_str("-")?;
        }
        f.write_str(&digits[..split])?;
        f.write_str(".")?;
        f.write_str(&digits[split..])
    }
}

fn pow10(exp: u32) -> Result<i256, DecimalError> {
    let max = u32::from(MAX_DECIMAL_PRECISION) * 2;
    if exp > max {
        return Err(DecimalError::ScaleOutOfRange {
            scale: i8::try_from(exp).unwrap_or(i8::MAX),
        });
    }
    Ok(POW10_BASE.wrapping_pow(exp))
}

fn digit_count_i256(mut value: i256) -> u8 {
    if value == i256::ZERO {
        return 1;
    }
    if value < i256::ZERO {
        value = value.wrapping_neg();
    }
    let mut count: u8 = 0;
    while value != i256::ZERO {
        value = value.wrapping_div(POW10_BASE);
        count += 1;
    }
    count
}

fn digit_buffer(mut value: i256) -> String {
    if value == i256::ZERO {
        return "0".to_owned();
    }
    if value < i256::ZERO {
        value = value.wrapping_neg();
    }
    let mut buf = Vec::new();
    let ten = POW10_BASE;
    let mut current = value;
    while current != i256::ZERO {
        let rem = current.wrapping_rem(ten);
        let digit = rem
            .to_i128()
            .expect("remainder from decimal division fits in i128") as i32;
        buf.push((b'0' + digit as u8) as char);
        current = current.wrapping_div(ten);
    }
    buf.iter().rev().collect()
}

fn scale_within_bounds(scale: i16) -> bool {
    let max = MAX_DECIMAL_PRECISION as i16;
    (-max..=max).contains(&scale)
}
