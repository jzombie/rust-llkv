//! Decimal utilities shared across LLKV crates.
//!
//! The runtime stores decimal values using Arrow's `Decimal128` semantics.
//! This module provides a lightweight helper type for manipulating those
//! values without pulling in heavier dependencies.

use std::fmt;
use std::str::FromStr;

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

impl FromStr for DecimalValue {
    type Err = DecimalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let (int_part, frac_part) = match s.split_once('.') {
            Some((i, f)) => (i, f),
            None => (s, ""),
        };

        let scale = frac_part.len();
        if scale > MAX_DECIMAL_PRECISION as usize {
            return Err(DecimalError::ScaleOutOfRange { scale: scale as i8 });
        }

        let combined = format!("{}{}", int_part, frac_part);
        let value = combined
            .parse::<i128>()
            .map_err(|_| DecimalError::Overflow)?;

        Self::new(value, scale as i8)
    }
}

impl PartialOrd for DecimalValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DecimalValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.scale == other.scale {
            return self.value.cmp(&other.value);
        }

        let max_scale = std::cmp::max(self.scale, other.scale);
        let scale_diff_self = (max_scale - self.scale) as u32;
        let scale_diff_other = (max_scale - other.scale) as u32;

        let l_i256 = i256::from_i128(self.value);
        let r_i256 = i256::from_i128(other.value);

        // Use wrapping_pow/mul because i256 handles the overflow of i128 range
        // and we are just comparing.
        let l_scaled = l_i256.wrapping_mul(POW10_BASE.wrapping_pow(scale_diff_self));
        let r_scaled = r_i256.wrapping_mul(POW10_BASE.wrapping_pow(scale_diff_other));

        l_scaled.cmp(&r_scaled)
    }
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
