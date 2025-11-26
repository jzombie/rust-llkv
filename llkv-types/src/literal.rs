//! Untyped literal values plus helpers for converting them into native types.
//!
//! Literals capture query parameters before a table knows the concrete Arrow
//! type of each column. Conversion helpers here defer type checking until the
//! caller can perform schema-aware coercion.

use std::ops::Bound;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int16Array,
    Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray, StructArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType};

use llkv_result::Error;
use time::{Date, Month};

use crate::decimal::DecimalValue;
use crate::interval::IntervalValue;

/// A literal value that has not yet been coerced into a specific native
/// type. This allows for type inference to be deferred until the column
/// type is known.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Int128(i128),
    Float64(f64),
    /// Decimal literal stored as scaled integer with fixed precision.
    Decimal128(DecimalValue),
    String(String),
    Boolean(bool),
    /// Date literal stored as days since the Unix epoch (1970-01-01).
    Date32(i32),
    /// Struct literal with field names and nested literals
    Struct(Vec<(String, Box<Literal>)>),
    /// Interval literal with mixed calendar and sub-day precision.
    Interval(IntervalValue),
    // Other types like Bytes, etc. can be added here.
}

macro_rules! impl_from_for_literal {
    ($variant:ident, $($t:ty),*) => {
        $(
            impl From<$t> for Literal {
                fn from(v: $t) -> Self {
                    Literal::$variant(v.into())
                }
            }
        )*
    };
}

impl_from_for_literal!(Int128, i8, i16, i32, i64, i128, u8, u16, u32, u64);
impl_from_for_literal!(Float64, f32, f64);
impl_from_for_literal!(String, String);
impl_from_for_literal!(Boolean, bool);
impl_from_for_literal!(Decimal128, DecimalValue);
impl_from_for_literal!(Interval, IntervalValue);

impl From<&str> for Literal {
    fn from(v: &str) -> Self {
        Literal::String(v.to_string())
    }
}

impl From<Vec<(String, Literal)>> for Literal {
    fn from(fields: Vec<(String, Literal)>) -> Self {
        let boxed_fields = fields
            .into_iter()
            .map(|(name, lit)| (name, Box::new(lit)))
            .collect();
        Literal::Struct(boxed_fields)
    }
}

impl Literal {
    /// Human-friendly rendering used in plan/debug output.
    pub fn format_display(&self) -> String {
        match self {
            Literal::Int128(i) => i.to_string(),
            Literal::Float64(f) => f.to_string(),
            Literal::Decimal128(d) => d.to_string(),
            Literal::Boolean(b) => b.to_string(),
            Literal::String(s) => format!("\"{}\"", escape_string(s)),
            Literal::Date32(days) => format!("DATE '{}'", format_date32(*days)),
            Literal::Interval(interval) => format!(
                "INTERVAL {{ months: {}, days: {}, nanos: {} }}",
                interval.months, interval.days, interval.nanos
            ),
            Literal::Null => "NULL".to_string(),
            Literal::Struct(fields) => {
                let field_strs: Vec<_> = fields
                    .iter()
                    .map(|(name, lit)| format!("{}: {}", name, lit.format_display()))
                    .collect();
                format!("{{{}}}", field_strs.join(", "))
            }
        }
    }
}

fn format_date32(days: i32) -> String {
    let julian = match epoch_julian_day().checked_add(days) {
        Some(value) => value,
        None => return days.to_string(),
    };

    match Date::from_julian_day(julian) {
        Ok(date) => {
            let (year, month, day) = date.to_calendar_date();
            let month_number = month as u8;
            format!("{:04}-{:02}-{:02}", year, month_number, day)
        }
        Err(_) => days.to_string(),
    }
}

fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
}

fn escape_string(value: &str) -> String {
    value.chars().flat_map(|c| c.escape_default()).collect()
}

/// Error converting a `Literal` into a concrete native type.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralCastError {
    /// Tried to coerce a non-integer literal into an integer native type.
    TypeMismatch {
        expected: &'static str,
        got: &'static str,
    },
    /// Integer value does not fit in the destination type.
    OutOfRange { target: &'static str, value: i128 },
    /// Float value does not fit in the destination type.
    FloatOutOfRange { target: &'static str, value: f64 },
}

impl std::fmt::Display for LiteralCastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiteralCastError::TypeMismatch { expected, got } => {
                write!(f, "expected {}, got {}", expected, got)
            }
            LiteralCastError::OutOfRange { target, value } => {
                write!(f, "value {} out of range for {}", value, target)
            }
            LiteralCastError::FloatOutOfRange { target, value } => {
                write!(f, "value {} out of range for {}", value, target)
            }
        }
    }
}

impl std::error::Error for LiteralCastError {}

/// Extension methods for working with `Literal`.
pub trait LiteralExt {
    fn type_name(&self) -> &'static str;
    fn to_string_owned(&self) -> Result<String, LiteralCastError>;
    fn to_native<T>(&self) -> Result<T, LiteralCastError>
    where
        T: FromLiteral + Copy + 'static;
    fn from_array_ref(array: &ArrayRef, index: usize) -> llkv_result::Result<Literal>;
    fn bound_to_native<T>(bound: &Bound<Literal>) -> Result<Bound<T::Native>, LiteralCastError>
    where
        T: ArrowPrimitiveType,
        T::Native: FromLiteral + Copy;
}

impl LiteralExt for Literal {
    fn type_name(&self) -> &'static str {
        match self {
            Literal::Int128(_) => "integer",
            Literal::Float64(_) => "float",
            Literal::Decimal128(_) => "decimal",
            Literal::String(_) => "string",
            Literal::Boolean(_) => "boolean",
            Literal::Date32(_) => "date",
            Literal::Null => "null",
            Literal::Struct(_) => "struct",
            Literal::Interval(_) => "interval",
        }
    }

    fn to_string_owned(&self) -> Result<String, LiteralCastError> {
        match self {
            Literal::String(s) => Ok(s.clone()),
            Literal::Null => Err(LiteralCastError::TypeMismatch {
                expected: "string",
                got: "null",
            }),
            Literal::Date32(_) => Err(LiteralCastError::TypeMismatch {
                expected: "string",
                got: "date",
            }),
            other => Err(LiteralCastError::TypeMismatch {
                expected: "string",
                got: other.type_name(),
            }),
        }
    }

    fn to_native<T>(&self) -> Result<T, LiteralCastError>
    where
        T: FromLiteral + Copy + 'static,
    {
        T::from_literal(self)
    }

    fn from_array_ref(array: &ArrayRef, index: usize) -> llkv_result::Result<Literal> {
        if array.is_null(index) {
            return Ok(Literal::Null);
        }

        match array.data_type() {
            DataType::Int8 => {
                let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::Int16 => {
                let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::Int32 => {
                let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::UInt8 => {
                let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::UInt16 => {
                let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::UInt32 => {
                let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::UInt64 => {
                let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                Ok(Literal::Int128(arr.value(index) as i128))
            }
            DataType::Float32 => {
                let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
                Ok(Literal::Float64(arr.value(index) as f64))
            }
            DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                Ok(Literal::Float64(arr.value(index)))
            }
            DataType::Utf8 => {
                let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
                Ok(Literal::String(arr.value(index).to_string()))
            }
            DataType::LargeUtf8 => {
                let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
                Ok(Literal::String(arr.value(index).to_string()))
            }
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                Ok(Literal::Boolean(arr.value(index)))
            }
            DataType::Date32 => {
                let arr = array.as_any().downcast_ref::<Date32Array>().unwrap();
                Ok(Literal::Date32(arr.value(index)))
            }
            DataType::Decimal128(_, scale) => {
                let arr = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
                let val = arr.value(index);
                let decimal = DecimalValue::new(val, *scale).map_err(|err| {
                    Error::InvalidArgumentError(format!(
                        "invalid decimal value for literal conversion: {err}"
                    ))
                })?;
                Ok(Literal::Decimal128(decimal))
            }
            DataType::Struct(fields) => {
                let struct_array =
                    array
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .ok_or_else(|| {
                            Error::InvalidArgumentError("failed to downcast struct array".into())
                        })?;
                let mut members = Vec::with_capacity(fields.len());
                for (idx, field) in fields.iter().enumerate() {
                    let child = struct_array.column(idx);
                    let literal = Literal::from_array_ref(child, index)?;
                    members.push((field.name().clone(), Box::new(literal)));
                }
                Ok(Literal::Struct(members))
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported type for literal conversion: {other:?}"
            ))),
        }
    }

    fn bound_to_native<T>(bound: &Bound<Literal>) -> Result<Bound<T::Native>, LiteralCastError>
    where
        T: ArrowPrimitiveType,
        T::Native: FromLiteral + Copy,
    {
        Ok(match bound {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(l) => Bound::Included(T::Native::from_literal(l)?),
            Bound::Excluded(l) => Bound::Excluded(T::Native::from_literal(l)?),
        })
    }
}

/// Helper trait implemented for primitive types that can be produced from a `Literal`.
pub trait FromLiteral: Sized {
    fn from_literal(lit: &Literal) -> Result<Self, LiteralCastError>;
}

macro_rules! impl_from_literal_int {
    ($($ty:ty),* $(,)?) => {
        $(
            impl FromLiteral for $ty {
                fn from_literal(lit: &Literal) -> Result<Self, LiteralCastError> {
                    match lit {
                        Literal::Int128(i) => <$ty>::try_from(*i).map_err(|_| {
                            LiteralCastError::OutOfRange {
                                target: std::any::type_name::<$ty>(),
                                value: *i,
                            }
                        }),
                        Literal::Float64(_) => Err(LiteralCastError::TypeMismatch {
                            expected: "integer",
                            got: "float",
                        }),
                        Literal::Boolean(_) => Err(LiteralCastError::TypeMismatch {
                            expected: "integer",
                            got: "boolean",
                        }),
                        Literal::String(_) => Err(LiteralCastError::TypeMismatch {
                            expected: "integer",
                            got: "string",
                        }),
                        Literal::Date32(_) => Err(LiteralCastError::TypeMismatch {
                            expected: "integer",
                            got: "date",
                        }),
                        Literal::Struct(_) => Err(LiteralCastError::TypeMismatch {
                            expected: "integer",
                            got: "struct",
                        }),
                        Literal::Interval(_) => Err(LiteralCastError::TypeMismatch {
                            expected: "integer",
                            got: "interval",
                        }),
                        Literal::Decimal128(decimal) => {
                            if decimal.scale() == 0 {
                                let raw = decimal.raw_value();
                                <$ty>::try_from(raw).map_err(|_| LiteralCastError::OutOfRange {
                                    target: std::any::type_name::<$ty>(),
                                    value: raw,
                                })
                            } else {
                                Err(LiteralCastError::TypeMismatch {
                                    expected: "integer",
                                    got: "decimal",
                                })
                            }
                        }
                        Literal::Null => Err(LiteralCastError::TypeMismatch {
                            expected: "integer",
                            got: "null",
                        }),
                    }
                }
            }
        )*
    };
}

impl_from_literal_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, usize);

impl FromLiteral for f32 {
    fn from_literal(lit: &Literal) -> Result<Self, LiteralCastError> {
        let value = match lit {
            Literal::Float64(f) => *f,
            Literal::Int128(i) => *i as f64,
            Literal::Decimal128(d) => d.to_f64(),
            Literal::Boolean(_) => {
                return Err(LiteralCastError::TypeMismatch {
                    expected: "float",
                    got: "boolean",
                });
            }
            Literal::String(_) => {
                return Err(LiteralCastError::TypeMismatch {
                    expected: "float",
                    got: "string",
                });
            }
            Literal::Struct(_) => {
                return Err(LiteralCastError::TypeMismatch {
                    expected: "float",
                    got: "struct",
                });
            }
            Literal::Interval(_) => {
                return Err(LiteralCastError::TypeMismatch {
                    expected: "float",
                    got: "interval",
                });
            }
            Literal::Null => {
                return Err(LiteralCastError::TypeMismatch {
                    expected: "float",
                    got: "null",
                });
            }
            Literal::Date32(_) => {
                return Err(LiteralCastError::TypeMismatch {
                    expected: "float",
                    got: "date",
                });
            }
        };

        let casted = value as f32;
        if casted.is_finite() {
            Ok(casted)
        } else {
            Err(LiteralCastError::FloatOutOfRange {
                target: "f32",
                value,
            })
        }
    }
}

impl FromLiteral for f64 {
    fn from_literal(lit: &Literal) -> Result<Self, LiteralCastError> {
        match lit {
            Literal::Float64(f) => Ok(*f),
            Literal::Int128(i) => Ok(*i as f64),
            Literal::Decimal128(d) => Ok(d.to_f64()),
            Literal::Boolean(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "boolean",
            }),
            Literal::String(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "string",
            }),
            Literal::Struct(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "struct",
            }),
            Literal::Interval(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "interval",
            }),
            Literal::Null => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "null",
            }),
            Literal::Date32(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "date",
            }),
        }
    }
}

impl FromLiteral for bool {
    fn from_literal(lit: &Literal) -> Result<Self, LiteralCastError> {
        match lit {
            Literal::Boolean(b) => Ok(*b),
            Literal::Int128(i) => match *i {
                0 => Ok(false),
                1 => Ok(true),
                value => Err(LiteralCastError::OutOfRange {
                    target: "bool",
                    value,
                }),
            },
            Literal::Float64(_) => Err(LiteralCastError::TypeMismatch {
                expected: "bool",
                got: "float",
            }),
            Literal::String(s) => {
                let normalized = s.trim().to_ascii_lowercase();
                match normalized.as_str() {
                    "true" | "t" | "1" => Ok(true),
                    "false" | "f" | "0" => Ok(false),
                    _ => Err(LiteralCastError::TypeMismatch {
                        expected: "bool",
                        got: "string",
                    }),
                }
            }
            Literal::Date32(_) => Err(LiteralCastError::TypeMismatch {
                expected: "bool",
                got: "date",
            }),
            Literal::Struct(_) => Err(LiteralCastError::TypeMismatch {
                expected: "bool",
                got: "struct",
            }),
            Literal::Interval(_) => Err(LiteralCastError::TypeMismatch {
                expected: "bool",
                got: "interval",
            }),
            Literal::Decimal128(_) => Err(LiteralCastError::TypeMismatch {
                expected: "bool",
                got: "decimal",
            }),
            Literal::Null => Err(LiteralCastError::TypeMismatch {
                expected: "bool",
                got: "null",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_literal_roundtrip() {
        let lit = Literal::from(true);
        assert_eq!(lit, Literal::Boolean(true));
        assert!(lit.to_native::<bool>().unwrap());
        assert!(!Literal::Boolean(false).to_native::<bool>().unwrap());
    }

    #[test]
    fn boolean_literal_rejects_integer_cast() {
        let lit = Literal::Boolean(true);
        let err = lit.to_native::<i32>().unwrap_err();
        assert!(matches!(
            err,
            LiteralCastError::TypeMismatch {
                expected: "integer",
                got: "boolean",
            }
        ));
    }
}
