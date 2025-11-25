// TODO: Move entirely to llkv-types?

//! Untyped literal values plus helpers for converting them into native types.
//!
//! Literals capture query parameters before a table knows the concrete Arrow
//! type of each column. Conversion helpers here defer type checking until the
//! caller can perform schema-aware coercion.

use arrow::datatypes::ArrowPrimitiveType;
use std::ops::Bound;

pub use llkv_types::interval::IntervalValue;
pub use llkv_types::literal::Literal;

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
                        Literal::Integer(i) => <$ty>::try_from(*i).map_err(|_| {
                            LiteralCastError::OutOfRange {
                                target: std::any::type_name::<$ty>(),
                                value: *i,
                            }
                        }),
                        Literal::Float(_) => Err(LiteralCastError::TypeMismatch {
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
                        Literal::Decimal(decimal) => {
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
            Literal::Float(f) => *f,
            Literal::Integer(i) => *i as f64,
            Literal::Decimal(d) => d.to_f64(),
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
        let cast = value as f32;
        if value.is_finite() && !cast.is_finite() {
            return Err(LiteralCastError::FloatOutOfRange {
                target: "f32",
                value,
            });
        }
        Ok(cast)
    }
}

impl FromLiteral for bool {
    fn from_literal(lit: &Literal) -> Result<Self, LiteralCastError> {
        match lit {
            Literal::Boolean(b) => Ok(*b),
            Literal::Integer(i) => match *i {
                0 => Ok(false),
                1 => Ok(true),
                value => Err(LiteralCastError::OutOfRange {
                    target: "bool",
                    value,
                }),
            },
            Literal::Float(_) => Err(LiteralCastError::TypeMismatch {
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
            Literal::Decimal(_) => Err(LiteralCastError::TypeMismatch {
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

impl FromLiteral for f64 {
    fn from_literal(lit: &Literal) -> Result<Self, LiteralCastError> {
        match lit {
            Literal::Float(f) => Ok(*f),
            Literal::Integer(i) => Ok(*i as f64),
            Literal::Decimal(d) => Ok(d.to_f64()),
            Literal::Boolean(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "boolean",
            }),
            Literal::String(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "string",
            }),
            Literal::Date32(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "date",
            }),
            Literal::Struct(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "struct",
            }),
            Literal::Null => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "null",
            }),
            Literal::Interval(_) => Err(LiteralCastError::TypeMismatch {
                expected: "float",
                got: "interval",
            }),
        }
    }
}

fn literal_type_name(lit: &Literal) -> &'static str {
    match lit {
        Literal::Integer(_) => "integer",
        Literal::Float(_) => "float",
        Literal::Decimal(_) => "decimal",
        Literal::String(_) => "string",
        Literal::Boolean(_) => "boolean",
        Literal::Date32(_) => "date",
        Literal::Null => "null",
        Literal::Struct(_) => "struct",
        Literal::Interval(_) => "interval",
    }
}

/// Convert a `Literal` into an owned `String`.
pub fn literal_to_string(lit: &Literal) -> Result<String, LiteralCastError> {
    match lit {
        Literal::String(s) => Ok(s.clone()),
        Literal::Null => Err(LiteralCastError::TypeMismatch {
            expected: "string",
            got: "null",
        }),
        Literal::Date32(_) => Err(LiteralCastError::TypeMismatch {
            expected: "string",
            got: "date",
        }),
        _ => Err(LiteralCastError::TypeMismatch {
            expected: "string",
            got: literal_type_name(lit),
        }),
    }
}

/// Convert a `Literal` into a concrete native type `T`.
pub fn literal_to_native<T>(lit: &Literal) -> Result<T, LiteralCastError>
where
    T: FromLiteral + Copy + 'static,
{
    T::from_literal(lit)
}

/// Convert a bound of `Literal` into a bound of `T::Native`.
///
/// Kept generic over `T: ArrowPrimitiveType` so callers (like the table
/// crate) can use `bound_to_native::<T>()` with `T::Native` inferred.
/// Error type is `LiteralCastError` since this crate is independent of
/// table-layer errors.
pub fn bound_to_native<T>(bound: &Bound<Literal>) -> Result<Bound<T::Native>, LiteralCastError>
where
    T: ArrowPrimitiveType,
    T::Native: FromLiteral + Copy,
{
    Ok(match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(l) => Bound::Included(literal_to_native::<T::Native>(l)?),
        Bound::Excluded(l) => Bound::Excluded(literal_to_native::<T::Native>(l)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_literal_roundtrip() {
        let lit = Literal::from(true);
        assert_eq!(lit, Literal::Boolean(true));
        assert!(literal_to_native::<bool>(&lit).unwrap());
        assert!(!literal_to_native::<bool>(&Literal::Boolean(false)).unwrap());
    }

    #[test]
    fn boolean_literal_rejects_integer_cast() {
        let lit = Literal::Boolean(true);
        let err = literal_to_native::<i32>(&lit).unwrap_err();
        assert!(matches!(
            err,
            LiteralCastError::TypeMismatch {
                expected: "integer",
                got: "boolean",
            }
        ));
    }
}
