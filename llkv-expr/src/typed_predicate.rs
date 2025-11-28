//! Build and evaluate fully typed predicates derived from logical expressions.
//!
//! The conversion utilities bridge the logical [`crate::expr::Operator`] values that
//! operate on untyped [`crate::literal::Literal`] instances and the concrete predicate
//! evaluators needed by execution code.

use std::cmp::Ordering;
use std::fmt;
use std::ops::Bound;

use arrow::datatypes::ArrowPrimitiveType;

use crate::expr::Operator;
use crate::literal::{FromLiteral, Literal, LiteralCastError, LiteralExt};

/// Value that can participate in typed predicate evaluation.
pub trait PredicateValue: Clone {
    type Borrowed<'a>: ?Sized
    where
        Self: 'a;

    fn borrowed(value: &Self) -> &Self::Borrowed<'_>;
    fn equals(value: &Self::Borrowed<'_>, target: &Self) -> bool;
    fn compare(value: &Self::Borrowed<'_>, target: &Self) -> Option<Ordering>;
    fn contains(value: &Self::Borrowed<'_>, target: &Self, case_sensitive: bool) -> bool {
        let _ = (value, target, case_sensitive);
        false
    }
    fn starts_with(value: &Self::Borrowed<'_>, target: &Self, case_sensitive: bool) -> bool {
        let _ = (value, target, case_sensitive);
        false
    }
    fn ends_with(value: &Self::Borrowed<'_>, target: &Self, case_sensitive: bool) -> bool {
        let _ = (value, target, case_sensitive);
        false
    }
}

/// Fully typed predicate ready to be matched against borrowed values.
#[derive(Debug, Clone)]
pub enum Predicate<V>
where
    V: PredicateValue,
{
    All,
    Equals(V),
    GreaterThan(V),
    GreaterThanOrEquals(V),
    LessThan(V),
    LessThanOrEquals(V),
    Range {
        lower: Option<Bound<V>>,
        upper: Option<Bound<V>>,
    },
    In(Vec<V>),
    StartsWith {
        pattern: V,
        case_sensitive: bool,
    },
    EndsWith {
        pattern: V,
        case_sensitive: bool,
    },
    Contains {
        pattern: V,
        case_sensitive: bool,
    },
}

impl<V> Predicate<V>
where
    V: PredicateValue,
{
    /// Return `true` when `value` satisfies the predicate variant.
    pub fn matches(&self, value: &V::Borrowed<'_>) -> bool {
        match self {
            Predicate::All => true,
            Predicate::Equals(target) => V::equals(value, target),
            Predicate::GreaterThan(target) => {
                matches!(V::compare(value, target), Some(Ordering::Greater))
            }
            Predicate::GreaterThanOrEquals(target) => {
                matches!(
                    V::compare(value, target),
                    Some(Ordering::Greater | Ordering::Equal)
                )
            }
            Predicate::LessThan(target) => {
                matches!(V::compare(value, target), Some(Ordering::Less))
            }
            Predicate::LessThanOrEquals(target) => matches!(
                V::compare(value, target),
                Some(Ordering::Less | Ordering::Equal)
            ),
            Predicate::Range { lower, upper } => {
                if let Some(bound) = lower
                    && !match bound {
                        Bound::Included(target) => matches!(
                            V::compare(value, target),
                            Some(Ordering::Greater | Ordering::Equal)
                        ),
                        Bound::Excluded(target) => {
                            matches!(V::compare(value, target), Some(Ordering::Greater))
                        }
                        Bound::Unbounded => true,
                    }
                {
                    return false;
                }

                if let Some(bound) = upper
                    && !match bound {
                        Bound::Included(target) => matches!(
                            V::compare(value, target),
                            Some(Ordering::Less | Ordering::Equal)
                        ),
                        Bound::Excluded(target) => {
                            matches!(V::compare(value, target), Some(Ordering::Less))
                        }
                        Bound::Unbounded => true,
                    }
                {
                    return false;
                }

                true
            }
            Predicate::In(values) => values.iter().any(|target| V::equals(value, target)),
            Predicate::StartsWith {
                pattern,
                case_sensitive,
            } => V::starts_with(value, pattern, *case_sensitive),
            Predicate::EndsWith {
                pattern,
                case_sensitive,
            } => V::ends_with(value, pattern, *case_sensitive),
            Predicate::Contains {
                pattern,
                case_sensitive,
            } => V::contains(value, pattern, *case_sensitive),
        }
    }
}

macro_rules! impl_predicate_value_for_primitive {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl PredicateValue for $ty {
                type Borrowed<'a> = Self where Self: 'a;

                fn borrowed(value: &Self) -> &Self::Borrowed<'_> {
                    value
                }

                fn equals(value: &Self::Borrowed<'_>, target: &Self) -> bool {
                    *value == *target
                }

                fn compare(value: &Self::Borrowed<'_>, target: &Self) -> Option<Ordering> {
                    value.partial_cmp(target)
                }
            }
        )+
    };
}

impl_predicate_value_for_primitive!(u64, u32, u16, u8, i64, i32, i16, i8, f64, f32, bool);

impl PredicateValue for String {
    type Borrowed<'a>
        = str
    where
        Self: 'a;

    fn borrowed(value: &Self) -> &Self::Borrowed<'_> {
        value.as_str()
    }

    fn equals(value: &Self::Borrowed<'_>, target: &Self) -> bool {
        value == target.as_str()
    }

    fn compare(value: &Self::Borrowed<'_>, target: &Self) -> Option<Ordering> {
        Some(value.cmp(target.as_str()))
    }

    fn contains(value: &Self::Borrowed<'_>, target: &Self, case_sensitive: bool) -> bool {
        if case_sensitive {
            value.contains(target.as_str())
        } else {
            value.to_lowercase().contains(&target.to_lowercase())
        }
    }

    fn starts_with(value: &Self::Borrowed<'_>, target: &Self, case_sensitive: bool) -> bool {
        if case_sensitive {
            value.starts_with(target.as_str())
        } else {
            value.to_lowercase().starts_with(&target.to_lowercase())
        }
    }

    fn ends_with(value: &Self::Borrowed<'_>, target: &Self, case_sensitive: bool) -> bool {
        if case_sensitive {
            value.ends_with(target.as_str())
        } else {
            value.to_lowercase().ends_with(&target.to_lowercase())
        }
    }
}

/// Error building a typed predicate from a logical operator.
#[derive(Debug, Clone)]
pub enum PredicateBuildError {
    LiteralCast(LiteralCastError),
    UnsupportedOperator(&'static str),
}

impl fmt::Display for PredicateBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredicateBuildError::LiteralCast(err) => write!(f, "literal cast error: {err}"),
            PredicateBuildError::UnsupportedOperator(op) => {
                write!(f, "unsupported operator for typed predicate: {op}")
            }
        }
    }
}

impl std::error::Error for PredicateBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PredicateBuildError::LiteralCast(err) => Some(err),
            PredicateBuildError::UnsupportedOperator(_) => None,
        }
    }
}

impl From<LiteralCastError> for PredicateBuildError {
    fn from(err: LiteralCastError) -> Self {
        PredicateBuildError::LiteralCast(err)
    }
}

/// Convert a logical operator into a predicate for fixed-width Arrow types.
///
/// # Errors
///
/// Returns [`PredicateBuildError::LiteralCast`] when the provided literal cannot be coerced into
/// the target native type or [`PredicateBuildError::UnsupportedOperator`] when the operator is not
/// supported for fixed-width values.
pub fn build_fixed_width_predicate<T>(
    op: &Operator<'_>,
) -> Result<Predicate<T::Native>, PredicateBuildError>
where
    T: ArrowPrimitiveType,
    T::Native: FromLiteral + Copy + PredicateValue,
{
    match op {
        Operator::Equals(lit) => Ok(Predicate::Equals(
            lit.to_native::<T::Native>()
                .map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThan(lit) => Ok(Predicate::GreaterThan(
            lit.to_native::<T::Native>()
                .map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThanOrEquals(lit) => Ok(Predicate::GreaterThanOrEquals(
            lit.to_native::<T::Native>()
                .map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThan(lit) => Ok(Predicate::LessThan(
            lit.to_native::<T::Native>()
                .map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThanOrEquals(lit) => Ok(Predicate::LessThanOrEquals(
            lit.to_native::<T::Native>()
                .map_err(PredicateBuildError::from)?,
        )),
        Operator::Range { lower, upper } => {
            let lb =
                match Literal::bound_to_native::<T>(lower).map_err(PredicateBuildError::from)? {
                    Bound::Unbounded => None,
                    other => Some(other),
                };
            let ub =
                match Literal::bound_to_native::<T>(upper).map_err(PredicateBuildError::from)? {
                    Bound::Unbounded => None,
                    other => Some(other),
                };

            if lb.is_none() && ub.is_none() {
                Ok(Predicate::All)
            } else {
                Ok(Predicate::Range {
                    lower: lb,
                    upper: ub,
                })
            }
        }
        Operator::In(values) => {
            let mut natives = Vec::with_capacity(values.len());
            for lit in *values {
                natives.push(
                    lit.to_native::<T::Native>()
                        .map_err(PredicateBuildError::from)?,
                );
            }
            Ok(Predicate::In(natives))
        }
        _ => Err(PredicateBuildError::UnsupportedOperator(
            "operator lacks typed literal support",
        )),
    }
}

fn parse_bool_bound(bound: &Bound<Literal>) -> Result<Option<Bound<bool>>, PredicateBuildError> {
    Ok(match bound {
        Bound::Unbounded => None,
        Bound::Included(lit) => Some(Bound::Included(
            lit.to_native::<bool>().map_err(PredicateBuildError::from)?,
        )),
        Bound::Excluded(lit) => Some(Bound::Excluded(
            lit.to_native::<bool>().map_err(PredicateBuildError::from)?,
        )),
    })
}

/// Convert a logical operator into a predicate over boolean values.
///
/// # Errors
///
/// Returns [`PredicateBuildError::LiteralCast`] when the literal cannot be interpreted as a
/// boolean or [`PredicateBuildError::UnsupportedOperator`] when string-specific predicates are
/// attempted.
pub fn build_bool_predicate(op: &Operator<'_>) -> Result<Predicate<bool>, PredicateBuildError> {
    match op {
        Operator::Equals(lit) => Ok(Predicate::Equals(
            lit.to_native::<bool>().map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThan(lit) => Ok(Predicate::GreaterThan(
            lit.to_native::<bool>().map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThanOrEquals(lit) => Ok(Predicate::GreaterThanOrEquals(
            lit.to_native::<bool>().map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThan(lit) => Ok(Predicate::LessThan(
            lit.to_native::<bool>().map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThanOrEquals(lit) => Ok(Predicate::LessThanOrEquals(
            lit.to_native::<bool>().map_err(PredicateBuildError::from)?,
        )),
        Operator::Range { lower, upper } => {
            let lb = parse_bool_bound(lower)?;
            let ub = parse_bool_bound(upper)?;
            if lb.is_none() && ub.is_none() {
                Ok(Predicate::All)
            } else {
                Ok(Predicate::Range {
                    lower: lb,
                    upper: ub,
                })
            }
        }
        Operator::In(values) => {
            let mut natives = Vec::with_capacity(values.len());
            for lit in *values {
                natives.push(lit.to_native::<bool>().map_err(PredicateBuildError::from)?);
            }
            Ok(Predicate::In(natives))
        }
        _ => Err(PredicateBuildError::UnsupportedOperator(
            "operator lacks boolean literal support",
        )),
    }
}

fn parse_string_bound(
    bound: &Bound<Literal>,
) -> Result<Option<Bound<String>>, PredicateBuildError> {
    match bound {
        Bound::Unbounded => Ok(None),
        Bound::Included(lit) => lit
            .to_string_owned()
            .map(|s| Some(Bound::Included(s)))
            .map_err(PredicateBuildError::from),
        Bound::Excluded(lit) => lit
            .to_string_owned()
            .map(|s| Some(Bound::Excluded(s)))
            .map_err(PredicateBuildError::from),
    }
}

/// Convert a logical operator into a predicate over UTF-8 string values.
///
/// # Errors
///
/// Returns [`PredicateBuildError::LiteralCast`] when literals cannot be converted into strings or
/// [`PredicateBuildError::UnsupportedOperator`] when the operator is not yet implemented for
/// strings.
pub fn build_var_width_predicate(
    op: &Operator<'_>,
) -> Result<Predicate<String>, PredicateBuildError> {
    match op {
        Operator::Equals(lit) => Ok(Predicate::Equals(
            lit.to_string_owned().map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThan(lit) => Ok(Predicate::GreaterThan(
            lit.to_string_owned().map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThanOrEquals(lit) => Ok(Predicate::GreaterThanOrEquals(
            lit.to_string_owned().map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThan(lit) => Ok(Predicate::LessThan(
            lit.to_string_owned().map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThanOrEquals(lit) => Ok(Predicate::LessThanOrEquals(
            lit.to_string_owned().map_err(PredicateBuildError::from)?,
        )),
        Operator::Range { lower, upper } => {
            let lb = parse_string_bound(lower)?;
            let ub = parse_string_bound(upper)?;
            if lb.is_none() && ub.is_none() {
                Ok(Predicate::All)
            } else {
                Ok(Predicate::Range {
                    lower: lb,
                    upper: ub,
                })
            }
        }
        Operator::In(values) => {
            let mut out = Vec::with_capacity(values.len());
            for lit in *values {
                out.push(lit.to_string_owned().map_err(PredicateBuildError::from)?);
            }
            Ok(Predicate::In(out))
        }
        Operator::StartsWith {
            pattern,
            case_sensitive,
        } => Ok(Predicate::StartsWith {
            pattern: pattern.to_string(),
            case_sensitive: *case_sensitive,
        }),
        Operator::EndsWith {
            pattern,
            case_sensitive,
        } => Ok(Predicate::EndsWith {
            pattern: pattern.to_string(),
            case_sensitive: *case_sensitive,
        }),
        Operator::Contains {
            pattern,
            case_sensitive,
        } => Ok(Predicate::Contains {
            pattern: pattern.to_string(),
            case_sensitive: *case_sensitive,
        }),
        Operator::IsNull | Operator::IsNotNull => Err(PredicateBuildError::UnsupportedOperator(
            "operator lacks string literal support",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Literal;
    use std::ops::Bound;

    #[test]
    fn predicate_matches_equals() {
        let op = Operator::Equals(42_i64.into());
        let predicate = build_fixed_width_predicate::<arrow::datatypes::Int64Type>(&op).unwrap();
        let forty_two: i64 = 42;
        let seven: i64 = 7;
        assert!(predicate.matches(&forty_two));
        assert!(!predicate.matches(&seven));
    }

    #[test]
    fn predicate_range_limits() {
        let op = Operator::Range {
            lower: Bound::Included(10.into()),
            upper: Bound::Excluded(20.into()),
        };
        let predicate = build_fixed_width_predicate::<arrow::datatypes::Int32Type>(&op).unwrap();
        assert!(predicate.matches(&10));
        assert!(predicate.matches(&19));
        assert!(!predicate.matches(&9));
        assert!(!predicate.matches(&20));
    }

    #[test]
    fn predicate_in_operator() {
        let values = [1.into(), 2.into(), 3.into()];
        let op = Operator::In(&values);
        let predicate = build_fixed_width_predicate::<arrow::datatypes::UInt8Type>(&op).unwrap();
        let two: u8 = 2;
        let five: u8 = 5;
        assert!(predicate.matches(&two));
        assert!(!predicate.matches(&five));
    }

    #[test]
    fn unsupported_operator_errors() {
        let op = Operator::starts_with("foo".to_string(), true);
        let err = build_fixed_width_predicate::<arrow::datatypes::UInt32Type>(&op).unwrap_err();
        assert!(matches!(err, PredicateBuildError::UnsupportedOperator(_)));
    }

    #[test]
    fn literal_cast_error_propagates() {
        let op = Operator::Equals("foo".into());
        let err = build_fixed_width_predicate::<arrow::datatypes::UInt16Type>(&op).unwrap_err();
        assert!(matches!(err, PredicateBuildError::LiteralCast(_)));
    }

    #[test]
    fn empty_bounds_map_to_all() {
        let op = Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        };
        let predicate = build_fixed_width_predicate::<arrow::datatypes::UInt32Type>(&op).unwrap();
        assert!(predicate.matches(&123u32));
    }

    #[test]
    fn matches_all_for_empty_in_list() {
        let values: [Literal; 0] = [];
        let op = Operator::In(&values);
        let predicate = build_fixed_width_predicate::<arrow::datatypes::Float32Type>(&op).unwrap();
        assert!(!predicate.matches(&1.23f32));
    }

    #[test]
    fn string_predicate_equals() {
        let op = Operator::Equals("foo".into());
        let predicate = build_var_width_predicate(&op).unwrap();
        assert!(predicate.matches("foo"));
        assert!(!predicate.matches("bar"));
    }

    #[test]
    fn string_predicate_range() {
        let op = Operator::Range {
            lower: Bound::Included("alpha".into()),
            upper: Bound::Excluded("omega".into()),
        };
        let predicate = build_var_width_predicate(&op).unwrap();
        assert!(predicate.matches("delta"));
        assert!(!predicate.matches("zzz"));
    }

    #[test]
    fn string_predicate_in_and_patterns() {
        let vals = ["x".into(), "y".into()];
        let op = Operator::In(&vals);
        let predicate = build_var_width_predicate(&op).unwrap();
        assert!(predicate.matches("x"));
        assert!(!predicate.matches("z"));

        let sw_sensitive =
            build_var_width_predicate(&Operator::starts_with("pre".to_string(), true))
                .expect("starts with predicate");
        assert!(sw_sensitive.matches("prefix"));
        assert!(!sw_sensitive.matches("Prefix"));

        let sw_insensitive =
            build_var_width_predicate(&Operator::starts_with("Pre".to_string(), false))
                .expect("starts with predicate");
        assert!(sw_insensitive.matches("prefix"));
        assert!(sw_insensitive.matches("Prefix"));

        let ew_sensitive = build_var_width_predicate(&Operator::ends_with("suf".to_string(), true))
            .expect("ends with predicate");
        assert!(ew_sensitive.matches("datsuf"));
        assert!(!ew_sensitive.matches("datSuf"));

        let ew_insensitive =
            build_var_width_predicate(&Operator::ends_with("SUF".to_string(), false))
                .expect("ends with predicate");
        assert!(ew_insensitive.matches("datsuf"));
        assert!(ew_insensitive.matches("datSuf"));

        let ct_sensitive = build_var_width_predicate(&Operator::contains("mid".to_string(), true))
            .expect("contains predicate");
        assert!(ct_sensitive.matches("amidst"));
        assert!(!ct_sensitive.matches("aMidst"));

        let ct_insensitive =
            build_var_width_predicate(&Operator::contains("MiD".to_string(), false))
                .expect("contains predicate");
        assert!(ct_insensitive.matches("amidst"));
        assert!(ct_insensitive.matches("aMidst"));
    }
}
