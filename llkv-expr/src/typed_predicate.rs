use std::fmt;
use std::ops::Bound;

use arrow::datatypes::ArrowPrimitiveType;
use llkv_column_map::store::FilterPrimitive;

use crate::expr::Operator;
use crate::literal::{FromLiteral, LiteralCastError, bound_to_native, literal_to_native};

#[derive(Debug, Clone)]
pub enum Predicate<T>
where
    T: ArrowPrimitiveType + FilterPrimitive,
{
    All,
    Equals(T::Native),
    GreaterThan(T::Native),
    GreaterThanOrEquals(T::Native),
    LessThan(T::Native),
    LessThanOrEquals(T::Native),
    Range {
        lower: Option<Bound<T::Native>>,
        upper: Option<Bound<T::Native>>,
    },
    In(Vec<T::Native>),
}

impl<T> Predicate<T>
where
    T: ArrowPrimitiveType + FilterPrimitive,
{
    pub fn matches(&self, value: T::Native) -> bool {
        match self {
            Predicate::All => true,
            Predicate::Equals(target) => value == *target,
            Predicate::GreaterThan(target) => value > *target,
            Predicate::GreaterThanOrEquals(target) => value >= *target,
            Predicate::LessThan(target) => value < *target,
            Predicate::LessThanOrEquals(target) => value <= *target,
            Predicate::Range { lower, upper } => {
                if let Some(limit) = lower {
                    match limit {
                        Bound::Included(bound) => {
                            if value < *bound {
                                return false;
                            }
                        }
                        Bound::Excluded(bound) => {
                            if value <= *bound {
                                return false;
                            }
                        }
                        Bound::Unbounded => {}
                    }
                }
                if let Some(limit) = upper {
                    match limit {
                        Bound::Included(bound) => {
                            if value > *bound {
                                return false;
                            }
                        }
                        Bound::Excluded(bound) => {
                            if value >= *bound {
                                return false;
                            }
                        }
                        Bound::Unbounded => {}
                    }
                }
                true
            }
            Predicate::In(values) => values.contains(&value),
        }
    }
}

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

pub fn build_predicate<T>(op: &Operator<'_>) -> Result<Predicate<T>, PredicateBuildError>
where
    T: ArrowPrimitiveType + FilterPrimitive,
    T::Native: FromLiteral + Copy,
{
    match op {
        Operator::Equals(lit) => Ok(Predicate::Equals(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThan(lit) => Ok(Predicate::GreaterThan(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThanOrEquals(lit) => Ok(Predicate::GreaterThanOrEquals(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThan(lit) => Ok(Predicate::LessThan(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThanOrEquals(lit) => Ok(Predicate::LessThanOrEquals(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::Range { lower, upper } => {
            let lb = match bound_to_native::<T>(lower).map_err(PredicateBuildError::from)? {
                Bound::Unbounded => None,
                other => Some(other),
            };
            let ub = match bound_to_native::<T>(upper).map_err(PredicateBuildError::from)? {
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
                natives
                    .push(literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?);
            }
            Ok(Predicate::In(natives))
        }
        _ => Err(PredicateBuildError::UnsupportedOperator(
            "operator lacks typed literal support",
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
        let predicate = build_predicate::<arrow::datatypes::Int64Type>(&op).unwrap();
        assert!(predicate.matches(42));
        assert!(!predicate.matches(7));
    }

    #[test]
    fn predicate_range_limits() {
        let op = Operator::Range {
            lower: Bound::Included(10.into()),
            upper: Bound::Excluded(20.into()),
        };
        let predicate = build_predicate::<arrow::datatypes::Int32Type>(&op).unwrap();
        assert!(predicate.matches(10));
        assert!(predicate.matches(19));
        assert!(!predicate.matches(9));
        assert!(!predicate.matches(20));
    }

    #[test]
    fn predicate_in_operator() {
        let values = [1.into(), 2.into(), 3.into()];
        let op = Operator::In(&values);
        let predicate = build_predicate::<arrow::datatypes::UInt8Type>(&op).unwrap();
        assert!(predicate.matches(2));
        assert!(!predicate.matches(5));
    }

    #[test]
    fn unsupported_operator_errors() {
        let op = Operator::StartsWith("foo");
        let err = build_predicate::<arrow::datatypes::UInt32Type>(&op).unwrap_err();
        assert!(matches!(err, PredicateBuildError::UnsupportedOperator(_)));
    }

    #[test]
    fn literal_cast_error_propagates() {
        let op = Operator::Equals("foo".into());
        let err = build_predicate::<arrow::datatypes::UInt16Type>(&op).unwrap_err();
        assert!(matches!(err, PredicateBuildError::LiteralCast(_)));
    }

    #[test]
    fn empty_bounds_map_to_all() {
        let op = Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        };
        let predicate = build_predicate::<arrow::datatypes::UInt32Type>(&op).unwrap();
        assert!(predicate.matches(123));
    }

    #[test]
    fn matches_all_for_empty_in_list() {
        let values: [Literal; 0] = [];
        let op = Operator::In(&values);
        let predicate = build_predicate::<arrow::datatypes::Float32Type>(&op).unwrap();
        assert!(!predicate.matches(1.23));
    }
}
