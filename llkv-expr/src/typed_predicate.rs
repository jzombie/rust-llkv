use std::fmt;
use std::ops::Bound;

use arrow::datatypes::ArrowPrimitiveType;
use llkv_column_map::store::FilterPrimitive;

use crate::expr::Operator;
use crate::literal::{
    FromLiteral, Literal, LiteralCastError, bound_to_native, literal_to_native, literal_to_string,
};

#[derive(Debug, Clone)]
pub enum FixedWidthPredicate<T>
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

impl<T> FixedWidthPredicate<T>
where
    T: ArrowPrimitiveType + FilterPrimitive,
{
    pub fn matches(&self, value: T::Native) -> bool {
        match self {
            FixedWidthPredicate::All => true,
            FixedWidthPredicate::Equals(target) => value == *target,
            FixedWidthPredicate::GreaterThan(target) => value > *target,
            FixedWidthPredicate::GreaterThanOrEquals(target) => value >= *target,
            FixedWidthPredicate::LessThan(target) => value < *target,
            FixedWidthPredicate::LessThanOrEquals(target) => value <= *target,
            FixedWidthPredicate::Range { lower, upper } => {
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
            FixedWidthPredicate::In(values) => values.contains(&value),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VarWidthPredicate {
    All,
    Equals(String),
    GreaterThan(String),
    GreaterThanOrEquals(String),
    LessThan(String),
    LessThanOrEquals(String),
    Range {
        lower: Option<Bound<String>>,
        upper: Option<Bound<String>>,
    },
    In(Vec<String>),
    StartsWith(String),
    EndsWith(String),
    Contains(String),
}

impl VarWidthPredicate {
    pub fn matches(&self, value: &str) -> bool {
        match self {
            VarWidthPredicate::All => true,
            VarWidthPredicate::Equals(target) => value == target,
            VarWidthPredicate::GreaterThan(target) => value > target.as_str(),
            VarWidthPredicate::GreaterThanOrEquals(target) => value >= target.as_str(),
            VarWidthPredicate::LessThan(target) => value < target.as_str(),
            VarWidthPredicate::LessThanOrEquals(target) => value <= target.as_str(),
            VarWidthPredicate::Range { lower, upper } => {
                if let Some(bound) = lower {
                    match bound {
                        Bound::Included(min) => {
                            if value < min.as_str() {
                                return false;
                            }
                        }
                        Bound::Excluded(min) => {
                            if value <= min.as_str() {
                                return false;
                            }
                        }
                        Bound::Unbounded => {}
                    }
                }
                if let Some(bound) = upper {
                    match bound {
                        Bound::Included(max) => {
                            if value > max.as_str() {
                                return false;
                            }
                        }
                        Bound::Excluded(max) => {
                            if value >= max.as_str() {
                                return false;
                            }
                        }
                        Bound::Unbounded => {}
                    }
                }
                true
            }
            VarWidthPredicate::In(values) => values.iter().any(|v| v == value),
            VarWidthPredicate::StartsWith(prefix) => value.starts_with(prefix),
            VarWidthPredicate::EndsWith(suffix) => value.ends_with(suffix),
            VarWidthPredicate::Contains(substr) => value.contains(substr),
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

pub fn build_fixed_width_predicate<T>(
    op: &Operator<'_>,
) -> Result<FixedWidthPredicate<T>, PredicateBuildError>
where
    T: ArrowPrimitiveType + FilterPrimitive,
    T::Native: FromLiteral + Copy,
{
    match op {
        Operator::Equals(lit) => Ok(FixedWidthPredicate::Equals(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThan(lit) => Ok(FixedWidthPredicate::GreaterThan(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThanOrEquals(lit) => Ok(FixedWidthPredicate::GreaterThanOrEquals(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThan(lit) => Ok(FixedWidthPredicate::LessThan(
            literal_to_native::<T::Native>(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThanOrEquals(lit) => Ok(FixedWidthPredicate::LessThanOrEquals(
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
                Ok(FixedWidthPredicate::All)
            } else {
                Ok(FixedWidthPredicate::Range {
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
            Ok(FixedWidthPredicate::In(natives))
        }
        _ => Err(PredicateBuildError::UnsupportedOperator(
            "operator lacks typed literal support",
        )),
    }
}

fn parse_string_bound(
    bound: &Bound<Literal>,
) -> Result<Option<Bound<String>>, PredicateBuildError> {
    match bound {
        Bound::Unbounded => Ok(None),
        Bound::Included(lit) => literal_to_string(lit)
            .map(|s| Some(Bound::Included(s)))
            .map_err(PredicateBuildError::from),
        Bound::Excluded(lit) => literal_to_string(lit)
            .map(|s| Some(Bound::Excluded(s)))
            .map_err(PredicateBuildError::from),
    }
}

pub fn build_var_width_predicate(
    op: &Operator<'_>,
) -> Result<VarWidthPredicate, PredicateBuildError> {
    match op {
        Operator::Equals(lit) => Ok(VarWidthPredicate::Equals(
            literal_to_string(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThan(lit) => Ok(VarWidthPredicate::GreaterThan(
            literal_to_string(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::GreaterThanOrEquals(lit) => Ok(VarWidthPredicate::GreaterThanOrEquals(
            literal_to_string(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThan(lit) => Ok(VarWidthPredicate::LessThan(
            literal_to_string(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::LessThanOrEquals(lit) => Ok(VarWidthPredicate::LessThanOrEquals(
            literal_to_string(lit).map_err(PredicateBuildError::from)?,
        )),
        Operator::Range { lower, upper } => {
            let lb = parse_string_bound(lower)?;
            let ub = parse_string_bound(upper)?;
            if lb.is_none() && ub.is_none() {
                Ok(VarWidthPredicate::All)
            } else {
                Ok(VarWidthPredicate::Range {
                    lower: lb,
                    upper: ub,
                })
            }
        }
        Operator::In(values) => {
            let mut out = Vec::with_capacity(values.len());
            for lit in *values {
                out.push(literal_to_string(lit).map_err(PredicateBuildError::from)?);
            }
            Ok(VarWidthPredicate::In(out))
        }
        Operator::StartsWith(prefix) => Ok(VarWidthPredicate::StartsWith(prefix.to_string())),
        Operator::EndsWith(suffix) => Ok(VarWidthPredicate::EndsWith(suffix.to_string())),
        Operator::Contains(substr) => Ok(VarWidthPredicate::Contains(substr.to_string())),
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
        assert!(predicate.matches(42));
        assert!(!predicate.matches(7));
    }

    #[test]
    fn predicate_range_limits() {
        let op = Operator::Range {
            lower: Bound::Included(10.into()),
            upper: Bound::Excluded(20.into()),
        };
        let predicate = build_fixed_width_predicate::<arrow::datatypes::Int32Type>(&op).unwrap();
        assert!(predicate.matches(10));
        assert!(predicate.matches(19));
        assert!(!predicate.matches(9));
        assert!(!predicate.matches(20));
    }

    #[test]
    fn predicate_in_operator() {
        let values = [1.into(), 2.into(), 3.into()];
        let op = Operator::In(&values);
        let predicate = build_fixed_width_predicate::<arrow::datatypes::UInt8Type>(&op).unwrap();
        assert!(predicate.matches(2));
        assert!(!predicate.matches(5));
    }

    #[test]
    fn unsupported_operator_errors() {
        let op = Operator::StartsWith("foo");
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
        assert!(predicate.matches(123));
    }

    #[test]
    fn matches_all_for_empty_in_list() {
        let values: [Literal; 0] = [];
        let op = Operator::In(&values);
        let predicate = build_fixed_width_predicate::<arrow::datatypes::Float32Type>(&op).unwrap();
        assert!(!predicate.matches(1.23));
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

        let sw =
            build_var_width_predicate(&Operator::StartsWith("pre")).expect("starts with predicate");
        assert!(sw.matches("prefix"));
        assert!(!sw.matches("post"));

        let ew =
            build_var_width_predicate(&Operator::EndsWith("suf")).expect("ends with predicate");
        assert!(ew.matches("endsuf"));
        assert!(!ew.matches("suffixless"));

        let ct = build_var_width_predicate(&Operator::Contains("mid")).expect("contains predicate");
        assert!(ct.matches("amidst"));
        assert!(!ct.matches("edge"));
    }
}
