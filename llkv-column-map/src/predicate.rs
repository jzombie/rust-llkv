//! Typed predicate primitives used by the column map.
//!
//! These predicates originated in `llkv-expr` but now live alongside the
//! column-store filter kernels so the crate can evaluate predicates without
//! depending on the planner-facing expression crate.

use std::cmp::Ordering;
use std::ops::Bound;

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
