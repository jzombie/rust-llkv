//! Type-aware, Arrow-native predicate AST.
#![forbid(unsafe_code)]

use arrow_array::{Array, Scalar};
use std::ops::Bound;
use std::sync::Arc;

/// Logical expression over predicates.
#[derive(Clone, Debug)]
pub enum Expr<'a, F> {
    And(Vec<Expr<'a, F>>),
    Or(Vec<Expr<'a, F>>),
    Not(Box<Expr<'a, F>>),
    Pred(Filter<'a, F>),
}

impl<'a, F> Expr<'a, F> {
    /// Build an AND of filters.
    #[inline]
    pub fn all_of(fs: Vec<Filter<'a, F>>) -> Expr<'a, F> {
        Expr::And(fs.into_iter().map(Expr::Pred).collect())
    }

    /// Build an OR of filters.
    #[inline]
    pub fn any_of(fs: Vec<Filter<'a, F>>) -> Expr<'a, F> {
        Expr::Or(fs.into_iter().map(Expr::Pred).collect())
    }

    /// Wrap an expression in a logical NOT.
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn not(e: Expr<'a, F>) -> Expr<'a, F> {
        Expr::Not(Box::new(e))
    }
}

/// Single predicate against a field.
#[derive(Debug, Clone)]
pub struct Filter<'a, F> {
    pub field_id: F,
    pub op: Operator<'a>,
}

type DynScalar = Scalar<Arc<dyn Array>>;

/// Comparison/matching operators over Arrow's Scalar.
#[derive(Debug, Clone)]
pub enum Operator<'a> {
    Equals(DynScalar),
    Range {
        lower: Bound<DynScalar>,
        upper: Bound<DynScalar>,
    },
    GreaterThan(DynScalar),
    GreaterThanOrEquals(DynScalar),
    LessThan(DynScalar),
    LessThanOrEquals(DynScalar),
    In(&'a [DynScalar]),
    StartsWith(&'a str),
    EndsWith(&'a str),
    Contains(&'a str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Scalar, StringArray};

    fn make_scalar<T: Array + 'static>(value: T) -> DynScalar {
        Scalar::new(Arc::new(value) as Arc<dyn Array>)
    }

    #[test]
    fn build_simple_exprs() {
        let f1 = Filter {
            field_id: 1,
            op: Operator::Equals(make_scalar(StringArray::from(vec!["abc"]))),
        };
        let f2 = Filter {
            field_id: 2,
            op: Operator::LessThan(make_scalar(StringArray::from(vec!["zzz"]))),
        };
        let all = Expr::all_of(vec![f1.clone(), f2.clone()]);
        let any = Expr::any_of(vec![f1.clone(), f2.clone()]);
        let not_all = Expr::not(all);
        match any {
            Expr::Or(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected Or"),
        }
        match not_all {
            Expr::Not(inner) => match *inner {
                Expr::And(v) => assert_eq!(v.len(), 2),
                _ => panic!("expected And inside Not"),
            },
            _ => panic!("expected Not"),
        }
    }

    #[test]
    fn complex_nested_shape() {
        // f1: id=1 == "a"
        // f2: id=2 <  "zzz"
        // f3: id=3 in ["x","y","z"]
        // f4: id=4 starts_with "pre"
        let f1 = Filter {
            field_id: 1u32,
            op: Operator::Equals(make_scalar(StringArray::from(vec!["a"]))),
        };
        let f2 = Filter {
            field_id: 2u32,
            op: Operator::LessThan(make_scalar(StringArray::from(vec!["zzz"]))),
        };
        let in_values = [
            make_scalar(StringArray::from(vec!["x"])),
            make_scalar(StringArray::from(vec!["y"])),
            make_scalar(StringArray::from(vec!["z"])),
        ];
        let f3 = Filter {
            field_id: 3u32,
            op: Operator::In(&in_values),
        };
        let f4 = Filter {
            field_id: 4u32,
            op: Operator::StartsWith("pre"),
        };

        // ( f1 AND ( f2 OR NOT f3 ) )  OR  ( NOT f1 AND f4 )
        let left = Expr::And(vec![
            Expr::Pred(f1.clone()),
            Expr::Or(vec![
                Expr::Pred(f2.clone()),
                Expr::not(Expr::Pred(f3.clone())),
            ]),
        ]);
        let right = Expr::And(vec![
            Expr::not(Expr::Pred(f1.clone())),
            Expr::Pred(f4.clone()),
        ]);
        let top = Expr::Or(vec![left, right]);

        // Shape checks
        match top {
            Expr::Or(branches) => {
                assert_eq!(branches.len(), 2);
                match &branches[0] {
                    Expr::And(v) => {
                        assert_eq!(v.len(), 2);
                        // AND: [Pred(f1), OR(...)]
                        match &v[0] {
                            Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 1),
                            _ => panic!("expected Pred(f1) in left-AND[0]"),
                        }
                        match &v[1] {
                            Expr::Or(or_vec) => {
                                assert_eq!(or_vec.len(), 2);
                                match &or_vec[0] {
                                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 2),
                                    _ => panic!("expected Pred(f2) in left-AND[1].OR[0]"),
                                }
                                match &or_vec[1] {
                                    Expr::Not(inner) => match inner.as_ref() {
                                        Expr::Pred(Filter { field_id, .. }) => {
                                            assert_eq!(*field_id, 3)
                                        }
                                        _ => panic!("expected Not(Pred(f3)) in left-AND[1].OR[1]"),
                                    },
                                    _ => panic!("expected Not(...) in left-AND[1].OR[1]"),
                                }
                            }
                            _ => panic!("expected OR in left-AND[1]"),
                        }
                    }
                    _ => panic!("expected AND on left branch of top OR"),
                }
                match &branches[1] {
                    Expr::And(v) => {
                        assert_eq!(v.len(), 2);
                        // AND: [Not(f1), Pred(f4)]
                        match &v[0] {
                            Expr::Not(inner) => match inner.as_ref() {
                                Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 1),
                                _ => panic!("expected Not(Pred(f1)) in right-AND[0]"),
                            },
                            _ => panic!("expected Not(...) in right-AND[0]"),
                        }
                        match &v[1] {
                            Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 4),
                            _ => panic!("expected Pred(f4) in right-AND[1]"),
                        }
                    }
                    _ => panic!("expected AND on right branch of top OR"),
                }
            }
            _ => panic!("expected top-level OR"),
        }
    }

    #[test]
    fn range_bounds_roundtrip() {
        // [aaa, bbb)
        let f = Filter {
            field_id: 7u32,
            op: Operator::Range {
                lower: Bound::Included(make_scalar(StringArray::from(vec!["aaa"]))),
                upper: Bound::Excluded(make_scalar(StringArray::from(vec!["bbb"]))),
            },
        };

        match &f.op {
            Operator::Range { lower, upper } => {
                if let Bound::Included(l) = lower {
                    let l_arr = l.clone().into_inner();
                    let l_val = l_arr
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .value(0);
                    assert_eq!(l_val, "aaa");
                } else {
                    panic!("lower bound should be Included");
                }

                if let Bound::Excluded(u) = upper {
                    let u_arr = u.clone().into_inner();
                    let u_val = u_arr
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .value(0);
                    assert_eq!(u_val, "bbb");
                } else {
                    panic!("upper bound should be Excluded");
                }
            }
            _ => panic!("expected Range operator"),
        }
    }

    #[test]
    fn helper_builders_preserve_structure_and_order() {
        let f1 = Filter {
            field_id: 1u32,
            op: Operator::Equals(make_scalar(StringArray::from(vec!["a"]))),
        };
        let f2 = Filter {
            field_id: 2u32,
            op: Operator::Equals(make_scalar(StringArray::from(vec!["b"]))),
        };
        let f3 = Filter {
            field_id: 3u32,
            op: Operator::Equals(make_scalar(StringArray::from(vec!["c"]))),
        };

        let and_expr = Expr::all_of(vec![f1.clone(), f2.clone(), f3.clone()]);
        match and_expr {
            Expr::And(v) => {
                assert_eq!(v.len(), 3);
                // Expect Pred(1), Pred(2), Pred(3) in order
                match &v[0] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 1),
                    _ => panic!(),
                }
                match &v[1] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 2),
                    _ => panic!(),
                }
                match &v[2] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 3),
                    _ => panic!(),
                }
            }
            _ => panic!("expected And"),
        }

        let or_expr = Expr::any_of(vec![f3.clone(), f2.clone(), f1.clone()]);
        match or_expr {
            Expr::Or(v) => {
                assert_eq!(v.len(), 3);
                // Expect Pred(3), Pred(2), Pred(1) in order
                match &v[0] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 3),
                    _ => panic!(),
                }
                match &v[1] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 2),
                    _ => panic!(),
                }
                match &v[2] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, 1),
                    _ => panic!(),
                }
            }
            _ => panic!("expected Or"),
        }
    }

    #[test]
    fn set_and_pattern_ops_hold_borrowed_slices() {
        let in_values = [
            make_scalar(StringArray::from(vec!["aa"])),
            make_scalar(StringArray::from(vec!["bb"])),
            make_scalar(StringArray::from(vec!["cc"])),
        ];
        let f_in = Filter {
            field_id: 9u32,
            op: Operator::In(&in_values),
        };
        match f_in.op {
            Operator::In(arr) => {
                assert_eq!(arr.len(), 3);
            }
            _ => panic!("expected In"),
        }

        let f_sw = Filter {
            field_id: 10u32,
            op: Operator::StartsWith("pre"),
        };
        let f_ew = Filter {
            field_id: 11u32,
            op: Operator::EndsWith("suf"),
        };
        let f_ct = Filter {
            field_id: 12u32,
            op: Operator::Contains("mid"),
        };

        match f_sw.op {
            Operator::StartsWith(b) => assert_eq!(b, "pre"),
            _ => panic!(),
        }
        match f_ew.op {
            Operator::EndsWith(b) => assert_eq!(b, "suf"),
            _ => panic!(),
        }
        match f_ct.op {
            Operator::Contains(b) => assert_eq!(b, "mid"),
            _ => panic!(),
        }
    }

    #[test]
    fn generic_field_id_works_with_strings() {
        // Demonstrate F = &'static str
        let f1 = Filter {
            field_id: "name",
            op: Operator::Equals(make_scalar(StringArray::from(vec!["alice"]))),
        };
        let f2 = Filter {
            field_id: "status",
            op: Operator::GreaterThanOrEquals(make_scalar(StringArray::from(vec!["active"]))),
        };
        let expr = Expr::all_of(vec![f1.clone(), f2.clone()]);

        match expr {
            Expr::And(v) => {
                assert_eq!(v.len(), 2);
                match &v[0] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, "name"),
                    _ => panic!("expected Pred(name)"),
                }
                match &v[1] {
                    Expr::Pred(Filter { field_id, .. }) => assert_eq!(*field_id, "status"),
                    _ => panic!("expected Pred(status)"),
                }
            }
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn very_deep_not_chain() {
        // Build Not(Not(...Not(Pred)...)) of depth 64
        let base = Expr::Pred(Filter {
            field_id: 42u32,
            op: Operator::Equals(make_scalar(StringArray::from(vec!["x"]))),
        });
        let mut expr = base;
        for _ in 0..64 {
            expr = Expr::not(expr);
        }

        // Count nested NOTs
        let mut count = 0usize;
        let mut cur = &expr;
        loop {
            match cur {
                Expr::Not(inner) => {
                    count += 1;
                    cur = inner;
                }
                Expr::Pred(Filter { field_id, .. }) => {
                    assert_eq!(*field_id, 42);
                    break;
                }
                _ => panic!("unexpected node inside deep NOT chain"),
            }
        }
        assert_eq!(count, 64);
    }
}
