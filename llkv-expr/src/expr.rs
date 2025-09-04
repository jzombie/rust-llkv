//! Buffer-only predicate AST.
#![forbid(unsafe_code)]

use std::ops::Bound;

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

/// Single predicate against a field. Byte semantics are adapter-defined.
#[derive(Debug, Clone)]
pub struct Filter<'a, F> {
    pub field_id: F,
    pub op: Operator<'a>,
}

/// Comparison/matching operators over raw byte slices.
#[derive(Debug, Clone)]
pub enum Operator<'a> {
    // Equality
    Equals(&'a [u8]),

    Range {
        lower: Bound<&'a [u8]>,
        upper: Bound<&'a [u8]>,
    },

    // Simple comparisons (can be implemented as special cases of Range if needed)
    GreaterThan(&'a [u8]),
    GreaterThanOrEquals(&'a [u8]),
    LessThan(&'a [u8]),
    LessThanOrEquals(&'a [u8]),

    // Set & pattern matching
    In(&'a [&'a [u8]]),
    StartsWith(&'a [u8]),
    EndsWith(&'a [u8]),
    Contains(&'a [u8]),
}

#[cfg(test)]
mod tests {
    use super::*;
    type TestFieldId = u32;
    use std::ops::Bound;
    use std::ptr;

    #[test]
    fn build_simple_exprs() {
        let f1 = Filter {
            field_id: 1,
            op: Operator::Equals(b"abc"),
        };
        let f2 = Filter {
            field_id: 2,
            op: Operator::LessThan(b"zzz"),
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
            op: Operator::Equals(b"a"),
        };
        let f2 = Filter {
            field_id: 2u32,
            op: Operator::LessThan(b"zzz"),
        };
        let f3 = Filter {
            field_id: 3u32,
            op: Operator::In(&[b"x", b"y", b"z"]),
        };
        let f4 = Filter {
            field_id: 4u32,
            op: Operator::StartsWith(b"pre"),
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
                lower: Bound::Included(b"aaa"),
                upper: Bound::Excluded(b"bbb"),
            },
        };

        match f.op {
            Operator::Range { lower, upper } => {
                match lower {
                    Bound::Included(b) => assert_eq!(b, b"aaa"),
                    _ => panic!("lower bound should be Included"),
                }
                match upper {
                    Bound::Excluded(b) => assert_eq!(b, b"bbb"),
                    _ => panic!("upper bound should be Excluded"),
                }
            }
            _ => panic!("expected Range operator"),
        }
    }

    #[test]
    fn helper_builders_preserve_structure_and_order() {
        let f1 = Filter {
            field_id: 1u32,
            op: Operator::Equals(b"a"),
        };
        let f2 = Filter {
            field_id: 2u32,
            op: Operator::Equals(b"b"),
        };
        let f3 = Filter {
            field_id: 3u32,
            op: Operator::Equals(b"c"),
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

        // empty lists are allowed => empty And/Or
        match Expr::<TestFieldId>::all_of(vec![]) {
            Expr::And(v) => assert!(v.is_empty()),
            _ => panic!(),
        }
        match Expr::<TestFieldId>::any_of(vec![]) {
            Expr::Or(v) => assert!(v.is_empty()),
            _ => panic!(),
        }

        // not wraps exactly once
        let pred = Expr::Pred(f1.clone());
        let not_pred = Expr::not(pred);
        match not_pred {
            Expr::Not(inner) => match *inner {
                Expr::Pred(Filter { field_id, .. }) => assert_eq!(field_id, 1),
                _ => panic!("Not should contain the original Pred"),
            },
            _ => panic!("expected Not"),
        }
    }

    #[test]
    fn set_and_pattern_ops_hold_borrowed_slices() {
        // Using 'static byte literals to check pointer identity is preserved.
        let a: &'static [u8] = b"aa";
        let b_: &'static [u8] = b"bb";
        let c: &'static [u8] = b"cc";

        let f_in = Filter {
            field_id: 9u32,
            op: Operator::In(&[a, b_, c]),
        };
        match f_in.op {
            Operator::In(arr) => {
                assert_eq!(arr.len(), 3);
                // Pointer identity check
                assert!(ptr::eq(arr[0].as_ptr(), a.as_ptr()));
                assert!(ptr::eq(arr[1].as_ptr(), b_.as_ptr()));
                assert!(ptr::eq(arr[2].as_ptr(), c.as_ptr()));
            }
            _ => panic!("expected In"),
        }

        let f_sw = Filter {
            field_id: 10u32,
            op: Operator::StartsWith(b"pre"),
        };
        let f_ew = Filter {
            field_id: 11u32,
            op: Operator::EndsWith(b"suf"),
        };
        let f_ct = Filter {
            field_id: 12u32,
            op: Operator::Contains(b"mid"),
        };

        match f_sw.op {
            Operator::StartsWith(b) => assert_eq!(b, b"pre"),
            _ => panic!(),
        }
        match f_ew.op {
            Operator::EndsWith(b) => assert_eq!(b, b"suf"),
            _ => panic!(),
        }
        match f_ct.op {
            Operator::Contains(b) => assert_eq!(b, b"mid"),
            _ => panic!(),
        }
    }

    #[test]
    fn generic_field_id_works_with_strings() {
        // Demonstrate F = &'static str
        let f1 = Filter {
            field_id: "name",
            op: Operator::Equals(b"alice"),
        };
        let f2 = Filter {
            field_id: "status",
            op: Operator::GreaterThanOrEquals(b"active"),
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
            op: Operator::Equals(b"x"),
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
