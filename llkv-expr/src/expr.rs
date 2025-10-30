//! Type-aware, Arrow-native predicate AST.
//!
//! This module defines a small predicate-expression AST that is decoupled
//! from Arrow's concrete scalar types by using `Literal`. Concrete typing
//! is deferred to the consumer (e.g., a table/scan layer) which knows the
//! column types and can coerce `Literal` into native values.

#![forbid(unsafe_code)]

pub use crate::literal::*;
use arrow::datatypes::DataType;
use std::ops::Bound;

/// Logical expression over predicates.
#[derive(Clone, Debug)]
pub enum Expr<'a, F> {
    And(Vec<Expr<'a, F>>),
    Or(Vec<Expr<'a, F>>),
    Not(Box<Expr<'a, F>>),
    Pred(Filter<'a, F>),
    Compare {
        left: ScalarExpr<F>,
        op: CompareOp,
        right: ScalarExpr<F>,
    },
    InList {
        expr: ScalarExpr<F>,
        list: Vec<ScalarExpr<F>>,
        negated: bool,
    },
    /// Check if a scalar expression IS NULL or IS NOT NULL.
    /// For simple column references, prefer `Pred(Filter { op: IsNull/IsNotNull })` for optimization.
    /// This variant handles complex expressions like `(col1 + col2) IS NULL`.
    IsNull {
        expr: ScalarExpr<F>,
        negated: bool,
    },
    /// A literal boolean value (true/false).
    /// Used for conditions that are always true or always false (e.g., empty IN lists).
    Literal(bool),
    /// Correlated subquery evaluated in a boolean context.
    Exists(SubqueryExpr),
}

/// Metadata describing a correlated subquery.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SubqueryId(pub u32);

/// Correlated subquery used within a predicate expression.
#[derive(Clone, Debug)]
pub struct SubqueryExpr {
    /// Identifier referencing the subquery definition attached to the parent filter.
    pub id: SubqueryId,
    /// True when the SQL contained `NOT EXISTS`.
    pub negated: bool,
}

/// Scalar subquery evaluated as part of a scalar expression.
#[derive(Clone, Debug)]
pub struct ScalarSubqueryExpr {
    /// Identifier referencing the subquery definition attached to the parent projection.
    pub id: SubqueryId,
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

/// Arithmetic scalar expression that can reference multiple fields.
#[derive(Clone, Debug)]
pub enum ScalarExpr<F> {
    Column(F),
    Literal(Literal),
    Binary {
        left: Box<ScalarExpr<F>>,
        op: BinaryOp,
        right: Box<ScalarExpr<F>>,
    },
    /// Aggregate function call (e.g., COUNT(*), SUM(col), etc.)
    /// This is used in expressions like COUNT(*) + 1
    Aggregate(AggregateCall<F>),
    /// Extract a field from a struct expression.
    /// For example: `user.address.city` would be represented as
    /// GetField { base: GetField { base: Column(user), field_name: "address" }, field_name: "city" }
    GetField {
        base: Box<ScalarExpr<F>>,
        field_name: String,
    },
    /// Explicit type cast to an Arrow data type.
    Cast {
        expr: Box<ScalarExpr<F>>,
        data_type: DataType,
    },
    /// Comparison producing a boolean (1/0) result.
    Compare {
        left: Box<ScalarExpr<F>>,
        op: CompareOp,
        right: Box<ScalarExpr<F>>,
    },
    /// First non-null expression in the provided list.
    Coalesce(Vec<ScalarExpr<F>>),
    /// Scalar subquery evaluated per input row.
    ScalarSubquery(ScalarSubqueryExpr),
    /// SQL CASE expression with optional operand and ELSE branch.
    Case {
        /// Optional operand for simple CASE (e.g., `CASE x WHEN ...`).
        operand: Option<Box<ScalarExpr<F>>>,
        /// Ordered (WHEN, THEN) branches.
        branches: Vec<(ScalarExpr<F>, ScalarExpr<F>)>,
        /// Optional ELSE result.
        else_expr: Option<Box<ScalarExpr<F>>>,
    },
}

/// Aggregate function call within a scalar expression
#[derive(Clone, Debug)]
pub enum AggregateCall<F> {
    CountStar,
    Count { column: F, distinct: bool },
    Sum { column: F, distinct: bool },
    Avg { column: F, distinct: bool },
    Min(F),
    Max(F),
    CountNulls(F),
}

impl<F> ScalarExpr<F> {
    #[inline]
    pub fn column(field: F) -> Self {
        Self::Column(field)
    }

    #[inline]
    pub fn literal<L: Into<Literal>>(lit: L) -> Self {
        Self::Literal(lit.into())
    }

    #[inline]
    pub fn binary(left: Self, op: BinaryOp, right: Self) -> Self {
        Self::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    #[inline]
    pub fn aggregate(call: AggregateCall<F>) -> Self {
        Self::Aggregate(call)
    }

    #[inline]
    pub fn get_field(base: Self, field_name: String) -> Self {
        Self::GetField {
            base: Box::new(base),
            field_name,
        }
    }

    #[inline]
    pub fn cast(expr: Self, data_type: DataType) -> Self {
        Self::Cast {
            expr: Box::new(expr),
            data_type,
        }
    }

    #[inline]
    pub fn compare(left: Self, op: CompareOp, right: Self) -> Self {
        Self::Compare {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    #[inline]
    pub fn coalesce(exprs: Vec<Self>) -> Self {
        Self::Coalesce(exprs)
    }

    #[inline]
    pub fn scalar_subquery(id: SubqueryId) -> Self {
        Self::ScalarSubquery(ScalarSubqueryExpr { id })
    }

    #[inline]
    pub fn case(
        operand: Option<Self>,
        branches: Vec<(Self, Self)>,
        else_expr: Option<Self>,
    ) -> Self {
        Self::Case {
            operand: operand.map(Box::new),
            branches,
            else_expr: else_expr.map(Box::new),
        }
    }
}

/// Arithmetic operator for [`ScalarExpr`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}

/// Comparison operator for scalar expressions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

/// Single predicate against a field.
#[derive(Debug, Clone)]
pub struct Filter<'a, F> {
    pub field_id: F,
    pub op: Operator<'a>,
}

/// Comparison/matching operators over untyped `Literal`s.
///
/// `In` accepts a borrowed slice of `Literal`s to avoid allocations in the
/// common case of small, static IN lists built at call sites.
#[derive(Debug, Clone)]
pub enum Operator<'a> {
    Equals(Literal),
    Range {
        lower: Bound<Literal>,
        upper: Bound<Literal>,
    },
    GreaterThan(Literal),
    GreaterThanOrEquals(Literal),
    LessThan(Literal),
    LessThanOrEquals(Literal),
    In(&'a [Literal]),
    StartsWith {
        pattern: &'a str,
        case_sensitive: bool,
    },
    EndsWith {
        pattern: &'a str,
        case_sensitive: bool,
    },
    Contains {
        pattern: &'a str,
        case_sensitive: bool,
    },
    IsNull,
    IsNotNull,
}

impl<'a> Operator<'a> {
    #[inline]
    pub fn starts_with(pattern: &'a str, case_sensitive: bool) -> Self {
        Operator::StartsWith {
            pattern,
            case_sensitive,
        }
    }

    #[inline]
    pub fn ends_with(pattern: &'a str, case_sensitive: bool) -> Self {
        Operator::EndsWith {
            pattern,
            case_sensitive,
        }
    }

    #[inline]
    pub fn contains(pattern: &'a str, case_sensitive: bool) -> Self {
        Operator::Contains {
            pattern,
            case_sensitive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_exprs() {
        let f1 = Filter {
            field_id: 1,
            op: Operator::Equals("abc".into()),
        };
        let f2 = Filter {
            field_id: 2,
            op: Operator::LessThan("zzz".into()),
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
            op: Operator::Equals("a".into()),
        };
        let f2 = Filter {
            field_id: 2u32,
            op: Operator::LessThan("zzz".into()),
        };
        let in_values = ["x".into(), "y".into(), "z".into()];
        let f3 = Filter {
            field_id: 3u32,
            op: Operator::In(&in_values),
        };
        let f4 = Filter {
            field_id: 4u32,
            op: Operator::starts_with("pre", true),
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
                            Expr::Pred(Filter { field_id, .. }) => {
                                assert_eq!(*field_id, 1)
                            }
                            _ => panic!("expected Pred(f1) in left-AND[0]"),
                        }
                        match &v[1] {
                            Expr::Or(or_vec) => {
                                assert_eq!(or_vec.len(), 2);
                                match &or_vec[0] {
                                    Expr::Pred(Filter { field_id, .. }) => {
                                        assert_eq!(*field_id, 2)
                                    }
                                    _ => panic!("expected Pred(f2) in left-AND[1].OR[0]"),
                                }
                                match &or_vec[1] {
                                    Expr::Not(inner) => match inner.as_ref() {
                                        Expr::Pred(Filter { field_id, .. }) => {
                                            assert_eq!(*field_id, 3)
                                        }
                                        _ => panic!(
                                            "expected Not(Pred(f3)) in \
                                             left-AND[1].OR[1]"
                                        ),
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
                                Expr::Pred(Filter { field_id, .. }) => {
                                    assert_eq!(*field_id, 1)
                                }
                                _ => panic!("expected Not(Pred(f1)) in right-AND[0]"),
                            },
                            _ => panic!("expected Not(...) in right-AND[0]"),
                        }
                        match &v[1] {
                            Expr::Pred(Filter { field_id, .. }) => {
                                assert_eq!(*field_id, 4)
                            }
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
                lower: Bound::Included("aaa".into()),
                upper: Bound::Excluded("bbb".into()),
            },
        };

        match &f.op {
            Operator::Range { lower, upper } => {
                if let Bound::Included(l) = lower {
                    assert_eq!(*l, Literal::String("aaa".to_string()));
                } else {
                    panic!("lower bound should be Included");
                }

                if let Bound::Excluded(u) = upper {
                    assert_eq!(*u, Literal::String("bbb".to_string()));
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
            op: Operator::Equals("a".into()),
        };
        let f2 = Filter {
            field_id: 2u32,
            op: Operator::Equals("b".into()),
        };
        let f3 = Filter {
            field_id: 3u32,
            op: Operator::Equals("c".into()),
        };

        let and_expr = Expr::all_of(vec![f1.clone(), f2.clone(), f3.clone()]);
        match and_expr {
            Expr::And(v) => {
                assert_eq!(v.len(), 3);
                // Expect Pred(1), Pred(2), Pred(3) in order
                match &v[0] {
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, 1)
                    }
                    _ => panic!(),
                }
                match &v[1] {
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, 2)
                    }
                    _ => panic!(),
                }
                match &v[2] {
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, 3)
                    }
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
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, 3)
                    }
                    _ => panic!(),
                }
                match &v[1] {
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, 2)
                    }
                    _ => panic!(),
                }
                match &v[2] {
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, 1)
                    }
                    _ => panic!(),
                }
            }
            _ => panic!("expected Or"),
        }
    }

    #[test]
    fn set_and_pattern_ops_hold_borrowed_slices() {
        let in_values = ["aa".into(), "bb".into(), "cc".into()];
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
            op: Operator::starts_with("pre", true),
        };
        let f_ew = Filter {
            field_id: 11u32,
            op: Operator::ends_with("suf", true),
        };
        let f_ct = Filter {
            field_id: 12u32,
            op: Operator::contains("mid", true),
        };

        match f_sw.op {
            Operator::StartsWith {
                pattern: b,
                case_sensitive,
            } => {
                assert_eq!(b, "pre");
                assert!(case_sensitive);
            }
            _ => panic!(),
        }
        match f_ew.op {
            Operator::EndsWith {
                pattern: b,
                case_sensitive,
            } => {
                assert_eq!(b, "suf");
                assert!(case_sensitive);
            }
            _ => panic!(),
        }
        match f_ct.op {
            Operator::Contains {
                pattern: b,
                case_sensitive,
            } => {
                assert_eq!(b, "mid");
                assert!(case_sensitive);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn generic_field_id_works_with_strings() {
        // Demonstrate F = &'static str
        let f1 = Filter {
            field_id: "name",
            op: Operator::Equals("alice".into()),
        };
        let f2 = Filter {
            field_id: "status",
            op: Operator::GreaterThanOrEquals("active".into()),
        };
        let expr = Expr::all_of(vec![f1.clone(), f2.clone()]);

        match expr {
            Expr::And(v) => {
                assert_eq!(v.len(), 2);
                match &v[0] {
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, "name")
                    }
                    _ => panic!("expected Pred(name)"),
                }
                match &v[1] {
                    Expr::Pred(Filter { field_id, .. }) => {
                        assert_eq!(*field_id, "status")
                    }
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
            op: Operator::Equals("x".into()),
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

    #[test]
    fn literal_construction() {
        let f = Filter {
            field_id: "my_u64_col",
            op: Operator::Range {
                lower: Bound::Included(150.into()),
                upper: Bound::Excluded(300.into()),
            },
        };

        match f.op {
            Operator::Range { lower, upper } => {
                assert_eq!(lower, Bound::Included(Literal::Integer(150)));
                assert_eq!(upper, Bound::Excluded(Literal::Integer(300)));
            }
            _ => panic!("Expected a range operator"),
        }

        let f2 = Filter {
            field_id: "my_str_col",
            op: Operator::Equals("hello".into()),
        };

        match f2.op {
            Operator::Equals(lit) => {
                assert_eq!(lit, Literal::String("hello".to_string()));
            }
            _ => panic!("Expected an equals operator"),
        }
    }
}
