//! Expression normalization logic.
//!
//! This module provides functions to normalize expressions, such as:
//! - Flattening nested AND/OR expressions.
//! - Pushing down NOT operators.
//! - Normalizing comparisons (e.g., converting `Compare` to `Pred` where possible).

use crate::expr::{CompareOp, Expr, Filter, Operator, ScalarExpr};
use crate::literal::Literal;

/// Normalize a predicate expression.
///
/// This applies several simplifications:
/// - Flattens nested AND/OR chains.
/// - Pushes NOT down to the leaves (De Morgan's laws).
/// - Converts `Compare` expressions to `Pred` (Filter) when one side is a column and the other is a literal.
/// - Simplifies boolean literals.
pub fn normalize_predicate<'expr, F: Clone>(expr: Expr<'expr, F>) -> Expr<'expr, F> {
    normalize_expr(expr)
}

fn normalize_expr<'expr, F: Clone>(expr: Expr<'expr, F>) -> Expr<'expr, F> {
    match expr {
        Expr::And(children) => {
            let mut normalized = Vec::with_capacity(children.len());
            for child in children {
                let child = normalize_expr(child);
                match child {
                    Expr::And(nested) => normalized.extend(nested),
                    other => normalized.push(other),
                }
            }
            Expr::And(normalized)
        }
        Expr::Or(children) => {
            let mut normalized = Vec::with_capacity(children.len());
            for child in children {
                let child = normalize_expr(child);
                match child {
                    Expr::Or(nested) => normalized.extend(nested),
                    other => normalized.push(other),
                }
            }
            Expr::Or(normalized)
        }
        Expr::Not(inner) => normalize_negated(*inner),
        Expr::Compare { left, op, right } => normalize_compare(left, op, right),
        other => other,
    }
}

fn normalize_compare<'expr, F: Clone>(
    left: ScalarExpr<F>,
    op: CompareOp,
    right: ScalarExpr<F>,
) -> Expr<'expr, F> {
    match (left, right) {
        (ScalarExpr::Column(field_id), ScalarExpr::Literal(lit))
            if !matches!(lit, Literal::Null) =>
        {
            match op {
                CompareOp::Eq => Expr::Pred(Filter {
                    field_id,
                    op: Operator::Equals(lit),
                }),
                CompareOp::Gt => Expr::Pred(Filter {
                    field_id,
                    op: Operator::GreaterThan(lit),
                }),
                CompareOp::GtEq => Expr::Pred(Filter {
                    field_id,
                    op: Operator::GreaterThanOrEquals(lit),
                }),
                CompareOp::Lt => Expr::Pred(Filter {
                    field_id,
                    op: Operator::LessThan(lit),
                }),
                CompareOp::LtEq => Expr::Pred(Filter {
                    field_id,
                    op: Operator::LessThanOrEquals(lit),
                }),
                CompareOp::NotEq => Expr::Not(Box::new(Expr::Pred(Filter {
                    field_id,
                    op: Operator::Equals(lit),
                }))),
            }
        }
        (ScalarExpr::Literal(lit), ScalarExpr::Column(field_id))
            if !matches!(lit, Literal::Null) =>
        {
            match op {
                CompareOp::Eq => Expr::Pred(Filter {
                    field_id,
                    op: Operator::Equals(lit),
                }),
                CompareOp::Gt => Expr::Pred(Filter {
                    field_id,
                    op: Operator::LessThan(lit),
                }),
                CompareOp::GtEq => Expr::Pred(Filter {
                    field_id,
                    op: Operator::LessThanOrEquals(lit),
                }),
                CompareOp::Lt => Expr::Pred(Filter {
                    field_id,
                    op: Operator::GreaterThan(lit),
                }),
                CompareOp::LtEq => Expr::Pred(Filter {
                    field_id,
                    op: Operator::GreaterThanOrEquals(lit),
                }),
                CompareOp::NotEq => Expr::Not(Box::new(Expr::Pred(Filter {
                    field_id,
                    op: Operator::Equals(lit),
                }))),
            }
        }
        (left, right) => Expr::Compare { left, op, right },
    }
}

fn normalize_negated<'expr, F: Clone>(inner: Expr<'expr, F>) -> Expr<'expr, F> {
    match inner {
        Expr::Not(nested) => normalize_expr(*nested),
        Expr::And(children) => {
            let mapped = children
                .into_iter()
                .map(|child| normalize_expr(Expr::Not(Box::new(child))))
                .collect();
            Expr::Or(mapped)
        }
        Expr::Or(children) => {
            let mapped = children
                .into_iter()
                .map(|child| normalize_expr(Expr::Not(Box::new(child))))
                .collect();
            Expr::And(mapped)
        }
        Expr::Compare { left, op, right } => {
            let negated_op = match op {
                CompareOp::Eq => CompareOp::NotEq,
                CompareOp::NotEq => CompareOp::Eq,
                CompareOp::Gt => CompareOp::LtEq,
                CompareOp::GtEq => CompareOp::Lt,
                CompareOp::Lt => CompareOp::GtEq,
                CompareOp::LtEq => CompareOp::Gt,
            };
            normalize_compare(left, negated_op, right)
        }
        Expr::Literal(value) => Expr::Literal(!value),
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr,
            negated: !negated,
        },
        other => Expr::Not(Box::new(normalize_expr(other))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Literal;

    // Simple integer ID for testing
    type TestId = usize;

    #[test]
    fn normalize_not_between_expands_to_or() {
        let field: TestId = 7;
        let column = ScalarExpr::Column(field);
        let lower = ScalarExpr::Literal(Literal::Integer(5));
        let upper = ScalarExpr::Literal(Literal::Null);

        let between = Expr::And(vec![
            Expr::Compare {
                left: column.clone(),
                op: CompareOp::GtEq,
                right: lower,
            },
            Expr::Compare {
                left: column.clone(),
                op: CompareOp::LtEq,
                right: upper,
            },
        ]);

        let normalized = normalize_predicate(Expr::Not(Box::new(between)));

        let Expr::Or(children) = normalized else {
            panic!("expected OR after normalization");
        };
        assert_eq!(children.len(), 2);

        match &children[0] {
            Expr::Pred(Filter {
                op: Operator::LessThan(_),
                ..
            }) => {}
            other => panic!("left branch should be Pred(LessThan), got {other:?}"),
        }

        match &children[1] {
            Expr::Compare {
                op: CompareOp::Gt,
                right: ScalarExpr::Literal(Literal::Null),
                ..
            } => {}
            other => panic!("right branch should be Compare(Gt, Null), got {other:?}"),
        }
    }

    #[test]
    fn normalize_flips_literal_bool() {
        let normalized = normalize_predicate(Expr::<TestId>::Not(Box::new(Expr::Literal(true))));
        assert!(matches!(normalized, Expr::Literal(false)));
    }
}
