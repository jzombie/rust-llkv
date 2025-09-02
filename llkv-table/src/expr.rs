//! Buffer-only predicate AST.
//!
//! All payloads are borrowed &[u8]. Negation is an expression node,
//! not a dedicated "Ne" operator.

#![forbid(unsafe_code)]

use crate::types::FieldId;

/// Logical expression over predicates.
#[derive(Clone, Debug)]
pub enum Expr<'a> {
    And(Vec<Expr<'a>>),
    Or(Vec<Expr<'a>>),
    Not(Box<Expr<'a>>),
    Pred(Filter<'a>),
}

impl<'a> Expr<'a> {
    /// Build an AND of filters.
    #[inline]
    pub fn all_of(fs: Vec<Filter<'a>>) -> Expr<'a> {
        Expr::And(fs.into_iter().map(Expr::Pred).collect())
    }

    /// Build an OR of filters.
    #[inline]
    pub fn any_of(fs: Vec<Filter<'a>>) -> Expr<'a> {
        Expr::Or(fs.into_iter().map(Expr::Pred).collect())
    }

    /// Wrap an expression in a logical NOT.
    #[inline]
    pub fn not(e: Expr<'a>) -> Expr<'a> {
        Expr::Not(Box::new(e))
    }
}

/// Single predicate against a field. Byte semantics are adapter-defined.
#[derive(Debug, Clone)]
pub struct Filter<'a> {
    pub field: FieldId,
    pub op: Operator<'a>,
}

/// Comparison/matching operators over raw byte slices.
#[derive(Debug, Clone)]
pub enum Operator<'a> {
    // Equality
    Equals(&'a [u8]),

    // Range comparisons
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

    #[test]
    fn build_simple_exprs() {
        let f1 = Filter {
            field: 1,
            op: Operator::Equals(b"abc"),
        };
        let f2 = Filter {
            field: 2,
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
}
