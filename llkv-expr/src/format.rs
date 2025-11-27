//! Formatting utilities for expression and predicate trees.

use std::fmt::Display;
use std::ops::Bound;

use crate::expr::{Filter, Operator};
use crate::{Expr, Literal, ScalarExpr};

impl<'a, F> Expr<'a, F>
where
    F: Display + Copy,
{
    /// Render a predicate expression as a human-readable string.
    pub fn format_display(&self) -> String {
        use Expr::*;

        // Iterative postorder traversal using work/result stack pattern.
        // This uses a two-pass approach: first collect nodes in postorder, then format them.
        // This avoids stack overflow on deeply nested expressions (50k+ nodes).
        let mut traverse_stack = Vec::new();
        let mut postorder = Vec::new();
        traverse_stack.push(self);

        while let Some(node) = traverse_stack.pop() {
            postorder.push(node);
            match node {
                And(children) | Or(children) => {
                    for child in children {
                        traverse_stack.push(child);
                    }
                }
                Not(inner) => traverse_stack.push(inner),
                Pred(_)
                | Compare { .. }
                | InList { .. }
                | IsNull { .. }
                | Literal(_)
                | Exists(_) => {}
            }
        }

        let mut result_stack: Vec<String> = Vec::new();
        for node in postorder.into_iter().rev() {
            match node {
                And(children) => {
                    if children.is_empty() {
                        result_stack.push("TRUE".to_string());
                    } else {
                        let mut parts = Vec::with_capacity(children.len());
                        for _ in 0..children.len() {
                            parts.push(result_stack.pop().unwrap_or_default());
                        }
                        parts.reverse();
                        result_stack.push(parts.join(" AND "));
                    }
                }
                Or(children) => {
                    if children.is_empty() {
                        result_stack.push("FALSE".to_string());
                    } else {
                        let mut parts = Vec::with_capacity(children.len());
                        for _ in 0..children.len() {
                            parts.push(result_stack.pop().unwrap_or_default());
                        }
                        parts.reverse();
                        result_stack.push(parts.join(" OR "));
                    }
                }
                Not(_) => {
                    let inner = result_stack.pop().unwrap_or_default();
                    result_stack.push(format!("NOT ({inner})"));
                }
                Pred(filter) => {
                    result_stack.push(format_filter(filter));
                }
                Compare { left, op, right } => {
                    result_stack.push(format!(
                        "{} {} {}",
                        left.format_display(),
                        op.as_str(),
                        right.format_display()
                    ));
                }
                InList {
                    expr,
                    list,
                    negated,
                } => {
                    let expr_str = expr.format_display();
                    let mut parts = Vec::with_capacity(list.len());
                    for value in list {
                        parts.push(value.format_display());
                    }
                    let keyword = if *negated { "NOT IN" } else { "IN" };
                    result_stack.push(format!("{} {} ({})", expr_str, keyword, parts.join(", ")));
                }
                IsNull { expr, negated } => {
                    let expr_str = expr.format_display();
                    let keyword = if *negated { "IS NOT NULL" } else { "IS NULL" };
                    result_stack.push(format!("{} {}", expr_str, keyword));
                }
                Literal(value) => {
                    result_stack.push(if *value {
                        "TRUE".to_string()
                    } else {
                        "FALSE".to_string()
                    });
                }
                Exists(_) => {
                    result_stack.push("EXISTS(...)".to_string());
                }
            }
        }

        result_stack.pop().unwrap_or_default()
    }
}

impl<F> ScalarExpr<F>
where
    F: Display + Copy,
{
    /// Render a scalar expression as a human-readable string.
    pub fn format_display(&self) -> String {
        use ScalarExpr::*;

        let mut traverse_stack = Vec::new();
        let mut postorder = Vec::new();
        traverse_stack.push(self);

        while let Some(node) = traverse_stack.pop() {
            postorder.push(node);
            match node {
                Column(_) | Literal(_) => {}
                Binary { left, right, .. } | Compare { left, right, .. } => {
                    traverse_stack.push(right);
                    traverse_stack.push(left);
                }
                Not(inner) => traverse_stack.push(inner),
                IsNull { expr, .. } => traverse_stack.push(expr),
                Aggregate(_) => {}
                GetField { base, .. } => traverse_stack.push(base),
                Cast { expr, .. } => traverse_stack.push(expr),
                Case {
                    operand,
                    branches,
                    else_expr,
                } => {
                    if let Some(op) = operand {
                        traverse_stack.push(op);
                    }
                    if let Some(else_expr) = else_expr {
                        traverse_stack.push(else_expr);
                    }
                    for (when_expr, then_expr) in branches {
                        traverse_stack.push(then_expr);
                        traverse_stack.push(when_expr);
                    }
                }
                Coalesce(items) => {
                    for item in items {
                        traverse_stack.push(item);
                    }
                }
                Random => {}
                ScalarSubquery(_) => {}
            }
        }

        let mut result_stack: Vec<String> = Vec::new();
        for node in postorder.into_iter().rev() {
            match node {
                Column(fid) => result_stack.push(format!("col#{}", fid)),
                Literal(lit) => result_stack.push(lit.format_display()),
                Aggregate(_agg) => result_stack.push("AGG".to_string()),
                GetField { field_name, .. } => {
                    let base = result_stack.pop().unwrap_or_default();
                    result_stack.push(format!("{base}.{field_name}"));
                }
                Cast { data_type, .. } => {
                    let value = result_stack.pop().unwrap_or_default();
                    result_stack.push(format!("CAST({value} AS {data_type:?})"));
                }
                Binary { op, .. } => {
                    let right = result_stack.pop().unwrap_or_default();
                    let left = result_stack.pop().unwrap_or_default();
                    result_stack.push(format!("({} {} {})", left, op.as_str(), right));
                }
                Compare { op, .. } => {
                    let right = result_stack.pop().unwrap_or_default();
                    let left = result_stack.pop().unwrap_or_default();
                    result_stack.push(format!("({} {} {})", left, op.as_str(), right));
                }
                Not(_) => {
                    let operand = result_stack.pop().unwrap_or_default();
                    result_stack.push(format!("(NOT {})", operand));
                }
                IsNull { negated, .. } => {
                    let operand = result_stack.pop().unwrap_or_default();
                    if *negated {
                        result_stack.push(format!("({operand} IS NOT NULL)"));
                    } else {
                        result_stack.push(format!("({operand} IS NULL)"));
                    }
                }
                Case {
                    branches,
                    else_expr,
                    ..
                } => {
                    let mut parts = Vec::with_capacity(branches.len() + 2);
                    for _ in 0..branches.len() {
                        let then_str = result_stack.pop().unwrap_or_default();
                        let when_str = result_stack.pop().unwrap_or_default();
                        parts.push(format!("WHEN {when_str} THEN {then_str}"));
                    }
                    if let Some(_else_expr) = else_expr {
                        parts.push(format!("ELSE {}", result_stack.pop().unwrap_or_default()));
                    }

                    parts.push("END".to_string());
                    let mut output = parts.join(" ");
                    output.insert_str(0, "CASE ");
                    result_stack.push(output);
                }
                Coalesce(items) => {
                    let mut args = Vec::with_capacity(items.len());
                    for _ in 0..items.len() {
                        args.push(result_stack.pop().unwrap_or_default());
                    }
                    args.reverse();
                    result_stack.push(format!("COALESCE({})", args.join(", ")));
                }
                Random => {
                    result_stack.push("RANDOM()".to_string());
                }
                ScalarSubquery(sub) => {
                    result_stack.push(format!("(SCALAR_SUBQUERY#{})", sub.id.0));
                }
            }
        }

        result_stack.pop().unwrap_or_default()
    }
}

fn format_filter<F: Display>(filter: &Filter<'_, F>) -> String {
    format!("field#{} {}", filter.field_id, format_operator(&filter.op))
}

fn format_operator(op: &Operator<'_>) -> String {
    match op {
        Operator::Equals(lit) => format!("= {}", lit.format_display()),
        Operator::Range { lower, upper } => format!(
            "IN {} .. {}",
            format_range_bound_lower(lower),
            format_range_bound_upper(upper)
        ),
        Operator::GreaterThan(lit) => format!("> {}", lit.format_display()),
        Operator::GreaterThanOrEquals(lit) => format!(">= {}", lit.format_display()),
        Operator::LessThan(lit) => format!("< {}", lit.format_display()),
        Operator::LessThanOrEquals(lit) => format!("<= {}", lit.format_display()),
        Operator::In(values) => {
            let rendered: Vec<String> = values.iter().map(|lit| lit.format_display()).collect();
            format!("IN {{{}}}", rendered.join(", "))
        }
        Operator::StartsWith {
            pattern,
            case_sensitive,
        } => format_pattern_op("STARTS WITH", pattern, *case_sensitive),
        Operator::EndsWith {
            pattern,
            case_sensitive,
        } => format_pattern_op("ENDS WITH", pattern, *case_sensitive),
        Operator::Contains {
            pattern,
            case_sensitive,
        } => format_pattern_op("CONTAINS", pattern, *case_sensitive),
        Operator::IsNull => "IS NULL".to_string(),
        Operator::IsNotNull => "IS NOT NULL".to_string(),
    }
}

fn format_pattern_op(op_name: &str, pattern: &str, case_sensitive: bool) -> String {
    let mut rendered = format!("{} \"{}\"", op_name, escape_string(pattern));
    if !case_sensitive {
        rendered.push_str(" (case-insensitive)");
    }
    rendered
}

fn format_range_bound_lower(bound: &Bound<Literal>) -> String {
    match bound {
        Bound::Unbounded => "-inf".to_string(),
        Bound::Included(lit) => format!("[{}", lit.format_display()),
        Bound::Excluded(lit) => format!("({}", lit.format_display()),
    }
}

fn format_range_bound_upper(bound: &Bound<Literal>) -> String {
    match bound {
        Bound::Unbounded => "+inf".to_string(),
        Bound::Included(lit) => format!("{}]", lit.format_display()),
        Bound::Excluded(lit) => format!("{})", lit.format_display()),
    }
}

fn escape_string(value: &str) -> String {
    value.chars().flat_map(|c| c.escape_default()).collect()
}
