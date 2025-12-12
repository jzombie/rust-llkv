
fn simplify_expr<F: std::hash::Hash + Eq + Clone>(expr: Expr<'static, F>) -> Expr<'static, F> {
    match expr {
        Expr::And(exprs) => Expr::And(exprs.into_iter().map(simplify_expr).collect()),
        Expr::Or(exprs) => Expr::Or(exprs.into_iter().map(simplify_expr).collect()),
        Expr::Not(expr) => Expr::Not(Box::new(simplify_expr(*expr))),
        Expr::Pred(filter) => Expr::Pred(filter),
        Expr::Compare { left, op, right } => Expr::Compare {
            left: llkv_compute::eval::ScalarEvaluator::simplify(&left),
            op,
            right: llkv_compute::eval::ScalarEvaluator::simplify(&right),
        },
        Expr::InList { expr, list, negated } => Expr::InList {
            expr: llkv_compute::eval::ScalarEvaluator::simplify(&expr),
            list: list
                .into_iter()
                .map(|e| llkv_compute::eval::ScalarEvaluator::simplify(&e))
                .collect(),
            negated,
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: llkv_compute::eval::ScalarEvaluator::simplify(&expr),
            negated,
        },
        Expr::Literal(b) => Expr::Literal(b),
        Expr::Exists(sub) => Expr::Exists(sub),
    }
}
