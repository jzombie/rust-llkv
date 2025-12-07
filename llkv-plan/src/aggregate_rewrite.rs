use llkv_expr::expr::{AggregateCall, BinaryOp, CompareOp, Expr, Operator, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_result::Result;

use crate::plans::{AggregateExpr, AggregateFunction, SelectPlan, SelectProjection};
use crate::schema::PlanSchema;

/// Rewritten aggregate state for complex aggregate projections.
#[derive(Clone, Debug)]
pub struct AggregateRewrite {
    pub aggregates: Vec<AggregateExpr>,
    pub final_expressions: Vec<ScalarExpr<String>>, // projections after aggregate substitution
    pub pre_aggregate_expressions: Vec<ScalarExpr<String>>, // expressions evaluated before aggregation
    pub final_names: Vec<String>,
    pub rewritten_having: Option<ScalarExpr<String>>, // HAVING rewritten to use aggregate outputs
}

struct AggVisitor {
    aggregates: Vec<AggregateExpr>,
    pre_agg_projections: Vec<ScalarExpr<String>>,
}

impl AggVisitor {
    fn new() -> Self {
        Self {
            aggregates: Vec::new(),
            pre_agg_projections: Vec::new(),
        }
    }

    fn visit(&mut self, expr: &ScalarExpr<String>) -> ScalarExpr<String> {
        match expr {
            ScalarExpr::Column(c) => ScalarExpr::Column(c.clone()),
            ScalarExpr::Literal(l) => ScalarExpr::Literal(l.clone()),
            ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
                left: Box::new(self.visit(left)),
                op: *op,
                right: Box::new(self.visit(right)),
            },
            ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(self.visit(e))),
            ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
                expr: Box::new(self.visit(expr)),
                negated: *negated,
            },
            ScalarExpr::Aggregate(call) => {
                let (arg_expr, distinct, func) = match call {
                    AggregateCall::CountStar => {
                        let alias = format!("_agg_res_{}", self.aggregates.len());
                        self.aggregates.push(AggregateExpr::CountStar {
                            alias: alias.clone(),
                            distinct: false,
                        });
                        return ScalarExpr::Column(alias);
                    }
                    AggregateCall::Count { expr, distinct } => {
                        (expr, *distinct, AggregateFunction::Count)
                    }
                    AggregateCall::Sum { expr, distinct } => {
                        (expr, *distinct, AggregateFunction::SumInt64)
                    }
                    AggregateCall::Total { expr, distinct } => {
                        (expr, *distinct, AggregateFunction::TotalInt64)
                    }
                    AggregateCall::Min(expr) => (expr, false, AggregateFunction::MinInt64),
                    AggregateCall::Max(expr) => (expr, false, AggregateFunction::MaxInt64),
                    AggregateCall::CountNulls(expr) => (expr, false, AggregateFunction::CountNulls),
                    AggregateCall::GroupConcat { expr, distinct, separator: _ } => {
                        (expr, *distinct, AggregateFunction::GroupConcat)
                    }
                    AggregateCall::Avg { expr, distinct } => {
                        let arg_idx = self.pre_agg_projections.len();
                        self.pre_agg_projections.push(*expr.clone());
                        let arg_col_name = format!("_agg_arg_{}", arg_idx);

                        let sum_alias = format!("_agg_res_{}", self.aggregates.len());
                        self.aggregates.push(AggregateExpr::Column {
                            column: arg_col_name.clone(),
                            alias: sum_alias.clone(),
                            function: AggregateFunction::SumInt64,
                            distinct: *distinct,
                        });

                        let count_alias = format!("_agg_res_{}", self.aggregates.len());
                        self.aggregates.push(AggregateExpr::Column {
                            column: arg_col_name,
                            alias: count_alias.clone(),
                            function: AggregateFunction::Count,
                            distinct: *distinct,
                        });

                        return ScalarExpr::Binary {
                            left: Box::new(ScalarExpr::Column(sum_alias)),
                            op: llkv_expr::expr::BinaryOp::Divide,
                            right: Box::new(ScalarExpr::Column(count_alias)),
                        };
                    }
                };

                let arg_idx = self.pre_agg_projections.len();
                self.pre_agg_projections.push(*arg_expr.clone());
                let arg_col_name = format!("_agg_arg_{}", arg_idx);

                let alias = format!("_agg_res_{}", self.aggregates.len());
                self.aggregates.push(AggregateExpr::Column {
                    column: arg_col_name,
                    alias: alias.clone(),
                    function: func,
                    distinct,
                });

                ScalarExpr::Column(alias)
            }
            ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
                base: Box::new(self.visit(base)),
                field_name: field_name.clone(),
            },
            ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
                expr: Box::new(self.visit(expr)),
                data_type: data_type.clone(),
            },
            ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
                left: Box::new(self.visit(left)),
                op: *op,
                right: Box::new(self.visit(right)),
            },
            ScalarExpr::Coalesce(exprs) => ScalarExpr::Coalesce(exprs.iter().map(|e| self.visit(e)).collect()),
            ScalarExpr::ScalarSubquery(s) => ScalarExpr::ScalarSubquery(s.clone()),
            ScalarExpr::Case { operand, branches, else_expr } => ScalarExpr::Case {
                operand: operand.as_ref().map(|e| Box::new(self.visit(e))),
                branches: branches.iter().map(|(w, t)| (self.visit(w), self.visit(t))).collect(),
                else_expr: else_expr.as_ref().map(|e| Box::new(self.visit(e))),
            },
            ScalarExpr::Random => ScalarExpr::Random,
        }
    }
}

/// Rewrite projections and optional additional expressions (e.g., HAVING) so
/// aggregates are extracted and replaced with column references to aggregate
/// outputs. Returns the aggregate specs, rewritten projections, pre-agg
/// expressions, and rewritten additional expressions.
pub fn extract_complex_aggregates(
    projections: &[ScalarExpr<String>],
    additional_exprs: &[ScalarExpr<String>],
) -> (
    Vec<AggregateExpr>,
    Vec<ScalarExpr<String>>,
    Vec<ScalarExpr<String>>,
    Vec<ScalarExpr<String>>,
) {
    let mut visitor = AggVisitor::new();
    let rewritten = projections.iter().map(|p| visitor.visit(p)).collect();
    let additional_rewritten = additional_exprs
        .iter()
        .map(|e| visitor.visit(e))
        .collect();
    (
        visitor.aggregates,
        rewritten,
        visitor.pre_agg_projections,
        additional_rewritten,
    )
}

pub(crate) fn expr_to_scalar_expr(expr: &Expr<'static, String>) -> ScalarExpr<String> {
    match expr {
        Expr::And(list) => {
            if list.is_empty() {
                return ScalarExpr::Literal(Literal::Boolean(true));
            }
            let mut iter = list.iter();
            let first = expr_to_scalar_expr(iter.next().unwrap());
            iter.fold(first, |acc, e| ScalarExpr::Binary {
                left: Box::new(acc),
                op: BinaryOp::And,
                right: Box::new(expr_to_scalar_expr(e)),
            })
        }
        Expr::Or(list) => {
            if list.is_empty() {
                return ScalarExpr::Literal(Literal::Boolean(false));
            }
            let mut iter = list.iter();
            let first = expr_to_scalar_expr(iter.next().unwrap());
            iter.fold(first, |acc, e| ScalarExpr::Binary {
                left: Box::new(acc),
                op: BinaryOp::Or,
                right: Box::new(expr_to_scalar_expr(e)),
            })
        }
        Expr::Not(e) => ScalarExpr::Not(Box::new(expr_to_scalar_expr(e))),
        Expr::Pred(f) => {
            let col = ScalarExpr::Column(f.field_id.clone());
            match &f.op {
                Operator::Equals(l) => ScalarExpr::Compare {
                    left: Box::new(col),
                    op: CompareOp::Eq,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                },
                Operator::GreaterThan(l) => ScalarExpr::Compare {
                    left: Box::new(col),
                    op: CompareOp::Gt,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                },
                Operator::GreaterThanOrEquals(l) => ScalarExpr::Compare {
                    left: Box::new(col),
                    op: CompareOp::GtEq,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                },
                Operator::LessThan(l) => ScalarExpr::Compare {
                    left: Box::new(col),
                    op: CompareOp::Lt,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                },
                Operator::LessThanOrEquals(l) => ScalarExpr::Compare {
                    left: Box::new(col),
                    op: CompareOp::LtEq,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                },
                Operator::IsNull => ScalarExpr::IsNull {
                    expr: Box::new(col),
                    negated: false,
                },
                Operator::IsNotNull => ScalarExpr::IsNull {
                    expr: Box::new(col),
                    negated: true,
                },
                Operator::In(list) => {
                    if list.is_empty() {
                        return ScalarExpr::Literal(Literal::Boolean(false));
                    }
                    let mut iter = list.iter();
                    let first = ScalarExpr::Compare {
                        left: Box::new(col.clone()),
                        op: CompareOp::Eq,
                        right: Box::new(ScalarExpr::Literal(iter.next().unwrap().clone())),
                    };
                    iter.fold(first, |acc, l| ScalarExpr::Binary {
                        left: Box::new(acc),
                        op: BinaryOp::Or,
                        right: Box::new(ScalarExpr::Compare {
                            left: Box::new(col.clone()),
                            op: CompareOp::Eq,
                            right: Box::new(ScalarExpr::Literal(l.clone())),
                        }),
                    })
                }
                _ => ScalarExpr::Literal(Literal::Boolean(true)),
            }
        }
        Expr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(left.clone()),
            op: *op,
            right: Box::new(right.clone()),
        },
        Expr::InList { expr, list, negated } => {
            if list.is_empty() {
                return ScalarExpr::Literal(Literal::Boolean(*negated));
            }
            let mut iter = list.iter();
            let first = ScalarExpr::Compare {
                left: Box::new(expr.clone()),
                op: CompareOp::Eq,
                right: Box::new(iter.next().unwrap().clone()),
            };
            let combined = iter.fold(first, |acc, l| ScalarExpr::Binary {
                left: Box::new(acc),
                op: BinaryOp::Or,
                right: Box::new(ScalarExpr::Compare {
                    left: Box::new(expr.clone()),
                    op: CompareOp::Eq,
                    right: Box::new(l.clone()),
                }),
            });
            if *negated {
                ScalarExpr::Not(Box::new(combined))
            } else {
                combined
            }
        }
        Expr::IsNull { expr, negated } => ScalarExpr::IsNull {
            expr: Box::new(expr.clone()),
            negated: *negated,
        },
        Expr::Literal(b) => ScalarExpr::Literal(Literal::Boolean(*b)),
        Expr::Exists(_) => ScalarExpr::Literal(Literal::Boolean(false)),
    }
}

fn build_single_projection_exprs(
    plan: &SelectPlan,
    schema: &PlanSchema,
) -> Vec<(ScalarExpr<String>, String)> {
    let mut out = Vec::new();
    for proj in &plan.projections {
        match proj {
            SelectProjection::AllColumns => {
                for col in schema.columns.iter() {
                    out.push((ScalarExpr::Column(col.name.clone()), col.name.clone()));
                }
            }
            SelectProjection::AllColumnsExcept { exclude } => {
                for col in schema.columns.iter() {
                    if !exclude.contains(&col.name) {
                        out.push((ScalarExpr::Column(col.name.clone()), col.name.clone()));
                    }
                }
            }
            SelectProjection::Column { name, alias } => {
                let final_name = alias.clone().unwrap_or_else(|| name.clone());
                out.push((ScalarExpr::Column(name.clone()), final_name));
            }
            SelectProjection::Computed { expr, alias } => {
                out.push((expr.clone(), alias.clone()));
            }
        }
    }
    out
}

/// Build aggregate rewrite metadata for single-table plans using resolved schema names.
pub fn build_single_aggregate_rewrite(
    plan: &SelectPlan,
    schema: &PlanSchema,
) -> Result<Option<AggregateRewrite>> {
    let projections = build_single_projection_exprs(plan, schema);
    let (projection_exprs, final_names): (Vec<_>, Vec<_>) = projections.into_iter().unzip();

    let having_exprs: Vec<ScalarExpr<String>> = plan
        .having
        .as_ref()
        .map(|h| vec![expr_to_scalar_expr(h)])
        .unwrap_or_default();

    let (aggregates, final_exprs, pre_agg_exprs, rewritten_having) =
        extract_complex_aggregates(&projection_exprs, &having_exprs);

    if aggregates.is_empty() {
        return Ok(None);
    }

    Ok(Some(AggregateRewrite {
        aggregates,
        final_expressions: final_exprs,
        pre_aggregate_expressions: pre_agg_exprs,
        final_names,
        rewritten_having: rewritten_having.into_iter().next(),
    }))
}
