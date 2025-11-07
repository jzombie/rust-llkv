//! Bytecode-style programs for predicate evaluation and domain analysis.

use std::ops::Bound;
use std::sync::Arc;

use llkv_expr::{CompareOp, Expr, Filter, Operator, ScalarExpr, literal::Literal};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::types::FieldId;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ExprKey(*const ());

impl ExprKey {
    #[inline]
    pub(crate) fn new(expr: &Expr<'_, FieldId>) -> Self {
        Self(expr as *const _ as *const ())
    }
}

#[derive(Debug)]
pub(crate) struct ProgramSet<'expr> {
    pub(crate) eval: EvalProgram,
    pub(crate) domains: DomainRegistry,
    _root_expr: Arc<Expr<'expr, FieldId>>,
}

impl<'expr> ProgramSet<'expr> {}

#[derive(Debug, Default)]
pub(crate) struct EvalProgram {
    pub(crate) ops: Vec<EvalOp>,
}

#[derive(Debug)]
pub(crate) enum EvalOp {
    PushPredicate(OwnedFilter),
    PushCompare {
        left: ScalarExpr<FieldId>,
        right: ScalarExpr<FieldId>,
        op: CompareOp,
    },
    PushInList {
        expr: ScalarExpr<FieldId>,
        list: Vec<ScalarExpr<FieldId>>,
        negated: bool,
    },
    PushIsNull {
        expr: ScalarExpr<FieldId>,
        negated: bool,
    },
    PushLiteral(bool),
    FusedAnd {
        field_id: FieldId,
        filters: Vec<OwnedFilter>,
    },
    And {
        child_count: usize,
    },
    Or {
        child_count: usize,
    },
    Not {
        domain: DomainProgramId,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct OwnedFilter {
    pub(crate) field_id: FieldId,
    pub(crate) op: OwnedOperator,
}

#[derive(Debug, Clone)]
pub(crate) enum OwnedOperator {
    Equals(Literal),
    Range {
        lower: Bound<Literal>,
        upper: Bound<Literal>,
    },
    GreaterThan(Literal),
    GreaterThanOrEquals(Literal),
    LessThan(Literal),
    LessThanOrEquals(Literal),
    In(Vec<Literal>),
    StartsWith {
        pattern: String,
        case_sensitive: bool,
    },
    EndsWith {
        pattern: String,
        case_sensitive: bool,
    },
    Contains {
        pattern: String,
        case_sensitive: bool,
    },
    IsNull,
    IsNotNull,
}

impl OwnedOperator {
    pub(crate) fn to_operator(&self) -> Operator<'_> {
        match self {
            Self::Equals(lit) => Operator::Equals(lit.clone()),
            Self::Range { lower, upper } => Operator::Range {
                lower: lower.clone(),
                upper: upper.clone(),
            },
            Self::GreaterThan(lit) => Operator::GreaterThan(lit.clone()),
            Self::GreaterThanOrEquals(lit) => Operator::GreaterThanOrEquals(lit.clone()),
            Self::LessThan(lit) => Operator::LessThan(lit.clone()),
            Self::LessThanOrEquals(lit) => Operator::LessThanOrEquals(lit.clone()),
            Self::In(values) => Operator::In(values.as_slice()),
            Self::StartsWith {
                pattern,
                case_sensitive,
            } => Operator::StartsWith {
                pattern: pattern.as_str(),
                case_sensitive: *case_sensitive,
            },
            Self::EndsWith {
                pattern,
                case_sensitive,
            } => Operator::EndsWith {
                pattern: pattern.as_str(),
                case_sensitive: *case_sensitive,
            },
            Self::Contains {
                pattern,
                case_sensitive,
            } => Operator::Contains {
                pattern: pattern.as_str(),
                case_sensitive: *case_sensitive,
            },
            Self::IsNull => Operator::IsNull,
            Self::IsNotNull => Operator::IsNotNull,
        }
    }
}

impl OwnedFilter {}

impl<'a> From<&'a Operator<'a>> for OwnedOperator {
    fn from(op: &'a Operator<'a>) -> Self {
        match op {
            Operator::Equals(lit) => Self::Equals(lit.clone()),
            Operator::Range { lower, upper } => Self::Range {
                lower: lower.clone(),
                upper: upper.clone(),
            },
            Operator::GreaterThan(lit) => Self::GreaterThan(lit.clone()),
            Operator::GreaterThanOrEquals(lit) => Self::GreaterThanOrEquals(lit.clone()),
            Operator::LessThan(lit) => Self::LessThan(lit.clone()),
            Operator::LessThanOrEquals(lit) => Self::LessThanOrEquals(lit.clone()),
            Operator::In(values) => Self::In((*values).to_vec()),
            Operator::StartsWith {
                pattern,
                case_sensitive,
            } => Self::StartsWith {
                pattern: (*pattern).to_string(),
                case_sensitive: *case_sensitive,
            },
            Operator::EndsWith {
                pattern,
                case_sensitive,
            } => Self::EndsWith {
                pattern: (*pattern).to_string(),
                case_sensitive: *case_sensitive,
            },
            Operator::Contains {
                pattern,
                case_sensitive,
            } => Self::Contains {
                pattern: (*pattern).to_string(),
                case_sensitive: *case_sensitive,
            },
            Operator::IsNull => Self::IsNull,
            Operator::IsNotNull => Self::IsNotNull,
        }
    }
}

impl<'a> From<&'a Filter<'a, FieldId>> for OwnedFilter {
    fn from(filter: &'a Filter<'a, FieldId>) -> Self {
        Self {
            field_id: filter.field_id,
            op: OwnedOperator::from(&filter.op),
        }
    }
}

pub(crate) type DomainProgramId = u32;

#[derive(Debug, Default)]
pub(crate) struct DomainRegistry {
    programs: Vec<DomainProgram>,
    index: FxHashMap<ExprKey, DomainProgramId>,
    root: Option<DomainProgramId>,
}

impl DomainRegistry {
    pub(crate) fn domain(&self, id: DomainProgramId) -> Option<&DomainProgram> {
        self.programs.get(id as usize)
    }

    fn ensure(&mut self, expr: &Expr<'_, FieldId>) -> DomainProgramId {
        let key = ExprKey::new(expr);
        if let Some(existing) = self.index.get(&key) {
            return *existing;
        }
        let id = self.programs.len() as DomainProgramId;
        let program = compile_domain(expr);
        self.programs.push(program);
        self.index.insert(key, id);
        id
    }
}

#[derive(Debug, Default)]
pub(crate) struct DomainProgram {
    pub(crate) ops: Vec<DomainOp>,
}

#[derive(Debug)]
pub(crate) enum DomainOp {
    PushFieldAll(FieldId),
    PushCompareDomain {
        left: ScalarExpr<FieldId>,
        right: ScalarExpr<FieldId>,
        op: CompareOp,
        fields: Vec<FieldId>,
    },
    PushInListDomain {
        expr: ScalarExpr<FieldId>,
        list: Vec<ScalarExpr<FieldId>>,
        fields: Vec<FieldId>,
        negated: bool,
    },
    PushIsNullDomain {
        expr: ScalarExpr<FieldId>,
        fields: Vec<FieldId>,
        negated: bool,
    },
    PushLiteralFalse,
    PushAllRows,
    Union {
        child_count: usize,
    },
    Intersect {
        child_count: usize,
    },
}

#[derive(Debug)]
pub(crate) struct ProgramCompiler<'expr> {
    root: Arc<Expr<'expr, FieldId>>,
}

impl<'expr> ProgramCompiler<'expr> {
    pub(crate) fn new(root: Arc<Expr<'expr, FieldId>>) -> Self {
        Self { root }
    }

    pub(crate) fn compile(self) -> LlkvResult<ProgramSet<'expr>> {
        let ProgramCompiler { root } = self;
        let mut domains = DomainRegistry::default();
        let eval_ops = {
            let root_ref = root.as_ref();
            compile_eval(root_ref, &mut domains)?
        };
        let root_domain = {
            let root_ref = root.as_ref();
            domains.ensure(root_ref)
        };
        domains.root = Some(root_domain);
        Ok(ProgramSet {
            eval: EvalProgram { ops: eval_ops },
            domains,
            _root_expr: root,
        })
    }
}

pub(crate) fn normalize_predicate<'expr>(expr: Expr<'expr, FieldId>) -> Expr<'expr, FieldId> {
    normalize_expr(expr)
}

fn normalize_expr<'expr>(expr: Expr<'expr, FieldId>) -> Expr<'expr, FieldId> {
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
        other => other,
    }
}

fn normalize_negated<'expr>(inner: Expr<'expr, FieldId>) -> Expr<'expr, FieldId> {
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
    use llkv_expr::literal::Literal;

    #[test]
    fn normalize_not_between_expands_to_or() {
        let field: FieldId = 7;
        let column = ScalarExpr::column(field);
        let lower = ScalarExpr::literal(5_i64);
        let upper = ScalarExpr::literal(Literal::Null);

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
            Expr::Not(inner) => match inner.as_ref() {
                Expr::Compare {
                    op: CompareOp::GtEq,
                    ..
                } => {}
                other => panic!("unexpected left branch: {other:?}"),
            },
            other => panic!("left branch should be NOT(compare), got {other:?}"),
        }

        match &children[1] {
            Expr::Not(inner) => match inner.as_ref() {
                Expr::Compare {
                    op: CompareOp::LtEq,
                    ..
                } => {}
                other => panic!("unexpected right branch: {other:?}"),
            },
            other => panic!("right branch should be NOT(compare), got {other:?}"),
        }
    }

    #[test]
    fn normalize_flips_literal_bool() {
        let normalized = normalize_predicate(Expr::Not(Box::new(Expr::Literal(true))));
        assert!(matches!(normalized, Expr::Literal(false)));
    }
}

#[derive(Clone, Copy)]
enum EvalVisit<'expr> {
    Enter(&'expr Expr<'expr, FieldId>),
    Exit(&'expr Expr<'expr, FieldId>),
    EmitFused { key: ExprKey, field_id: FieldId },
}

type PredicateVec = Vec<OwnedFilter>;

fn compile_eval<'expr>(
    expr: &'expr Expr<'expr, FieldId>,
    domains: &mut DomainRegistry,
) -> LlkvResult<Vec<EvalOp>> {
    let mut ops = Vec::new();
    let mut stack = vec![EvalVisit::Enter(expr)];
    let mut fused: FxHashMap<ExprKey, PredicateVec> = FxHashMap::default();

    while let Some(frame) = stack.pop() {
        match frame {
            EvalVisit::Enter(node) => match node {
                Expr::And(children) => {
                    if children.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "AND expression requires at least one predicate".into(),
                        ));
                    }
                    if let Some((field_id, filters)) = gather_fused(children) {
                        let key = ExprKey::new(node);
                        fused.insert(key, filters);
                        stack.push(EvalVisit::EmitFused { key, field_id });
                    } else {
                        stack.push(EvalVisit::Exit(node));
                        for child in children.iter().rev() {
                            stack.push(EvalVisit::Enter(child));
                        }
                    }
                }
                Expr::Or(children) => {
                    if children.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "OR expression requires at least one predicate".into(),
                        ));
                    }
                    stack.push(EvalVisit::Exit(node));
                    for child in children.iter().rev() {
                        stack.push(EvalVisit::Enter(child));
                    }
                }
                Expr::Not(inner) => {
                    stack.push(EvalVisit::Exit(node));
                    stack.push(EvalVisit::Enter(inner));
                }
                _ => stack.push(EvalVisit::Exit(node)),
            },
            EvalVisit::Exit(node) => match node {
                Expr::Pred(filter) => {
                    ops.push(EvalOp::PushPredicate(OwnedFilter::from(filter)));
                }
                Expr::Compare { left, op, right } => {
                    ops.push(EvalOp::PushCompare {
                        left: left.clone(),
                        right: right.clone(),
                        op: *op,
                    });
                }
                Expr::InList {
                    expr,
                    list,
                    negated,
                } => {
                    ops.push(EvalOp::PushInList {
                        expr: expr.clone(),
                        list: list.clone(),
                        negated: *negated,
                    });
                }
                Expr::IsNull { expr, negated } => {
                    ops.push(EvalOp::PushIsNull {
                        expr: expr.clone(),
                        negated: *negated,
                    });
                }
                Expr::Literal(value) => ops.push(EvalOp::PushLiteral(*value)),
                Expr::And(children) => ops.push(EvalOp::And {
                    child_count: children.len(),
                }),
                Expr::Or(children) => ops.push(EvalOp::Or {
                    child_count: children.len(),
                }),
                Expr::Not(inner) => {
                    let id = domains.ensure(inner);
                    ops.push(EvalOp::Not { domain: id });
                }
                Expr::Exists(_) => {
                    return Err(Error::InvalidArgumentError(
                        "EXISTS predicates are not supported in storage evaluation".into(),
                    ));
                }
            },
            EvalVisit::EmitFused { key, field_id } => {
                let filters = fused
                    .remove(&key)
                    .ok_or_else(|| Error::Internal("missing fused predicate metadata".into()))?;
                ops.push(EvalOp::FusedAnd { field_id, filters });
            }
        }
    }

    Ok(ops)
}

fn gather_fused<'expr>(
    children: &'expr [Expr<'expr, FieldId>],
) -> Option<(FieldId, Vec<OwnedFilter>)> {
    if children.is_empty() {
        return None;
    }
    let mut field: Option<FieldId> = None;
    let mut out: Vec<OwnedFilter> = Vec::with_capacity(children.len());
    for child in children {
        match child {
            Expr::Pred(filter) => {
                if let Some(expected) = field {
                    if expected != filter.field_id {
                        return None;
                    }
                } else {
                    field = Some(filter.field_id);
                }
                out.push(OwnedFilter::from(filter));
            }
            _ => return None,
        }
    }
    field.map(|fid| (fid, out))
}

#[derive(Clone, Copy)]
enum DomainVisit<'expr> {
    Enter(&'expr Expr<'expr, FieldId>),
    Exit(&'expr Expr<'expr, FieldId>),
}

fn compile_domain(expr: &Expr<'_, FieldId>) -> DomainProgram {
    let mut ops = Vec::new();
    let mut stack = vec![DomainVisit::Enter(expr)];

    while let Some(frame) = stack.pop() {
        match frame {
            DomainVisit::Enter(node) => match node {
                Expr::And(children) | Expr::Or(children) => {
                    stack.push(DomainVisit::Exit(node));
                    for child in children.iter().rev() {
                        stack.push(DomainVisit::Enter(child));
                    }
                }
                Expr::Not(inner) => {
                    stack.push(DomainVisit::Exit(node));
                    stack.push(DomainVisit::Enter(inner));
                }
                _ => stack.push(DomainVisit::Exit(node)),
            },
            DomainVisit::Exit(node) => match node {
                Expr::Pred(filter) => ops.push(DomainOp::PushFieldAll(filter.field_id)),
                Expr::Compare { left, op, right } => ops.push(DomainOp::PushCompareDomain {
                    left: left.clone(),
                    right: right.clone(),
                    op: *op,
                    fields: collect_fields([left, right]),
                }),
                Expr::InList {
                    expr,
                    list,
                    negated,
                } => {
                    let mut exprs: Vec<&ScalarExpr<FieldId>> = Vec::with_capacity(list.len() + 1);
                    exprs.push(expr);
                    exprs.extend(list.iter());
                    ops.push(DomainOp::PushInListDomain {
                        expr: expr.clone(),
                        list: list.clone(),
                        fields: collect_fields(exprs),
                        negated: *negated,
                    });
                }
                Expr::IsNull { expr, negated } => {
                    ops.push(DomainOp::PushIsNullDomain {
                        expr: expr.clone(),
                        fields: collect_fields([expr]),
                        negated: *negated,
                    });
                }
                Expr::Literal(value) => {
                    if *value {
                        ops.push(DomainOp::PushAllRows);
                    } else {
                        ops.push(DomainOp::PushLiteralFalse);
                    }
                }
                Expr::And(children) => {
                    if children.len() > 1 {
                        ops.push(DomainOp::Intersect {
                            child_count: children.len(),
                        });
                    }
                }
                Expr::Or(children) => {
                    if children.len() > 1 {
                        ops.push(DomainOp::Union {
                            child_count: children.len(),
                        });
                    }
                }
                Expr::Not(_) => {
                    // Domain equals child domain; no-op.
                }
                Expr::Exists(_) => {
                    panic!("EXISTS predicates should not reach storage domain evaluation stage");
                }
            },
        }
    }

    DomainProgram { ops }
}

fn collect_fields<'expr>(
    exprs: impl IntoIterator<Item = &'expr ScalarExpr<FieldId>>,
) -> Vec<FieldId> {
    let mut seen: FxHashSet<FieldId> = FxHashSet::default();
    let mut stack: Vec<&'expr ScalarExpr<FieldId>> = exprs.into_iter().collect();
    while let Some(expr) = stack.pop() {
        match expr {
            ScalarExpr::Column(fid) => {
                seen.insert(*fid);
            }
            ScalarExpr::Literal(_) => {}
            ScalarExpr::Binary { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            ScalarExpr::Compare { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            ScalarExpr::Aggregate(agg) => match agg {
                llkv_expr::expr::AggregateCall::CountStar => {}
                llkv_expr::expr::AggregateCall::Count { expr, .. }
                | llkv_expr::expr::AggregateCall::Sum { expr, .. }
                | llkv_expr::expr::AggregateCall::Total { expr, .. }
                | llkv_expr::expr::AggregateCall::Avg { expr, .. }
                | llkv_expr::expr::AggregateCall::Min(expr)
                | llkv_expr::expr::AggregateCall::Max(expr)
                | llkv_expr::expr::AggregateCall::CountNulls(expr)
                | llkv_expr::expr::AggregateCall::GroupConcat { expr, .. } => {
                    stack.push(expr.as_ref());
                }
            },
            ScalarExpr::GetField { base, .. } => {
                stack.push(base);
            }
            ScalarExpr::Cast { expr, .. } => {
                stack.push(expr.as_ref());
            }
            ScalarExpr::Not(expr) => {
                stack.push(expr.as_ref());
            }
            ScalarExpr::IsNull { expr, .. } => {
                stack.push(expr.as_ref());
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                if let Some(inner) = operand.as_deref() {
                    stack.push(inner);
                }
                for (when_expr, then_expr) in branches {
                    stack.push(when_expr);
                    stack.push(then_expr);
                }
                if let Some(inner) = else_expr.as_deref() {
                    stack.push(inner);
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    stack.push(item);
                }
            }
            ScalarExpr::Random => {
                // Random does not reference any fields
            }
            ScalarExpr::ScalarSubquery(_) => {
                // Scalar subqueries are resolved separately at planning time
            }
        }
    }
    let mut fields: Vec<FieldId> = seen.into_iter().collect();
    fields.sort_unstable();
    fields
}
