use crate::eval::ScalarExprTypeExt;
use arrow::datatypes::DataType;
use llkv_expr::literal::Literal;
use llkv_expr::{Expr, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::{FieldId, LogicalFieldId, TableId};
use rustc_hash::FxHashMap;

pub fn computed_expr_requires_numeric(expr: &ScalarExpr<FieldId>) -> bool {
    match expr {
        ScalarExpr::Literal(_) => false,
        ScalarExpr::Column(_) => true,
        ScalarExpr::Binary { .. } => true,
        ScalarExpr::Compare { .. } => true,
        ScalarExpr::Aggregate(_) => false, // Aggregates are computed separately
        ScalarExpr::GetField { .. } => false, // GetField requires raw arrays, not numeric conversion
        ScalarExpr::Cast { expr, .. } => computed_expr_requires_numeric(expr),
        ScalarExpr::Not(expr) => computed_expr_requires_numeric(expr),
        ScalarExpr::IsNull { expr, .. } => computed_expr_requires_numeric(expr),
        ScalarExpr::Case { .. } => true,
        ScalarExpr::Coalesce(items) => items.iter().any(computed_expr_requires_numeric),
        ScalarExpr::Random => true,
        ScalarExpr::ScalarSubquery(_) => false,
    }
}

pub fn computed_expr_prefers_float(
    expr: &ScalarExpr<FieldId>,
    table_id: TableId,
    lfid_dtypes: &FxHashMap<LogicalFieldId, DataType>,
) -> LlkvResult<bool> {
    match expr {
        ScalarExpr::Literal(lit) => literal_prefers_float(lit),
        ScalarExpr::Column(fid) => {
            let lfid = LogicalFieldId::for_user(table_id, *fid);
            let dtype = lfid_dtypes
                .get(&lfid)
                .ok_or_else(|| Error::Internal("missing dtype for computed column".into()))?;
            Ok(matches!(
                dtype,
                DataType::Float16 | DataType::Float32 | DataType::Float64
            ))
        }
        ScalarExpr::Binary { left, right, .. } => {
            Ok(
                computed_expr_prefers_float(left.as_ref(), table_id, lfid_dtypes)?
                    || computed_expr_prefers_float(right.as_ref(), table_id, lfid_dtypes)?,
            )
        }
        ScalarExpr::Compare { .. } => Ok(false),
        ScalarExpr::Aggregate(_) => Ok(false),
        ScalarExpr::GetField { base, field_name } => {
            let dtype = get_field_dtype(base.as_ref(), field_name, table_id, lfid_dtypes)?;
            Ok(matches!(
                dtype,
                DataType::Float16 | DataType::Float32 | DataType::Float64
            ))
        }
        ScalarExpr::Cast { expr, data_type } => {
            if matches!(
                data_type,
                DataType::Float16 | DataType::Float32 | DataType::Float64
            ) {
                Ok(true)
            } else {
                computed_expr_prefers_float(expr.as_ref(), table_id, lfid_dtypes)
            }
        }
        ScalarExpr::Not(expr) => computed_expr_prefers_float(expr.as_ref(), table_id, lfid_dtypes),

        ScalarExpr::IsNull { expr, .. } => {
            let _ = computed_expr_prefers_float(expr.as_ref(), table_id, lfid_dtypes)?;
            Ok(false)
        }
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(inner) = operand.as_deref()
                && computed_expr_prefers_float(inner, table_id, lfid_dtypes)?
            {
                return Ok(true);
            }
            for (when_expr, then_expr) in branches {
                if computed_expr_prefers_float(when_expr, table_id, lfid_dtypes)?
                    || computed_expr_prefers_float(then_expr, table_id, lfid_dtypes)?
                {
                    return Ok(true);
                }
            }
            if let Some(inner) = else_expr.as_deref()
                && computed_expr_prefers_float(inner, table_id, lfid_dtypes)?
            {
                return Ok(true);
            }
            Ok(false)
        }
        ScalarExpr::Coalesce(items) => {
            for item in items {
                if computed_expr_prefers_float(item, table_id, lfid_dtypes)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        ScalarExpr::Random => Ok(true),
        ScalarExpr::ScalarSubquery(_) => Ok(false),
    }
}

pub fn scalar_expr_contains_coalesce(expr: &ScalarExpr<FieldId>) -> bool {
    match expr {
        ScalarExpr::Coalesce(_) => true,
        ScalarExpr::Binary { left, right, .. } | ScalarExpr::Compare { left, right, .. } => {
            scalar_expr_contains_coalesce(left) || scalar_expr_contains_coalesce(right)
        }
        ScalarExpr::Not(expr) => scalar_expr_contains_coalesce(expr),
        ScalarExpr::IsNull { expr, .. } => scalar_expr_contains_coalesce(expr),
        ScalarExpr::Cast { expr, .. } => scalar_expr_contains_coalesce(expr),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            operand
                .as_deref()
                .map(scalar_expr_contains_coalesce)
                .unwrap_or(false)
                || branches.iter().any(|(when_expr, then_expr)| {
                    scalar_expr_contains_coalesce(when_expr)
                        || scalar_expr_contains_coalesce(then_expr)
                })
                || else_expr
                    .as_deref()
                    .map(scalar_expr_contains_coalesce)
                    .unwrap_or(false)
        }
        ScalarExpr::GetField { base, .. } => scalar_expr_contains_coalesce(base),
        ScalarExpr::Aggregate(_)
        | ScalarExpr::Column(_)
        | ScalarExpr::Literal(_)
        | ScalarExpr::Random
        | ScalarExpr::ScalarSubquery(_) => false,
    }
}

pub fn literal_prefers_float(literal: &Literal) -> LlkvResult<bool> {
    match literal {
        Literal::Float64(_) => Ok(true),
        Literal::Decimal128(_) => Ok(true),
        Literal::Struct(fields) => {
            for (_, nested) in fields {
                if literal_prefers_float(nested.as_ref())? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Literal::Int128(_) | Literal::Boolean(_) | Literal::String(_) | Literal::Null => Ok(false),
        Literal::Date32(_) => Ok(false),
        Literal::Interval(_) => Ok(false),
    }
}

pub fn get_field_dtype(
    expr: &ScalarExpr<FieldId>,
    field_name: &str,
    table_id: TableId,
    lfid_dtypes: &FxHashMap<LogicalFieldId, DataType>,
) -> LlkvResult<DataType> {
    let base_dtype = match expr {
        ScalarExpr::Column(fid) => {
            let lfid = LogicalFieldId::for_user(table_id, *fid);
            lfid_dtypes
                .get(&lfid)
                .cloned()
                .ok_or_else(|| Error::Internal("missing dtype for computed column".into()))?
        }
        ScalarExpr::GetField {
            base: inner_base,
            field_name: inner_field,
        } => get_field_dtype(inner_base.as_ref(), inner_field, table_id, lfid_dtypes)?,
        _ => {
            return Err(Error::InvalidArgumentError(
                "GetField base must be a column or another GetField".into(),
            ));
        }
    };

    if let DataType::Struct(fields) = base_dtype {
        fields
            .iter()
            .find(|f| f.name() == field_name)
            .map(|f| f.data_type().clone())
            .ok_or_else(|| {
                Error::InvalidArgumentError(format!("Field '{}' not found in struct", field_name))
            })
    } else {
        Err(Error::InvalidArgumentError(
            "GetField can only be applied to struct types".into(),
        ))
    }
}

/// Infer the result type for a scalar expression given table-level field dtypes.
pub fn computed_expr_result_type(
    expr: &ScalarExpr<FieldId>,
    table_id: TableId,
    lfid_dtypes: &FxHashMap<LogicalFieldId, DataType>,
) -> LlkvResult<Option<DataType>> {
    let mut resolver = |fid: FieldId| {
        let lfid = LogicalFieldId::for_user(table_id, fid);
        lfid_dtypes.get(&lfid).cloned()
    };
    Ok(expr.infer_result_type(&mut resolver))
}

#[derive(Default, Clone, Copy)]
struct FieldPredicateStats {
    total: usize,
    contains: usize,
}

/// Lightweight cache used to decide when to fuse predicates for a field.
#[derive(Default)]
pub struct PredicateFusionCache {
    per_field: FxHashMap<FieldId, FieldPredicateStats>,
}

impl PredicateFusionCache {
    pub fn from_expr(expr: &Expr<'_, FieldId>) -> Self {
        let mut cache = Self::default();
        cache.record_expr(expr);
        cache
    }

    fn record_expr(&mut self, expr: &Expr<'_, FieldId>) {
        let mut stack = vec![expr];

        while let Some(node) = stack.pop() {
            match node {
                Expr::Pred(filter) => {
                    let entry = self.per_field.entry(filter.field_id).or_default();
                    entry.total += 1;
                    if matches!(filter.op, llkv_expr::Operator::Contains { .. }) {
                        entry.contains += 1;
                    }
                }
                Expr::And(children) | Expr::Or(children) => {
                    for child in children {
                        stack.push(child);
                    }
                }
                Expr::Not(inner) => stack.push(inner),
                Expr::Compare { .. }
                | Expr::InList { .. }
                | Expr::IsNull { .. }
                | Expr::Literal(_)
                | Expr::Exists(_) => {}
            }
        }
    }

    pub fn should_fuse(&self, field_id: FieldId, dtype: &DataType) -> bool {
        let Some(stats) = self.per_field.get(&field_id) else {
            return false;
        };

        match dtype {
            DataType::Utf8 | DataType::LargeUtf8 => stats.contains >= 1 && stats.total >= 2,
            _ => stats.total >= 2,
        }
    }
}
