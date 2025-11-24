use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::DataType;
use llkv_expr::literal::Literal;
use llkv_expr::{AggregateCall, BinaryOp, CompareOp, DecimalValue, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::array::NumericArray;
use crate::date::{add_interval_to_date32, parse_date32_literal, subtract_interval_from_date32};
use crate::kernels::{compute_binary, compute_binary_scalar};
use crate::numeric::{NumericKind, NumericValue};

/// Mapping from field identifiers to the numeric Arrow array used for evaluation.
pub type NumericArrayMap<F> = FxHashMap<F, NumericArray>;

/// Intermediate representation for vectorized evaluators.
enum VectorizedExpr {
    Array(NumericArray),
    Scalar(Option<NumericValue>),
}

impl VectorizedExpr {
    fn materialize(self, len: usize, kind: NumericKind) -> ArrayRef {
        match self {
            VectorizedExpr::Array(array) => array.to_aligned_array_ref(kind),
            VectorizedExpr::Scalar(Some(value)) => {
                let target_kind = match (value.kind(), kind) {
                    (_, NumericKind::String) => NumericKind::String,
                    (NumericKind::String, _) => NumericKind::String,
                    (NumericKind::Float, _) => NumericKind::Float,
                    (NumericKind::Decimal, _) => NumericKind::Decimal,
                    (NumericKind::Integer, NumericKind::Float) => NumericKind::Float,
                    (NumericKind::Integer, NumericKind::Decimal) => NumericKind::Decimal,
                    (NumericKind::Integer, NumericKind::Integer) => NumericKind::Integer,
                };
                let values = vec![Some(value); len];
                let array = NumericArray::from_numeric_values(values, target_kind);
                array.to_aligned_array_ref(kind)
            }
            VectorizedExpr::Scalar(None) => {
                let values = vec![None; len];
                let array = NumericArray::from_numeric_values(values, kind);
                array.to_aligned_array_ref(kind)
            }
        }
    }
}

/// Represents an affine transformation `scale * field + offset`.
#[derive(Clone, Copy, Debug)]
pub struct AffineExpr<F> {
    pub field: F,
    pub scale: f64,
    pub offset: f64,
}

/// Internal accumulator representing a partially merged affine expression.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct AffineState<F> {
    field: Option<F>,
    scale: f64,
    offset: f64,
}

// TODO: Place in impl?
/// Combine field identifiers while tracking whether multiple fields were encountered.
#[allow(dead_code)]
fn merge_field<F: Eq + Copy>(lhs: Option<F>, rhs: Option<F>) -> Option<Option<F>> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => {
            if a == b {
                Some(Some(a))
            } else {
                None
            }
        }
        (Some(a), None) => Some(Some(a)),
        (None, Some(b)) => Some(Some(b)),
        (None, None) => Some(None),
    }
}

/// Centralizes the numeric kernels applied to scalar expressions so they can be
/// tuned without touching the surrounding table scan logic.
pub struct ScalarEvaluator;

impl ScalarEvaluator {
    /// Collect every field referenced by the scalar expression into `acc`.
    pub fn collect_fields<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>, acc: &mut FxHashSet<F>) {
        match expr {
            ScalarExpr::Column(fid) => {
                acc.insert(*fid);
            }
            ScalarExpr::Literal(_) => {}
            ScalarExpr::Binary { left, right, .. } => {
                Self::collect_fields(left, acc);
                Self::collect_fields(right, acc);
            }
            ScalarExpr::Compare { left, right, .. } => {
                Self::collect_fields(left, acc);
                Self::collect_fields(right, acc);
            }
            ScalarExpr::Not(inner) => {
                Self::collect_fields(inner, acc);
            }
            ScalarExpr::IsNull { expr, .. } => {
                Self::collect_fields(expr, acc);
            }
            ScalarExpr::Aggregate(agg) => {
                // Collect fields referenced by the aggregate expression
                match agg {
                    AggregateCall::CountStar => {}
                    AggregateCall::Count { expr, .. }
                    | AggregateCall::Sum { expr, .. }
                    | AggregateCall::Total { expr, .. }
                    | AggregateCall::Avg { expr, .. }
                    | AggregateCall::Min(expr)
                    | AggregateCall::Max(expr)
                    | AggregateCall::CountNulls(expr)
                    | AggregateCall::GroupConcat { expr, .. } => {
                        Self::collect_fields(expr, acc);
                    }
                }
            }
            ScalarExpr::GetField { base, .. } => {
                // Collect fields from the base expression
                Self::collect_fields(base, acc);
            }
            ScalarExpr::Cast { expr, .. } => {
                Self::collect_fields(expr, acc);
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                if let Some(inner) = operand.as_deref() {
                    Self::collect_fields(inner, acc);
                }
                for (when_expr, then_expr) in branches {
                    Self::collect_fields(when_expr, acc);
                    Self::collect_fields(then_expr, acc);
                }
                if let Some(inner) = else_expr.as_deref() {
                    Self::collect_fields(inner, acc);
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    Self::collect_fields(item, acc);
                }
            }
            ScalarExpr::Random => {
                // Random does not reference any fields
            }
            ScalarExpr::ScalarSubquery(_) => {
                // Scalar subqueries don't directly reference fields from the outer query
            }
        }
    }

    pub fn prepare_numeric_arrays<F: Hash + Eq + Copy>(
        arrays: &FxHashMap<F, ArrayRef>,
        _row_count: usize,
    ) -> NumericArrayMap<F> {
        let mut result = FxHashMap::default();
        for (field, array) in arrays {
            if let Ok(numeric) = NumericArray::try_from_arrow(array) {
                result.insert(*field, numeric);
            }
        }
        result
    }

    /// Attempts to represent the expression as `scale * column + offset`.
    /// Returns `None` when the expression depends on multiple columns or is non-linear.
    pub fn extract_affine<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> Option<AffineExpr<F>> {
        let simplified = Self::simplify(expr);
        Self::extract_affine_simplified(&simplified)
    }

    /// Variant of `extract_affine` that assumes `expr` is already simplified.
    pub fn extract_affine_simplified<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
    ) -> Option<AffineExpr<F>> {
        let state = Self::affine_state(expr)?;
        let field = state.field?;
        Some(AffineExpr {
            field,
            scale: state.scale,
            offset: state.offset,
        })
    }

    fn affine_state<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> Option<AffineState<F>> {
        match expr {
            ScalarExpr::Column(fid) => Some(AffineState {
                field: Some(*fid),
                scale: 1.0,
                offset: 0.0,
            }),
            ScalarExpr::Literal(_) => {
                let value = Self::literal_numeric_value(expr)?.as_f64();
                Some(AffineState {
                    field: None,
                    scale: 0.0,
                    offset: value,
                })
            }
            ScalarExpr::Aggregate(_) => None,
            ScalarExpr::GetField { .. } => None,
            ScalarExpr::Binary { left, op, right } => {
                let left_state = Self::affine_state(left)?;
                let right_state = Self::affine_state(right)?;
                match op {
                    BinaryOp::Add => Self::affine_add(left_state, right_state),
                    BinaryOp::Subtract => Self::affine_sub(left_state, right_state),
                    BinaryOp::Multiply => Self::affine_mul(left_state, right_state),
                    BinaryOp::Divide => Self::affine_div(left_state, right_state),
                    _ => None,
                }
            }
            ScalarExpr::Cast { expr, .. } => Self::affine_state(expr),
            _ => None,
        }
    }

    /// Infer the numeric kind of an expression using only the kinds of its referenced columns.
    pub fn infer_result_kind_from_types<F, R>(
        expr: &ScalarExpr<F>,
        resolve_kind: &mut R,
    ) -> Option<NumericKind>
    where
        F: Hash + Eq + Copy,
        R: FnMut(F) -> Option<NumericKind>,
    {
        match expr {
            ScalarExpr::Literal(_) => Self::literal_numeric_value(expr).map(|v| v.kind()),
            ScalarExpr::Column(fid) => resolve_kind(*fid),
            ScalarExpr::Binary { left, op, right } => {
                let left_kind = Self::infer_result_kind_from_types(left, resolve_kind)?;
                let right_kind = Self::infer_result_kind_from_types(right, resolve_kind)?;
                Some(Self::binary_result_kind(*op, left_kind, right_kind))
            }
            ScalarExpr::Compare { .. } => Some(NumericKind::Integer),
            ScalarExpr::Not(_) => Some(NumericKind::Integer),
            ScalarExpr::IsNull { .. } => Some(NumericKind::Integer),
            ScalarExpr::Aggregate(_) => Some(NumericKind::Float),
            ScalarExpr::GetField { .. } => None,
            ScalarExpr::Cast { expr, data_type } => {
                let target_kind = Self::kind_for_data_type(data_type);
                target_kind.or_else(|| Self::infer_result_kind_from_types(expr, resolve_kind))
            }
            ScalarExpr::Case {
                branches,
                else_expr,
                ..
            } => {
                let mut result_kind = NumericKind::Integer;
                for (_, then_expr) in branches {
                    let kind = Self::infer_result_kind_from_types(then_expr, resolve_kind)?;
                    if matches!(kind, NumericKind::Float) {
                        result_kind = NumericKind::Float;
                        break;
                    }
                }
                if result_kind != NumericKind::Float
                    && let Some(inner) = else_expr.as_deref()
                    && let Some(kind) = Self::infer_result_kind_from_types(inner, resolve_kind)
                    && matches!(kind, NumericKind::Float)
                {
                    result_kind = NumericKind::Float;
                }
                Some(result_kind)
            }
            ScalarExpr::Coalesce(items) => {
                let mut result_kind = NumericKind::Integer;
                for item in items {
                    let kind = Self::infer_result_kind_from_types(item, resolve_kind)?;
                    if matches!(kind, NumericKind::Float) {
                        result_kind = NumericKind::Float;
                        break;
                    }
                }
                Some(result_kind)
            }
            ScalarExpr::Random => Some(NumericKind::Float),
            ScalarExpr::ScalarSubquery(_) => Some(NumericKind::Float),
        }
    }

    fn binary_result_kind(
        op: BinaryOp,
        lhs_kind: NumericKind,
        rhs_kind: NumericKind,
    ) -> NumericKind {
        let lhs_value = match lhs_kind {
            NumericKind::Integer => NumericValue::Int(1),
            NumericKind::Float => NumericValue::Float(1.0),
            NumericKind::Decimal => NumericValue::Decimal(DecimalValue::from_i64(1)),
            NumericKind::String => NumericValue::String("".to_string()),
        };
        let rhs_value = match rhs_kind {
            NumericKind::Integer => NumericValue::Int(1),
            NumericKind::Float => NumericValue::Float(1.0),
            NumericKind::Decimal => NumericValue::Decimal(DecimalValue::from_i64(1)),
            NumericKind::String => NumericValue::String("".to_string()),
        };

        Self::apply_binary_values(op, lhs_value, rhs_value)
            .unwrap_or(NumericValue::Float(0.0))
            .kind()
    }

    /// Evaluate a scalar expression for the row at `idx` using the provided numeric arrays.
    pub fn evaluate_value<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        idx: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<Option<NumericValue>> {
        match expr {
            ScalarExpr::Column(fid) => {
                let array = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal("missing column for field".into()))?;
                Ok(array.value(idx))
            }
            ScalarExpr::Literal(_) => Ok(Self::literal_numeric_value(expr)),
            ScalarExpr::Binary { left, op, right } => {
                if let Some(result) =
                    Self::try_evaluate_date_interval_binary(left, *op, right, idx, arrays)?
                {
                    return Ok(result);
                }
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                Ok(Self::apply_binary(*op, l, r))
            }
            ScalarExpr::Compare { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                match (l, r) {
                    (Some(lhs), Some(rhs)) => {
                        let result = Self::compare(*op, lhs, rhs);
                        Ok(Some(NumericValue::Int(result as i64)))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Not(inner) => {
                let value = Self::evaluate_value(inner, idx, arrays)?;
                match value {
                    Some(v) => {
                        let is_truthy = Self::truthy_numeric(v);
                        Ok(Some(NumericValue::Int(if is_truthy { 0 } else { 1 })))
                    }
                    None => Ok(None),
                }
            }
            ScalarExpr::IsNull { expr, negated } => {
                let value = Self::evaluate_value(expr, idx, arrays)?;
                let is_null = value.is_none();
                // XOR-style comparison keeps negated IS NULL readable.
                let condition_holds = is_null != *negated;
                Ok(Some(NumericValue::Int(if condition_holds { 1 } else { 0 })))
            }
            ScalarExpr::Aggregate(_) => Err(Error::Internal(
                "Aggregate expressions should not appear in row-level evaluation".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::Internal(
                "GetField expressions should not be evaluated in numeric kernels".into(),
            )),
            ScalarExpr::Cast { expr, data_type } => {
                if matches!(data_type, DataType::Date32) {
                    return Self::evaluate_cast_date32_value(expr, idx, arrays);
                }

                let value = Self::evaluate_value(expr, idx, arrays)?;
                let target_kind = Self::kind_for_data_type(data_type).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unsupported cast target type {:?}",
                        data_type
                    ))
                })?;
                Self::cast_numeric_value_to_kind(value, target_kind)
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand_value = match operand.as_deref() {
                    Some(op) => Some(Self::evaluate_value(op, idx, arrays)?),
                    None => None,
                };

                for (when_expr, then_expr) in branches {
                    let matched = if let Some(op_val_opt) = &operand_value {
                        let when_val = Self::evaluate_value(when_expr, idx, arrays)?;
                        match (op_val_opt, &when_val) {
                            (Some(op_val), Some(branch_val)) => {
                                Self::numeric_equals(op_val.clone(), branch_val.clone())
                            }
                            _ => false,
                        }
                    } else {
                        let cond_val = Self::evaluate_value(when_expr, idx, arrays)?;
                        cond_val.is_some_and(Self::truthy_numeric)
                    };

                    if matched {
                        return Self::evaluate_value(then_expr, idx, arrays);
                    }
                }

                if let Some(else_expr) = else_expr.as_deref() {
                    Self::evaluate_value(else_expr, idx, arrays)
                } else {
                    Ok(None)
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    if let Some(value) = Self::evaluate_value(item, idx, arrays)? {
                        return Ok(Some(value));
                    }
                }
                Ok(None)
            }
            ScalarExpr::Random => Ok(Some(NumericValue::Float(rand::random::<f64>()))),
            ScalarExpr::ScalarSubquery(_) => Err(Error::Internal(
                "Scalar subquery evaluation requires a separate execution context".into(),
            )),
        }
    }

    /// Evaluate a scalar expression for every row in the batch.
    #[allow(dead_code)]
    pub fn evaluate_batch<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<ArrayRef> {
        let simplified = Self::simplify(expr);
        Self::evaluate_batch_simplified(&simplified, len, arrays)
    }

    /// Evaluate a scalar expression that has already been simplified.
    pub fn evaluate_batch_simplified<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<ArrayRef> {
        if let ScalarExpr::Cast {
            expr: inner,
            data_type,
        } = expr
            && matches!(data_type, DataType::Date32)
        {
            return Self::evaluate_cast_date32_batch(inner, len, arrays);
        }

        let preferred = Self::infer_result_kind(expr, arrays);
        if let Some(vectorized) = Self::try_evaluate_vectorized(expr, len, arrays, preferred)? {
            let result = vectorized.materialize(len, preferred);
            return Ok(result);
        }

        let mut values: Vec<Option<NumericValue>> = Vec::with_capacity(len);
        for idx in 0..len {
            values.push(Self::evaluate_value(expr, idx, arrays)?);
        }
        let array = NumericArray::from_numeric_values(values, preferred);
        Ok(array.to_aligned_array_ref(preferred))
    }

    fn try_evaluate_vectorized<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
        _preferred: NumericKind,
    ) -> LlkvResult<Option<VectorizedExpr>> {
        if Self::expr_contains_interval(expr) {
            return Ok(None);
        }
        match expr {
            ScalarExpr::Column(fid) => {
                let array = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal("missing column for field".into()))?;
                Ok(Some(VectorizedExpr::Array(array.clone())))
            }
            ScalarExpr::Literal(_) => Ok(Some(VectorizedExpr::Scalar(
                Self::literal_numeric_value(expr),
            ))),
            ScalarExpr::Binary { left, op, right } => {
                let left_kind = Self::infer_result_kind(left, arrays);
                let right_kind = Self::infer_result_kind(right, arrays);

                let left_vec = Self::try_evaluate_vectorized(left, len, arrays, left_kind)?;
                let right_vec = Self::try_evaluate_vectorized(right, len, arrays, right_kind)?;

                match (left_vec, right_vec) {
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Scalar(rhs))) => Ok(
                        Some(VectorizedExpr::Scalar(Self::apply_binary(*op, lhs, rhs))),
                    ),
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let array = compute_binary(&lhs, &rhs, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let array = compute_binary_scalar(&lhs, rhs, *op, true)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let array = compute_binary_scalar(&rhs, lhs, *op, false)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Compare { .. } => Ok(None),
            ScalarExpr::Not(_) => Ok(None),
            ScalarExpr::IsNull { .. } => Ok(None),
            ScalarExpr::Aggregate(_) => Err(Error::Internal(
                "Aggregate expressions should not appear in row-level evaluation".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::Internal(
                "GetField expressions should not be evaluated in numeric kernels".into(),
            )),
            ScalarExpr::Cast { expr, data_type } => {
                if matches!(data_type, DataType::Date32) {
                    return Ok(None);
                }

                let inner_kind = Self::infer_result_kind(expr, arrays);
                let inner_vec = Self::try_evaluate_vectorized(expr, len, arrays, inner_kind)?;
                let target_kind = Self::kind_for_data_type(data_type).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unsupported cast target type {:?}",
                        data_type
                    ))
                })?;

                match inner_vec {
                    Some(VectorizedExpr::Scalar(value)) => Ok(Some(VectorizedExpr::Scalar(
                        Self::cast_numeric_value_to_kind(value, target_kind)?,
                    ))),
                    Some(VectorizedExpr::Array(array)) => Ok(Some(VectorizedExpr::Array(
                        Self::cast_numeric_array_to_kind(&array, target_kind)?,
                    ))),
                    None => Ok(None),
                }
            }
            ScalarExpr::Case { .. } => Ok(None),
            ScalarExpr::Coalesce(_) => Ok(None),
            ScalarExpr::Random => {
                // Generate array of random float values
                let values: Vec<f64> = (0..len).map(|_| rand::random::<f64>()).collect();
                let array = Float64Array::from(values);
                Ok(Some(VectorizedExpr::Array(NumericArray::new_float(
                    Arc::new(array),
                ))))
            }
            ScalarExpr::ScalarSubquery(_) => Ok(None),
        }
    }

    /// Returns the column referenced by an expression when it's a direct or additive identity passthrough.
    pub fn passthrough_column<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> Option<F> {
        match Self::simplify(expr) {
            ScalarExpr::Column(fid) => Some(fid),
            _ => None,
        }
    }

    /// Simplify an expression by constant folding and identity removal.
    pub fn simplify<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> ScalarExpr<F> {
        match expr {
            ScalarExpr::Binary { left, op, right } => {
                let l = Self::simplify(left);
                let r = Self::simplify(right);

                // Constant folding
                if let (ScalarExpr::Literal(_lv), ScalarExpr::Literal(_rv)) = (&l, &r) {
                    let ln = Self::literal_numeric_value(&l);
                    let rn = Self::literal_numeric_value(&r);
                    if let (Some(ln_val), Some(rn_val)) = (ln, rn) {
                        if let Some(res) = Self::apply_binary_literal(*op, ln_val, rn_val) {
                            return res;
                        }
                    }
                }

                // Identity removal (e.g. x + 0 = x)
                if *op == BinaryOp::Add {
                    if let ScalarExpr::Literal(_lit) = &r {
                        if let Some(NumericValue::Int(0)) = Self::literal_numeric_value(&r) {
                            return l;
                        }
                        if let Some(NumericValue::Float(f)) = Self::literal_numeric_value(&r) {
                            if f == 0.0 {
                                return l;
                            }
                        }
                    }
                    if let ScalarExpr::Literal(_lit) = &l {
                        if let Some(NumericValue::Int(0)) = Self::literal_numeric_value(&l) {
                            return r;
                        }
                        if let Some(NumericValue::Float(f)) = Self::literal_numeric_value(&l) {
                            if f == 0.0 {
                                return r;
                            }
                        }
                    }
                }

                ScalarExpr::Binary {
                    left: Box::new(l),
                    op: *op,
                    right: Box::new(r),
                }
            }
            ScalarExpr::Cast { expr, data_type } => {
                let inner = Self::simplify(expr);
                // Remove redundant casts?
                ScalarExpr::Cast {
                    expr: Box::new(inner),
                    data_type: data_type.clone(),
                }
            }
            _ => expr.clone(),
        }
    }

    fn infer_result_kind<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        arrays: &NumericArrayMap<F>,
    ) -> NumericKind {
        match expr {
            ScalarExpr::Column(fid) => arrays
                .get(fid)
                .map(|a| a.kind())
                .unwrap_or(NumericKind::Float),
            ScalarExpr::Literal(lit) => match lit {
                Literal::Integer(_) => NumericKind::Integer,
                Literal::Float(_) => NumericKind::Float,
                Literal::Decimal(_) => NumericKind::Decimal,
                Literal::String(_) => NumericKind::String,
                _ => NumericKind::Float,
            },
            ScalarExpr::Binary { left, op, right } => {
                if *op == BinaryOp::Divide {
                    return NumericKind::Float;
                }
                let l = Self::infer_result_kind(left, arrays);
                let r = Self::infer_result_kind(right, arrays);
                match (l, r) {
                    (NumericKind::String, _) | (_, NumericKind::String) => NumericKind::String,
                    (NumericKind::Float, _) | (_, NumericKind::Float) => NumericKind::Float,
                    (NumericKind::Decimal, _) | (_, NumericKind::Decimal) => NumericKind::Decimal,
                    (NumericKind::Integer, NumericKind::Integer) => NumericKind::Integer,
                }
            }
            ScalarExpr::Compare { .. } => NumericKind::Integer, // Boolean 0/1
            ScalarExpr::Not(_) => NumericKind::Integer,         // Boolean 0/1
            ScalarExpr::IsNull { .. } => NumericKind::Integer,  // Boolean 0/1
            ScalarExpr::Aggregate(_) => NumericKind::Float,
            ScalarExpr::GetField { .. } => NumericKind::Float,
            ScalarExpr::Cast { expr, data_type } => {
                let target_kind = Self::kind_for_data_type(data_type);
                target_kind.unwrap_or_else(|| Self::infer_result_kind(expr, arrays))
            }
            ScalarExpr::Case {
                branches,
                else_expr,
                ..
            } => {
                let mut result_kind = NumericKind::Integer;
                let mut has_decimal = false;
                for (_, then_expr) in branches {
                    match Self::infer_result_kind(then_expr, arrays) {
                        NumericKind::Float => {
                            result_kind = NumericKind::Float;
                            break;
                        }
                        NumericKind::Decimal => {
                            has_decimal = true;
                        }
                        _ => {}
                    }
                }
                if result_kind != NumericKind::Float
                    && let Some(inner) = else_expr.as_deref()
                {
                    match Self::infer_result_kind(inner, arrays) {
                        NumericKind::Float => result_kind = NumericKind::Float,
                        NumericKind::Decimal => has_decimal = true,
                        _ => {}
                    }
                }
                if result_kind != NumericKind::Float && has_decimal {
                    result_kind = NumericKind::Decimal;
                }
                result_kind
            }
            ScalarExpr::Coalesce(items) => {
                let mut result_kind = NumericKind::Integer;
                let mut has_decimal = false;
                for item in items {
                    match Self::infer_result_kind(item, arrays) {
                        NumericKind::Float => {
                            result_kind = NumericKind::Float;
                            break;
                        }
                        NumericKind::Decimal => {
                            has_decimal = true;
                        }
                        _ => {}
                    }
                }
                if result_kind != NumericKind::Float && has_decimal {
                    result_kind = NumericKind::Decimal;
                }
                result_kind
            }
            ScalarExpr::Random => NumericKind::Float,
            ScalarExpr::ScalarSubquery(sub) => {
                Self::kind_for_data_type(&sub.data_type).unwrap_or(NumericKind::Float)
            }
        }
    }

    fn expr_contains_interval<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> bool {
        match expr {
            ScalarExpr::Literal(Literal::Interval(_)) => true,
            ScalarExpr::Binary { left, right, .. } => {
                Self::expr_contains_interval(left) || Self::expr_contains_interval(right)
            }
            _ => false,
        }
    }

    fn try_evaluate_date_interval_binary<F: Hash + Eq + Copy>(
        left: &ScalarExpr<F>,
        op: BinaryOp,
        right: &ScalarExpr<F>,
        idx: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<Option<Option<NumericValue>>> {
        // Check for DATE +/- INTERVAL
        if let ScalarExpr::Literal(Literal::Interval(interval)) = right {
            if op == BinaryOp::Add || op == BinaryOp::Subtract {
                // Left must be a date
                let left_val = Self::evaluate_value(left, idx, arrays)?;
                if let Some(NumericValue::Int(days)) = left_val {
                    // Treat int as days since epoch (Date32)
                    let result_days = if op == BinaryOp::Add {
                        add_interval_to_date32(days as i32, *interval)?
                    } else {
                        subtract_interval_from_date32(days as i32, *interval)?
                    };
                    return Ok(Some(Some(NumericValue::Int(result_days as i64))));
                }
            }
        }
        Ok(None)
    }

    fn evaluate_option_numeric_and(
        lhs: Option<NumericValue>,
        rhs: Option<NumericValue>,
    ) -> Option<NumericValue> {
        // SQL AND logic with NULLs:
        // TRUE AND TRUE = TRUE
        // TRUE AND FALSE = FALSE
        // TRUE AND NULL = NULL
        // FALSE AND ... = FALSE
        // NULL AND FALSE = FALSE
        // NULL AND NULL = NULL

        let l_truthy = lhs.as_ref().map(|v| Self::truthy_numeric(v.clone()));
        let r_truthy = rhs.as_ref().map(|v| Self::truthy_numeric(v.clone()));

        match (l_truthy, r_truthy) {
            (Some(false), _) => Some(NumericValue::Int(0)),
            (_, Some(false)) => Some(NumericValue::Int(0)),
            (Some(true), Some(true)) => Some(NumericValue::Int(1)),
            _ => None,
        }
    }

    fn evaluate_option_numeric_or(
        lhs: Option<NumericValue>,
        rhs: Option<NumericValue>,
    ) -> Option<NumericValue> {
        // SQL OR logic with NULLs:
        // TRUE OR ... = TRUE
        // FALSE OR TRUE = TRUE
        // FALSE OR FALSE = FALSE
        // FALSE OR NULL = NULL
        // NULL OR TRUE = TRUE
        // NULL OR NULL = NULL

        let l_truthy = lhs.as_ref().map(|v| Self::truthy_numeric(v.clone()));
        let r_truthy = rhs.as_ref().map(|v| Self::truthy_numeric(v.clone()));

        match (l_truthy, r_truthy) {
            (Some(true), _) => Some(NumericValue::Int(1)),
            (_, Some(true)) => Some(NumericValue::Int(1)),
            (Some(false), Some(false)) => Some(NumericValue::Int(0)),
            _ => None,
        }
    }

    fn evaluate_cast_date32_value<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        idx: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<Option<NumericValue>> {
        let val = Self::evaluate_value(expr, idx, arrays)?;
        match val {
            Some(NumericValue::String(s)) => {
                let days = parse_date32_literal(&s)?;
                Ok(Some(NumericValue::Int(days as i64)))
            }
            Some(NumericValue::Int(i)) => Ok(Some(NumericValue::Int(i))),
            _ => Err(Error::InvalidArgumentError(
                "Cannot cast non-string/int to Date32".into(),
            )),
        }
    }

    fn evaluate_cast_date32_batch<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<ArrayRef> {
        // Fallback to row-by-row for now
        let mut builder = arrow::array::Date32Builder::with_capacity(len);
        for idx in 0..len {
            let val = Self::evaluate_cast_date32_value(expr, idx, arrays)?;
            match val {
                Some(NumericValue::Int(i)) => builder.append_value(i as i32),
                _ => builder.append_null(),
            }
        }
        Ok(Arc::new(builder.finish()))
    }

    pub fn kind_for_data_type(dt: &DataType) -> Option<NumericKind> {
        match dt {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _) => Some(NumericKind::Integer),
            DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64 => Some(NumericKind::Float),
            DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => Some(NumericKind::Decimal),
            DataType::Utf8 | DataType::LargeUtf8 => Some(NumericKind::String),
            _ => None,
        }
    }

    fn cast_numeric_array_to_kind(
        array: &NumericArray,
        target: NumericKind,
    ) -> LlkvResult<NumericArray> {
        // TODO: Implement vectorized cast
        // For now, just rebuild
        let len = array.len();
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let val = array.value(i);
            values.push(Self::cast_numeric_value_to_kind(val, target)?);
        }
        Ok(NumericArray::from_numeric_values(values, target))
    }

    fn literal_numeric_value<F>(expr: &ScalarExpr<F>) -> Option<NumericValue> {
        match expr {
            ScalarExpr::Literal(lit) => match lit {
                Literal::Integer(i) => Some(NumericValue::Int((*i).try_into().unwrap_or(0))),
                Literal::Float(f) => Some(NumericValue::Float(*f)),
                Literal::Decimal(d) => Some(NumericValue::Decimal(*d)),
                Literal::String(s) => Some(NumericValue::String(s.clone())),
                Literal::Boolean(b) => Some(NumericValue::Int(if *b { 1 } else { 0 })),
                Literal::Null => None,
                Literal::Date32(d) => Some(NumericValue::Int(*d as i64)),
                _ => None,
            },
            _ => None,
        }
    }

    fn truthy_numeric(val: NumericValue) -> bool {
        match val {
            NumericValue::Int(i) => i != 0,
            NumericValue::Float(f) => f != 0.0,
            NumericValue::Decimal(d) => d.raw_value() != 0,
            NumericValue::String(s) => !s.is_empty(),
        }
    }

    fn numeric_equals(lhs: NumericValue, rhs: NumericValue) -> bool {
        match (lhs, rhs) {
            (NumericValue::Int(a), NumericValue::Int(b)) => a == b,
            (NumericValue::Float(a), NumericValue::Float(b)) => (a - b).abs() < f64::EPSILON,
            (NumericValue::Decimal(a), NumericValue::Decimal(b)) => a == b,
            (NumericValue::String(a), NumericValue::String(b)) => a == b,
            (NumericValue::Int(a), NumericValue::Float(b)) => (a as f64 - b).abs() < f64::EPSILON,
            (NumericValue::Float(a), NumericValue::Int(b)) => (a - b as f64).abs() < f64::EPSILON,
            _ => false,
        }
    }

    pub fn compare(op: CompareOp, lhs: NumericValue, rhs: NumericValue) -> bool {
        match op {
            CompareOp::Eq => Self::numeric_equals(lhs, rhs),
            CompareOp::NotEq => !Self::numeric_equals(lhs, rhs),
            CompareOp::Lt => lhs.to_f64() < rhs.to_f64(),
            CompareOp::LtEq => lhs.to_f64() <= rhs.to_f64(),
            CompareOp::Gt => lhs.to_f64() > rhs.to_f64(),
            CompareOp::GtEq => lhs.to_f64() >= rhs.to_f64(),
        }
    }

    #[allow(dead_code)]
    fn affine_add<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        let merged_field = merge_field(lhs.field, rhs.field)?;
        if merged_field.is_none() {
            // Both constant
            return Some(AffineState {
                field: None,
                scale: 0.0,
                offset: lhs.offset + rhs.offset,
            });
        }
        Some(AffineState {
            field: merged_field,
            scale: lhs.scale + rhs.scale,
            offset: lhs.offset + rhs.offset,
        })
    }

    #[allow(dead_code)]
    fn affine_sub<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        let merged_field = merge_field(lhs.field, rhs.field)?;
        if merged_field.is_none() {
            return Some(AffineState {
                field: None,
                scale: 0.0,
                offset: lhs.offset - rhs.offset,
            });
        }
        Some(AffineState {
            field: merged_field,
            scale: lhs.scale - rhs.scale,
            offset: lhs.offset - rhs.offset,
        })
    }

    #[allow(dead_code)]
    fn affine_mul<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        if lhs.field.is_some() && rhs.field.is_some() {
            return None; // Non-linear
        }
        if lhs.field.is_none() {
            let factor = lhs.offset;
            return Some(AffineState {
                field: rhs.field,
                scale: rhs.scale * factor,
                offset: rhs.offset * factor,
            });
        }
        if rhs.field.is_none() {
            let factor = rhs.offset;
            return Some(AffineState {
                field: lhs.field,
                scale: lhs.scale * factor,
                offset: lhs.offset * factor,
            });
        }
        None
    }

    #[allow(dead_code)]
    fn affine_div<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        if rhs.field.is_some() {
            return None;
        }
        let denom = rhs.offset;
        if denom == 0.0 {
            return None;
        }
        Some(AffineState {
            field: lhs.field,
            scale: lhs.scale / denom,
            offset: lhs.offset / denom,
        })
    }

    fn apply_binary_literal<F>(
        op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue,
    ) -> Option<ScalarExpr<F>> {
        match op {
            BinaryOp::Add => Some(Self::literal_from_numeric(Self::add_values(lhs, rhs))),
            BinaryOp::Subtract => Some(Self::literal_from_numeric(Self::sub_values(lhs, rhs))),
            BinaryOp::Multiply => Some(Self::literal_from_numeric(Self::mul_values(lhs, rhs))),
            BinaryOp::Divide => Self::div_values(lhs, rhs).map(Self::literal_from_numeric),
            BinaryOp::Modulo => Self::mod_values(lhs, rhs).map(Self::literal_from_numeric),
            BinaryOp::And => {
                let truthy = Self::truthy_numeric(lhs) && Self::truthy_numeric(rhs);
                Some(ScalarExpr::literal(if truthy { 1 } else { 0 }))
            }
            BinaryOp::Or => {
                let truthy = Self::truthy_numeric(lhs) || Self::truthy_numeric(rhs);
                Some(ScalarExpr::literal(if truthy { 1 } else { 0 }))
            }
            BinaryOp::BitwiseShiftLeft => {
                let lhs_i64 = match lhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let rhs_i64 = match rhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let result = lhs_i64.wrapping_shl(rhs_i64 as u32);
                Some(ScalarExpr::literal(result))
            }
            BinaryOp::BitwiseShiftRight => {
                let lhs_i64 = match lhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let rhs_i64 = match rhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let result = lhs_i64.wrapping_shr(rhs_i64 as u32);
                Some(ScalarExpr::literal(result))
            }
        }
    }

    fn literal_from_numeric<F>(value: NumericValue) -> ScalarExpr<F> {
        match value {
            NumericValue::Int(i) => ScalarExpr::literal(i),
            NumericValue::Float(f) => ScalarExpr::literal(f),
            NumericValue::Decimal(d) => ScalarExpr::Literal(Literal::Decimal(d)),
            NumericValue::String(s) => ScalarExpr::Literal(Literal::String(s)),
        }
    }

    /// Apply an arithmetic kernel. Returns `None` when the computation results in a null (e.g. divide by zero).
    pub fn apply_binary(
        op: BinaryOp,
        lhs: Option<NumericValue>,
        rhs: Option<NumericValue>,
    ) -> Option<NumericValue> {
        match op {
            BinaryOp::And => Self::evaluate_option_numeric_and(lhs, rhs),
            BinaryOp::Or => Self::evaluate_option_numeric_or(lhs, rhs),
            _ => match (lhs, rhs) {
                (Some(lv), Some(rv)) => Self::apply_binary_values(op, lv, rv),
                _ => None,
            },
        }
    }

    fn apply_binary_values(
        op: BinaryOp,
        lhs: NumericValue,
        rhs: NumericValue,
    ) -> Option<NumericValue> {
        match op {
            BinaryOp::Add => Some(Self::add_values(lhs, rhs)),
            BinaryOp::Subtract => Some(Self::sub_values(lhs, rhs)),
            BinaryOp::Multiply => Some(Self::mul_values(lhs, rhs)),
            BinaryOp::Divide => Self::div_values(lhs, rhs),
            BinaryOp::Modulo => Self::mod_values(lhs, rhs),
            BinaryOp::And => Some(NumericValue::Int(
                if Self::truthy_numeric(lhs) && Self::truthy_numeric(rhs) {
                    1
                } else {
                    0
                },
            )),
            BinaryOp::Or => Some(NumericValue::Int(
                if Self::truthy_numeric(lhs) || Self::truthy_numeric(rhs) {
                    1
                } else {
                    0
                },
            )),
            BinaryOp::BitwiseShiftLeft => {
                let lhs_i64 = match lhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let rhs_i64 = match rhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let result = lhs_i64.wrapping_shl(rhs_i64 as u32);
                Some(NumericValue::Int(result))
            }
            BinaryOp::BitwiseShiftRight => {
                let lhs_i64 = match lhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let rhs_i64 = match rhs {
                    NumericValue::Int(i) => i,
                    NumericValue::Float(f) => f as i64,
                    NumericValue::Decimal(d) => d.to_f64() as i64,
                    NumericValue::String(_) => return None,
                };
                let result = lhs_i64.wrapping_shr(rhs_i64 as u32);
                Some(NumericValue::Int(result))
            }
        }
    }

    fn add_values(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
        lhs.add(&rhs).unwrap_or_else(|_| {
            // Fallback to float on error (e.g. overflow)
            NumericValue::Float(lhs.as_f64() + rhs.as_f64())
        })
    }

    fn sub_values(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
        lhs.sub(&rhs)
            .unwrap_or_else(|_| NumericValue::Float(lhs.as_f64() - rhs.as_f64()))
    }

    fn mul_values(lhs: NumericValue, rhs: NumericValue) -> NumericValue {
        lhs.mul(&rhs)
            .unwrap_or_else(|_| NumericValue::Float(lhs.as_f64() * rhs.as_f64()))
    }

    fn div_values(lhs: NumericValue, rhs: NumericValue) -> Option<NumericValue> {
        lhs.div(&rhs).ok()
    }

    fn mod_values(lhs: NumericValue, rhs: NumericValue) -> Option<NumericValue> {
        lhs.rem(&rhs).ok()
    }

    fn cast_numeric_value_to_kind(
        value: Option<NumericValue>,
        target: NumericKind,
    ) -> LlkvResult<Option<NumericValue>> {
        match value {
            None => Ok(None),
            Some(NumericValue::Int(v)) => Ok(Some(match target {
                NumericKind::Integer => NumericValue::Int(v),
                NumericKind::Float => NumericValue::Float(v as f64),
                NumericKind::Decimal => NumericValue::Decimal(DecimalValue::from_i64(v)),
                NumericKind::String => NumericValue::String(v.to_string()),
            })),
            Some(NumericValue::Float(v)) => Ok(Some(match target {
                NumericKind::Integer => NumericValue::Int(v as i64),
                NumericKind::Float => NumericValue::Float(v),
                NumericKind::Decimal => NumericValue::Decimal(DecimalValue::from_i64(v as i64)),
                NumericKind::String => NumericValue::String(v.to_string()),
            })),
            Some(NumericValue::Decimal(v)) => Ok(Some(match target {
                NumericKind::Integer => NumericValue::Int(v.to_f64() as i64),
                NumericKind::Float => NumericValue::Float(v.to_f64()),
                NumericKind::Decimal => NumericValue::Decimal(v),
                NumericKind::String => NumericValue::String(v.to_string()),
            })),
            Some(NumericValue::String(v)) => Ok(Some(match target {
                NumericKind::Integer => NumericValue::Int(v.parse().unwrap_or(0)),
                NumericKind::Float => NumericValue::Float(v.parse().unwrap_or(0.0)),
                NumericKind::Decimal => NumericValue::Decimal(DecimalValue::from_i64(0)),
                NumericKind::String => NumericValue::String(v),
            })),
        }
    }
}
