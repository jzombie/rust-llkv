use std::hash::Hash;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, UInt32Array, new_null_array};
use arrow::compute::kernels::cast;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{concat, is_not_null, take};
use arrow::datatypes::{DataType, IntervalMonthDayNanoType};
use llkv_expr::literal::{Literal, LiteralExt};
use llkv_types::IntervalValue;
use llkv_expr::{AggregateCall, BinaryOp, CompareOp, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::date::{add_interval_to_date32, parse_date32_literal, subtract_interval_from_date32};
use crate::kernels::{compute_binary, get_common_type};

/// Mapping from field identifiers to the numeric Arrow array used for evaluation.
pub type NumericArrayMap<F> = FxHashMap<F, ArrayRef>;

/// Intermediate representation for vectorized evaluators.
enum VectorizedExpr {
    Array(ArrayRef),
    Scalar(ArrayRef),
}

impl VectorizedExpr {
    fn materialize(self, len: usize, target_type: DataType) -> ArrayRef {
        match self {
            VectorizedExpr::Array(array) => {
                if array.data_type() == &target_type {
                    array
                } else {
                    cast::cast(&array, &target_type).unwrap_or(array)
                }
            }
            VectorizedExpr::Scalar(scalar_array) => {
                if scalar_array.is_empty() {
                    return new_null_array(&target_type, len);
                }
                if scalar_array.is_null(0) {
                    return new_null_array(scalar_array.data_type(), len);
                }

                // Expand scalar to array of length len
                let indices = UInt32Array::from(vec![0; len]);
                take(&scalar_array, &indices, None)
                    .unwrap_or_else(|_| new_null_array(scalar_array.data_type(), len))
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

/// Centralizes the numeric kernels applied to scalar expressions so they can be
/// tuned without touching the surrounding table scan logic.
pub struct ScalarEvaluator;

impl ScalarEvaluator {
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
        arrays.clone()
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
            ScalarExpr::Literal(lit) => {
                let arr = Self::literal_to_array(lit);
                let val = cast::cast(&arr, &DataType::Float64).ok()?;
                let val = val.as_any().downcast_ref::<Float64Array>()?;
                if val.is_null(0) {
                    return None;
                }
                Some(AffineState {
                    field: None,
                    scale: 0.0,
                    offset: val.value(0),
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
    pub fn infer_result_type<F, R>(expr: &ScalarExpr<F>, resolve_type: &mut R) -> Option<DataType>
    where
        F: Hash + Eq + Copy,
        R: FnMut(F) -> Option<DataType>,
    {
        match expr {
            ScalarExpr::Literal(lit) => Some(Self::literal_type(lit)),
            ScalarExpr::Column(fid) => resolve_type(*fid),
            ScalarExpr::Binary { left, op, right } => {
                let left_type = Self::infer_result_type(left, resolve_type)?;
                let right_type = Self::infer_result_type(right, resolve_type)?;
                Some(Self::binary_result_type(*op, left_type, right_type))
            }
            ScalarExpr::Compare { .. } => Some(DataType::Boolean),
            ScalarExpr::Not(_) => Some(DataType::Boolean),
            ScalarExpr::IsNull { .. } => Some(DataType::Boolean),
            ScalarExpr::Aggregate(_) => Some(DataType::Float64), // TODO: Fix aggregate types
            ScalarExpr::GetField { .. } => None,
            ScalarExpr::Cast { data_type, .. } => Some(data_type.clone()),
            ScalarExpr::Case {
                branches,
                else_expr,
                ..
            } => {
                let mut types = Vec::new();
                for (_, then_expr) in branches {
                    if let Some(t) = Self::infer_result_type(then_expr, resolve_type) {
                        types.push(t);
                    }
                }
                if let Some(else_expr) = else_expr {
                    if let Some(t) = Self::infer_result_type(else_expr, resolve_type) {
                        types.push(t);
                    }
                } else {
                    // Implicit ELSE NULL
                    types.push(DataType::Null);
                }

                if types.is_empty() {
                    return None;
                }

                let mut common = types[0].clone();
                for t in &types[1..] {
                    common = get_common_type(&common, t);
                }

                Some(common)
            }
            ScalarExpr::Coalesce(items) => {
                let mut types = Vec::new();
                for item in items {
                    if let Some(t) = Self::infer_result_type(item, resolve_type) {
                        types.push(t);
                    }
                }
                if types.is_empty() {
                    return None;
                }
                let mut common = types[0].clone();
                for t in &types[1..] {
                    common = get_common_type(&common, t);
                }
                Some(common)
            }
            ScalarExpr::Random => Some(DataType::Float64),
            ScalarExpr::ScalarSubquery(sub) => Some(sub.data_type.clone()),
        }
    }

    fn literal_type(lit: &Literal) -> DataType {
        match lit {
            Literal::Null => DataType::Null,
            Literal::Boolean(_) => DataType::Boolean,
            Literal::Integer(_) => DataType::Int64, // Default to Int64 for literals
            Literal::Float(_) => DataType::Float64,
            Literal::Decimal(d) => DataType::Decimal128(d.precision(), d.scale()),
            Literal::String(_) => DataType::Utf8,
            Literal::Date32(_) => DataType::Date32,
            Literal::Interval(_) => {
                DataType::Interval(arrow::datatypes::IntervalUnit::MonthDayNano)
            }
            Literal::Struct(_) => DataType::Struct(arrow::datatypes::Fields::empty()), // TODO: Infer struct fields
        }
    }

    fn binary_result_type(_op: BinaryOp, lhs: DataType, rhs: DataType) -> DataType {
        get_common_type(&lhs, &rhs)
    }

    /// Evaluate a scalar expression for the row at `idx` using the provided numeric arrays.
    pub fn evaluate_value<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        idx: usize,
        arrays: &NumericArrayMap<F>,
    ) -> LlkvResult<ArrayRef> {
        match expr {
            ScalarExpr::Column(fid) => {
                let array = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal("missing column for field".into()))?;
                Ok(array.slice(idx, 1))
            }
            ScalarExpr::Literal(lit) => Ok(Self::literal_to_array(lit)),
            ScalarExpr::Binary { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                Self::evaluate_binary_scalar(&l, *op, &r)
            }
            ScalarExpr::Compare { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                crate::kernels::compute_compare(&l, *op, &r)
            }
            ScalarExpr::Not(expr) => {
                let val = Self::evaluate_value(expr, idx, arrays)?;
                let bool_arr = cast::cast(&val, &DataType::Boolean)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                let bool_arr = bool_arr
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                let result = arrow::compute::kernels::boolean::not(bool_arr)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                Ok(Arc::new(result))
            }
            ScalarExpr::IsNull { expr, negated } => {
                let val = Self::evaluate_value(expr, idx, arrays)?;
                let is_null = val.is_null(0);
                let result = if *negated { !is_null } else { is_null };
                Ok(Arc::new(arrow::array::BooleanArray::from(vec![result])))
            }
            ScalarExpr::Cast { expr, data_type } => {
                let val = Self::evaluate_value(expr, idx, arrays)?;
                cast::cast(&val, data_type).map_err(|e| Error::Internal(e.to_string()))
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand_val = if let Some(op) = operand {
                    Some(Self::evaluate_value(op, idx, arrays)?)
                } else {
                    None
                };

                for (when_expr, then_expr) in branches {
                    let when_val = Self::evaluate_value(when_expr, idx, arrays)?;

                    let is_match = if let Some(op_val) = &operand_val {
                        // Simple CASE: operand = when_val
                        // If either is null, result is null (false for condition)
                        if op_val.is_null(0) || when_val.is_null(0) {
                            false
                        } else {
                            let eq =
                                crate::kernels::compute_compare(op_val, CompareOp::Eq, &when_val)?;
                            let bool_arr = eq
                                .as_any()
                                .downcast_ref::<arrow::array::BooleanArray>()
                                .unwrap();
                            bool_arr.value(0)
                        }
                    } else {
                        // Searched CASE: when_val is boolean condition
                        if when_val.is_null(0) {
                            false
                        } else {
                            let bool_arr = cast::cast(&when_val, &DataType::Boolean)
                                .map_err(|e| Error::Internal(e.to_string()))?;
                            let bool_arr = bool_arr
                                .as_any()
                                .downcast_ref::<arrow::array::BooleanArray>()
                                .unwrap();
                            bool_arr.value(0)
                        }
                    };

                    if is_match {
                        return Self::evaluate_value(then_expr, idx, arrays);
                    }
                }
                if let Some(else_expr) = else_expr {
                    Self::evaluate_value(else_expr, idx, arrays)
                } else {
                    Ok(new_null_array(&DataType::Null, 1))
                }
            }
            ScalarExpr::Coalesce(items) => {
                for item in items {
                    let val = Self::evaluate_value(item, idx, arrays)?;
                    if !val.is_null(0) && val.data_type() != &DataType::Null {
                        return Ok(val);
                    }
                }
                Ok(new_null_array(&DataType::Null, 1))
            }
            ScalarExpr::Random => {
                let val = rand::random::<f64>();
                Ok(Arc::new(Float64Array::from(vec![val])))
            }
            _ => Err(Error::Internal("Unsupported scalar expression".into())),
        }
    }

    fn literal_to_array(lit: &Literal) -> ArrayRef {
        match lit {
            Literal::Null => new_null_array(&DataType::Null, 1),
            Literal::Boolean(b) => Arc::new(arrow::array::BooleanArray::from(vec![*b])),
            Literal::Integer(i) => Arc::new(arrow::array::Int64Array::from(vec![*i as i64])),
            Literal::Float(f) => Arc::new(Float64Array::from(vec![*f])),
            Literal::Decimal(d) => {
                let array = arrow::array::Decimal128Array::from(vec![Some(d.raw_value())])
                    .with_precision_and_scale(d.precision(), d.scale())
                    .unwrap();
                Arc::new(array)
            }
            Literal::String(s) => Arc::new(arrow::array::StringArray::from(vec![s.clone()])),
            Literal::Date32(d) => Arc::new(arrow::array::Date32Array::from(vec![*d])),
            Literal::Interval(i) => {
                let val = IntervalMonthDayNanoType::make_value(i.months, i.days, i.nanos);
                Arc::new(arrow::array::IntervalMonthDayNanoArray::from(vec![val]))
            }
            Literal::Struct(_) => {
                new_null_array(&DataType::Struct(arrow::datatypes::Fields::empty()), 1)
            }
        }
    }

    fn evaluate_binary_scalar(
        lhs: &ArrayRef,
        op: BinaryOp,
        rhs: &ArrayRef,
    ) -> LlkvResult<ArrayRef> {
        compute_binary(lhs, rhs, op)
    }

    /// Evaluate a scalar expression for every row in the batch.
    #[allow(dead_code)]
    pub fn evaluate_batch<F: Hash + Eq + Copy + std::fmt::Debug>(
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
        let preferred = Self::infer_result_type_from_arrays(expr, arrays);

        if len == 0 {
            return Ok(new_null_array(&preferred, 0));
        }

        if let Some(vectorized) =
            Self::try_evaluate_vectorized(expr, len, arrays, preferred.clone())?
        {
            let result = vectorized.materialize(len, preferred);
            return Ok(result);
        }

        let mut values = Vec::with_capacity(len);
        for idx in 0..len {
            let val = Self::evaluate_value(expr, idx, arrays)?;
            if val.data_type() != &preferred {
                let casted = cast::cast(&val, &preferred).map_err(|e| {
                    Error::Internal(format!(
                        "Failed to cast row {}: {} (Val type: {:?}, Preferred: {:?})",
                        idx,
                        e,
                        val.data_type(),
                        preferred
                    ))
                })?;
                values.push(casted);
            } else {
                values.push(val);
            }
        }
        concat(&values.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
            .map_err(|e| Error::Internal(e.to_string()))
    }

    fn try_evaluate_vectorized<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        len: usize,
        arrays: &NumericArrayMap<F>,
        _target_type: DataType,
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
            ScalarExpr::Literal(lit) => {
                let array = Self::literal_to_array(lit);
                Ok(Some(VectorizedExpr::Scalar(array)))
            }
            ScalarExpr::Binary { left, op, right } => {
                let left_type = Self::infer_result_type_from_arrays(left, arrays);
                let right_type = Self::infer_result_type_from_arrays(right, arrays);

                let left_vec = Self::try_evaluate_vectorized(left, len, arrays, left_type)?;
                let right_vec = Self::try_evaluate_vectorized(right, len, arrays, right_type)?;

                match (left_vec, right_vec) {
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let result = compute_binary(&lhs, &rhs, *op)?;
                        Ok(Some(VectorizedExpr::Scalar(result)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let array = compute_binary(&lhs, &rhs, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let rhs_expanded = VectorizedExpr::Scalar(rhs)
                            .materialize(lhs.len(), lhs.data_type().clone());
                        let array = compute_binary(&lhs, &rhs_expanded, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let lhs_expanded = VectorizedExpr::Scalar(lhs)
                            .materialize(rhs.len(), rhs.data_type().clone());
                        let array = compute_binary(&lhs_expanded, &rhs, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Cast { expr, data_type } => {
                let inner_type = Self::infer_result_type_from_arrays(expr, arrays);
                let inner_vec = Self::try_evaluate_vectorized(expr, len, arrays, inner_type)?;

                match inner_vec {
                    Some(VectorizedExpr::Scalar(array)) => {
                        let casted = cast::cast(&array, data_type)
                            .map_err(|e| Error::Internal(e.to_string()))?;
                        Ok(Some(VectorizedExpr::Scalar(casted)))
                    }
                    Some(VectorizedExpr::Array(array)) => {
                        let casted = cast::cast(&array, data_type)
                            .map_err(|e| Error::Internal(e.to_string()))?;
                        Ok(Some(VectorizedExpr::Array(casted)))
                    }
                    None => Ok(None),
                }
            }
            ScalarExpr::Coalesce(items) => {
                let mut evaluated_items = Vec::with_capacity(items.len());
                let mut types = Vec::with_capacity(items.len());

                for item in items {
                    let item_type = Self::infer_result_type_from_arrays(item, arrays);
                    // If any item cannot be vectorized, we cannot vectorize the whole Coalesce
                    let vec_expr = match Self::try_evaluate_vectorized(
                        item,
                        len,
                        arrays,
                        item_type.clone(),
                    )? {
                        Some(v) => v,
                        None => return Ok(None),
                    };

                    let array = vec_expr.materialize(len, item_type.clone());
                    types.push(array.data_type().clone());
                    evaluated_items.push(array);
                }

                if evaluated_items.is_empty() {
                    return Ok(Some(VectorizedExpr::Array(new_null_array(
                        &DataType::Null,
                        len,
                    ))));
                }

                // Determine common type
                let mut common_type = types[0].clone();
                for t in &types[1..] {
                    common_type = get_common_type(&common_type, t);
                }

                // Cast all arrays to common type
                let mut casted_arrays = Vec::with_capacity(evaluated_items.len());
                for array in evaluated_items {
                    if array.data_type() != &common_type {
                        let casted = cast::cast(&array, &common_type)
                            .map_err(|e| Error::Internal(e.to_string()))?;
                        casted_arrays.push(casted);
                    } else {
                        casted_arrays.push(array);
                    }
                }

                let mut result = casted_arrays[0].clone();
                for next_array in &casted_arrays[1..] {
                    let mask = is_not_null(&result).map_err(|e| Error::Internal(e.to_string()))?;
                    // result = zip(mask, result, next_array)
                    // if mask is true (result is not null), keep result.
                    // if mask is false (result is null), take next_array.
                    result = zip(&mask, &result, next_array)
                        .map_err(|e| Error::Internal(e.to_string()))?;
                }
                Ok(Some(VectorizedExpr::Array(result)))
            }
            ScalarExpr::Random => {
                let values: Vec<f64> = (0..len).map(|_| rand::random::<f64>()).collect();
                let array = Float64Array::from(values);
                Ok(Some(VectorizedExpr::Array(Arc::new(array))))
            }
            _ => Ok(None),
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
                // TODO: Restore constant folding using ScalarValue
                ScalarExpr::Binary {
                    left: Box::new(l),
                    op: *op,
                    right: Box::new(r),
                }
            }
            ScalarExpr::Cast { expr, data_type } => {
                let inner = Self::simplify(expr);
                ScalarExpr::Cast {
                    expr: Box::new(inner),
                    data_type: data_type.clone(),
                }
            }
            _ => expr.clone(),
        }
    }

    pub fn evaluate_constant_literal_expr<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
    ) -> LlkvResult<Option<Literal>> {
        let simplified = Self::simplify(expr);

        if let Some(literal) = Self::evaluate_constant_literal_non_numeric(&simplified)? {
            return Ok(Some(literal));
        }

        if let ScalarExpr::Literal(lit) = &simplified {
            return Ok(Some(lit.clone()));
        }

        let arrays = NumericArrayMap::default();
        let array = Self::evaluate_value(&simplified, 0, &arrays)?;
        if array.is_null(0) {
            return Ok(None);
        }
        Ok(Some(Literal::from_array_ref(&array, 0)?))
    }

    pub fn evaluate_constant_literal_non_numeric<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
    ) -> LlkvResult<Option<Literal>> {
        match expr {
            ScalarExpr::Literal(lit) => Ok(Some(lit.clone())),
            ScalarExpr::Cast {
                expr,
                data_type: DataType::Date32,
            } => {
                let inner = Self::evaluate_constant_literal_non_numeric(expr)?;
                match inner {
                    Some(Literal::Null) => Ok(Some(Literal::Null)),
                    Some(Literal::String(text)) => {
                        let days = parse_date32_literal(&text)?;
                        Ok(Some(Literal::Date32(days)))
                    }
                    Some(Literal::Date32(days)) => Ok(Some(Literal::Date32(days))),
                    Some(other) => Err(Error::InvalidArgumentError(format!(
                        "cannot cast literal of type {} to DATE",
                        other.type_name()
                    ))),
                    None => Ok(None),
                }
            }
            ScalarExpr::Cast { .. } => Ok(None),
            ScalarExpr::Binary { left, op, right } => {
                let left_lit = match Self::evaluate_constant_literal_non_numeric(left)? {
                    Some(lit) => lit,
                    None => return Ok(None),
                };
                let right_lit = match Self::evaluate_constant_literal_non_numeric(right)? {
                    Some(lit) => lit,
                    None => return Ok(None),
                };

                if matches!(left_lit, Literal::Null) || matches!(right_lit, Literal::Null) {
                    return Ok(Some(Literal::Null));
                }

                match op {
                    BinaryOp::Add => match (&left_lit, &right_lit) {
                        (Literal::Date32(days), Literal::Interval(interval))
                        | (Literal::Interval(interval), Literal::Date32(days)) => {
                            let adjusted = add_interval_to_date32(*days, *interval)?;
                            Ok(Some(Literal::Date32(adjusted)))
                        }
                        (Literal::Interval(left), Literal::Interval(right)) => {
                            let sum = left.checked_add(*right).ok_or_else(|| {
                                Error::InvalidArgumentError(
                                    "interval addition overflow during constant folding".into(),
                                )
                            })?;
                            Ok(Some(Literal::Interval(sum)))
                        }
                        _ => Ok(None),
                    },
                    BinaryOp::Subtract => match (&left_lit, &right_lit) {
                        (Literal::Date32(days), Literal::Interval(interval)) => {
                            let adjusted = subtract_interval_from_date32(*days, *interval)?;
                            Ok(Some(Literal::Date32(adjusted)))
                        }
                        (Literal::Interval(left), Literal::Interval(right)) => {
                            let diff = left.checked_sub(*right).ok_or_else(|| {
                                Error::InvalidArgumentError(
                                    "interval subtraction overflow during constant folding".into(),
                                )
                            })?;
                            Ok(Some(Literal::Interval(diff)))
                        }
                        (Literal::Date32(lhs), Literal::Date32(rhs)) => {
                            let delta = i64::from(*lhs) - i64::from(*rhs);
                            if delta < i64::from(i32::MIN) || delta > i64::from(i32::MAX) {
                                return Err(Error::InvalidArgumentError(
                                    "DATE subtraction overflowed day precision".into(),
                                ));
                            }
                            Ok(Some(Literal::Interval(IntervalValue::new(
                                0,
                                delta as i32,
                                0,
                            ))))
                        }
                        _ => Ok(None),
                    },
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    pub fn is_supported_numeric(dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::UInt64
                | DataType::UInt32
                | DataType::UInt16
                | DataType::UInt8
                | DataType::Int64
                | DataType::Int32
                | DataType::Int16
                | DataType::Int8
                | DataType::Float64
                | DataType::Float32
        )
    }

    // TODO: Move to ScalarExpr impl?
    fn infer_result_type_from_arrays<F: Hash + Eq + Copy>(
        expr: &ScalarExpr<F>,
        arrays: &NumericArrayMap<F>,
    ) -> DataType {
        let mut resolver = |fid| arrays.get(&fid).map(|a| a.data_type().clone());
        Self::infer_result_type(expr, &mut resolver).unwrap_or(DataType::Float64)
    }

    // TODO: Move to ScalarExpr impl?
    fn expr_contains_interval<F: Hash + Eq + Copy>(expr: &ScalarExpr<F>) -> bool {
        match expr {
            ScalarExpr::Literal(Literal::Interval(_)) => true,
            ScalarExpr::Binary { left, right, .. } => {
                Self::expr_contains_interval(left) || Self::expr_contains_interval(right)
            }
            _ => false,
        }
    }

    #[allow(dead_code)]
    fn affine_add<F: Eq + Copy>(
        lhs: AffineState<F>,
        rhs: AffineState<F>,
    ) -> Option<AffineState<F>> {
        let merged_field = Self::merge_field(lhs.field, rhs.field)?;
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
        let merged_field = Self::merge_field(lhs.field, rhs.field)?;
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
}
