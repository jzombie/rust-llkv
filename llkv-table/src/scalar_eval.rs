use std::{iter, sync::Arc};

use arrow::array::{Array, ArrayRef, Float64Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{BinaryOp, CompareOp, ScalarExpr};
use llkv_result::{Error, Result as LlkvResult};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::types::FieldId;

/// Mapping from field identifiers to the numeric Arrow array used for evaluation.
pub type NumericArrayMap = FxHashMap<FieldId, Arc<Float64Array>>;

#[derive(Clone, Copy, Debug)]
pub struct AffineExpr {
    pub field: FieldId,
    pub scale: f64,
    pub offset: f64,
}

#[derive(Clone, Copy, Debug)]
struct AffineState {
    field: Option<FieldId>,
    scale: f64,
    offset: f64,
}

fn merge_field(lhs: Option<FieldId>, rhs: Option<FieldId>) -> Option<Option<FieldId>> {
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

enum VectorizedExpr {
    Array(Arc<Float64Array>),
    Scalar(Option<f64>),
}

impl VectorizedExpr {
    fn materialize(self, len: usize) -> ArrayRef {
        match self {
            VectorizedExpr::Array(array) => array as ArrayRef,
            VectorizedExpr::Scalar(Some(value)) => {
                let iter = iter::repeat_n(Some(value), len);
                let array = Float64Array::from_iter(iter);
                Arc::new(array) as ArrayRef
            }
            VectorizedExpr::Scalar(None) => {
                let iter = iter::repeat_n(None, len);
                let array = Float64Array::from_iter(iter);
                Arc::new(array) as ArrayRef
            }
        }
    }
}

/// Centralizes the numeric kernels applied to scalar expressions so they can be
/// tuned without touching the surrounding table scan logic.
pub struct NumericKernels;

impl NumericKernels {
    /// Collect every field referenced by the scalar expression into `acc`.
    pub fn collect_fields(expr: &ScalarExpr<FieldId>, acc: &mut FxHashSet<FieldId>) {
        match expr {
            ScalarExpr::Column(fid) => {
                acc.insert(*fid);
            }
            ScalarExpr::Literal(_) => {}
            ScalarExpr::Binary { left, right, .. } => {
                Self::collect_fields(left, acc);
                Self::collect_fields(right, acc);
            }
            ScalarExpr::Aggregate(agg) => {
                // Collect fields referenced by the aggregate
                match agg {
                    llkv_expr::expr::AggregateCall::CountStar => {}
                    llkv_expr::expr::AggregateCall::Count(fid)
                    | llkv_expr::expr::AggregateCall::Sum(fid)
                    | llkv_expr::expr::AggregateCall::Min(fid)
                    | llkv_expr::expr::AggregateCall::Max(fid)
                    | llkv_expr::expr::AggregateCall::CountNulls(fid) => {
                        acc.insert(*fid);
                    }
                }
            }
        }
    }

    /// Ensure each referenced column has a `Float64Array`, casting on the fly when needed.
    pub fn prepare_numeric_arrays(
        lfids: &[LogicalFieldId],
        arrays: &[ArrayRef],
        needed_fields: &FxHashSet<FieldId>,
    ) -> LlkvResult<NumericArrayMap> {
        let mut out: NumericArrayMap = FxHashMap::default();
        if needed_fields.is_empty() {
            return Ok(out);
        }
        for (lfid, array) in lfids.iter().zip(arrays.iter()) {
            let fid = lfid.field_id();
            if !needed_fields.contains(&fid) {
                continue;
            }
            let f64_array = if matches!(array.data_type(), DataType::Float64) {
                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("expected Float64 array".into()))?
                    .clone()
            } else {
                let casted = cast(array.as_ref(), &DataType::Float64)
                    .map_err(|e| Error::Internal(format!("cast to Float64 failed: {e}")))?;
                casted
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("cast produced non-Float64 array".into()))?
                    .clone()
            };
            out.insert(fid, Arc::new(f64_array));
        }
        Ok(out)
    }

    /// Evaluate a scalar expression for the row at `idx` using the provided numeric arrays.
    pub fn evaluate_value(
        expr: &ScalarExpr<FieldId>,
        idx: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<Option<f64>> {
        match expr {
            ScalarExpr::Column(fid) => {
                let arr = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal(format!("missing column for field {fid}")))?;
                if arr.is_null(idx) {
                    Ok(None)
                } else {
                    Ok(Some(arr.value(idx)))
                }
            }
            ScalarExpr::Literal(lit) => match lit {
                llkv_expr::literal::Literal::Float(f) => Ok(Some(*f)),
                llkv_expr::literal::Literal::Integer(i) => Ok(Some(*i as f64)),
                llkv_expr::literal::Literal::String(_) => Err(Error::InvalidArgumentError(
                    "String literals are not supported in numeric expressions".into(),
                )),
            },
            ScalarExpr::Binary { left, op, right } => {
                let l = Self::evaluate_value(left, idx, arrays)?;
                let r = Self::evaluate_value(right, idx, arrays)?;
                Ok(Self::apply_binary(*op, l, r))
            }
            ScalarExpr::Aggregate(_) => Err(Error::Internal(
                "Aggregate expressions should not appear in row-level evaluation".into(),
            )),
        }
    }

    /// Evaluate a scalar expression for every row in the batch.
    #[allow(dead_code)]
    pub fn evaluate_batch(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<ArrayRef> {
        let simplified = Self::simplify(expr);
        Self::evaluate_batch_simplified(&simplified, len, arrays)
    }

    /// Evaluate a scalar expression that has already been simplified.
    pub fn evaluate_batch_simplified(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<ArrayRef> {
        if let Some(vectorized) = Self::try_evaluate_vectorized(expr, len, arrays)? {
            return Ok(vectorized.materialize(len));
        }

        let mut values: Vec<Option<f64>> = Vec::with_capacity(len);
        for idx in 0..len {
            values.push(Self::evaluate_value(expr, idx, arrays)?);
        }
        Ok(Arc::new(Float64Array::from(values)) as ArrayRef)
    }

    fn try_evaluate_vectorized(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<Option<VectorizedExpr>> {
        match expr {
            ScalarExpr::Column(fid) => {
                let arr = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal(format!("missing column for field {fid}")))?;
                Ok(Some(VectorizedExpr::Array(Arc::clone(arr))))
            }
            ScalarExpr::Literal(lit) => match lit {
                llkv_expr::literal::Literal::Float(f) => Ok(Some(VectorizedExpr::Scalar(Some(*f)))),
                llkv_expr::literal::Literal::Integer(i) => {
                    Ok(Some(VectorizedExpr::Scalar(Some(*i as f64))))
                }
                llkv_expr::literal::Literal::String(_) => Ok(None),
            },
            ScalarExpr::Binary { left, op, right } => {
                let left_vec = Self::try_evaluate_vectorized(left, len, arrays)?;
                let right_vec = Self::try_evaluate_vectorized(right, len, arrays)?;

                match (left_vec, right_vec) {
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Scalar(rhs))) => Ok(
                        Some(VectorizedExpr::Scalar(Self::apply_binary(*op, lhs, rhs))),
                    ),
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let array =
                            Self::compute_binary_array_array(lhs.as_ref(), rhs.as_ref(), len, *op)?;
                        Ok(Some(VectorizedExpr::Array(array)))
                    }
                    (Some(VectorizedExpr::Array(lhs)), Some(VectorizedExpr::Scalar(rhs))) => {
                        let value =
                            Self::compute_binary_array_scalar(lhs.as_ref(), rhs, len, *op, true)?;
                        Ok(Some(value))
                    }
                    (Some(VectorizedExpr::Scalar(lhs)), Some(VectorizedExpr::Array(rhs))) => {
                        let value =
                            Self::compute_binary_array_scalar(rhs.as_ref(), lhs, len, *op, false)?;
                        Ok(Some(value))
                    }
                    _ => Ok(None),
                }
            }
            ScalarExpr::Aggregate(_) => Err(Error::Internal(
                "Aggregate expressions should not appear in row-level evaluation".into(),
            )),
        }
    }

    fn compute_binary_array_array(
        left: &Float64Array,
        right: &Float64Array,
        len: usize,
        op: BinaryOp,
    ) -> LlkvResult<Arc<Float64Array>> {
        if left.len() != len || right.len() != len {
            return Err(Error::Internal("scalar expression length mismatch".into()));
        }

        let iter = (0..len).map(|idx| {
            let lhs = if left.is_null(idx) {
                None
            } else {
                Some(left.value(idx))
            };
            let rhs = if right.is_null(idx) {
                None
            } else {
                Some(right.value(idx))
            };
            Self::apply_binary(op, lhs, rhs)
        });

        let array = Float64Array::from_iter(iter);
        Ok(Arc::new(array))
    }

    fn compute_binary_array_scalar(
        array: &Float64Array,
        scalar: Option<f64>,
        len: usize,
        op: BinaryOp,
        array_is_left: bool,
    ) -> LlkvResult<VectorizedExpr> {
        if array.len() != len {
            return Err(Error::Internal("scalar expression length mismatch".into()));
        }

        if scalar.is_none() {
            return Ok(VectorizedExpr::Scalar(None));
        }
        let scalar_value = scalar.expect("checked above");

        if array_is_left && matches!(op, BinaryOp::Divide | BinaryOp::Modulo) && scalar_value == 0.0
        {
            return Ok(VectorizedExpr::Scalar(None));
        }

        let iter = (0..len).map(|idx| {
            let array_val = if array.is_null(idx) {
                None
            } else {
                Some(array.value(idx))
            };
            let (lhs, rhs) = if array_is_left {
                (array_val, Some(scalar_value))
            } else {
                (Some(scalar_value), array_val)
            };
            Self::apply_binary(op, lhs, rhs)
        });

        let array = Float64Array::from_iter(iter);
        Ok(VectorizedExpr::Array(Arc::new(array)))
    }

    /// Returns the column referenced by an expression when it's a direct or additive identity passthrough.
    pub fn passthrough_column(expr: &ScalarExpr<FieldId>) -> Option<FieldId> {
        match Self::simplify(expr) {
            ScalarExpr::Column(fid) => Some(fid),
            _ => None,
        }
    }

    fn literal_numeric_value(expr: &ScalarExpr<FieldId>) -> Option<f64> {
        if let ScalarExpr::Literal(lit) = expr {
            match lit {
                llkv_expr::literal::Literal::Float(f) => Some(*f),
                llkv_expr::literal::Literal::Integer(i) => Some(*i as f64),
                llkv_expr::literal::Literal::String(_) => None,
            }
        } else {
            None
        }
    }

    fn literal_is_value(expr: &ScalarExpr<FieldId>, expected: f64) -> bool {
        matches!(Self::literal_numeric_value(expr), Some(v) if (v - expected).abs() == 0.0)
    }

    /// Recursively simplify the expression by folding literals and eliminating identity operations.
    pub fn simplify(expr: &ScalarExpr<FieldId>) -> ScalarExpr<FieldId> {
        match expr {
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) | ScalarExpr::Aggregate(_) => {
                expr.clone()
            }
            ScalarExpr::Binary { left, op, right } => {
                let left_s = Self::simplify(left);
                let right_s = Self::simplify(right);

                if let (Some(lv), Some(rv)) = (
                    Self::literal_numeric_value(&left_s),
                    Self::literal_numeric_value(&right_s),
                ) && let Some(lit) = Self::apply_binary_literal(*op, lv, rv)
                {
                    return lit;
                }

                match op {
                    BinaryOp::Add => {
                        if Self::literal_is_value(&left_s, 0.0) {
                            return right_s;
                        }
                        if Self::literal_is_value(&right_s, 0.0) {
                            return left_s;
                        }
                    }
                    BinaryOp::Subtract => {
                        if Self::literal_is_value(&right_s, 0.0) {
                            return left_s;
                        }
                    }
                    BinaryOp::Multiply => {
                        if Self::literal_is_value(&left_s, 1.0) {
                            return right_s;
                        }
                        if Self::literal_is_value(&right_s, 1.0) {
                            return left_s;
                        }
                        if Self::literal_is_value(&left_s, 0.0)
                            || Self::literal_is_value(&right_s, 0.0)
                        {
                            return ScalarExpr::literal(0);
                        }
                    }
                    BinaryOp::Divide => {
                        if Self::literal_is_value(&right_s, 1.0) {
                            return left_s;
                        }
                    }
                    BinaryOp::Modulo => {}
                }

                ScalarExpr::binary(left_s, *op, right_s)
            }
        }
    }

    /// Attempts to represent the expression as `scale * column + offset`.
    /// Returns `None` when the expression depends on multiple columns or is non-linear.
    #[allow(dead_code)]
    pub fn extract_affine(expr: &ScalarExpr<FieldId>) -> Option<AffineExpr> {
        let simplified = Self::simplify(expr);
        Self::extract_affine_simplified(&simplified)
    }

    /// Variant of \[`extract_affine`\] that assumes `expr` is already simplified.
    pub fn extract_affine_simplified(expr: &ScalarExpr<FieldId>) -> Option<AffineExpr> {
        let state = Self::affine_state(expr)?;
        let field = state.field?;
        Some(AffineExpr {
            field,
            scale: state.scale,
            offset: state.offset,
        })
    }

    fn affine_state(expr: &ScalarExpr<FieldId>) -> Option<AffineState> {
        match expr {
            ScalarExpr::Column(fid) => Some(AffineState {
                field: Some(*fid),
                scale: 1.0,
                offset: 0.0,
            }),
            ScalarExpr::Literal(_) => {
                let value = Self::literal_numeric_value(expr)?;
                Some(AffineState {
                    field: None,
                    scale: 0.0,
                    offset: value,
                })
            }
            ScalarExpr::Aggregate(_) => None, // Aggregates not supported in affine transformations
            ScalarExpr::Binary { left, op, right } => {
                let left_state = Self::affine_state(left)?;
                let right_state = Self::affine_state(right)?;
                match op {
                    BinaryOp::Add => Self::affine_add(left_state, right_state),
                    BinaryOp::Subtract => Self::affine_sub(left_state, right_state),
                    BinaryOp::Multiply => Self::affine_mul(left_state, right_state),
                    BinaryOp::Divide => Self::affine_div(left_state, right_state),
                    BinaryOp::Modulo => None,
                }
            }
        }
    }

    fn affine_add(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
        let field = merge_field(lhs.field, rhs.field)?;
        Some(AffineState {
            field,
            scale: lhs.scale + rhs.scale,
            offset: lhs.offset + rhs.offset,
        })
    }

    fn affine_sub(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
        let neg_rhs = AffineState {
            field: rhs.field,
            scale: -rhs.scale,
            offset: -rhs.offset,
        };
        Self::affine_add(lhs, neg_rhs)
    }

    fn affine_mul(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
        if rhs.field.is_none() {
            let factor = rhs.offset;
            return Some(AffineState {
                field: lhs.field,
                scale: lhs.scale * factor,
                offset: lhs.offset * factor,
            });
        }
        if lhs.field.is_none() {
            let factor = lhs.offset;
            return Some(AffineState {
                field: rhs.field,
                scale: rhs.scale * factor,
                offset: rhs.offset * factor,
            });
        }
        None
    }

    fn affine_div(lhs: AffineState, rhs: AffineState) -> Option<AffineState> {
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

    fn apply_binary_literal(op: BinaryOp, lhs: f64, rhs: f64) -> Option<ScalarExpr<FieldId>> {
        match op {
            BinaryOp::Add => Some(Self::literal_from_f64(lhs + rhs)),
            BinaryOp::Subtract => Some(Self::literal_from_f64(lhs - rhs)),
            BinaryOp::Multiply => Some(Self::literal_from_f64(lhs * rhs)),
            BinaryOp::Divide => {
                if rhs == 0.0 {
                    None
                } else {
                    Some(Self::literal_from_f64(lhs / rhs))
                }
            }
            BinaryOp::Modulo => {
                if rhs == 0.0 {
                    None
                } else {
                    Some(Self::literal_from_f64(lhs % rhs))
                }
            }
        }
    }

    fn literal_from_f64(value: f64) -> ScalarExpr<FieldId> {
        if value.fract() == 0.0 {
            ScalarExpr::literal(value as i64)
        } else {
            ScalarExpr::literal(value)
        }
    }

    /// Apply an arithmetic kernel. Returns `None` when the computation results in a null (e.g. divide by zero).
    pub fn apply_binary(op: BinaryOp, lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
        match (lhs, rhs) {
            (Some(lv), Some(rv)) => match op {
                BinaryOp::Add => Some(lv + rv),
                BinaryOp::Subtract => Some(lv - rv),
                BinaryOp::Multiply => Some(lv * rv),
                BinaryOp::Divide => {
                    if rv == 0.0 {
                        None
                    } else {
                        Some(lv / rv)
                    }
                }
                BinaryOp::Modulo => {
                    if rv == 0.0 {
                        None
                    } else {
                        Some(lv % rv)
                    }
                }
            },
            (Some(_), None) | (None, Some(_)) => None,
            (None, None) => None,
        }
    }

    /// Compare two numeric values using the provided operator.
    pub fn compare(op: CompareOp, lhs: f64, rhs: f64) -> bool {
        match op {
            CompareOp::Eq => lhs == rhs,
            CompareOp::NotEq => lhs != rhs,
            CompareOp::Lt => lhs < rhs,
            CompareOp::LtEq => lhs <= rhs,
            CompareOp::Gt => lhs > rhs,
            CompareOp::GtEq => lhs >= rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use llkv_expr::Literal;

    fn array(values: &[Option<f64>]) -> Arc<Float64Array> {
        Arc::new(Float64Array::from(values.to_vec()))
    }

    #[test]
    fn vectorized_add_columns() {
        const F1: FieldId = 1;
        const F2: FieldId = 2;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, array(&[Some(1.0), Some(2.0), None, Some(-1.0)]));
        arrays.insert(F2, array(&[Some(5.0), Some(-1.0), Some(3.0), Some(4.0)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::column(F2),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 6.0);
        assert_eq!(result.value(1), 1.0);
        assert!(result.is_null(2));
        assert_eq!(result.value(3), 3.0);
    }

    #[test]
    fn vectorized_multiply_literal() {
        const F1: FieldId = 10;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, array(&[Some(1.0), Some(-2.5), Some(0.0), None]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Multiply,
            ScalarExpr::literal(3),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 3.0);
        assert_eq!(result.value(1), -7.5);
        assert_eq!(result.value(2), 0.0);
        assert!(result.is_null(3));
    }

    #[test]
    fn vectorized_add_column_scalar_literal() {
        const F1: FieldId = 11;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, array(&[Some(2.0), None, Some(-5.5)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(4),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.value(0), 6.0);
        assert!(result.is_null(1));
        assert!((result.value(2) - (-1.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn vectorized_literal_minus_column() {
        const F1: FieldId = 12;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, array(&[Some(3.0), Some(-2.0), None]));

        let expr = ScalarExpr::binary(
            ScalarExpr::literal(10),
            BinaryOp::Subtract,
            ScalarExpr::column(F1),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.value(0), 7.0);
        assert_eq!(result.value(1), 12.0);
        assert!(result.is_null(2));
    }

    #[test]
    fn vectorized_divide_by_zero_yields_null() {
        const NUM: FieldId = 20;
        const DEN: FieldId = 21;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(NUM, array(&[Some(4.0), Some(9.0), Some(5.0), Some(-6.0)]));
        arrays.insert(DEN, array(&[Some(2.0), Some(0.0), None, Some(-3.0)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(NUM),
            BinaryOp::Divide,
            ScalarExpr::column(DEN),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 2.0);
        assert!(result.is_null(1));
        assert!(result.is_null(2));
        assert_eq!(result.value(3), 2.0);
    }

    #[test]
    fn vectorized_divide_by_zero_literal_rhs_yields_nulls() {
        const F1: FieldId = 22;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(F1, array(&[Some(1.0), Some(-4.0), None]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Divide,
            ScalarExpr::literal(0),
        );

        let result = NumericKernels::evaluate_batch(&expr, 3, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.is_null(0));
        assert!(result.is_null(1));
        assert!(result.is_null(2));
    }

    #[test]
    fn vectorized_modulo_literals() {
        let expr = ScalarExpr::binary(
            ScalarExpr::literal(13),
            BinaryOp::Modulo,
            ScalarExpr::literal(5),
        );

        let simplified = NumericKernels::simplify(&expr);
        let ScalarExpr::Literal(Literal::Integer(value)) = simplified else {
            panic!("expected literal result");
        };
        assert_eq!(value, 3);
    }

    #[test]
    fn vectorized_modulo_column_rhs_zero_yields_null() {
        const NUM: FieldId = 23;
        const DEN: FieldId = 24;
        let mut arrays: NumericArrayMap = NumericArrayMap::default();
        arrays.insert(NUM, array(&[Some(4.0), Some(7.0), None, Some(-6.0)]));
        arrays.insert(DEN, array(&[Some(2.0), Some(0.0), Some(3.0), Some(-4.0)]));

        let expr = ScalarExpr::binary(
            ScalarExpr::column(NUM),
            BinaryOp::Modulo,
            ScalarExpr::column(DEN),
        );

        let result = NumericKernels::evaluate_batch(&expr, 4, &arrays).unwrap();
        let result = result
            .as_ref()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.value(0), 0.0);
        assert!(result.is_null(1));
        assert!(result.is_null(2));
        assert_eq!(result.value(3), -6.0 % -4.0);
    }

    #[test]
    fn passthrough_detects_identity_ops() {
        const F1: FieldId = 99;

        let expr_add = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(0),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_add), Some(F1));

        let expr_sub = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Subtract,
            ScalarExpr::literal(0),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_sub), Some(F1));

        let expr_mul = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Multiply,
            ScalarExpr::literal(1),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_mul), Some(F1));

        let expr_div = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Divide,
            ScalarExpr::literal(1),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_div), Some(F1));

        // Non-identity literal should not passthrough.
        let expr_add_two = ScalarExpr::binary(
            ScalarExpr::column(F1),
            BinaryOp::Add,
            ScalarExpr::literal(2),
        );
        assert_eq!(NumericKernels::passthrough_column(&expr_add_two), None);
    }
}
