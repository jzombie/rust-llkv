use std::sync::Arc;

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
        }
    }

    /// Evaluate a scalar expression for every row in the batch.
    pub fn evaluate_batch(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<ArrayRef> {
        if let Some(array) = Self::try_evaluate_vectorized(expr, len, arrays)? {
            return Ok(array);
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
    ) -> LlkvResult<Option<ArrayRef>> {
        match expr {
            ScalarExpr::Column(fid) => {
                let arr = arrays
                    .get(fid)
                    .ok_or_else(|| Error::Internal(format!("missing column for field {fid}")))?;
                Ok(Some(Arc::clone(arr) as ArrayRef))
            }
            ScalarExpr::Literal(lit) => match lit {
                llkv_expr::literal::Literal::Float(f) => {
                    let array = Float64Array::from(vec![Some(*f); len]);
                    Ok(Some(Arc::new(array) as ArrayRef))
                }
                llkv_expr::literal::Literal::Integer(i) => {
                    let array = Float64Array::from(vec![Some(*i as f64); len]);
                    Ok(Some(Arc::new(array) as ArrayRef))
                }
                llkv_expr::literal::Literal::String(_) => Ok(None),
            },
            ScalarExpr::Binary { left, op, right } => {
                if *op == BinaryOp::Add {
                    if Self::literal_is_zero(left) {
                        return Self::try_evaluate_vectorized(right, len, arrays);
                    }
                    if Self::literal_is_zero(right) {
                        return Self::try_evaluate_vectorized(left, len, arrays);
                    }
                }
                let left_array = Self::try_evaluate_vectorized(left, len, arrays)?;
                let right_array = Self::try_evaluate_vectorized(right, len, arrays)?;

                match (left_array, right_array) {
                    (Some(left_arr), Some(right_arr)) => {
                        let left_float = left_arr
                            .as_ref()
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| Error::Internal("expected Float64 array".into()))?;
                        let right_float = right_arr
                            .as_ref()
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| Error::Internal("expected Float64 array".into()))?;

                        if left_float.len() != len || right_float.len() != len {
                            return Err(Error::Internal(
                                "scalar expression length mismatch".into(),
                            ));
                        }

                        let op = *op;
                        let values: Vec<Option<f64>> = (0..len)
                            .map(|idx| {
                                if left_float.is_null(idx) || right_float.is_null(idx) {
                                    None
                                } else {
                                    let lhs = left_float.value(idx);
                                    let rhs = right_float.value(idx);
                                    match op {
                                        BinaryOp::Add => Some(lhs + rhs),
                                        BinaryOp::Subtract => Some(lhs - rhs),
                                        BinaryOp::Multiply => Some(lhs * rhs),
                                        BinaryOp::Divide => {
                                            if rhs == 0.0 {
                                                None
                                            } else {
                                                Some(lhs / rhs)
                                            }
                                        }
                                    }
                                }
                            })
                            .collect();

                        let array = Float64Array::from(values);
                        Ok(Some(Arc::new(array) as ArrayRef))
                    }
                    _ => Ok(None),
                }
            }
        }
    }

    #[inline]
    fn literal_is_zero(expr: &ScalarExpr<FieldId>) -> bool {
        matches!(Self::literal_numeric_value(expr), Some(v) if v == 0.0)
    }

    /// Returns the column referenced by an expression when it's a direct or additive identity passthrough.
    pub fn passthrough_column(expr: &ScalarExpr<FieldId>) -> Option<FieldId> {
        match expr {
            ScalarExpr::Column(fid) => Some(*fid),
            ScalarExpr::Binary {
                left,
                op: BinaryOp::Add,
                right,
            } => {
                if Self::literal_is_zero(left) {
                    Self::passthrough_column(right)
                } else if Self::literal_is_zero(right) {
                    Self::passthrough_column(left)
                } else {
                    None
                }
            }
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
}
