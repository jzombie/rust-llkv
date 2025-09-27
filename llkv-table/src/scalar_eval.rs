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

                Ok(None)
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
