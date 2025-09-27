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
        let mut values: Vec<Option<f64>> = Vec::with_capacity(len);
        for idx in 0..len {
            values.push(Self::evaluate_value(expr, idx, arrays)?);
        }
        Ok(Arc::new(Float64Array::from(values)) as ArrayRef)
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
            _ => None,
        }
    }

    /// Apply a comparison kernel.
    pub fn compare(op: CompareOp, left: f64, right: f64) -> bool {
        match op {
            CompareOp::Eq => left == right,
            CompareOp::NotEq => left != right,
            CompareOp::Lt => left < right,
            CompareOp::LtEq => left <= right,
            CompareOp::Gt => left > right,
            CompareOp::GtEq => left >= right,
        }
    }
}