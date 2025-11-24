//! Numeric scalar expression evaluation utilities for table scans.
//!
//! This module now delegates to `llkv_compute::eval::ScalarEvaluator` to centralize
//! compute logic.

use arrow::array::ArrayRef;
use arrow::datatypes::DataType;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{CompareOp, ScalarExpr};
use llkv_result::Result as LlkvResult;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::types::FieldId;

// Re-export types used by consumers of this module
pub use llkv_compute::eval::{AffineExpr, ScalarEvaluator};
pub use llkv_compute::{
    NumericArray, NumericKind, NumericValue, compute_binary, compute_binary_scalar,
};

pub type NumericArrayMap = llkv_compute::eval::NumericArrayMap<FieldId>;

/// Wrapper around `ScalarEvaluator` to maintain backward compatibility
/// with the existing `NumericKernels` API in `llkv-table`.
pub struct NumericKernels;

impl NumericKernels {
    /// Collect every field referenced by the scalar expression into `acc`.
    pub fn collect_fields(expr: &ScalarExpr<FieldId>, acc: &mut FxHashSet<FieldId>) {
        ScalarEvaluator::collect_fields(expr, acc)
    }

    /// Evaluate a scalar expression for the row at `idx` using the provided numeric arrays.
    pub fn evaluate_value(
        expr: &ScalarExpr<FieldId>,
        idx: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<Option<NumericValue>> {
        ScalarEvaluator::evaluate_value(expr, idx, arrays)
    }

    /// Evaluate a scalar expression for every row in the batch.
    pub fn evaluate_batch(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<ArrayRef> {
        ScalarEvaluator::evaluate_batch(expr, len, arrays)
    }

    /// Evaluate a scalar expression that has already been simplified.
    pub fn evaluate_batch_simplified(
        expr: &ScalarExpr<FieldId>,
        len: usize,
        arrays: &NumericArrayMap,
    ) -> LlkvResult<ArrayRef> {
        ScalarEvaluator::evaluate_batch_simplified(expr, len, arrays)
    }

    /// Returns the column referenced by an expression when it's a direct or additive identity passthrough.
    pub fn passthrough_column(expr: &ScalarExpr<FieldId>) -> Option<FieldId> {
        ScalarEvaluator::passthrough_column(expr)
    }

    /// Simplify an expression by constant folding and identity removal.
    pub fn simplify(expr: &ScalarExpr<FieldId>) -> ScalarExpr<FieldId> {
        ScalarEvaluator::simplify(expr)
    }

    /// Attempts to represent the expression as `scale * column + offset`.
    pub fn extract_affine(expr: &ScalarExpr<FieldId>) -> Option<AffineExpr<FieldId>> {
        ScalarEvaluator::extract_affine(expr)
    }

    /// Infer the numeric kind of an expression using only the kinds of its referenced columns.
    pub fn infer_result_kind_from_types<F>(
        expr: &ScalarExpr<FieldId>,
        resolve_kind: &mut F,
    ) -> Option<NumericKind>
    where
        F: FnMut(FieldId) -> Option<NumericKind>,
    {
        ScalarEvaluator::infer_result_kind_from_types(expr, resolve_kind)
    }

    /// Map an Arrow `DataType` to the corresponding numeric kind when supported.
    pub fn kind_for_data_type(dtype: &DataType) -> Option<NumericKind> {
        ScalarEvaluator::kind_for_data_type(dtype)
    }

    /// Compare two numeric values using the provided operator.
    pub fn compare(op: CompareOp, lhs: NumericValue, rhs: NumericValue) -> bool {
        ScalarEvaluator::compare(op, lhs, rhs)
    }

    /// Ensure each referenced column is materialized as a `NumericArray`, casting as needed.
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
            let numeric = NumericArray::try_from_arrow(array)?;
            out.insert(fid, numeric);
        }
        Ok(out)
    }
}
