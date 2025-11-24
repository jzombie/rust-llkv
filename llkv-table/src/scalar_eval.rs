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
pub use llkv_compute::compute_binary;
pub use llkv_compute::eval::{AffineExpr, ScalarEvaluator};

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
    ) -> LlkvResult<ArrayRef> {
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

    /// Infer the result type of an expression using the types of its referenced columns.
    pub fn infer_result_type<F>(
        expr: &ScalarExpr<FieldId>,
        resolve_type: &mut F,
    ) -> Option<DataType>
    where
        F: FnMut(FieldId) -> Option<DataType>,
    {
        ScalarEvaluator::infer_result_type(expr, resolve_type)
    }
}
