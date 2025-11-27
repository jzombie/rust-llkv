//! Numeric scalar expression evaluation utilities for table scans.
//!
//! This module now delegates to `llkv_compute::eval::ScalarEvaluator` to centralize
//! compute logic.

use crate::types::FieldId;

// Re-export types used by consumers of this module
pub use llkv_compute::compute_binary;
pub use llkv_compute::eval::{AffineExpr, ScalarEvaluator};

pub type NumericArrayMap = llkv_compute::eval::NumericArrayMap<FieldId>;

// TODO: Migrate to llkv-compute
/// Wrapper around `ScalarEvaluator` to maintain backward compatibility
/// with the existing `NumericKernels` API in `llkv-table`.
pub type NumericKernels = ScalarEvaluator;
