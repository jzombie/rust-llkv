//! Configuration flags controlling scan behavior.
//!
//! Options capture sorting, pagination, row-id emission, and null-handling.
//! They are intentionally lightweight so callers can build them inline.

use super::*;
use crate::store::scan::ranges::IntRanges;

/// Hint flags that shape [`crate::store::scan::ScanBuilder`] iteration semantics.
///
/// Options control ordering, pagination, row-id emission, and null handling while remaining
/// lightweight enough to build inline around hot scan paths.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    pub sorted: bool,
    pub reverse: bool,
    pub with_row_ids: bool,
    // NOTE: Null emission relies on `with_row_ids`; richer output wiring is tracked separately.
    /// Maximum number of items to return. Applies across chunks/runs.
    /// None means unbounded.
    pub limit: Option<usize>,
    /// Number of leading items to skip before emitting. Applies across chunks/runs.
    /// Defaults to 0.
    pub offset: usize,
    /// When true, include rows missing from the target column (nulls).
    /// Requires `with_row_ids=true` and `anchor_row_id_field` set.
    pub include_nulls: bool,
    /// When including nulls, place them before non-nulls (NULLS FIRST).
    pub nulls_first: bool,
    /// Row-id column LogicalFieldId defining the row universe for null detection.
    pub anchor_row_id_field: Option<LogicalFieldId>,
    /// Optional range bounds for pruning chunks.
    pub ranges: IntRanges,
}
