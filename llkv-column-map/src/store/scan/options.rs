use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    pub sorted: bool,
    pub reverse: bool,
    pub with_row_ids: bool,
    // TODO: Include ability to include nulls in output (will require using `with_row_ids`)
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
}
