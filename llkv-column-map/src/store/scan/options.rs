use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    pub sorted: bool,
    pub reverse: bool,
    pub with_row_ids: bool,
    // TODO: Can this be inferred w/o perf cost?
    /// When `with_row_ids` is true, the LogicalFieldId of the row-id column to pair with.
    pub row_id_field: Option<LogicalFieldId>,
    // TODO: Include ability to include nulls in output (will require using `with_row_ids`)
    /// Maximum number of items to return. Applies across chunks/runs.
    /// None means unbounded.
    pub limit: Option<usize>,
    /// Number of leading items to skip before emitting. Applies across chunks/runs.
    /// Defaults to 0.
    pub offset: usize,
}
