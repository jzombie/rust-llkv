use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    pub sorted: bool,
    pub reverse: bool,
    pub with_row_ids: bool,
    // TODO: Can this be inferred w/o perf cost?
    /// When `with_row_ids` is true, the LogicalFieldId of the row-id column to pair with.
    pub row_id_field: Option<LogicalFieldId>,
}
