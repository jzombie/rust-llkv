use super::column_entry::ColumnEntry;

/// Top-level directory of all columns.
///
/// The `Manifest` maps logical columns (by `field_id`) to the current
/// `ColumnIndex` for each column. It is small and updated when a
/// column index is replaced (e.g., after sealing or compaction).
#[derive(Debug, Clone)]
pub struct Manifest {
    pub columns: Vec<ColumnEntry>,
}
