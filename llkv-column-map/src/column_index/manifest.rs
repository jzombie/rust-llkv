use super::column_entry::ColumnEntry;
use bitcode::{Decode, Encode};

/// Top-level directory of all columns.
///
/// The `Manifest` maps **logical columns** (by `field_id`) to the *current*
/// `ColumnIndex` for each column. The `Manifest` itself is small and updated
/// when a column’s index is replaced (e.g., after sealing new segments or
/// performing compaction).
///
/// Encoding: serialized as a typed blob via `bitcode`.
#[derive(Debug, Clone, Encode, Decode)]
pub struct Manifest {
    pub columns: Vec<ColumnEntry>,
}
