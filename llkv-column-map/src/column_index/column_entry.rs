use crate::types::{LogicalFieldId, PhysicalKey};
use bitcode::{Decode, Encode};

/// Pointer from a logical column to its current `ColumnIndex`.
///
/// This indirection makes `ColumnIndex` immutable: when segments change, a new
/// `ColumnIndex` is written and the `Manifest` is updated to point at it.
/// Readers can then fetch the exact `ColumnIndex` version referenced by the
/// `Manifest` they opened.
#[derive(Debug, Clone, Encode, Decode)]
pub struct ColumnEntry {
    pub field_id: LogicalFieldId,
    /// Physical key of the current ColumnIndex blob for this column.
    pub column_index_physical_key: PhysicalKey,
}
