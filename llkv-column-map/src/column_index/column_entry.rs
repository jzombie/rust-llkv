use crate::types::{LogicalFieldId, PhysicalKey};
use bitcode::{Decode, Encode};

#[derive(Debug, Clone, Encode, Decode)]
pub struct ColumnEntry {
    pub field_id: LogicalFieldId,
    /// Physical key of the current ColumnIndex blob for this column.
    pub column_index_physical_key: PhysicalKey,
}
