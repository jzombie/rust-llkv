use crate::types::{LogicalFieldId, PhysicalKey};

/// One column listed in the Manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnEntry {
    pub field_id: LogicalFieldId,
    pub column_index_physical_key: PhysicalKey,
}
