use crate::types::{FieldId, LogicalFieldId, Namespace, TableId};

/// Reserved field id used by the table-level row-id shadow column.
pub const TABLE_ROW_ID_FIELD_ID: FieldId = u32::MAX;

// TODO: Dedupe in tests and benches
/// Sets the shadow row_id tag on a LogicalFieldId using the Namespace enum.
pub fn rowid_fid(fid: LogicalFieldId) -> LogicalFieldId {
    fid.with_namespace(Namespace::RowIdShadow)
}

/// Logical field identifier for the table-level row_id shadow column.
pub fn table_rowid_fid(table_id: TableId) -> LogicalFieldId {
    LogicalFieldId::from_parts(Namespace::RowIdShadow, table_id, TABLE_ROW_ID_FIELD_ID)
}
