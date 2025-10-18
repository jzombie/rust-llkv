//! Common types for the table core.

#![forbid(unsafe_code)]

use arrow::datatypes::DataType;

pub use llkv_column_map::types::{FieldId, ROW_ID_FIELD_ID, RowId, TableId};

/// Column definition with assigned field id used when registering table metadata.
#[derive(Clone, Debug)]
pub struct TableColumn {
    pub field_id: FieldId,
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub check_expr: Option<String>,
}
