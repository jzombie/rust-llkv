#![forbid(unsafe_code)]

use crate::constraints::ConstraintId;
use crate::constraints::ForeignKeyAction;
use crate::types::{FieldId, TableId};

/// Read-only view describing a resolved foreign key constraint.
#[derive(Clone, Debug)]
pub struct ForeignKeyView {
    pub constraint_id: ConstraintId,
    pub constraint_name: Option<String>,
    pub referencing_table_id: TableId,
    pub referencing_table_display: String,
    pub referencing_table_canonical: String,
    pub referencing_field_ids: Vec<FieldId>,
    pub referencing_column_names: Vec<String>,
    pub referenced_table_id: TableId,
    pub referenced_table_display: String,
    pub referenced_table_canonical: String,
    pub referenced_field_ids: Vec<FieldId>,
    pub referenced_column_names: Vec<String>,
    pub on_delete: ForeignKeyAction,
    pub on_update: ForeignKeyAction,
}
