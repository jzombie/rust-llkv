//! Helper types used internally by RuntimeContext

use llkv_plan::PlanValue;
use llkv_table::{FieldId, InsertColumnConstraint, InsertMultiColumnUnique, InsertUniqueColumn};

/// Represents how a column assignment should be materialized during UPDATE/INSERT.
pub(crate) enum PreparedAssignmentValue {
    Literal(PlanValue),
    Expression { expr_index: usize },
}

// TODO: Move to llkv-table?
#[derive(Debug, Clone)]
pub(crate) struct TableConstraintContext {
    pub(crate) schema_field_ids: Vec<FieldId>,
    pub(crate) column_constraints: Vec<InsertColumnConstraint>,
    pub(crate) unique_columns: Vec<InsertUniqueColumn>,
    pub(crate) multi_column_uniques: Vec<InsertMultiColumnUnique>,
    pub(crate) primary_key: Option<InsertMultiColumnUnique>,
}
