//! Constraint metadata definitions and helpers for persistence.
//!
//! This module defines the storage-friendly representations of constraints
//! that are persisted through the system catalog. The structures intentionally
//! avoid heavyweight data (like table names) to keep serialized blobs small and
//! cache friendly.

#![forbid(unsafe_code)]

use bitcode::{Decode, Encode};

use crate::types::{FieldId, RowId, TableId};

/// Controls when constraint validation occurs for new writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Encode, Decode)]
pub enum ConstraintEnforcementMode {
    /// Perform all constraint checks before rows become visible.
    Immediate,
    /// Defer checks so callers can load data quickly and validate later.
    Deferred,
}

impl ConstraintEnforcementMode {
    #[inline]
    pub const fn is_immediate(self) -> bool {
        matches!(self, ConstraintEnforcementMode::Immediate)
    }

    #[inline]
    pub const fn is_deferred(self) -> bool {
        matches!(self, ConstraintEnforcementMode::Deferred)
    }
}

impl Default for ConstraintEnforcementMode {
    fn default() -> Self {
        ConstraintEnforcementMode::Immediate
    }
}

/// Identifier assigned to a persisted constraint.
///
/// Constraint IDs are scoped per-table and limited to 32 bits so they can be
/// embedded into catalog row IDs without collisions.
pub type ConstraintId = u32;

const TABLE_ID_BITS: u32 = (std::mem::size_of::<TableId>() * 8) as u32;
const CONSTRAINT_ID_BITS: u32 = 64 - TABLE_ID_BITS;
const CONSTRAINT_ID_MASK: u64 = (1u64 << CONSTRAINT_ID_BITS) - 1;

const _: () = assert!(
    std::mem::size_of::<ConstraintId>() * 8 <= CONSTRAINT_ID_BITS as usize,
    "ConstraintId does not fit within allocated row id bits"
);

/// Pack a table/constraint pair into a catalog row identifier.
#[inline]
pub fn encode_constraint_row_id(table_id: TableId, constraint_id: ConstraintId) -> RowId {
    ((table_id as u64) << CONSTRAINT_ID_BITS) | (constraint_id as u64 & CONSTRAINT_ID_MASK)
}

/// Decode a catalog row identifier into its table/constraint components.
#[inline]
pub fn decode_constraint_row_id(row_id: RowId) -> (TableId, ConstraintId) {
    let constraint_id = (row_id & CONSTRAINT_ID_MASK) as ConstraintId;
    let table_id = (row_id >> CONSTRAINT_ID_BITS) as TableId;
    (table_id, constraint_id)
}

/// Persisted state of a constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Encode, Decode)]
pub enum ConstraintState {
    Active,
    Dropped,
}

impl ConstraintState {
    #[inline]
    pub fn is_active(self) -> bool {
        matches!(self, ConstraintState::Active)
    }
}

/// Persisted constraint record stored in the system catalog.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct ConstraintRecord {
    pub constraint_id: ConstraintId,
    pub kind: ConstraintKind,
    pub state: ConstraintState,
    /// Monotonic revision counter maintained by the metadata manager.
    pub revision: u64,
    /// Timestamp (microseconds since epoch) when this record was last updated.
    pub last_modified_micros: u64,
}

impl ConstraintRecord {
    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }
}

/// Logical description of a constraint.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub enum ConstraintKind {
    PrimaryKey(PrimaryKeyConstraint),
    Unique(UniqueConstraint),
    ForeignKey(ForeignKeyConstraint),
    Check(CheckConstraint),
}

impl ConstraintKind {
    /// Return the field IDs participating on the referencing side of the constraint.
    pub fn field_ids(&self) -> &[FieldId] {
        match self {
            ConstraintKind::PrimaryKey(payload) => &payload.field_ids,
            ConstraintKind::Unique(payload) => &payload.field_ids,
            ConstraintKind::ForeignKey(payload) => &payload.referencing_field_ids,
            ConstraintKind::Check(payload) => &payload.field_ids,
        }
    }
}

/// Primary key definition.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct PrimaryKeyConstraint {
    pub field_ids: Vec<FieldId>,
}

/// Unique constraint definition (single or multi-column).
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct UniqueConstraint {
    pub field_ids: Vec<FieldId>,
}

/// Foreign key definition referencing another table.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct ForeignKeyConstraint {
    pub referencing_field_ids: Vec<FieldId>,
    pub referenced_table: TableId,
    pub referenced_field_ids: Vec<FieldId>,
    pub on_delete: ForeignKeyAction,
    pub on_update: ForeignKeyAction,
}

/// Foreign key referential action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Encode, Decode)]
pub enum ForeignKeyAction {
    NoAction,
    Restrict,
}

/// Serialized check constraint definition.
///
/// The expression payload is referenced indirectly via an identifier so the
/// metadata layer can resolve the actual expression lazily.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct CheckConstraint {
    pub field_ids: Vec<FieldId>,
    pub expression_ref: ConstraintExpressionRef,
}

/// Reference to a stored constraint expression payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encode, Decode)]
pub struct ConstraintExpressionRef(pub u64);

impl ConstraintExpressionRef {
    pub const NONE: ConstraintExpressionRef = ConstraintExpressionRef(0);

    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraint_row_id_round_trip() {
        let samples: &[(TableId, ConstraintId)] = &[
            (1, 1),
            (42, 7),
            (TableId::MAX, ConstraintId::MAX),
            (123, 987_654),
        ];

        for &(table_id, constraint_id) in samples {
            let row_id = encode_constraint_row_id(table_id, constraint_id);
            let (decoded_table, decoded_constraint) = decode_constraint_row_id(row_id);
            assert_eq!(decoded_table, table_id);
            assert_eq!(decoded_constraint, constraint_id);
        }
    }

    #[test]
    fn constraint_record_active_state() {
        let record = ConstraintRecord {
            constraint_id: 1,
            kind: ConstraintKind::PrimaryKey(PrimaryKeyConstraint {
                field_ids: vec![1, 2],
            }),
            state: ConstraintState::Active,
            revision: 5,
            last_modified_micros: 123,
        };
        assert!(record.is_active());
    }
}
