//! Core type definitions for the storage engine.

// FIXME: Since upgrading to `rustc 1.90.0 (1159e78c4 2025-09-14)`, this seems
// to be needed to workaround parenthesis errors in `LogicalFieldId`, which
// creep up regardless of comments being added or not. This is possibly a bug
// with Clippy or `modular_bitfield`, or a small incompatibility issue.
#![allow(unused_parens)]

use modular_bitfield::prelude::*;

/// Defines the category of data a `LogicalFieldId` refers to.
/// This enum uses 16 bits, allowing for up to 65,536 distinct namespaces.
#[derive(Specifier, Debug, PartialEq, Eq, Clone, Copy)]
#[bits = 16]
pub enum Namespace {
    /// Standard user-defined column data.
    UserData = 0,
    /// Internal shadow column for row IDs.
    RowIdShadow = 1,
    /// Highest sentinel reserved for future expansion.
    Reserved = 0xFFFF,
}

/// Identifier for a logical table within a [`Namespace`].
///
/// `TableId` consumes the middle 16 bits inside [`LogicalFieldId`]. Using a
/// `u16` keeps IDs compact and (more importantly) guarantees they always fit in
/// that bitfield without extra checks.
pub type TableId = u16;

/// Logical column identifier within a table.
///
/// `FieldId` is stored inside [`LogicalFieldId::field_id`], which is a 32-bit
/// lane.  Keep this alias in sync with that width so table metadata and
/// runtime identifiers can round-trip without truncation.
pub type FieldId = u32;

/// Row identifier for persisted data.
///
/// `ColumnStore` emits row ids as Arrow `UInt64Array`s (see `core.rs`), so this
/// alias mirrors that width to avoid casts when marshalling data in and out of
/// the engine.
pub type RowId = u64;

/// A namespaced logical identifier for a column.
///
/// This 64-bit struct is designed to prevent ID collisions by partitioning the key space
/// into distinct namespaces, table IDs, and field IDs.
#[bitfield]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u64)]
pub struct LogicalFieldId {
    /// The specific field/column within a table (up to ~4.3 billion).
    pub field_id: B32,
    /// The table this field belongs to (up to 65,535).
    pub table_id: B16,
    /// The type of data this ID represents (up to 65,536 namespaces).
    pub namespace: Namespace,
}



impl LogicalFieldId {
    /// Build a logical field identifier from its namespace, table, and field components.
    #[inline]
    pub fn from_parts(namespace: Namespace, table_id: TableId, field_id: FieldId) -> Self {
        LogicalFieldId::new()
            .with_namespace(namespace)
            .with_table_id(table_id)
            .with_field_id(field_id)
    }

    /// Convenience constructor for user data columns.
    #[inline]
    pub fn for_user(table_id: TableId, field_id: FieldId) -> Self {
        Self::from_parts(Namespace::UserData, table_id, field_id)
    }

    /// Convenience constructor for user data columns in table 0.
    ///
    /// Many tests use table 0 by default; this method avoids repeating the table ID literal.
    #[inline]
    pub fn for_default_user(field_id: FieldId) -> Self {
        Self::for_user(0, field_id)
    }
}
