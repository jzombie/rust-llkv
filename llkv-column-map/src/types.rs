//! Core type definitions for the storage engine.

use modular_bitfield::prelude::*;

/// Opaque 64-bit address in the pager namespace.
/// Treated as an opaque handle by higher layers.
pub type PhysicalKey = u64;
/// Defines the category of data a `LogicalFieldId` refers to.
/// This enum uses 3 bits, allowing for up to 8 distinct namespaces.
#[derive(Specifier, Debug, PartialEq, Eq, Clone, Copy)]
#[bits = 3]
pub enum Namespace {
    /// Standard user-defined column data.
    UserData = 0,
    /// Internal shadow column for row IDs.
    RowIdShadow = 1,
    /// Internal shadow column for sort permutations.
    SortIndex = 2,
    // Up to 5 more namespaces can be added here.
    Reserved = 7,
}

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
    /// The table this field belongs to (up to ~536 million).
    pub table_id: B29,
    /// The type of data this ID represents (e.g., user data, shadow row ID).
    pub namespace: Namespace,
}

// Well-known keys for root objects.
pub(crate) const CATALOG_ROOT_PKEY: PhysicalKey = 0;
pub(crate) const CONFIG_ROOT_PKEY: PhysicalKey = 1;
