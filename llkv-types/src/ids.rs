//! Identifiers shared across LLKV crates.
//!
//! These types live in `llkv-types` so they can be reused without depending on
//! the storage-specific crates.

// FIXME: Since upgrading to `rustc 1.90.0 (1159e78c4 2025-09-14)`, this seems
// to be needed to workaround parenthesis errors in `LogicalFieldId`, which
// creep up regardless of comments being added or not. This is possibly a bug
// with Clippy or `modular_bitfield`, or a small incompatibility issue.
#![allow(unused_parens)]

use modular_bitfield::prelude::*;

#[inline]
fn rowid_shadow(fid: LogicalFieldId) -> LogicalFieldId {
    fid.with_namespace(LogicalStorageNamespace::RowIdShadow)
}

/// Category of data a column contains.
///
/// The `LogicalStorageNamespace` enum prevents ID collisions by segregating different types of
/// columns into distinct namespaces. Each namespace can contain up to 2^16 tables,
/// and each table can have up to 2^32 fields.
///
/// # Usage
///
/// Namespaces are embedded in [`LogicalFieldId`] to create globally unique column
/// identifiers. User code typically works with `UserData` columns, while system
/// components use the other namespaces for internal bookkeeping.
#[derive(Specifier, Debug, PartialEq, Eq, Clone, Copy)]
#[bits = 16]
pub enum LogicalStorageNamespace {
    /// User-defined table columns.
    ///
    /// This is the default namespace for regular table columns. When a table is created
    /// with columns like `name TEXT, age INT`, those columns use the `UserData` namespace.
    UserData = 0,

    /// Internal shadow column tracking row IDs.
    ///
    /// For each user column, the storage engine maintains a corresponding shadow column
    /// that stores the row ID for each value. This enables efficient row-level operations
    /// and join/filter optimizations.
    RowIdShadow = 1,

    /// MVCC metadata: transaction that created each row.
    ///
    /// Stores the transaction ID (`TxnId`) that inserted each row. Used for snapshot
    /// isolation to determine row visibility.
    TxnCreatedBy = 2,

    /// MVCC metadata: transaction that deleted each row.
    ///
    /// Stores the transaction ID that deleted each row, or `TXN_ID_NONE` if the row
    /// is not deleted. Used for snapshot isolation and garbage collection.
    TxnDeletedBy = 3,

    /// Reserved for future system use.
    ///
    /// The value `0xFFFF` is reserved as a sentinel to allow future expansion without
    /// breaking compatibility.
    Reserved = 0xFFFF,
}

/// Unique identifier for a table.
///
/// Table IDs are 16-bit unsigned integers, allowing up to 65,535 tables per database.
/// This type is embedded in [`LogicalFieldId`] to associate columns with tables.
///
/// # Special Values
///
/// - Table ID `0` is reserved for the system catalog
/// - User tables receive IDs starting from `1`
pub type TableId = u16;

/// Unique identifier for a column within a table.
///
/// Field IDs are 32-bit unsigned integers, allowing up to ~4.3 billion columns per table.
/// This type is stored in [`LogicalFieldId::field_id`] and must match that bitfield width.
///
/// # Special Values
///
/// - Field ID `0` (`ROW_ID_FIELD_ID`) is reserved for row ID columns
/// - Field ID `u32::MAX` is reserved for MVCC `created_by` columns
/// - Field ID `u32::MAX - 1` is reserved for MVCC `deleted_by` columns
/// - User columns receive IDs starting from `1`
pub type FieldId = u32;

/// Reserved field ID for row ID columns.
///
/// This constant is used for the synthetic row ID column that exists in all tables.
/// Row IDs are globally unique `u64` values that never change once assigned.
pub const ROW_ID_FIELD_ID: FieldId = 0;

/// Unique identifier for a row within a table.
///
/// Row IDs are 64-bit unsigned integers assigned sequentially on insert. They are:
/// - Globally unique within a table
/// - Never reused (even after deletion)
/// - Monotonically increasing (within append batches)
/// - Used for joins, filters, and row-level operations
pub type RowId = u64;

/// Globally unique identifier for a column in the storage engine.
///
/// A `LogicalFieldId` combines three components into a single 64-bit value:
/// - **Namespace** (16 bits): Category of data (user, system, MVCC)
/// - **Table ID** (16 bits): Which table the column belongs to
/// - **Field ID** (32 bits): Which column within the table
///
/// This design prevents ID collisions across different tables and data categories while
/// keeping identifiers compact and easy to pass around.
///
/// # Bit Layout
///
/// ```text
/// |-------- 64 bits total --------|
/// | namespace | table_id | field_id |
/// |  16 bits  | 16 bits  | 32 bits  |
/// ```
///
/// # Construction
///
/// Use the constructor methods rather than directly manipulating bits:
/// - [`LogicalFieldId::for_user`] - User-defined columns
/// - [`LogicalFieldId::for_mvcc_created_by`] - MVCC created_by metadata
/// - [`LogicalFieldId::for_mvcc_deleted_by`] - MVCC deleted_by metadata
/// - [`LogicalFieldId::from_parts`] - Custom construction
///
/// # Thread Safety
///
/// `LogicalFieldId` is `Copy` and thread-safe.
#[bitfield]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u64)]
pub struct LogicalFieldId {
    /// Column identifier within the table (32 bits).
    ///
    /// Supports up to ~4.3 billion columns per table. Field ID `0` is reserved for
    /// row ID columns.
    pub field_id: B32,

    /// Table identifier (16 bits).
    ///
    /// Supports up to 65,535 tables. Table ID `0` is reserved for the system catalog.
    pub table_id: B16,

    /// Data category (16 bits).
    ///
    /// Determines whether this ID refers to user data, system metadata, or MVCC tracking.
    pub namespace: LogicalStorageNamespace,
}

impl LogicalFieldId {
    /// Construct a `LogicalFieldId` from individual components.
    ///
    /// This is the most general constructor. Use the convenience methods
    /// ([`for_user`](Self::for_user), [`for_mvcc_created_by`](Self::for_mvcc_created_by), etc.)
    /// for common cases.
    #[inline]
    pub fn from_parts(
        namespace: LogicalStorageNamespace,
        table_id: TableId,
        field_id: FieldId,
    ) -> Self {
        LogicalFieldId::new()
            .with_namespace(namespace)
            .with_table_id(table_id)
            .with_field_id(field_id)
    }

    /// Create an ID for a user-defined column.
    ///
    /// This is the most common constructor for regular table columns. It uses the
    /// `UserData` namespace.
    #[inline]
    pub fn for_user(table_id: TableId, field_id: FieldId) -> Self {
        Self::from_parts(LogicalStorageNamespace::UserData, table_id, field_id)
    }

    /// Create an ID for a user column in table 0.
    ///
    /// This is a convenience method for tests and examples that use the default table ID.
    #[inline]
    pub fn for_user_table_0(field_id: FieldId) -> Self {
        Self::for_user(0, field_id)
    }

    /// Create an ID for the MVCC `created_by` column of a table.
    ///
    /// Each table has a `created_by` column that tracks which transaction inserted
    /// each row. The field ID is always `u32::MAX` as a sentinel value.
    #[inline]
    pub fn for_mvcc_created_by(table_id: TableId) -> Self {
        Self::from_parts(LogicalStorageNamespace::TxnCreatedBy, table_id, u32::MAX)
    }

    /// Create an ID for the MVCC `deleted_by` column of a table.
    ///
    /// Each table has a `deleted_by` column that tracks which transaction deleted
    /// each row (or `TXN_ID_NONE` if not deleted). The field ID is always `u32::MAX - 1`
    /// as a sentinel value.
    #[inline]
    pub fn for_mvcc_deleted_by(table_id: TableId) -> Self {
        Self::from_parts(
            LogicalStorageNamespace::TxnDeletedBy,
            table_id,
            u32::MAX - 1,
        )
    }
}

/// Convenience helper for constructing a user-space logical field id.
#[inline]
pub fn lfid(table_id: TableId, col_id: FieldId) -> LogicalFieldId {
    LogicalFieldId::for_user(table_id, col_id)
}

/// Logical field id for the table's row id shadow column, as a `u64` key.
#[inline]
pub fn rid_col(table_id: TableId, col_id: FieldId) -> u64 {
    let fid = lfid(table_id, col_id);
    rowid_shadow(fid).into()
}

/// Logical field id for the reserved table row id column, as a `u64` key.
#[inline]
pub fn rid_table(table_id: TableId) -> u64 {
    lfid(table_id, ROW_ID_FIELD_ID).into()
}
