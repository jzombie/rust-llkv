//! Reserved value ranges for system use.
//!
//! This module centralizes all "magic numbers" used by the system to prevent
//! accidental collisions with user data. All reserved values and validation
//! functions live here.
//!
//! # Adding New Reserved Values
//!
//! When adding a new reserved value:
//!
//! 1. **Add the constant** to the appropriate section below
//! 2. **Update the `is_reserved_*` function** to include the new value
//! 3. **Update the `*_message` function** to provide a clear error message
//! 4. **Add a test case** in the tests section
//! 5. **Document WHY** the value is reserved (don't just say "reserved for X")
//!
//! # Design Philosophy
//!
//! - **Use `0` for sentinel "first" values** (catalog table, row_id field)
//! - **Use `1` for common defaults** (TXN_ID_AUTO_COMMIT)
//! - **Use `u64::MAX` and nearby** for sentinel "none/infinity" values (TXN_ID_NONE, MVCC field IDs)
//! - **Use sparse ranges** for catalog fields (1, 10, 100, 101, 102...) to leave room for expansion
//! - **Always validate at allocation** - don't assume users won't try reserved values

use crate::types::{FieldId, RowId, TableId};

// =============================================================================
// TABLE ID RESERVATIONS
// =============================================================================

/// The system catalog table (reserved).
pub const CATALOG_TABLE_ID: TableId = 0;

/// Check if a table ID is reserved and cannot be used for user tables.
#[inline]
pub fn is_reserved_table_id(id: TableId) -> bool {
    id == CATALOG_TABLE_ID
}

/// Return the error message for attempting to use a reserved table ID.
#[inline]
pub fn reserved_table_id_message(id: TableId) -> String {
    format!("Table ID {} is reserved for system use", id)
}

// =============================================================================
// FIELD ID RESERVATIONS
// =============================================================================

/// The row_id field (reserved for internal row identifiers).
/// Present in all tables, always at FieldId 0.
pub const ROW_ID_FIELD_ID: FieldId = 0;

/// Sentinel FieldId for MVCC created_by column (u32::MAX).
/// Tracks which transaction created each row.
/// Uses sentinel value to avoid collision with user field IDs.
pub const MVCC_CREATED_BY_FIELD_ID: FieldId = u32::MAX;

/// Sentinel FieldId for MVCC deleted_by column (u32::MAX - 1).
/// Tracks which transaction deleted each row (TXN_ID_NONE if not deleted).
/// Uses sentinel value to avoid collision with user field IDs.
pub const MVCC_DELETED_BY_FIELD_ID: FieldId = u32::MAX - 1;

/// First FieldId available for user-defined columns.
/// System columns (row_id=0, MVCC columns=sentinels) are excluded from this range.
pub const FIRST_USER_FIELD_ID: FieldId = 1;

/// Check if a field ID is reserved and cannot be used for user columns.
#[inline]
pub fn is_reserved_field_id(id: FieldId) -> bool {
    id == ROW_ID_FIELD_ID 
        || id == MVCC_CREATED_BY_FIELD_ID 
        || id == MVCC_DELETED_BY_FIELD_ID
}

/// Check if a field ID is a system column (row_id or MVCC columns).
/// System columns are present in all tables and have special handling.
#[inline]
pub fn is_system_field(id: FieldId) -> bool {
    id == ROW_ID_FIELD_ID 
        || id == MVCC_CREATED_BY_FIELD_ID 
        || id == MVCC_DELETED_BY_FIELD_ID
}

/// Check if a field ID is an MVCC column (created_by or deleted_by).
/// MVCC columns track transaction visibility.
#[inline]
pub fn is_mvcc_field(id: FieldId) -> bool {
    id == MVCC_CREATED_BY_FIELD_ID || id == MVCC_DELETED_BY_FIELD_ID
}

/// Return the error message for attempting to use a reserved field ID.
#[inline]
pub fn reserved_field_id_message(id: FieldId) -> String {
    match id {
        ROW_ID_FIELD_ID => "Field ID 0 is reserved for row_id".to_string(),
        MVCC_CREATED_BY_FIELD_ID => format!("Field ID {} (u32::MAX) is reserved for MVCC created_by", id),
        MVCC_DELETED_BY_FIELD_ID => format!("Field ID {} (u32::MAX - 1) is reserved for MVCC deleted_by", id),
        _ => format!("Field ID {} is reserved for system use", id),
    }
}

// =============================================================================
// ROW ID RESERVATIONS (Catalog-specific)
// =============================================================================

/// Row ID reserved for the catalog's next_table_id singleton.
pub const CATALOG_NEXT_TABLE_ROW_ID: RowId = 0;

/// Row ID reserved for the catalog's next_txn_id singleton.
pub const CATALOG_NEXT_TXN_ROW_ID: RowId = 1;

/// Row ID reserved for the catalog's last_committed_txn_id singleton.
pub const CATALOG_LAST_COMMITTED_TXN_ROW_ID: RowId = 2;

/// Check if a row ID is reserved in the catalog table.
/// User tables can use any row ID, but catalog table has reserved rows.
#[inline]
pub fn is_reserved_catalog_row_id(id: RowId) -> bool {
    id == CATALOG_NEXT_TABLE_ROW_ID 
        || id == CATALOG_NEXT_TXN_ROW_ID 
        || id == CATALOG_LAST_COMMITTED_TXN_ROW_ID
}

/// Return the error message for attempting to use a reserved catalog row ID.
#[inline]
pub fn reserved_catalog_row_id_message(id: RowId) -> String {
    match id {
        CATALOG_NEXT_TABLE_ROW_ID => "Row ID 0 is reserved for catalog's next_table_id".to_string(),
        CATALOG_NEXT_TXN_ROW_ID => "Row ID 1 is reserved for catalog's next_txn_id".to_string(),
        CATALOG_LAST_COMMITTED_TXN_ROW_ID => "Row ID 2 is reserved for catalog's last_committed_txn_id".to_string(),
        _ => format!("Row ID {} is reserved in catalog table", id),
    }
}

// =============================================================================
// CATALOG FIELD ID RESERVATIONS (Internal catalog fields)
// =============================================================================

/// Catalog field for table metadata (Binary-encoded TableMeta).
pub const CATALOG_FIELD_TABLE_META_ID: u32 = 1;

/// Catalog field for column metadata (Binary-encoded ColMeta).
pub const CATALOG_FIELD_COL_META_ID: u32 = 10;

/// Catalog field for next table ID counter (UInt64).
pub const CATALOG_FIELD_NEXT_TABLE_ID: u32 = 100;

/// Catalog field for next transaction ID counter (UInt64).
pub const CATALOG_FIELD_NEXT_TXN_ID: u32 = 101;

/// Catalog field for last committed transaction ID (UInt64).
pub const CATALOG_FIELD_LAST_COMMITTED_TXN_ID: u32 = 102;

/// Check if a field ID is used by the catalog's internal structure.
#[inline]
pub fn is_catalog_internal_field(id: u32) -> bool {
    matches!(
        id,
        CATALOG_FIELD_TABLE_META_ID
            | CATALOG_FIELD_COL_META_ID
            | CATALOG_FIELD_NEXT_TABLE_ID
            | CATALOG_FIELD_NEXT_TXN_ID
            | CATALOG_FIELD_LAST_COMMITTED_TXN_ID
    )
}

// =============================================================================
// TRANSACTION ID RESERVATIONS
// =============================================================================

/// Transaction ID representing "no transaction" or "not deleted".
/// Uses u64::MAX to avoid collision with real transaction IDs.
pub const TXN_ID_NONE: u64 = u64::MAX;

/// Transaction ID for auto-commit (single-statement) transactions.
/// Always treated as committed.
pub const TXN_ID_AUTO_COMMIT: u64 = 1;

/// Minimum valid transaction ID for multi-statement transactions.
pub const TXN_ID_MIN_MULTI_STATEMENT: u64 = TXN_ID_AUTO_COMMIT + 1;

/// Check if a transaction ID is reserved (cannot be allocated).
#[inline]
pub fn is_reserved_txn_id(id: u64) -> bool {
    id == TXN_ID_NONE || id <= TXN_ID_AUTO_COMMIT
}

/// Return the error message for attempting to use a reserved transaction ID.
#[inline]
pub fn reserved_txn_id_message(id: u64) -> String {
    match id {
        TXN_ID_NONE => format!("Transaction ID {} (u64::MAX) is reserved for TXN_ID_NONE", id),
        0 => "Transaction ID 0 is invalid".to_string(),
        TXN_ID_AUTO_COMMIT => "Transaction ID 1 is reserved for TXN_ID_AUTO_COMMIT".to_string(),
        _ => format!("Transaction ID {} is reserved", id),
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_id_reservations() {
        assert!(is_reserved_table_id(CATALOG_TABLE_ID));
        assert!(!is_reserved_table_id(1));
        assert!(!is_reserved_table_id(100));
    }

    #[test]
    fn test_field_id_reservations() {
        assert!(is_reserved_field_id(ROW_ID_FIELD_ID));
        assert!(is_reserved_field_id(MVCC_CREATED_BY_FIELD_ID));
        assert!(is_reserved_field_id(MVCC_DELETED_BY_FIELD_ID));
        assert!(!is_reserved_field_id(FIRST_USER_FIELD_ID));
        assert!(!is_reserved_field_id(100));
    }

    #[test]
    fn test_system_field_helpers() {
        // System fields: row_id (0) and MVCC sentinels (u32::MAX, u32::MAX-1)
        assert!(is_system_field(ROW_ID_FIELD_ID));
        assert!(is_system_field(MVCC_CREATED_BY_FIELD_ID));
        assert!(is_system_field(MVCC_DELETED_BY_FIELD_ID));
        assert!(!is_system_field(FIRST_USER_FIELD_ID));
        assert!(!is_system_field(100));

        // MVCC fields: sentinels only (not row_id)
        assert!(!is_mvcc_field(ROW_ID_FIELD_ID));
        assert!(is_mvcc_field(MVCC_CREATED_BY_FIELD_ID));
        assert!(is_mvcc_field(MVCC_DELETED_BY_FIELD_ID));
        assert!(!is_mvcc_field(FIRST_USER_FIELD_ID));
    }

    #[test]
    fn test_first_user_field_id() {
        // User fields start at 1 (0 reserved for row_id, sentinels for MVCC)
        assert_eq!(FIRST_USER_FIELD_ID, 1);
        assert!(!is_system_field(FIRST_USER_FIELD_ID));
        assert!(!is_reserved_field_id(FIRST_USER_FIELD_ID));
    }

    #[test]
    fn test_catalog_row_id_reservations() {
        assert!(is_reserved_catalog_row_id(CATALOG_NEXT_TABLE_ROW_ID));
        assert!(is_reserved_catalog_row_id(CATALOG_NEXT_TXN_ROW_ID));
        assert!(is_reserved_catalog_row_id(CATALOG_LAST_COMMITTED_TXN_ROW_ID));
        assert!(!is_reserved_catalog_row_id(3));
        assert!(!is_reserved_catalog_row_id(100));
    }

    #[test]
    fn test_txn_id_reservations() {
        assert!(is_reserved_txn_id(TXN_ID_NONE));
        assert!(is_reserved_txn_id(0));
        assert!(is_reserved_txn_id(TXN_ID_AUTO_COMMIT));
        assert!(!is_reserved_txn_id(TXN_ID_MIN_MULTI_STATEMENT));
        assert!(!is_reserved_txn_id(100));
    }

    #[test]
    fn test_catalog_internal_fields() {
        assert!(is_catalog_internal_field(CATALOG_FIELD_TABLE_META_ID));
        assert!(is_catalog_internal_field(CATALOG_FIELD_COL_META_ID));
        assert!(is_catalog_internal_field(CATALOG_FIELD_NEXT_TABLE_ID));
        assert!(is_catalog_internal_field(CATALOG_FIELD_NEXT_TXN_ID));
        assert!(is_catalog_internal_field(CATALOG_FIELD_LAST_COMMITTED_TXN_ID));
        assert!(!is_catalog_internal_field(2));
        assert!(!is_catalog_internal_field(200));
    }
}
