/// Multi-Version Concurrency Control (MVCC) implementation
/// 
/// This module provides transaction IDs and row versioning for snapshot isolation.
/// Instead of copying entire tables, we track which transaction created/deleted each row.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global transaction ID counter
static GLOBAL_TXN_ID: AtomicU64 = AtomicU64::new(1);

/// Transaction ID type
pub type TxnId = u64;

/// Special transaction ID representing "not deleted"
pub const TXN_ID_NONE: TxnId = u64::MAX;

/// Special transaction ID for auto-commit (single-statement) transactions
/// Used when not in an explicit transaction
pub const TXN_ID_AUTO_COMMIT: TxnId = 1;

/// Transaction ID manager
#[derive(Clone)]
pub struct TxnIdManager {
    _phantom: (),
}

impl TxnIdManager {
    pub fn new() -> Self {
        Self { _phantom: () }
    }

    /// Allocate a new transaction ID
    pub fn next_txn_id(&self) -> TxnId {
        GLOBAL_TXN_ID.fetch_add(1, Ordering::SeqCst)
    }

    /// Get current transaction ID (for testing)
    pub fn current_txn_id(&self) -> TxnId {
        GLOBAL_TXN_ID.load(Ordering::SeqCst)
    }
}

impl Default for TxnIdManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Row version metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowVersion {
    /// Transaction ID that created this row version
    pub created_by: TxnId,
    /// Transaction ID that deleted this row version (TXN_ID_NONE if not deleted)
    pub deleted_by: TxnId,
}

impl RowVersion {
    /// Create a new row version created by the given transaction
    pub fn new(created_by: TxnId) -> Self {
        Self {
            created_by,
            deleted_by: TXN_ID_NONE,
        }
    }

    /// Mark this row as deleted by the given transaction
    pub fn delete(&mut self, deleted_by: TxnId) {
        self.deleted_by = deleted_by;
    }

    /// Check if this row version is visible to a transaction with the given ID
    /// 
    /// Visibility rules:
    /// - Row created_by <= snapshot_txn_id (row existed when transaction started)
    /// - Row deleted_by > snapshot_txn_id OR deleted_by == TXN_ID_NONE (row not deleted yet)
    pub fn is_visible(&self, snapshot_txn_id: TxnId) -> bool {
        self.created_by <= snapshot_txn_id
            && (self.deleted_by == TXN_ID_NONE || self.deleted_by > snapshot_txn_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_txn_id_manager() {
        let manager = TxnIdManager::new();
        let txn1 = manager.next_txn_id();
        let txn2 = manager.next_txn_id();
        assert!(txn2 > txn1);
    }

    #[test]
    fn test_row_visibility() {
        // Row created by transaction 10
        let mut row = RowVersion::new(10);

        // Transaction 5 started before row was created - should not see it
        assert!(!row.is_visible(5));

        // Transaction 10 created it - should see it
        assert!(row.is_visible(10));

        // Transaction 15 started after creation - should see it
        assert!(row.is_visible(15));

        // Now delete the row at transaction 20
        row.delete(20);

        // Transaction 15 started before deletion - should still see it
        assert!(row.is_visible(15));

        // Transaction 20 is doing the delete - should still see it (snapshot was before delete)
        assert!(row.is_visible(20));

        // Transaction 25 started after deletion - should NOT see it
        assert!(!row.is_visible(25));
    }

    #[test]
    fn test_never_deleted_row() {
        let row = RowVersion::new(10);
        
        // Should be visible to any transaction after creation
        assert!(row.is_visible(10));
        assert!(row.is_visible(100));
        assert!(row.is_visible(1000));
    }
}
