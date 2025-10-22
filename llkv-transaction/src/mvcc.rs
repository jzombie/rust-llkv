/// Multi-Version Concurrency Control (MVCC) utilities.
///
/// This module centralizes the transaction ID allocator, row-version metadata,
/// Arrow helpers for MVCC column construction, and visibility checks used across
/// the engine. The overarching goal is to allow transactions to operate directly
/// on the base storage without copying tables into a staging area.
use arrow::array::{ArrayRef, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::{
    CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, FIELD_ID_META_KEY, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::{FieldId, RowId};
use llkv_result::Error;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Transaction ID type.
pub type TxnId = u64;

/// Transaction ID representing "no transaction" or "not deleted".
pub const TXN_ID_NONE: TxnId = TxnId::MAX;

/// Transaction ID for auto-commit (single-statement) transactions.
pub const TXN_ID_AUTO_COMMIT: TxnId = 1;

/// Minimum valid transaction ID for multi-statement transactions.
pub const TXN_ID_MIN_MULTI_STATEMENT: TxnId = TXN_ID_AUTO_COMMIT + 1;

/// Check if a transaction ID is reserved (cannot be allocated).
#[inline]
pub fn is_reserved_txn_id(id: TxnId) -> bool {
    id == TXN_ID_NONE || id <= TXN_ID_AUTO_COMMIT
}

/// Return the error message for attempting to use a reserved transaction ID.
#[inline]
pub fn reserved_txn_id_message(id: TxnId) -> String {
    match id {
        TXN_ID_NONE => format!(
            "Transaction ID {} (u64::MAX) is reserved for TXN_ID_NONE",
            id
        ),
        0 => "Transaction ID 0 is invalid".to_string(),
        TXN_ID_AUTO_COMMIT => "Transaction ID 1 is reserved for TXN_ID_AUTO_COMMIT".to_string(),
        _ => format!("Transaction ID {} is reserved", id),
    }
}

/// Internal state shared across transaction ID managers.
#[derive(Debug)]
struct TxnIdManagerInner {
    /// Next transaction ID to allocate.
    next_txn_id: AtomicU64,
    /// Largest committed transaction ID (acts as snapshot watermark).
    last_committed: AtomicU64,
    /// Tracking map for transaction statuses.
    statuses: Mutex<FxHashMap<TxnId, TxnStatus>>,
}

impl TxnIdManagerInner {
    fn new() -> Self {
        Self::new_with_initial_txn_id(TXN_ID_AUTO_COMMIT + 1)
    }

    fn new_with_initial_txn_id(next_txn_id: TxnId) -> Self {
        Self::new_with_initial_state(next_txn_id, TXN_ID_AUTO_COMMIT)
    }

    fn new_with_initial_state(next_txn_id: TxnId, last_committed: TxnId) -> Self {
        let mut statuses = FxHashMap::with_capacity_and_hasher(1, Default::default());
        statuses.insert(TXN_ID_AUTO_COMMIT, TxnStatus::Committed);

        Self {
            next_txn_id: AtomicU64::new(next_txn_id),
            last_committed: AtomicU64::new(last_committed),
            statuses: Mutex::new(statuses),
        }
    }
}

/// Transaction ID manager that hands out IDs and tracks commit status.
#[derive(Clone, Debug)]
pub struct TxnIdManager {
    inner: Arc<TxnIdManagerInner>,
}

impl TxnIdManager {
    /// Create a new manager.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TxnIdManagerInner::new()),
        }
    }

    /// Create a new manager with a custom initial transaction ID.
    /// Used when loading persisted state from the catalog.
    pub fn new_with_initial_txn_id(next_txn_id: TxnId) -> Self {
        Self {
            inner: Arc::new(TxnIdManagerInner::new_with_initial_txn_id(next_txn_id)),
        }
    }

    /// Create a new manager with custom initial state.
    /// Used when loading persisted state from the catalog.
    pub fn new_with_initial_state(next_txn_id: TxnId, last_committed: TxnId) -> Self {
        Self {
            inner: Arc::new(TxnIdManagerInner::new_with_initial_state(
                next_txn_id,
                last_committed,
            )),
        }
    }

    /// Get the current next_txn_id value (for persistence).
    pub fn current_next_txn_id(&self) -> TxnId {
        self.inner.next_txn_id.load(Ordering::SeqCst)
    }

    /// Begin a new transaction and return its snapshot.
    ///
    /// The snapshot captures both the allocated transaction ID and the latest
    /// committed ID at the moment the transaction starts. These two values are
    /// required to evaluate row visibility rules.
    pub fn begin_transaction(&self) -> TransactionSnapshot {
        let snapshot_id = self.inner.last_committed.load(Ordering::SeqCst);
        let txn_id = self.inner.next_txn_id.fetch_add(1, Ordering::SeqCst);

        {
            let mut guard = self
                .inner
                .statuses
                .lock()
                .expect("txn status lock poisoned");
            guard.insert(txn_id, TxnStatus::Active);
        }

        TransactionSnapshot {
            txn_id,
            snapshot_id,
        }
    }

    // TODO: Is this method a good idea?  A transaction has to begin in order to determine the next ID?
    // NOTE: This helper preserves existing call sites that only need identifiers; revisit after
    // transaction lifecycle changes land.
    /// Convenience helper that returns only the allocated transaction ID.
    /// Prefer [`TxnIdManager::begin_transaction`] when a snapshot is required.
    pub fn next_txn_id(&self) -> TxnId {
        self.begin_transaction().txn_id
    }

    /// Return the status for a given transaction ID.
    pub fn status(&self, txn_id: TxnId) -> TxnStatus {
        if txn_id == TXN_ID_NONE {
            return TxnStatus::None;
        }
        if txn_id == TXN_ID_AUTO_COMMIT {
            return TxnStatus::Committed;
        }

        let guard = self
            .inner
            .statuses
            .lock()
            .expect("txn status lock poisoned");
        guard.get(&txn_id).copied().unwrap_or(TxnStatus::Committed)
    }

    /// Mark a transaction as committed and advance the global watermark.
    pub fn mark_committed(&self, txn_id: TxnId) {
        {
            let mut guard = self
                .inner
                .statuses
                .lock()
                .expect("txn status lock poisoned");
            guard.insert(txn_id, TxnStatus::Committed);
        }

        // Opportunistically advance the committed watermark. Exact ordering is
        // not critical; best effort progression keeps snapshots monotonic.
        let mut current = self.inner.last_committed.load(Ordering::SeqCst);
        loop {
            if txn_id <= current {
                break;
            }
            match self.inner.last_committed.compare_exchange(
                current,
                txn_id,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    /// Mark a transaction as aborted.
    pub fn mark_aborted(&self, txn_id: TxnId) {
        let mut guard = self
            .inner
            .statuses
            .lock()
            .expect("txn status lock poisoned");
        guard.insert(txn_id, TxnStatus::Aborted);
    }

    /// Return the latest committed transaction ID (snapshot watermark).
    pub fn last_committed(&self) -> TxnId {
        self.inner.last_committed.load(Ordering::SeqCst)
    }
}

impl Default for TxnIdManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata tracking when a row was created and deleted.
///
/// Each row in a table has associated `created_by` and `deleted_by` values stored
/// in MVCC metadata columns. These values determine which transactions can see the row.
///
/// # Visibility Rules
///
/// A row is visible to a transaction if:
/// 1. The creating transaction committed and its ID ≤ the snapshot ID
/// 2. The row is not deleted (`deleted_by == TXN_ID_NONE`), or the deleting
///    transaction either hasn't committed or has ID > snapshot ID
///
/// Special case: Rows created by the current transaction are visible to that
/// transaction immediately, even before commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowVersion {
    /// Transaction ID that created this row version.
    pub created_by: TxnId,
    /// Transaction ID that deleted this row version (or [`TXN_ID_NONE`] if still visible).
    pub deleted_by: TxnId,
}

impl RowVersion {
    /// Create a new row version associated with `created_by`.
    pub fn new(created_by: TxnId) -> Self {
        Self {
            created_by,
            deleted_by: TXN_ID_NONE,
        }
    }

    /// Soft-delete the row version by the given transaction.
    pub fn delete(&mut self, deleted_by: TxnId) {
        self.deleted_by = deleted_by;
    }

    /// Basic visibility check using only numeric ordering.
    ///
    /// This retains the previous behaviour and is primarily used in tests.
    pub fn is_visible(&self, snapshot_txn_id: TxnId) -> bool {
        self.created_by <= snapshot_txn_id
            && (self.deleted_by == TXN_ID_NONE || self.deleted_by > snapshot_txn_id)
    }

    /// Determine whether the row is visible for the supplied snapshot using full MVCC rules.
    pub fn is_visible_for(&self, manager: &TxnIdManager, snapshot: TransactionSnapshot) -> bool {
        tracing::trace!(
            "[MVCC] is_visible_for: created_by={}, deleted_by={}, snapshot.txn_id={}, snapshot.snapshot_id={}",
            self.created_by,
            self.deleted_by,
            snapshot.txn_id,
            snapshot.snapshot_id
        );

        // Rows created inside the current transaction are visible unless this
        // transaction also deleted them.
        // IMPORTANT: TXN_ID_AUTO_COMMIT is never treated as "current transaction"
        if self.created_by == snapshot.txn_id && snapshot.txn_id != TXN_ID_AUTO_COMMIT {
            let visible = self.deleted_by != snapshot.txn_id;
            tracing::trace!("[MVCC] created by current txn, visible={}", visible);
            return visible;
        }

        // Ignore rows whose creator has not committed yet.
        let creator_status = manager.status(self.created_by);
        tracing::trace!("[MVCC] creator_status={:?}", creator_status);
        if !creator_status.is_committed() {
            tracing::trace!("[MVCC] creator not committed, invisible");
            return false;
        }

        if self.created_by > snapshot.snapshot_id {
            tracing::trace!("[MVCC] created_by > snapshot_id, invisible");
            return false;
        }

        match self.deleted_by {
            TXN_ID_NONE => {
                tracing::trace!("[MVCC] not deleted, visible");
                true
            }
            tx if tx == snapshot.txn_id && snapshot.txn_id != TXN_ID_AUTO_COMMIT => {
                tracing::trace!("[MVCC] deleted by current txn, invisible");
                false
            }
            tx => {
                if !manager.status(tx).is_committed() {
                    // A different transaction marked the row deleted but has not
                    // committed; the row remains visible to others.
                    tracing::trace!("[MVCC] deleter not committed, visible");
                    return true;
                }
                let visible = tx > snapshot.snapshot_id;
                tracing::trace!("[MVCC] deleter committed, visible={}", visible);
                visible
            }
        }
    }

    /// Visibility check for foreign key validation.
    ///
    /// For FK checks, we want different semantics:
    /// - Rows created by current txn are visible (so FK checks see new parent rows)
    /// - Rows deleted by current txn are STILL VISIBLE for FK purposes
    ///   (uncommitted deletes shouldn't affect FK validation)
    ///
    /// This implements the SQL standard behavior where FK constraints are checked
    /// against the committed state plus uncommitted inserts, but ignoring uncommitted deletes.
    pub fn is_visible_for_fk_check(
        &self,
        manager: &TxnIdManager,
        snapshot: TransactionSnapshot,
    ) -> bool {
        tracing::trace!(
            "[MVCC-FK] is_visible_for_fk_check: created_by={}, deleted_by={}, snapshot.txn_id={}, snapshot.snapshot_id={}",
            self.created_by,
            self.deleted_by,
            snapshot.txn_id,
            snapshot.snapshot_id
        );

        // Rows created inside the current transaction are visible.
        // IMPORTANT: TXN_ID_AUTO_COMMIT is never treated as "current transaction"
        if self.created_by == snapshot.txn_id && snapshot.txn_id != TXN_ID_AUTO_COMMIT {
            tracing::trace!("[MVCC-FK] created by current txn, visible");
            return true;
        }

        // Ignore rows whose creator has not committed yet.
        let creator_status = manager.status(self.created_by);
        tracing::trace!("[MVCC-FK] creator_status={:?}", creator_status);
        if !creator_status.is_committed() {
            tracing::trace!("[MVCC-FK] creator not committed, invisible");
            return false;
        }

        if self.created_by > snapshot.snapshot_id {
            tracing::trace!("[MVCC-FK] created_by > snapshot_id, invisible");
            return false;
        }

        // For FK checks, treat rows deleted by the current transaction as still visible
        if self.deleted_by == snapshot.txn_id && snapshot.txn_id != TXN_ID_AUTO_COMMIT {
            tracing::trace!("[MVCC-FK] deleted by current txn, but still visible for FK check");
            return true;
        }

        match self.deleted_by {
            TXN_ID_NONE => {
                tracing::trace!("[MVCC-FK] not deleted, visible");
                true
            }
            tx => {
                if !manager.status(tx).is_committed() {
                    // A different transaction marked the row deleted but has not
                    // committed; the row remains visible to others.
                    tracing::trace!("[MVCC-FK] deleter not committed, visible");
                    return true;
                }
                let visible = tx > snapshot.snapshot_id;
                tracing::trace!("[MVCC-FK] deleter committed, visible={}", visible);
                visible
            }
        }
    }
}

/// Transaction metadata captured when a transaction begins.
///
/// A snapshot contains two key pieces of information:
/// - `txn_id`: The unique ID of this transaction (used for writes)
/// - `snapshot_id`: The highest committed transaction ID at the time this transaction started
///
/// The `snapshot_id` determines which rows are visible: rows created by transactions
/// with IDs ≤ `snapshot_id` are visible, while rows created by later transactions are not.
/// This implements snapshot isolation.
#[derive(Debug, Clone, Copy)]
pub struct TransactionSnapshot {
    /// The unique ID assigned to this transaction.
    pub txn_id: TxnId,
    /// The highest committed transaction ID when this transaction began.
    pub snapshot_id: TxnId,
}

/// Transaction status values tracked by the manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    Active,
    Committed,
    Aborted,
    None,
}

impl TxnStatus {
    pub fn is_committed(self) -> bool {
        matches!(self, TxnStatus::Committed)
    }

    pub fn is_active(self) -> bool {
        matches!(self, TxnStatus::Active)
    }

    pub fn is_aborted(self) -> bool {
        matches!(self, TxnStatus::Aborted)
    }
}

// ============================================================================
// Arrow helpers for MVCC column construction
// ============================================================================

/// Build MVCC columns (row_id, created_by, deleted_by) for INSERT/CTAS operations.
pub fn build_insert_mvcc_columns(
    row_count: usize,
    start_row_id: RowId,
    creator_txn_id: TxnId,
    deleted_marker: TxnId,
) -> (ArrayRef, ArrayRef, ArrayRef) {
    let mut row_builder = UInt64Builder::with_capacity(row_count);
    for offset in 0..row_count {
        row_builder.append_value(start_row_id + offset as u64);
    }

    let mut created_builder = UInt64Builder::with_capacity(row_count);
    let mut deleted_builder = UInt64Builder::with_capacity(row_count);
    for _ in 0..row_count {
        created_builder.append_value(creator_txn_id);
        deleted_builder.append_value(deleted_marker);
    }

    (
        Arc::new(row_builder.finish()) as ArrayRef,
        Arc::new(created_builder.finish()) as ArrayRef,
        Arc::new(deleted_builder.finish()) as ArrayRef,
    )
}

/// Build MVCC field definitions (row_id, created_by, deleted_by).
pub fn build_mvcc_fields() -> Vec<Field> {
    vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false),
    ]
}

/// Build field with field_id metadata for a user column.
pub fn build_field_with_metadata(
    name: &str,
    data_type: DataType,
    nullable: bool,
    field_id: FieldId,
) -> Field {
    let mut metadata = HashMap::with_capacity(1);
    metadata.insert(FIELD_ID_META_KEY.to_string(), field_id.to_string());
    Field::new(name, data_type, nullable).with_metadata(metadata)
}

/// Build DELETE batch with row_id and deleted_by columns.
pub fn build_delete_batch(
    row_ids: Vec<RowId>,
    deleted_by_txn_id: TxnId,
) -> llkv_result::Result<RecordBatch> {
    let row_count = row_ids.len();

    let fields = vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false),
    ];

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(row_ids)),
        Arc::new(UInt64Array::from(vec![deleted_by_txn_id; row_count])),
    ];

    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(Error::Arrow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_txn_id_manager_allocates_monotonic_ids() {
        let manager = TxnIdManager::new();
        let snapshot1 = manager.begin_transaction();
        let snapshot2 = manager.begin_transaction();
        assert!(snapshot2.txn_id > snapshot1.txn_id);
    }

    #[test]
    fn test_row_visibility_simple() {
        let manager = TxnIdManager::new();
        let writer_snapshot = manager.begin_transaction();
        let mut row = RowVersion::new(writer_snapshot.txn_id);

        // Newly created rows are visible to the creating transaction.
        assert!(row.is_visible(writer_snapshot.txn_id));
        assert!(row.is_visible_for(&manager, writer_snapshot));

        // After the writer commits, the row becomes visible to later snapshots.
        manager.mark_committed(writer_snapshot.txn_id);
        let committed = manager.last_committed();
        assert!(row.is_visible(committed));

        let reader_snapshot = manager.begin_transaction();
        assert!(row.is_visible_for(&manager, reader_snapshot));

        // Deleting transaction must commit before other readers stop seeing the row.
        let deleter_snapshot = manager.begin_transaction();
        row.delete(deleter_snapshot.txn_id);
        assert!(row.is_visible_for(&manager, reader_snapshot));

        manager.mark_committed(deleter_snapshot.txn_id);
        assert!(row.is_visible_for(&manager, reader_snapshot));

        let post_delete_snapshot = manager.begin_transaction();
        assert!(!row.is_visible_for(&manager, post_delete_snapshot));
    }
}
