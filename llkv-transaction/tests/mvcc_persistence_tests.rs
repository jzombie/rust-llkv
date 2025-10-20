#![forbid(unsafe_code)]

//! Integration tests for MVCC transaction persistence with durable pagers.
//!
//! These tests verify that transaction state (transaction IDs, commit status, etc.)
//! is properly persisted and can be recovered when reopening a store with a durable pager.
//!
//! ## Implementation Status
//!
//! **MVCC data (created_by, deleted_by) IS persisted** to the durable pager and can be read
//! after reopening the store.
//!
//! **Transaction manager state (next_txn_id, last_committed) IS persisted** to the system catalog.
//! When a store is reopened:
//! - `next_txn_id` is loaded from the catalog, ensuring transaction IDs don't overlap
//! - `last_committed` is loaded, ensuring the snapshot watermark is correctly restored
//! - Unknown transaction IDs (>= next_txn_id) default to TxnStatus::Committed for backwards compatibility
//!
//! This ensures:
//! - Committed transactions from a previous session remain visible
//! - Uncommitted transactions remain invisible (their created_by txn_id won't be in committed set)
//! - Transaction IDs are never reused across sessions
//! - MVCC visibility rules work correctly after store reopens
//!
//! ## Test Coverage
//!
//! All tests verify MVCC persistence with durable pagers (SimdRDrivePager):
//! - `test_mvcc_committed_transaction_persists_across_reopen`: Committed data remains visible
//! - `test_mvcc_uncommitted_transaction_not_visible_after_reopen`: Uncommitted data is invisible
//! - `test_mvcc_multiple_transactions_persist_correctly`: Multiple committed transactions work
//! - `test_mvcc_soft_delete_persists_across_reopen`: Soft deletes (deleted_by) are persisted

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use tempfile::TempDir;

use llkv_column_map::ColumnStore;
use llkv_column_map::store::{GatherNullPolicy, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_result::Result;
use llkv_storage::pager::simd_r_drive_pager::SimdRDrivePager;
use llkv_table::Table;
use llkv_table::types::{FieldId, TableId};
use llkv_transaction::mvcc::{RowVersion, TransactionSnapshot, TxnIdManager};

/// Helper to create a logical field ID for a user data column.
fn lfid(table_id: TableId, field_id: FieldId) -> LogicalFieldId {
    LogicalFieldId::from_parts(Namespace::UserData, table_id, field_id)
}

/// Helper to create a field with field_id metadata.
fn field_with_fid(name: &str, dt: DataType, fid: FieldId, nullable: bool) -> Field {
    use std::collections::HashMap;
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        llkv_table::constants::FIELD_ID_META_KEY.to_string(),
        fid.to_string(),
    )]))
}

/// Helper to open or create a SimdRDrivePager at the given path.
fn open_pager(path: &Path) -> Result<Arc<SimdRDrivePager>> {
    Ok(Arc::new(SimdRDrivePager::open(path)?))
}

/// Helper to open or create a Table with the given pager.
fn open_table(table_id: TableId, pager: Arc<SimdRDrivePager>) -> Result<Table<SimdRDrivePager>> {
    Table::from_id(table_id, pager)
}

/// Helper to insert a batch with MVCC columns into a table.
/// Note: MVCC columns should NOT have field_id metadata initially - Table.append() will add it.
#[allow(clippy::too_many_arguments)]
fn insert_batch_with_mvcc(
    table: &Table<SimdRDrivePager>,
    _table_id: TableId,
    row_ids: Vec<u64>,
    ids: Vec<i64>,
    names: Vec<&str>,
    created_by: Vec<u64>,
    deleted_by: Vec<u64>,
) -> Result<()> {
    use llkv_column_map::store::{CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME};

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        field_with_fid("id", DataType::Int64, 1, false),
        field_with_fid("name", DataType::Utf8, 2, false),
        // MVCC columns without field_id metadata - Table.append() will add the LogicalFieldId
        Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
        Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(row_ids)),
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(UInt64Array::from(created_by)),
            Arc::new(UInt64Array::from(deleted_by)),
        ],
    )
    .expect("RecordBatch creation");

    table.append(&batch)
}

/// Helper to verify MVCC visibility of rows.
fn verify_mvcc_visibility(
    store: &ColumnStore<SimdRDrivePager>,
    table_id: TableId,
    row_ids: &[u64],
    txn_manager: &TxnIdManager,
    snapshot: TransactionSnapshot,
    expected_visible: &[bool],
) -> Result<()> {
    let created_by_lfid = LogicalFieldId::for_mvcc_created_by(table_id);
    let deleted_by_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);

    // Gather MVCC columns
    let batch = store.gather_rows(
        &[created_by_lfid, deleted_by_lfid],
        row_ids,
        GatherNullPolicy::ErrorOnMissing,
    )?;

    let created_by_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let deleted_by_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();

    for (idx, &expected) in expected_visible.iter().enumerate() {
        let row_version = RowVersion {
            created_by: created_by_arr.value(idx),
            deleted_by: deleted_by_arr.value(idx),
        };
        let visible = row_version.is_visible_for(txn_manager, snapshot);
        assert_eq!(
            visible,
            expected,
            "Row {} visibility mismatch: expected {}, got {} (created_by={}, deleted_by={}, snapshot_txn_id={}, snapshot_id={})",
            row_ids[idx],
            expected,
            visible,
            row_version.created_by,
            row_version.deleted_by,
            snapshot.txn_id,
            snapshot.snapshot_id
        );
    }

    Ok(())
}

#[test]
fn test_mvcc_committed_transaction_persists_across_reopen() -> Result<()> {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc_persist.db");

    const TABLE_ID: TableId = 100;
    const TXN_ID: u64 = 2; // Transaction ID for our insert

    // ===== Scope 1: Create table, insert data with transaction, commit =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, pager)?;
        let txn_manager = TxnIdManager::new();

        // Simulate a transaction: begin, insert, commit
        let snapshot = txn_manager.begin_transaction();
        assert_eq!(snapshot.txn_id, TXN_ID);

        // Insert rows with MVCC metadata (created_by = TXN_ID, deleted_by = TXN_ID_NONE)
        insert_batch_with_mvcc(
            &table,
            TABLE_ID,
            vec![1, 2, 3],
            vec![100, 200, 300],
            vec!["alice", "bob", "charlie"],
            vec![TXN_ID, TXN_ID, TXN_ID],
            vec![u64::MAX, u64::MAX, u64::MAX], // TXN_ID_NONE
        )?;

        // Commit the transaction
        txn_manager.mark_committed(TXN_ID);

        // Verify rows are visible to a new snapshot after commit
        let post_commit_snapshot = txn_manager.begin_transaction();
        verify_mvcc_visibility(
            table.store(),
            TABLE_ID,
            &[1, 2, 3],
            &txn_manager,
            post_commit_snapshot,
            &[true, true, true],
        )?;

        // Scope ends: table and pager are dropped
    }

    // ===== Scope 2: Reopen the store and verify data is still visible =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, pager)?;
        let txn_manager = TxnIdManager::new();

        // Create a new snapshot (should see committed data)
        let snapshot = txn_manager.begin_transaction();

        // Verify the committed rows are visible
        verify_mvcc_visibility(
            table.store(),
            TABLE_ID,
            &[1, 2, 3],
            &txn_manager,
            snapshot,
            &[true, true, true],
        )?;

        // Verify we can read the actual data
        let id_lfid = lfid(TABLE_ID, 1);
        let name_lfid = lfid(TABLE_ID, 2);

        let batch = table.store().gather_rows(
            &[id_lfid, name_lfid],
            &[1, 2, 3],
            GatherNullPolicy::ErrorOnMissing,
        )?;

        let id_arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let name_arr = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(id_arr.value(0), 100);
        assert_eq!(id_arr.value(1), 200);
        assert_eq!(id_arr.value(2), 300);
        assert_eq!(name_arr.value(0), "alice");
        assert_eq!(name_arr.value(1), "bob");
        assert_eq!(name_arr.value(2), "charlie");
    }

    Ok(())
}

#[test]
fn test_mvcc_uncommitted_transaction_not_visible_after_reopen() -> Result<()> {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc_uncommitted.db");

    const TABLE_ID: TableId = 100;
    const TXN_ID: u64 = 2;

    // ===== Scope 1: Create table, insert data with transaction, DO NOT commit =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, Arc::clone(&pager))?;
        let txn_manager = TxnIdManager::new();

        // Simulate a transaction: begin, insert, but don't commit
        let snapshot = txn_manager.begin_transaction();
        assert_eq!(snapshot.txn_id, TXN_ID);

        // Insert rows with MVCC metadata (created_by = TXN_ID, deleted_by = TXN_ID_NONE)
        insert_batch_with_mvcc(
            &table,
            TABLE_ID,
            vec![1, 2],
            vec![100, 200],
            vec!["alice", "bob"],
            vec![TXN_ID, TXN_ID],
            vec![u64::MAX, u64::MAX],
        )?;

        // Do NOT commit - just let the scope end
        // txn_manager.mark_committed(TXN_ID); // <- OMITTED

        // Persist the current transaction state so that on reopen, the manager knows about txn_id=2
        let next_txn_id = txn_manager.current_next_txn_id();
        let last_committed = txn_manager.last_committed();
        let catalog = llkv_table::SysCatalog::new(table.store());
        catalog.put_next_txn_id(next_txn_id)?;
        catalog.put_last_committed_txn_id(last_committed)?;

        // Scope ends: table and pager are dropped, transaction is abandoned
    }

    // ===== Scope 2: Reopen the store and verify data is NOT visible =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, Arc::clone(&pager))?;

        // Load transaction state from catalog
        let catalog = llkv_table::SysCatalog::new(table.store());
        let loaded_next_txn_id = catalog
            .get_next_txn_id()?
            .expect("next_txn_id should be persisted");
        let loaded_last_committed = catalog
            .get_last_committed_txn_id()?
            .expect("last_committed should be persisted");

        let txn_manager = llkv_transaction::TxnIdManager::new_with_initial_state(
            loaded_next_txn_id,
            loaded_last_committed,
        );

        // Create a new snapshot
        let snapshot = txn_manager.begin_transaction();

        // IMPORTANT: Since the TxnIdManager is recreated from scratch and doesn't
        // persist transaction commit status, it doesn't know about TXN_ID=2.
        // The status() method will return TxnStatus::Committed for unknown txn ids
        // by default (see mvcc.rs line 102), so the rows WILL be visible.
        //
        // TODO: To fix this, we need to:
        // 1. Persist next_txn_id to the catalog
        // 2. On reopen, load next_txn_id from the catalog
        // 3. Treat txn_ids >= persisted next_txn_id as unknown/aborted
        //
        // For now, this test is ignored and documents the expected behavior once
        // transaction state persistence is implemented.
        verify_mvcc_visibility(
            table.store(),
            TABLE_ID,
            &[1, 2],
            &txn_manager,
            snapshot,
            &[false, false],
        )?;
    }

    Ok(())
}

#[test]
fn test_mvcc_multiple_transactions_persist_correctly() -> Result<()> {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc_multi_txn.db");

    const TABLE_ID: TableId = 100;

    // ===== Scope 1: Create table, run two transactions =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, pager)?;
        let txn_manager = TxnIdManager::new();

        // Transaction 1: Insert rows 1, 2
        let snapshot1 = txn_manager.begin_transaction();
        let txn1_id = snapshot1.txn_id;
        insert_batch_with_mvcc(
            &table,
            TABLE_ID,
            vec![1, 2],
            vec![100, 200],
            vec!["alice", "bob"],
            vec![txn1_id, txn1_id],
            vec![u64::MAX, u64::MAX],
        )?;
        txn_manager.mark_committed(txn1_id);

        // Transaction 2: Insert rows 3, 4
        let snapshot2 = txn_manager.begin_transaction();
        let txn2_id = snapshot2.txn_id;
        insert_batch_with_mvcc(
            &table,
            TABLE_ID,
            vec![3, 4],
            vec![300, 400],
            vec!["charlie", "david"],
            vec![txn2_id, txn2_id],
            vec![u64::MAX, u64::MAX],
        )?;
        txn_manager.mark_committed(txn2_id);

        // Verify all rows are visible
        let post_commit_snapshot = txn_manager.begin_transaction();
        verify_mvcc_visibility(
            table.store(),
            TABLE_ID,
            &[1, 2, 3, 4],
            &txn_manager,
            post_commit_snapshot,
            &[true, true, true, true],
        )?;

        // Persist the current transaction state
        let next_txn_id = txn_manager.current_next_txn_id();
        let last_committed = txn_manager.last_committed();
        let catalog = llkv_table::SysCatalog::new(table.store());
        catalog.put_next_txn_id(next_txn_id)?;
        catalog.put_last_committed_txn_id(last_committed)?;

        // Scope ends: table and pager are dropped
    }

    // ===== Scope 2: Reopen and verify all committed data is visible =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, Arc::clone(&pager))?;

        // Load transaction state from catalog
        let catalog = llkv_table::SysCatalog::new(table.store());
        let loaded_next_txn_id = catalog
            .get_next_txn_id()?
            .expect("next_txn_id should be persisted");
        let loaded_last_committed = catalog
            .get_last_committed_txn_id()?
            .expect("last_committed should be persisted");

        let txn_manager = llkv_transaction::TxnIdManager::new_with_initial_state(
            loaded_next_txn_id,
            loaded_last_committed,
        );

        // Create a new snapshot
        let snapshot = txn_manager.begin_transaction();

        // Verify all committed rows are visible
        verify_mvcc_visibility(
            table.store(),
            TABLE_ID,
            &[1, 2, 3, 4],
            &txn_manager,
            snapshot,
            &[true, true, true, true],
        )?;

        // Verify actual data
        let id_lfid = lfid(TABLE_ID, 1);
        let name_lfid = lfid(TABLE_ID, 2);

        let batch = table.store().gather_rows(
            &[id_lfid, name_lfid],
            &[1, 2, 3, 4],
            GatherNullPolicy::ErrorOnMissing,
        )?;

        let id_arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id_arr.value(0), 100);
        assert_eq!(id_arr.value(1), 200);
        assert_eq!(id_arr.value(2), 300);
        assert_eq!(id_arr.value(3), 400);
    }

    Ok(())
}

#[test]
fn test_mvcc_soft_delete_persists_across_reopen() -> Result<()> {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc_soft_delete.db");

    const TABLE_ID: TableId = 100;

    // ===== Scope 1: Insert data, then soft-delete it =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, pager)?;
        let txn_manager = TxnIdManager::new();

        // Transaction 1: Insert rows
        let snapshot1 = txn_manager.begin_transaction();
        let txn1_id = snapshot1.txn_id;
        insert_batch_with_mvcc(
            &table,
            TABLE_ID,
            vec![1, 2],
            vec![100, 200],
            vec!["alice", "bob"],
            vec![txn1_id, txn1_id],
            vec![u64::MAX, u64::MAX],
        )?;
        txn_manager.mark_committed(txn1_id);

        // Transaction 2: Soft-delete row 1
        let snapshot2 = txn_manager.begin_transaction();
        let txn2_id = snapshot2.txn_id;
        // Update row 1's deleted_by to txn2_id
        insert_batch_with_mvcc(
            &table,
            TABLE_ID,
            vec![1],
            vec![100],
            vec!["alice"],
            vec![txn1_id], // Keep original created_by
            vec![txn2_id], // Mark as deleted by txn2
        )?;
        txn_manager.mark_committed(txn2_id);

        // Verify: row 1 should be invisible, row 2 should be visible
        let post_delete_snapshot = txn_manager.begin_transaction();
        verify_mvcc_visibility(
            table.store(),
            TABLE_ID,
            &[1, 2],
            &txn_manager,
            post_delete_snapshot,
            &[false, true],
        )?;

        // Persist the current transaction state
        let next_txn_id = txn_manager.current_next_txn_id();
        let last_committed = txn_manager.last_committed();
        let catalog = llkv_table::SysCatalog::new(table.store());
        catalog.put_next_txn_id(next_txn_id)?;
        catalog.put_last_committed_txn_id(last_committed)?;

        // Scope ends
    }

    // ===== Scope 2: Reopen and verify soft-delete is persisted =====
    {
        let pager = open_pager(&path)?;
        let table = open_table(TABLE_ID, Arc::clone(&pager))?;

        // Load transaction state from catalog
        let catalog = llkv_table::SysCatalog::new(table.store());
        let loaded_next_txn_id = catalog
            .get_next_txn_id()?
            .expect("next_txn_id should be persisted");
        let loaded_last_committed = catalog
            .get_last_committed_txn_id()?
            .expect("last_committed should be persisted");

        let txn_manager = llkv_transaction::TxnIdManager::new_with_initial_state(
            loaded_next_txn_id,
            loaded_last_committed,
        );

        let snapshot = txn_manager.begin_transaction();

        // Verify row 1 is still invisible, row 2 is still visible
        verify_mvcc_visibility(
            table.store(),
            TABLE_ID,
            &[1, 2],
            &txn_manager,
            snapshot,
            &[false, true],
        )?;
    }

    Ok(())
}
