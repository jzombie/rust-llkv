//! MVCC row filtering and builder utilities.
//!
//! This module provides higher-level MVCC utilities that work with tables,
//! including row visibility filtering and MVCC column building for table operations.

use arrow::array::{Array, ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field};
use llkv_column_map::store::GatherNullPolicy;
use llkv_column_map::types::LogicalFieldId;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::catalog::MvccColumnBuilder;
use llkv_table::table::RowIdFilter;
use llkv_table::{FieldId, RowId, Table};
use simd_r_drive_entry_handle::EntryHandle;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::mvcc::{self, RowVersion, TXN_ID_AUTO_COMMIT, TXN_ID_NONE};
use crate::{TransactionSnapshot, TxnIdManager};

/// Builder for MVCC columns that delegates to the `mvcc` module functions.
///
/// This implements the `MvccColumnBuilder` trait from `llkv-table` and provides
/// MVCC column construction for INSERT operations and table creation.
pub struct TransactionMvccBuilder;

impl MvccColumnBuilder for TransactionMvccBuilder {
    fn build_insert_columns(
        &self,
        row_count: usize,
        start_row_id: RowId,
        creator_txn_id: u64,
        deleted_marker: u64,
    ) -> (ArrayRef, ArrayRef, ArrayRef) {
        mvcc::build_insert_mvcc_columns(row_count, start_row_id, creator_txn_id, deleted_marker)
    }

    fn mvcc_fields(&self) -> Vec<Field> {
        mvcc::build_mvcc_fields()
    }

    fn field_with_metadata(
        &self,
        name: &str,
        data_type: DataType,
        nullable: bool,
        field_id: FieldId,
    ) -> Field {
        mvcc::build_field_with_metadata(name, data_type, nullable, field_id)
    }
}

/// Filters a list of row IDs to return only those visible to the given snapshot.
///
/// This function:
/// 1. Retrieves MVCC metadata (created_by, deleted_by) for each row
/// 2. Applies snapshot visibility rules using the transaction manager
/// 3. Returns only row IDs that are visible to the snapshot
///
/// # Arguments
///
/// * `table` - The table to filter rows from
/// * `row_ids` - Row IDs to check for visibility
/// * `txn_manager` - Transaction ID manager for visibility checks
/// * `snapshot` - Transaction snapshot defining the visibility point
///
/// # Returns
///
/// A filtered vector of row IDs that are visible to the snapshot.
///
/// # Errors
///
/// Returns an error if MVCC column gathering fails (excluding NotFound, which
/// is treated as all rows being visible).
pub fn filter_row_ids_for_snapshot<P>(
    table: &Table<P>,
    row_ids: Vec<RowId>,
    txn_manager: &TxnIdManager,
    snapshot: TransactionSnapshot,
) -> Result<Vec<RowId>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    tracing::debug!(
        "[FILTER_ROWS] Filtering {} row IDs for snapshot txn_id={}, snapshot_id={}",
        row_ids.len(),
        snapshot.txn_id,
        snapshot.snapshot_id
    );

    if row_ids.is_empty() {
        return Ok(row_ids);
    }

    let table_id = table.table_id();
    let created_lfid = LogicalFieldId::for_mvcc_created_by(table_id);
    let deleted_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);
    let logical_fields: Arc<[LogicalFieldId]> = Arc::from([created_lfid, deleted_lfid]);

    if let Err(err) = table
        .store()
        .prepare_gather_context(logical_fields.as_ref())
    {
        match err {
            Error::NotFound => {
                tracing::trace!(
                    "[FILTER_ROWS] MVCC columns not found for table_id={}, treating all rows as visible",
                    table_id
                );
                return Ok(row_ids);
            }
            other => {
                tracing::error!(
                    "[FILTER_ROWS] Failed to prepare gather context: {:?}",
                    other
                );
                return Err(other);
            }
        }
    }

    let total_rows = row_ids.len();
    let mut stream = match table.stream_columns(
        Arc::clone(&logical_fields),
        row_ids,
        GatherNullPolicy::IncludeNulls,
    ) {
        Ok(stream) => stream,
        Err(err) => {
            tracing::error!("[FILTER_ROWS] stream_columns error: {:?}", err);
            return Err(err);
        }
    };

    let mut visible = Vec::with_capacity(total_rows);

    while let Some(chunk) = stream.next_batch()? {
        let batch = chunk.batch();
        let window = chunk.row_ids();

        if batch.num_columns() < 2 {
            tracing::debug!(
                "[FILTER_ROWS] version_batch has < 2 columns for table_id={}, returning window rows unfiltered",
                table_id
            );
            visible.extend_from_slice(window);
            continue;
        }

        let created_column = batch.column(0).as_any().downcast_ref::<UInt64Array>();
        let deleted_column = batch.column(1).as_any().downcast_ref::<UInt64Array>();

        if created_column.is_none() || deleted_column.is_none() {
            tracing::debug!(
                "[FILTER_ROWS] Failed to downcast MVCC columns for table_id={}, returning window rows unfiltered",
                table_id
            );
            visible.extend_from_slice(window);
            continue;
        }

        let created_column = created_column.unwrap();
        let deleted_column = deleted_column.unwrap();

        for (idx, row_id) in window.iter().enumerate() {
            let created_by = if created_column.is_null(idx) {
                TXN_ID_AUTO_COMMIT
            } else {
                created_column.value(idx)
            };
            let deleted_by = if deleted_column.is_null(idx) {
                TXN_ID_NONE
            } else {
                deleted_column.value(idx)
            };

            let version = RowVersion {
                created_by,
                deleted_by,
            };
            let is_visible = version.is_visible_for(txn_manager, snapshot);
            tracing::trace!(
                "[FILTER_ROWS] row_id={}: created_by={}, deleted_by={}, is_visible={}",
                row_id,
                created_by,
                deleted_by,
                is_visible
            );
            if is_visible {
                visible.push(*row_id);
            }
        }
    }

    tracing::debug!(
        "[FILTER_ROWS] Filtered from {} to {} visible rows",
        total_rows,
        visible.len()
    );
    Ok(visible)
}

/// A `RowIdFilter` implementation that applies MVCC visibility filtering.
///
/// This filter can be passed to table scan operations to automatically filter
/// out rows that are not visible to a particular transaction snapshot.
pub struct MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    txn_manager: Arc<TxnIdManager>,
    snapshot: TransactionSnapshot,
    _marker: PhantomData<fn(P)>,
}

impl<P> MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Creates a new MVCC row filter with the given transaction manager and snapshot.
    pub fn new(txn_manager: Arc<TxnIdManager>, snapshot: TransactionSnapshot) -> Self {
        Self {
            txn_manager,
            snapshot,
            _marker: PhantomData,
        }
    }
}

impl<P> RowIdFilter<P> for MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn filter(&self, table: &Table<P>, row_ids: Vec<RowId>) -> Result<Vec<RowId>> {
        tracing::trace!(
            "[MVCC_FILTER] filter() called with row_ids {:?}, snapshot txn={}, snapshot_id={}",
            row_ids,
            self.snapshot.txn_id,
            self.snapshot.snapshot_id
        );
        let result = filter_row_ids_for_snapshot(table, row_ids, &self.txn_manager, self.snapshot);
        if let Ok(ref visible) = result {
            tracing::trace!(
                "[MVCC_FILTER] filter() returning visible row_ids: {:?}",
                visible
            );
        }
        result
    }
}
