//! Constraint validation helpers for RuntimeContext.
//!
//! This module contains constraint-related validation logic:
//! - Delete conflict detection (MVCC)
//! - Visible child rows collection (for foreign key validation)

use arrow::array::{Array, UInt64Array};
use llkv_column_map::store::GatherNullPolicy;
use llkv_column_map::types::LogicalFieldId;
use llkv_executor::{translation, ExecutorTable};
use llkv_plan::PlanValue;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::{FieldId, RowId};
use llkv_transaction::{TransactionSnapshot, filter_row_ids_for_snapshot, TxnId};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

use crate::TXN_ID_NONE;

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Detect delete conflicts by checking if any rows are locked by other transactions.
    /// 
    /// This function is `pub(super)` because it's called by apply_delete in delete.rs.
    pub(super) fn detect_delete_conflicts(
        &self,
        table: &ExecutorTable<P>,
        display_name: &str,
        row_ids: &[RowId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() {
            return Ok(());
        }

        let table_id = table.table.table_id();
        let deleted_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);
        let logical_fields: Arc<[LogicalFieldId]> = Arc::from([deleted_lfid]);

        if let Err(err) = table
            .table
            .store()
            .prepare_gather_context(logical_fields.as_ref())
        {
            match err {
                Error::NotFound => return Ok(()),
                other => return Err(other),
            }
        }

        let mut stream = table.table.stream_columns(
            Arc::clone(&logical_fields),
            row_ids.to_vec(),
            GatherNullPolicy::IncludeNulls,
        )?;

        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let window = chunk.row_ids();
            let deleted_column = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::Internal(
                        "failed to read MVCC deleted_by column for conflict detection".into(),
                    )
                })?;

            for (idx, row_id) in window.iter().enumerate() {
                let deleted_by: TxnId = if deleted_column.is_null(idx) {
                    TXN_ID_NONE
                } else {
                    deleted_column.value(idx)
                };

                if deleted_by == TXN_ID_NONE || deleted_by == snapshot.txn_id {
                    continue;
                }

                let status = self.txn_manager.status(deleted_by);
                if !status.is_active() {
                    continue;
                }

                tracing::debug!(
                    "[MVCC] delete conflict: table='{}' row_id={} deleted_by={} status={:?} current_txn={}",
                    display_name,
                    row_id,
                    deleted_by,
                    status,
                    snapshot.txn_id
                );

                return Err(Error::TransactionContextError(format!(
                    "transaction conflict on table '{}' for row {}: row locked by transaction {} ({:?})",
                    display_name, row_id, deleted_by, status
                )));
            }
        }

        Ok(())
    }

    /// Collect visible child rows for foreign key validation.
    ///
    /// This function is `pub(super)` because it's called by foreign key validation
    /// in delete.rs and update.rs.
    pub(super) fn collect_visible_child_rows(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<(RowId, Vec<PlanValue>)>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let anchor_field = field_ids[0];
        let filter_expr = translation::expression::full_table_scan_filter(anchor_field);
        let raw_row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let visible_row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            raw_row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if visible_row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let logical_field_ids: Vec<LogicalFieldId> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let mut stream = match table.table.stream_columns(
            logical_field_ids.clone(),
            visible_row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); visible_row_ids.len()];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    if let Some(row) = rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }

        Ok(visible_row_ids.into_iter().zip(rows).collect())
    }

    /// Validate foreign key constraints for INSERT operations.
    ///
    /// This function is `pub(super)` because it's called by insert operations
    /// in insert.rs and update operations in update.rs (when inserting new row values).
    pub(super) fn check_foreign_keys_on_insert(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        let schema_field_ids: Vec<FieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| column.field_id)
            .collect();

        self.constraint_service.validate_insert_foreign_keys(
            table.table.table_id(),
            &schema_field_ids,
            column_order,
            rows,
            |request| {
                let parent_table = self.lookup_table(request.referenced_table_canonical)?;
                self.scan_multi_column_values(
                    parent_table.as_ref(),
                    request.referenced_field_ids,
                    snapshot,
                )
            },
        )
    }

    /// Validate foreign key constraints for UPDATE operations.
    /// 
    /// This function is `pub(super)` because it's called by update operations in update.rs.
    pub(super) fn check_foreign_keys_on_update(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        _canonical_name: &str,
        row_ids: &[RowId],
        updated_field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() || updated_field_ids.is_empty() {
            return Ok(());
        }

        self.constraint_service.validate_update_foreign_keys(
            table.table.table_id(),
            row_ids,
            updated_field_ids,
            |request| {
                self.collect_row_values_for_ids(
                    table,
                    request.referenced_row_ids,
                    request.referenced_field_ids,
                )
            },
            |request| {
                let child_table = self.lookup_table(request.referencing_table_canonical)?;
                self.collect_visible_child_rows(
                    child_table.as_ref(),
                    request.referencing_field_ids,
                    snapshot,
                )
            },
        )
    }

    /// Validate foreign key constraints for DELETE operations.
    /// 
    /// This function is `pub(super)` because it's called by apply_delete in delete.rs.
    pub(super) fn check_foreign_keys_on_delete(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        row_ids: &[RowId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() {
            return Ok(());
        }

        self.constraint_service.validate_delete_foreign_keys(
            table.table.table_id(),
            row_ids,
            |request| {
                self.collect_row_values_for_ids(
                    table,
                    request.referenced_row_ids,
                    request.referenced_field_ids,
                )
            },
            |request| {
                let child_table = self.lookup_table(request.referencing_table_canonical)?;
                self.collect_visible_child_rows(
                    child_table.as_ref(),
                    request.referencing_field_ids,
                    snapshot,
                )
            },
        )
    }
}
