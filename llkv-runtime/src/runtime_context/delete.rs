//! DELETE operation implementation for RuntimeContext.
//!
//! This module contains all logic for deleting rows from tables, including:
//! - Filtered and full table deletes
//! - Foreign key validation for deletes
//! - MVCC-based soft deletion

use crate::{RuntimeStatementResult, canonical_table_name};
use croaring::Treemap;
use llkv_executor::{ExecutorTable, translation};
use llkv_expr::Expr as LlkvExpr;
use llkv_plan::DeletePlan;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_transaction::{TransactionSnapshot, mvcc};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Delete rows according to a parsed DeletePlan.
    pub(crate) fn delete(
        &self,
        plan: DeletePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = match self.tables.read().unwrap().get(&canonical_name) {
            Some(table) => Arc::clone(table),
            None => {
                return Err(Error::NotFound);
            }
        };

        // Views are read-only - reject DELETE operations
        if self.is_view(table.table_id())? {
            return Err(Error::InvalidArgumentError(format!(
                "cannot modify view '{}'",
                display_name
            )));
        }

        match plan.filter {
            Some(filter) => self.delete_filtered_rows(
                table.as_ref(),
                display_name,
                canonical_name.clone(),
                filter,
                snapshot,
            ),
            None => self.delete_all_rows(table.as_ref(), display_name, canonical_name, snapshot),
        }
    }

    /// Delete rows from a table that match a filter expression.
    pub(super) fn delete_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let schema = table.schema.as_ref();
        let filter_expr = translation::expression::translate_predicate(filter, schema, |name| {
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{}'",
                name
            ))
        })?;
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        tracing::trace!(
            table = %display_name,
            rows = row_ids.cardinality(),
            "delete_filtered_rows collected row ids"
        );
        self.apply_delete(table, display_name, canonical_name, row_ids, snapshot, true)
    }

    /// Delete all rows from a table.
    pub(super) fn delete_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if total_rows == 0 {
            return Ok(RuntimeStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let anchor_field = table.schema.first_field_id().ok_or_else(|| {
            Error::InvalidArgumentError("DELETE requires a table with at least one column".into())
        })?;
        let filter_expr = translation::expression::full_table_scan_filter(anchor_field);
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        self.apply_delete(table, display_name, canonical_name, row_ids, snapshot, true)
    }

    /// Apply a delete operation (MVCC soft delete).
    ///
    /// This function is `pub(super)` because it's also called by UPDATE operations
    /// when replacing rows (delete old + insert new).
    pub(super) fn apply_delete(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        _canonical_name: String,
        row_ids: Treemap,
        snapshot: TransactionSnapshot,
        enforce_foreign_keys: bool,
    ) -> Result<RuntimeStatementResult<P>> {
        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        if enforce_foreign_keys {
            self.check_foreign_keys_on_delete(table, &display_name, &row_ids, snapshot)?;
        }

        self.detect_delete_conflicts(table, &display_name, &row_ids, snapshot)?;

        let removed = row_ids.cardinality();

        // Build DELETE batch using helper
        let batch = mvcc::build_delete_batch(row_ids, snapshot.txn_id)?;
        table.table.append(&batch)?;

        table.total_rows.fetch_sub(removed, Ordering::SeqCst);

        Ok(RuntimeStatementResult::Delete {
            table_name: display_name,
            rows_deleted: removed as usize,
        })
    }
}
