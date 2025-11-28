//! TRUNCATE operation implementation for RuntimeContext.
//!
//! This module contains logic for truncating tables (removing all rows), including:
//! - Foreign key validation for truncates
//! - MVCC-based soft deletion of all rows

use crate::{RuntimeStatementResult, canonical_table_name};
use llkv_executor::{ExecutorTable, translation};
use llkv_plan::TruncatePlan;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_transaction::TransactionSnapshot;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, atomic::Ordering};

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Truncate all rows from a table according to a parsed TruncatePlan.
    pub(crate) fn truncate(
        &self,
        plan: TruncatePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = match self.tables.read().unwrap().get(&canonical_name) {
            Some(table) => Arc::clone(table),
            None => {
                return Err(Error::NotFound);
            }
        };

        // Views are read-only - reject TRUNCATE operations
        if self.is_view(table.table_id())? {
            return Err(Error::InvalidArgumentError(format!(
                "cannot modify view '{}'",
                display_name
            )));
        }

        self.truncate_all_rows(table.as_ref(), display_name, canonical_name, snapshot)
    }

    /// Truncate all rows from a table (similar to DELETE without WHERE).
    fn truncate_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        // Check if this table is referenced by foreign keys (including self-references)
        // TRUNCATE is not allowed on tables with incoming FK references unless CASCADE is specified
        let table_id = table.table_id();
        let referencing_fks = self.constraint_service.referencing_foreign_keys(table_id)?;

        if !referencing_fks.is_empty() {
            return Err(Error::ConstraintError(
                "Violates foreign key constraint".to_string(),
            ));
        }

        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if total_rows == 0 {
            return Ok(RuntimeStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        // Get all row IDs from the table
        let anchor_field = table.schema.first_field_id().ok_or_else(|| {
            Error::InvalidArgumentError("TRUNCATE requires a table with at least one column".into())
        })?;
        let filter_expr = translation::expression::full_table_scan_filter(anchor_field);
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;

        // TRUNCATE always enforces foreign key checks (cannot truncate if other tables reference this one)
        self.apply_delete(table, display_name, canonical_name, row_ids, snapshot, true)
    }
}
