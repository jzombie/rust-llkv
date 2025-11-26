//! Constraint validation helpers for RuntimeContext.
//!
//! This module contains constraint-related validation logic:
//! - Delete conflict detection (MVCC)
//! - Visible child rows collection (for foreign key validation)
//! - Foreign key validation for INSERT, UPDATE, DELETE
//! - Constraint context building

use arrow::array::{Array, UInt64Array};
use croaring::Treemap;
use llkv_column_map::store::GatherNullPolicy;
use llkv_types::LogicalFieldId;
use llkv_executor::{ExecutorTable, translation};
use llkv_plan::PlanValue;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::{
    ConstraintColumnInfo, FieldId, InsertColumnConstraint, InsertMultiColumnUnique,
    InsertUniqueColumn, RowId,
};
use llkv_transaction::{TransactionSnapshot, TxnId, filter_row_ids_for_snapshot};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

use crate::TXN_ID_NONE;

use super::{RuntimeContext, TableConstraintContext};

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
        row_ids: &Treemap,
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
            row_ids,
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
            &visible_row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows =
            vec![Vec::with_capacity(field_ids.len()); visible_row_ids.cardinality() as usize];
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

        Ok(visible_row_ids.iter().zip(rows).collect())
    }

    /// Validate foreign key constraints for INSERT operations.
    ///
    /// This function is `pub(super)` because it's called by insert operations
    /// in insert.rs and update operations in update.rs (when inserting new row values).
    ///
    /// For foreign key validation, we use a modified snapshot that includes:
    /// - Rows inserted by the current transaction (so FK checks see new parent rows)
    /// - Rows deleted by the current transaction are treated as STILL VISIBLE for FK checks
    ///   (this matches SQL standard where uncommitted deletes don't affect FK validation)
    ///
    /// We achieve this by using a snapshot with a special txn_id marker that tells the
    /// MVCC layer to ignore deletes from the current transaction.
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
                self.scan_multi_column_values_for_fk_check(
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
        row_ids: &Treemap,
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
        row_ids: &Treemap,
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

    /// Build a constraint context for validation operations.
    ///
    /// This consolidates all constraint metadata (NOT NULL, CHECK, UNIQUE,
    /// multi-column UNIQUE, PRIMARY KEY) into a single context object
    /// used during INSERT, UPDATE, and table creation validation.
    pub(super) fn build_table_constraint_context(
        &self,
        table: &ExecutorTable<P>,
    ) -> Result<TableConstraintContext> {
        let schema_field_ids: Vec<FieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| column.field_id)
            .collect();

        let column_constraints: Vec<InsertColumnConstraint> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, column)| InsertColumnConstraint {
                schema_index: idx,
                column: ConstraintColumnInfo {
                    name: column.name.clone(),
                    field_id: column.field_id,
                    data_type: column.data_type.clone(),
                    nullable: column.nullable,
                    check_expr: column.check_expr.clone(),
                },
            })
            .collect();

        let unique_columns: Vec<InsertUniqueColumn> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.unique && !column.primary_key)
            .map(|(idx, column)| InsertUniqueColumn {
                schema_index: idx,
                field_id: column.field_id,
                name: column.name.clone(),
            })
            .collect();

        let mut multi_column_uniques: Vec<InsertMultiColumnUnique> = Vec::new();
        for constraint in table.multi_column_uniques() {
            if constraint.column_indices.is_empty() {
                continue;
            }

            let mut schema_indices = Vec::with_capacity(constraint.column_indices.len());
            let mut field_ids = Vec::with_capacity(constraint.column_indices.len());
            let mut column_names = Vec::with_capacity(constraint.column_indices.len());
            for &col_idx in &constraint.column_indices {
                let column = table.schema.columns.get(col_idx).ok_or_else(|| {
                    Error::Internal(format!(
                        "multi-column UNIQUE constraint references invalid column index {}",
                        col_idx
                    ))
                })?;
                schema_indices.push(col_idx);
                field_ids.push(column.field_id);
                column_names.push(column.name.clone());
            }

            multi_column_uniques.push(InsertMultiColumnUnique {
                schema_indices,
                field_ids,
                column_names,
            });
        }

        let primary_indices: Vec<usize> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.primary_key)
            .map(|(idx, _)| idx)
            .collect();

        let primary_key = if primary_indices.is_empty() {
            None
        } else {
            let mut field_ids = Vec::with_capacity(primary_indices.len());
            let mut column_names = Vec::with_capacity(primary_indices.len());
            for &idx in &primary_indices {
                let column = table.schema.columns.get(idx).ok_or_else(|| {
                    Error::Internal(format!(
                        "primary key references invalid column index {}",
                        idx
                    ))
                })?;
                field_ids.push(column.field_id);
                column_names.push(column.name.clone());
            }

            Some(InsertMultiColumnUnique {
                schema_indices: primary_indices.clone(),
                field_ids,
                column_names,
            })
        };

        Ok(TableConstraintContext {
            schema_field_ids,
            column_constraints,
            unique_columns,
            multi_column_uniques,
            primary_key,
        })
    }
}
