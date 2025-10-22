//! INSERT operation implementation for RuntimeContext.
//!
//! This module contains all logic for inserting rows into tables, including:
//! - Row insertion with constraint validation
//! - Batch insertion
//! - Foreign key validation for inserts

use crate::{RuntimeStatementResult, TXN_ID_NONE};
use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_executor::{build_array_for_column, normalize_insert_value_for_column, resolve_insert_columns, ExecutorTable};
use llkv_plan::PlanValue;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_transaction::{mvcc, TransactionSnapshot};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Insert rows into a table with full constraint and foreign key validation.
    pub(super) fn insert_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        mut rows: Vec<Vec<PlanValue>>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "INSERT requires at least one row".into(),
            ));
        }

        let column_order = resolve_insert_columns(&columns, table.schema.as_ref())?;
        let expected_len = column_order.len();
        for row in &rows {
            if row.len() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} values in INSERT row, found {}",
                    expected_len,
                    row.len()
                )));
            }
        }

        for row in rows.iter_mut() {
            for (position, value) in row.iter_mut().enumerate() {
                let schema_index = column_order
                    .get(position)
                    .copied()
                    .ok_or_else(|| Error::Internal("invalid INSERT column index mapping".into()))?;
                let column = table.schema.columns.get(schema_index).ok_or_else(|| {
                    Error::Internal(format!(
                        "INSERT column index {} out of bounds for table '{}'",
                        schema_index, display_name
                    ))
                })?;
                let normalized = normalize_insert_value_for_column(column, value.clone())?;
                *value = normalized;
            }
        }

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] Validating constraints for {} row(s) before insert",
                rows.len()
            );
            for (i, row) in rows.iter().enumerate() {
                tracing::trace!("[KEYS]   row[{}]: {:?}", i, row);
            }
        }

        self.constraint_service.validate_insert_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &constraint_ctx.unique_columns,
            &constraint_ctx.multi_column_uniques,
            primary_key_spec,
            &column_order,
            &rows,
            |field_id| self.scan_column_values(table, field_id, snapshot),
            |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
        )?;

        self.check_foreign_keys_on_insert(table, &display_name, &rows, &column_order, snapshot)?;

        let row_count = rows.len();
        let mut column_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_count); table.schema.columns.len()];
        for row in rows {
            for (idx, value) in row.into_iter().enumerate() {
                let dest_index = column_order[idx];
                column_values[dest_index].push(value);
            }
        }

        let start_row = table.next_row_id.load(Ordering::SeqCst);

        // Build MVCC columns using helper
        let (row_id_array, created_by_array, deleted_by_array) =
            mvcc::build_insert_mvcc_columns(row_count, start_row, snapshot.txn_id, TXN_ID_NONE);

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_values.len() + 3);
        arrays.push(row_id_array);
        arrays.push(created_by_array);
        arrays.push(deleted_by_array);

        let mut fields: Vec<Field> = Vec::with_capacity(column_values.len() + 3);
        fields.extend(mvcc::build_mvcc_fields());

        for (column, values) in table.schema.columns.iter().zip(column_values.into_iter()) {
            let array = build_array_for_column(&column.data_type, &values)?;
            let field = mvcc::build_field_with_metadata(
                &column.name,
                column.data_type.clone(),
                column.nullable,
                column.field_id,
            );
            arrays.push(array);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        tracing::trace!(
            table_name = %display_name,
            store_ptr = ?std::ptr::addr_of!(*table.table.store()),
            "About to call table.append"
        );
        table.table.append(&batch)?;
        table
            .next_row_id
            .store(start_row + row_count as u64, Ordering::SeqCst);
        table
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        self.record_table_with_new_rows(snapshot.txn_id, canonical_name);

        Ok(RuntimeStatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }

    /// Insert multiple batches of rows into a table.
    pub(super) fn insert_batches(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        batches: Vec<RecordBatch>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if batches.is_empty() {
            return Ok(RuntimeStatementResult::Insert {
                table_name: display_name,
                rows_inserted: 0,
            });
        }

        let expected_len = if columns.is_empty() {
            table.schema.columns.len()
        } else {
            columns.len()
        };
        let mut total_rows_inserted = 0usize;

        for batch in batches {
            if batch.num_columns() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} columns in INSERT batch, found {}",
                    expected_len,
                    batch.num_columns()
                )));
            }
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(row_count);
            for row_idx in 0..row_count {
                let mut row: Vec<PlanValue> = Vec::with_capacity(expected_len);
                for col_idx in 0..expected_len {
                    let array = batch.column(col_idx);
                    row.push(llkv_plan::plan_value_from_array(array, row_idx)?);
                }
                rows.push(row);
            }

            match self.insert_rows(
                table,
                display_name.clone(),
                canonical_name.clone(),
                rows,
                columns.clone(),
                snapshot,
            )? {
                RuntimeStatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(RuntimeStatementResult::Insert {
            table_name: display_name,
            rows_inserted: total_rows_inserted,
        })
    }
}
