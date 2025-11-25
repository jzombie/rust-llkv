//! General utility methods for RuntimeContext operations.
//!
//! This module contains shared helper functions used across multiple DML and DDL operations:
//! - Type coercion for INSERT/UPDATE value normalization
//! - Column scanning for INDEX creation and constraint validation
//! - Multi-column scanning for UNIQUE/PRIMARY KEY/FOREIGN KEY constraints
//! - Row collection for FK validation
//! - MVCC visibility filtering for transaction isolation
//! - Transaction state tracking
//! - Table cache management

use roaring::RoaringTreemap;
use arrow::array::{Array, UInt64Array};
use arrow::datatypes::{DataType, IntervalUnit};
use llkv_column_map::store::GatherNullPolicy;
use llkv_column_map::types::LogicalFieldId;
use llkv_compute::date::parse_date32_literal;
use llkv_compute::scalar::decimal::{
    align_decimal_to_scale, decimal_truthy, truncate_decimal_to_i64,
};
use llkv_executor::{ExecutorColumn, ExecutorTable, translation};
use llkv_plan::PlanValue;
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::{FieldId, RowId};
use llkv_transaction::{TransactionSnapshot, TxnId, filter_row_ids_for_snapshot};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

use crate::TXN_ID_AUTO_COMMIT;
use crate::TXN_ID_NONE;

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Coerce a PlanValue to match the expected column type.
    ///
    /// Used during INSERT and UPDATE operations to ensure type compatibility.
    pub(super) fn coerce_plan_value_for_column(
        &self,
        value: PlanValue,
        column: &ExecutorColumn,
    ) -> Result<PlanValue> {
        match value {
            PlanValue::Null => Ok(PlanValue::Null),
            PlanValue::Decimal(decimal) => match &column.data_type {
                DataType::Decimal128(precision, scale) => {
                    let aligned = align_decimal_to_scale(decimal, *precision, *scale).map_err(
                        |err| {
                            Error::InvalidArgumentError(format!(
                                "decimal literal {} incompatible with DECIMAL({}, {}) column '{}': {err}",
                                decimal, precision, scale, column.name
                            ))
                        },
                    )?;
                    Ok(PlanValue::Decimal(aligned))
                }
                DataType::Int64 => {
                    let coerced = truncate_decimal_to_i64(decimal).map_err(|err| {
                        Error::InvalidArgumentError(format!(
                            "decimal literal {} incompatible with INT column '{}': {err}",
                            decimal, column.name
                        ))
                    })?;
                    Ok(PlanValue::Integer(coerced))
                }
                DataType::Float64 => Ok(PlanValue::Float(decimal.to_f64())),
                DataType::Boolean => Ok(PlanValue::Integer(if decimal_truthy(decimal) {
                    1
                } else {
                    0
                })),
                DataType::Utf8 => Ok(PlanValue::String(decimal.to_string())),
                DataType::Date32 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign decimal literal to DATE column '{}'",
                    column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign decimal literal to STRUCT column '{}'",
                    column.name
                ))),
                other => Err(Error::InvalidArgumentError(format!(
                    "unsupported target type {:?} for DECIMAL literal in column '{}'",
                    other, column.name
                ))),
            },
            PlanValue::Integer(v) => match &column.data_type {
                DataType::Int64 => Ok(PlanValue::Integer(v)),
                DataType::Float64 => Ok(PlanValue::Float(v as f64)),
                DataType::Boolean => Ok(PlanValue::Integer(if v != 0 { 1 } else { 0 })),
                DataType::Utf8 => Ok(PlanValue::String(v.to_string())),
                DataType::Date32 => {
                    let casted = i32::try_from(v).map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "integer literal out of range for DATE column '{}'",
                            column.name
                        ))
                    })?;
                    Ok(PlanValue::Date32(casted))
                }
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign integer to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::Integer(v)),
            },
            PlanValue::Float(v) => match &column.data_type {
                DataType::Int64 => Ok(PlanValue::Integer(v as i64)),
                DataType::Float64 => Ok(PlanValue::Float(v)),
                DataType::Boolean => Ok(PlanValue::Integer(if v != 0.0 { 1 } else { 0 })),
                DataType::Utf8 => Ok(PlanValue::String(v.to_string())),
                DataType::Date32 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign floating-point value to DATE column '{}'",
                    column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign floating-point value to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::Float(v)),
            },
            PlanValue::String(s) => match &column.data_type {
                DataType::Boolean => {
                    let normalized = s.trim().to_ascii_lowercase();
                    match normalized.as_str() {
                        "true" | "t" | "1" => Ok(PlanValue::Integer(1)),
                        "false" | "f" | "0" => Ok(PlanValue::Integer(0)),
                        _ => Err(Error::InvalidArgumentError(format!(
                            "cannot assign string '{}' to BOOLEAN column '{}'",
                            s, column.name
                        ))),
                    }
                }
                DataType::Utf8 => Ok(PlanValue::String(s)),
                DataType::Date32 => {
                    let days = parse_date32_literal(&s)?;
                    Ok(PlanValue::Date32(days))
                }
                DataType::Int64 | DataType::Float64 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign string '{}' to numeric column '{}'",
                    s, column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign string to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::String(s)),
            },
            PlanValue::Struct(map) => match &column.data_type {
                DataType::Struct(_) => Ok(PlanValue::Struct(map)),
                _ => Err(Error::InvalidArgumentError(format!(
                    "cannot assign struct value to column '{}'",
                    column.name
                ))),
            },
            PlanValue::Interval(interval) => match &column.data_type {
                DataType::Interval(IntervalUnit::MonthDayNano) => Ok(PlanValue::Interval(interval)),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign INTERVAL literal to STRUCT column '{}'",
                    column.name
                ))),
                other => Err(Error::InvalidArgumentError(format!(
                    "unsupported target type {:?} for INTERVAL literal in column '{}'",
                    other, column.name
                ))),
            },
            PlanValue::Date32(days) => match &column.data_type {
                DataType::Date32 => Ok(PlanValue::Date32(days)),
                DataType::Int64 => Ok(PlanValue::Integer(i64::from(days))),
                DataType::Float64 => Ok(PlanValue::Float(days as f64)),
                DataType::Utf8 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign DATE literal to TEXT column '{}'",
                    column.name
                ))),
                DataType::Boolean => Err(Error::InvalidArgumentError(format!(
                    "cannot assign DATE literal to BOOLEAN column '{}'",
                    column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign DATE literal to STRUCT column '{}'",
                    column.name
                ))),
                other => Err(Error::InvalidArgumentError(format!(
                    "unsupported target type {:?} for DATE literal in column '{}'",
                    other, column.name
                ))),
            },
        }
    }

    /// Scan a single column and materialize values into memory.
    ///
    /// Used during CREATE INDEX operations and constraint validation.
    ///
    /// NOTE: Current implementation buffers the entire result set; convert to a
    /// streaming iterator once executor-side consumers support incremental consumption.
    pub(super) fn scan_column_values(
        &self,
        table: &ExecutorTable<P>,
        field_id: FieldId,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<PlanValue>> {
        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        // Create a filter that matches all rows (unbounded range)
        let match_all_filter = Filter {
            field_id,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        // Get all matching row_ids first
        let row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        // Apply MVCC filtering manually using filter_row_ids_for_snapshot
        let row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            row_ids.iter().collect(),
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Gather the column values for visible rows
        let logical_field_id = LogicalFieldId::for_user(table_id, field_id);
        let row_count = row_ids.len();
        let mut stream = match table.table.stream_columns(
            vec![logical_field_id],
            row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        // TODO: Don't buffer all values; make this streamable
        // NOTE: Values are accumulated eagerly; revisit when `llkv-plan` supports
        // incremental parameter binding.
        let mut values = Vec::with_capacity(row_count);
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }
            let array = batch.column(0);
            for row_idx in 0..batch.num_rows() {
                if let Ok(value) = llkv_plan::plan_value_from_array(array, row_idx) {
                    values.push(value);
                }
            }
        }

        Ok(values)
    }

    /// Scan a set of columns and materialize rows into memory.
    ///
    /// Used during constraint validation (multi-column UNIQUE, PRIMARY KEY)
    /// and CREATE INDEX operations.
    ///
    /// NOTE: Similar to [`Self::scan_column_values`], this buffers eagerly pending
    /// enhancements to the executor pipeline.
    pub(super) fn scan_multi_column_values(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<Vec<PlanValue>>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        let match_all_filter = Filter {
            field_id: field_ids[0],
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        let row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            row_ids.iter().collect(),
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let logical_field_ids: Vec<_> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let total_rows = row_ids.len();
        let mut stream = match table.table.stream_columns(
            logical_field_ids,
            row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); total_rows];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }

            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < rows.len(),
                        "stream chunk produced out-of-bounds row index"
                    );
                    if let Some(row) = rows.get_mut(target_index) {
                        match llkv_plan::plan_value_from_array(array, local_idx) {
                            Ok(value) => row.push(value),
                            Err(_) => row.push(PlanValue::Null),
                        }
                    }
                }
            }
        }

        Ok(rows)
    }

    /// Scan multi-column values with FK-aware visibility semantics.
    ///
    /// This is similar to [`Self::scan_multi_column_values`], but uses FK-specific
    /// MVCC visibility rules where rows deleted by the current transaction are treated
    /// as still visible (matching SQL standard FK constraint checking behavior).
    pub(super) fn scan_multi_column_values_for_fk_check(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<Vec<PlanValue>>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        let match_all_filter = Filter {
            field_id: field_ids[0],
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        let row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        // Use FK-specific filtering that treats deleted rows as still visible
        let row_ids = llkv_transaction::filter_row_ids_for_fk_check(
            table.table.as_ref(),
            row_ids.iter().collect(),
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let logical_field_ids: Vec<_> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let total_rows = row_ids.len();
        let mut stream = match table.table.stream_columns(
            logical_field_ids,
            row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); total_rows];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }

            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < rows.len(),
                        "stream chunk produced out-of-bounds row index"
                    );
                    if let Some(row) = rows.get_mut(target_index) {
                        match llkv_plan::plan_value_from_array(array, local_idx) {
                            Ok(value) => row.push(value),
                            Err(_) => row.push(PlanValue::Null),
                        }
                    }
                }
            }
        }

        Ok(rows)
    }

    /// Collect row values for specific row IDs and field IDs.
    ///
    /// Used during foreign key validation to gather referenced/referencing values.
    pub(super) fn collect_row_values_for_ids(
        &self,
        table: &ExecutorTable<P>,
        row_ids: &[RowId],
        field_ids: &[FieldId],
    ) -> Result<Vec<Vec<PlanValue>>> {
        if row_ids.is_empty() || field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let logical_field_ids: Vec<LogicalFieldId> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let mut stream = match table.table.stream_columns(
            logical_field_ids.clone(),
            row_ids.to_vec(),
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); row_ids.len()];
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

        Ok(rows)
    }

    /// Filter row IDs to only include those visible in the given snapshot.
    ///
    /// Used across all DML operations to ensure MVCC transaction isolation.
    pub(super) fn filter_visible_row_ids(
        &self,
        table: &ExecutorTable<P>,
        row_ids: RoaringTreemap,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<RowId>> {
        let row_ids_vec: Vec<RowId> = row_ids.iter().collect();
        filter_row_ids_for_snapshot(table.table.as_ref(), row_ids_vec, &self.txn_manager, snapshot)
    }

    /// Record that a transaction has inserted rows into a table.
    ///
    /// Used for deferred PRIMARY KEY validation at commit time.
    pub(super) fn record_table_with_new_rows(&self, txn_id: TxnId, canonical_name: String) {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return;
        }

        let mut guard = self.txn_tables_with_new_rows.write().unwrap();
        guard.entry(txn_id).or_default().insert(canonical_name);
    }

    /// Collect all rows created by a specific transaction.
    ///
    /// Used for PRIMARY KEY validation at commit time.
    pub(super) fn collect_rows_created_by_txn(
        &self,
        table: &ExecutorTable<P>,
        txn_id: TxnId,
    ) -> Result<Vec<Vec<PlanValue>>> {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return Ok(Vec::new());
        }

        if table.schema.columns.is_empty() {
            return Ok(Vec::new());
        }

        let Some(first_field_id) = table.schema.first_field_id() else {
            return Ok(Vec::new());
        };
        let filter_expr = translation::expression::full_table_scan_filter(first_field_id);

        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let mut logical_fields: Vec<LogicalFieldId> =
            Vec::with_capacity(table.schema.columns.len() + 2);
        logical_fields.push(LogicalFieldId::for_mvcc_created_by(table_id));
        logical_fields.push(LogicalFieldId::for_mvcc_deleted_by(table_id));
        for column in &table.schema.columns {
            logical_fields.push(LogicalFieldId::for_user(table_id, column.field_id));
        }

        let logical_fields: Arc<[LogicalFieldId]> = logical_fields.into();
        let mut stream = table.table.stream_columns(
            Arc::clone(&logical_fields),
            row_ids.iter().collect(),
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut rows = Vec::new();
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() < table.schema.columns.len() + 2 {
                continue;
            }

            let created_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("missing created_by column in MVCC data".into()))?;
            let deleted_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("missing deleted_by column in MVCC data".into()))?;

            for row_idx in 0..batch.num_rows() {
                let created_by = if created_col.is_null(row_idx) {
                    TXN_ID_AUTO_COMMIT
                } else {
                    created_col.value(row_idx)
                };
                if created_by != txn_id {
                    continue;
                }

                let deleted_by = if deleted_col.is_null(row_idx) {
                    TXN_ID_NONE
                } else {
                    deleted_col.value(row_idx)
                };
                if deleted_by != TXN_ID_NONE {
                    continue;
                }

                let mut row_values = Vec::with_capacity(table.schema.columns.len());
                for col_idx in 0..table.schema.columns.len() {
                    let array = batch.column(col_idx + 2);
                    let value = llkv_plan::plan_value_from_array(array, row_idx)?;
                    row_values.push(value);
                }
                rows.push(row_values);
            }
        }

        Ok(rows)
    }

    /// Validate primary key constraints for rows inserted by a transaction before commit.
    pub(crate) fn validate_primary_keys_for_commit(&self, txn_id: TxnId) -> Result<()> {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return Ok(());
        }

        let pending_tables = {
            let guard = self.txn_tables_with_new_rows.read().unwrap();
            guard.get(&txn_id).cloned()
        };

        let Some(tables) = pending_tables else {
            return Ok(());
        };

        for canonical_name in tables {
            let table = match self.lookup_table(&canonical_name) {
                Ok(table) => table,
                Err(Error::NotFound) => continue,
                Err(err) => return Err(err),
            };

            let constraint_ctx = self.build_table_constraint_context(table.as_ref())?;
            let Some(primary_key) = constraint_ctx.primary_key.as_ref() else {
                continue;
            };

            let new_rows = self.collect_rows_created_by_txn(table.as_ref(), txn_id)?;
            if new_rows.is_empty() {
                continue;
            }

            let column_order: Vec<usize> = (0..table.schema.columns.len()).collect();
            let table_for_fetch = Arc::clone(&table);
            let snapshot = self.default_snapshot();

            self.constraint_service.validate_primary_key_rows(
                &constraint_ctx.schema_field_ids,
                primary_key,
                &column_order,
                &new_rows,
                |field_ids| {
                    self.scan_multi_column_values(table_for_fetch.as_ref(), field_ids, snapshot)
                },
            )?;
        }

        Ok(())
    }

    /// Clear any per-transaction bookkeeping maintained by the runtime context.
    pub(crate) fn clear_transaction_state(&self, txn_id: TxnId) {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return;
        }

        let mut guard = self.txn_tables_with_new_rows.write().unwrap();
        guard.remove(&txn_id);
    }

    /// Remove a table from the executor cache.
    ///
    /// Forces the table to be reloaded from metadata on next access.
    /// Used after schema changes (ALTER TABLE, DROP COLUMN, etc.).
    pub(super) fn remove_table_entry(&self, canonical_name: &str) {
        let mut tables = self.tables.write().unwrap();
        if tables.remove(canonical_name).is_some() {
            tracing::trace!(
                "remove_table_entry: removed table '{}' from context cache",
                canonical_name
            );
        }
    }

    /// Check if a table is marked as dropped.
    pub fn is_table_marked_dropped(&self, canonical_name: &str) -> bool {
        self.dropped_tables.read().unwrap().contains(canonical_name)
    }
}
