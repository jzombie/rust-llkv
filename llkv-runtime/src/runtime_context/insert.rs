//! INSERT operation implementation for RuntimeContext.
//!
//! This module contains all logic for inserting rows into tables, including:
//! - Row insertion with constraint validation
//! - Batch insertion
//! - Foreign key validation for inserts

use crate::{RuntimeStatementResult, TXN_ID_NONE, canonical_table_name};
use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use croaring::Treemap;
use llkv_column_map::store::GatherNullPolicy;
use llkv_executor::{
    ExecutorTable, build_array_for_column, normalize_insert_value_for_column,
    resolve_insert_columns,
};
use llkv_plan::{InsertConflictAction, InsertPlan, InsertSource, PlanValue};
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::ConstraintEnforcementMode;
use llkv_transaction::{TransactionSnapshot, filter_row_ids_for_snapshot, mvcc};
use llkv_types::{LogicalFieldId, RowId};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::RuntimeContext;

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Insert operation - internal storage API. Use `RuntimeSession::execute_insert_plan()` instead.
    pub(crate) fn insert(
        &self,
        plan: InsertPlan,
        snapshot: TransactionSnapshot,
        constraint_mode: ConstraintEnforcementMode,
    ) -> Result<RuntimeStatementResult<P>> {
        let InsertPlan {
            table: table_name,
            columns,
            source,
            on_conflict,
        } = plan;
        let (display_name, canonical_name) = canonical_table_name(&table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        // Views are read-only - reject INSERT operations
        if self.is_view(table.table_id())? {
            return Err(Error::InvalidArgumentError(format!(
                "cannot modify view '{}'",
                display_name
            )));
        }

        // Targeted debug for 'keys' table only
        if display_name == "keys" {
            tracing::trace!(
                "\n[KEYS] INSERT starting - table_id={}, context_pager={:p}",
                table.table_id(),
                &*self.pager
            );
            tracing::trace!(
                "[KEYS] Table has {} columns, primary_key columns: {:?}",
                table.schema.columns.len(),
                table
                    .schema
                    .columns
                    .iter()
                    .filter(|c| c.primary_key)
                    .map(|c| &c.name)
                    .collect::<Vec<_>>()
            );
        }

        let ctx = InsertExecContext::new(
            table.as_ref(),
            display_name.clone(),
            canonical_name.clone(),
            columns,
            snapshot,
            constraint_mode,
        );

        let result = match source {
            InsertSource::Rows(rows) =>
                self.insert_rows_with_conflict(ctx, rows, on_conflict),
            InsertSource::Batches(batches) => self.insert_batches(ctx, batches),
            InsertSource::Select { .. } => Err(Error::Internal(
                "InsertSource::Select should be materialized before reaching RuntimeContext::insert"
                    .into(),
            )),
        };

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] INSERT completed: {:?}",
                result
                    .as_ref()
                    .map(|_| "OK")
                    .map_err(|e| format!("{:?}", e))
            );
        }

        if matches!(result, Err(Error::NotFound)) {
            panic!(
                "BUG: insert yielded Error::NotFound for table '{}'. \
                 This should never happen: insert should never return NotFound after successful table lookup. \
                 This indicates a logic error in the runtime.",
                display_name
            );
        }

        result
    }

    /// Insert rows with conflict resolution handling.
    pub(super) fn insert_rows_with_conflict(
        &self,
        ctx: InsertExecContext<'_, P>,
        rows: Vec<Vec<PlanValue>>,
        on_conflict: InsertConflictAction,
    ) -> Result<RuntimeStatementResult<P>> {
        match on_conflict {
            InsertConflictAction::None
            | InsertConflictAction::Abort
            | InsertConflictAction::Fail
            | InsertConflictAction::Rollback => {
                // Standard INSERT behavior - fail on constraint violation
                self.insert_rows(ctx.clone(), rows)
            }
            InsertConflictAction::Replace => self.insert_rows_or_replace(ctx.clone(), rows),
            InsertConflictAction::Ignore => self.insert_rows_or_ignore(ctx, rows),
        }
    }

    /// Insert rows into a table with full constraint and foreign key validation.
    pub(super) fn insert_rows(
        &self,
        ctx: InsertExecContext<'_, P>,
        mut rows: Vec<Vec<PlanValue>>,
    ) -> Result<RuntimeStatementResult<P>> {
        let InsertExecContext {
            table,
            display_name,
            canonical_name,
            columns,
            snapshot,
            constraint_mode,
        } = ctx;
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
                let val = std::mem::replace(value, PlanValue::Null);
                let normalized = normalize_insert_value_for_column(column, val)?;
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

        if constraint_mode.is_immediate() {
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

            self.check_foreign_keys_on_insert(
                table,
                &display_name,
                &rows,
                &column_order,
                snapshot,
            )?;
        }

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

    /// INSERT OR REPLACE: delete conflicting rows and insert new ones.
    ///
    /// SQLite's INSERT OR REPLACE semantics: If a UNIQUE or PRIMARY KEY constraint
    /// would be violated, delete the existing row(s) that conflict, then insert the new row.
    fn insert_rows_or_replace(
        &self,
        ctx: InsertExecContext<'_, P>,
        rows: Vec<Vec<PlanValue>>,
    ) -> Result<RuntimeStatementResult<P>> {
        let table = ctx.table;
        // Get the constraint context to know which columns have uniqueness constraints
        let constraint_ctx = self.build_table_constraint_context(table)?;

        // If columns is empty, it means "all columns in order"
        let columns_for_search = if ctx.columns.is_empty() {
            table
                .schema
                .columns
                .iter()
                .map(|c| c.name.clone())
                .collect()
        } else {
            ctx.columns.clone()
        };

        // Proactively find conflicting rows by scanning for matches on unique/primary key columns
        let row_ids = self.find_conflicting_rows(
            table,
            &rows,
            &columns_for_search,
            &constraint_ctx,
            ctx.snapshot,
        )?;

        // Delete conflicting rows BEFORE inserting (without foreign key enforcement since we're replacing)
        if !row_ids.is_empty() {
            self.apply_delete(
                table,
                ctx.display_name.clone(),
                ctx.canonical_name.clone(),
                row_ids.into_iter().collect(),
                ctx.snapshot,
                false, // Don't enforce foreign keys for REPLACE
            )?;
        }

        // Now insert the new rows
        self.insert_rows(ctx, rows)
    }

    /// INSERT OR IGNORE: insert rows, silently skipping any that violate constraints.
    fn insert_rows_or_ignore(
        &self,
        ctx: InsertExecContext<'_, P>,
        rows: Vec<Vec<PlanValue>>,
    ) -> Result<RuntimeStatementResult<P>> {
        // Try inserting rows one at a time, counting successes
        let mut rows_inserted = 0;

        for row in rows {
            match self.insert_rows(ctx.clone(), vec![row]) {
                Ok(_) => {
                    rows_inserted += 1;
                }
                Err(Error::ConstraintError(_)) => {
                    // Ignore constraint violations
                    continue;
                }
                Err(e) => {
                    // Other errors should still be raised
                    return Err(e);
                }
            }
        }

        Ok(RuntimeStatementResult::Insert {
            table_name: ctx.display_name.clone(),
            rows_inserted,
        })
    }

    /// Multiple batches of rows into a table.
    pub(super) fn insert_batches(
        &self,
        ctx: InsertExecContext<'_, P>,
        batches: Vec<RecordBatch>,
    ) -> Result<RuntimeStatementResult<P>> {
        let table = ctx.table;
        if batches.is_empty() {
            return Ok(RuntimeStatementResult::Insert {
                table_name: ctx.display_name.clone(),
                rows_inserted: 0,
            });
        }

        let expected_len = if ctx.columns.is_empty() {
            table.schema.columns.len()
        } else {
            ctx.columns.len()
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

            match self.insert_rows(ctx.clone(), rows)? {
                RuntimeStatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(RuntimeStatementResult::Insert {
            table_name: ctx.display_name.clone(),
            rows_inserted: total_rows_inserted,
        })
    }

    /// Find rows that conflict with the rows being inserted based on UNIQUE/PRIMARY KEY constraints.
    ///
    /// This scans the table to find existing rows that have matching values for UNIQUE columns,
    /// multi-column UNIQUE constraints, or PRIMARY KEY columns.
    fn find_conflicting_rows(
        &self,
        table: &ExecutorTable<P>,
        new_rows: &[Vec<PlanValue>],
        columns: &[String],
        constraint_ctx: &super::TableConstraintContext,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<RowId>> {
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        // Build a mapping from column name to position in the new_rows vectors
        let mut column_positions = std::collections::HashMap::new();
        for (idx, col_name) in columns.iter().enumerate() {
            column_positions.insert(col_name.as_str(), idx);
        }

        let mut conflicting_row_ids = Vec::new();
        let table_id = table.table_id();

        // Helper: get all visible row IDs
        let anchor_field = table
            .schema
            .first_field_id()
            .ok_or_else(|| Error::Internal("table must have at least one column".into()))?;
        let match_all_filter = Filter {
            field_id: anchor_field,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        let row_ids = match table.filter_row_ids(&filter_expr) {
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

        // Check single-column unique constraints
        for unique_col in &constraint_ctx.unique_columns {
            let col_name = &unique_col.name;
            if let Some(&col_pos) = column_positions.get(col_name.as_str()) {
                // Collect non-NULL values being inserted for this unique column
                let mut new_values = Vec::new();
                for row in new_rows {
                    if let Some(value) = row.get(col_pos)
                        && !matches!(value, PlanValue::Null)
                        && !new_values.contains(value)
                    {
                        new_values.push(value.clone());
                    }
                }

                if new_values.is_empty() {
                    continue;
                }

                // Scan existing rows for this column
                let logical_field_id = LogicalFieldId::for_user(table_id, unique_col.field_id);
                let mut stream = table.stream_columns(
                    vec![logical_field_id],
                    &row_ids,
                    GatherNullPolicy::IncludeNulls,
                )?;

                while let Some(chunk) = stream.next_batch()? {
                    let batch = chunk.batch();
                    if batch.num_columns() == 0 {
                        continue;
                    }
                    let array = batch.column(0);
                    let _base_idx = chunk.row_offset();

                    for local_idx in 0..batch.num_rows() {
                        if let Ok(existing_value) =
                            llkv_plan::plan_value_from_array(array, local_idx)
                            && new_values.contains(&existing_value)
                        {
                            let rid = chunk.row_ids()[local_idx];
                            if !conflicting_row_ids.contains(&rid) {
                                conflicting_row_ids.push(rid);
                            }
                        }
                    }
                }
            }
        }

        // Check primary key (single-column or multi-column)
        if let Some(pk) = &constraint_ctx.primary_key {
            self.find_multi_column_conflicts(
                table,
                &row_ids,
                new_rows,
                columns,
                pk,
                &mut conflicting_row_ids,
                snapshot,
            )?;
        }

        // Check multi-column unique constraints
        for unique_constraint in &constraint_ctx.multi_column_uniques {
            self.find_multi_column_conflicts(
                table,
                &row_ids,
                new_rows,
                columns,
                unique_constraint,
                &mut conflicting_row_ids,
                snapshot,
            )?;
        }

        Ok(conflicting_row_ids)
    }

    /// Helper to find conflicts for multi-column constraints (PRIMARY KEY or UNIQUE).
    #[allow(clippy::too_many_arguments)]
    fn find_multi_column_conflicts(
        &self,
        table: &ExecutorTable<P>,
        row_ids: &Treemap,
        new_rows: &[Vec<PlanValue>],
        columns: &[String],
        constraint: &llkv_table::InsertMultiColumnUnique,
        conflicting_row_ids: &mut Vec<RowId>,
        _snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if constraint.field_ids.is_empty() {
            return Ok(());
        }

        // Build a mapping from column name to position in new_rows
        let mut column_positions = std::collections::HashMap::new();
        for (idx, col_name) in columns.iter().enumerate() {
            column_positions.insert(col_name.as_str(), idx);
        }

        // Map constraint columns to positions in new_rows
        let mut constraint_col_positions = Vec::new();
        for &schema_idx in &constraint.schema_indices {
            if schema_idx < table.schema.columns.len() {
                let col_name = &table.schema.columns[schema_idx].name;
                if let Some(&pos) = column_positions.get(col_name.as_str()) {
                    constraint_col_positions.push(Some(pos));
                } else {
                    constraint_col_positions.push(None);
                }
            } else {
                constraint_col_positions.push(None);
            }
        }

        // Collect multi-column values from new rows
        let mut new_values = Vec::new();
        for row in new_rows {
            let mut constraint_value = Vec::new();
            let mut has_all_values = true;
            for pos_opt in &constraint_col_positions {
                if let Some(pos) = pos_opt {
                    if let Some(value) = row.get(*pos) {
                        constraint_value.push(value.clone());
                    } else {
                        has_all_values = false;
                        break;
                    }
                } else {
                    has_all_values = false;
                    break;
                }
            }
            if has_all_values && !new_values.contains(&constraint_value) {
                new_values.push(constraint_value);
            }
        }

        if new_values.is_empty() {
            return Ok(());
        }

        // Scan existing rows for these columns
        let table_id = table.table_id();
        let logical_field_ids: Vec<_> = constraint
            .field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let mut stream =
            table.stream_columns(logical_field_ids, row_ids, GatherNullPolicy::IncludeNulls)?;

        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }

            let num_rows = batch.num_rows();

            for local_idx in 0..num_rows {
                let mut existing_value = Vec::new();
                let mut has_all_values = true;

                for col_idx in 0..batch.num_columns() {
                    let array = batch.column(col_idx);
                    match llkv_plan::plan_value_from_array(array, local_idx) {
                        Ok(value) => existing_value.push(value),
                        Err(_) => {
                            has_all_values = false;
                            break;
                        }
                    }
                }

                if has_all_values && new_values.contains(&existing_value) {
                    let rid = chunk.row_ids()[local_idx];
                    if !conflicting_row_ids.contains(&rid) {
                        conflicting_row_ids.push(rid);
                    }
                }
            }
        }

        Ok(())
    }
}

pub(super) struct InsertExecContext<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: &'a ExecutorTable<P>,
    display_name: String,
    canonical_name: String,
    columns: Vec<String>,
    snapshot: TransactionSnapshot,
    constraint_mode: ConstraintEnforcementMode,
}

impl<'a, P> InsertExecContext<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(super) fn new(
        table: &'a ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
        constraint_mode: ConstraintEnforcementMode,
    ) -> Self {
        Self {
            table,
            display_name,
            canonical_name,
            columns,
            snapshot,
            constraint_mode,
        }
    }
}

impl<'a, P> Clone for InsertExecContext<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            table: self.table,
            display_name: self.display_name.clone(),
            canonical_name: self.canonical_name.clone(),
            columns: self.columns.clone(),
            snapshot: self.snapshot,
            constraint_mode: self.constraint_mode,
        }
    }
}
