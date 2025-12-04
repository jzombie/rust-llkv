//! UPDATE operation implementation for RuntimeContext.
//!
//! This module contains all logic for updating rows in tables, including:
//! - Filtered and full table updates
//! - Expression-based assignments
//! - Constraint validation for updates
//! - Foreign key validation for updates

use crate::{RuntimeStatementResult, canonical_table_name};
use arrow::array::ArrayRef;
use arrow::record_batch::RecordBatch;
use croaring::Treemap;
use llkv_column_map::store::GatherNullPolicy;
use llkv_executor::{
    ExecutorColumn, ExecutorTable, build_array_for_column, resolve_insert_columns, translation,
};
use llkv_expr::{Expr as LlkvExpr, ScalarExpr};
use llkv_plan::{AssignmentValue, ColumnAssignment, PlanValue, UpdatePlan};
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::table::ScanProjection;
use llkv_table::table::ScanStreamOptions;
use llkv_table::{
    ConstraintEnforcementMode, FieldId, RowStream, UniqueKey, build_composite_unique_key,
};
use llkv_transaction::{MvccRowIdFilter, TransactionSnapshot, filter_row_ids_for_snapshot, mvcc};
use llkv_types::LogicalFieldId;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{
    PreparedAssignmentValue, RuntimeContext, TableConstraintContext, insert::InsertExecContext,
};

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Update operation - internal storage API. Use `RuntimeSession::execute_update_plan()` instead.
    pub(crate) fn update(
        &self,
        plan: UpdatePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let UpdatePlan {
            table,
            assignments,
            filter,
        } = plan;
        let (display_name, canonical_name) = canonical_table_name(&table)?;
        let table = self.lookup_table(&canonical_name)?;

        // Views are read-only - reject UPDATE operations
        if self.is_view(table.table_id())? {
            return Err(Error::InvalidArgumentError(format!(
                "cannot modify view '{}'",
                display_name
            )));
        }
        if let Some(filter) = filter {
            self.update_filtered_rows(
                table.as_ref(),
                display_name,
                canonical_name,
                assignments,
                filter,
                snapshot,
            )
        } else {
            self.update_all_rows(
                table.as_ref(),
                display_name,
                canonical_name,
                assignments,
                snapshot,
            )
        }
    }

    /// Update rows in a table that match a filter expression.
    pub(super) fn update_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        assignments: Vec<ColumnAssignment>,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let schema = table.schema.as_ref();
        let filter_expr = translation::expression::translate_predicate(filter, schema, |name| {
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{}'",
                name
            ))
        })?;

        // Use a map to track column assignments. If a column appears multiple times,
        // the last assignment wins (SQLite-compatible behavior).
        let mut column_assignments: FxHashMap<String, (ExecutorColumn, PreparedAssignmentValue)> =
            FxHashMap::with_capacity_and_hasher(assignments.len(), Default::default());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;

            let prepared_value = match assignment.value {
                AssignmentValue::Literal(value) => PreparedAssignmentValue::Literal(value),
                AssignmentValue::Expression(expr) => {
                    let translated = translation::expression::translate_scalar_with(
                        &expr,
                        schema,
                        |name| {
                            Error::InvalidArgumentError(format!(
                                "Binder Error: does not have a column named '{}'",
                                name
                            ))
                        },
                        |name| {
                            Error::InvalidArgumentError(format!(
                                "unknown column '{}' in aggregate",
                                name
                            ))
                        },
                    )?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    PreparedAssignmentValue::Expression { expr_index }
                }
            };

            // Store in map - if column appears multiple times, last one wins (SQLite behavior)
            column_assignments.insert(normalized, (column.clone(), prepared_value));
        }

        // Convert map to vector for processing
        let prepared: Vec<(ExecutorColumn, PreparedAssignmentValue)> =
            column_assignments.into_values().collect();

        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.cardinality();
        let table_id = table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count as usize];

        {
            let mut stream =
                table.stream_columns(logical_fields, &row_ids, GatherNullPolicy::IncludeNulls)?;

            let mut emitted = 0usize;
            while let Some(chunk) = stream.next_chunk()? {
                let batch = chunk.record_batch();
                let base = emitted;
                let local_len = batch.num_rows();
                for col_idx in 0..batch.num_columns() {
                    let array = batch.column(col_idx);
                    for local_idx in 0..local_len {
                        let target_index = base + local_idx;
                        debug_assert!(
                            target_index < new_rows.len(),
                            "row stream produced out-of-range row index"
                        );
                        if let Some(row) = new_rows.get_mut(target_index) {
                            let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                            row.push(value);
                        }
                    }
                }
                emitted += local_len;
            }
        }
        debug_assert!(
            new_rows
                .iter()
                .all(|row| row.len() == table.schema.columns.len())
        );

        tracing::trace!(
            table = %display_name,
            row_count,
            rows = ?new_rows,
            "update_filtered_rows captured source rows"
        );

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();
        let mut original_primary_key_keys: Vec<Option<UniqueKey>> = Vec::new();
        if let Some(pk) = primary_key_spec {
            original_primary_key_keys.reserve(row_count as usize);
            for row in &new_rows {
                let mut values = Vec::with_capacity(pk.schema_indices.len());
                for &idx in &pk.schema_indices {
                    let value = row.get(idx).cloned().unwrap_or(PlanValue::Null);
                    values.push(value);
                }
                let key = build_composite_unique_key(&values, &pk.column_names)?;
                original_primary_key_keys.push(key);
            }
        }

        let column_positions: FxHashMap<FieldId, usize> = FxHashMap::from_iter(
            table
                .schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, column)| (column.field_id, idx)),
        );

        // Extract field IDs being updated for FK validation later
        let updated_field_ids: Vec<FieldId> =
            prepared.iter().map(|(column, _)| column.field_id).collect();

        for (column, value) in prepared {
            let column_index =
                column_positions
                    .get(&column.field_id)
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' missing in table schema during UPDATE",
                            column.name
                        ))
                    })?;

            let values = match value {
                PreparedAssignmentValue::Literal(lit) => vec![lit; row_count as usize],
                PreparedAssignmentValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count as usize {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };

            for (row_idx, new_value) in values.into_iter().enumerate() {
                if let Some(row) = new_rows.get_mut(row_idx) {
                    let coerced = self.coerce_plan_value_for_column(new_value, &column)?;
                    row[column_index] = coerced;
                }
            }
        }

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_order = resolve_insert_columns(&column_names, table.schema.as_ref())?;

        // Validate row-level constraints (NOT NULL, CHECK)
        self.constraint_service.validate_row_level_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &column_order,
            &new_rows,
        )?;

        // For UPDATE, validate UNIQUE constraints against existing rows EXCLUDING
        // the rows being updated (since they'll be deleted before new values are inserted).
        // This prevents false duplicate detection when primary key values don't change.
        let all_visible_row_ids = {
            let first_field = table
                .schema
                .first_field_id()
                .ok_or_else(|| Error::Internal("table has no columns for validation".into()))?;
            let filter_expr = translation::expression::full_table_scan_filter(first_field);
            let all_ids = table.filter_row_ids(&filter_expr)?;
            filter_row_ids_for_snapshot(table.table.as_ref(), all_ids, &self.txn_manager, snapshot)?
        };

        self.constraint_service.validate_insert_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &constraint_ctx.unique_columns,
            &constraint_ctx.multi_column_uniques,
            primary_key_spec,
            &column_order,
            &new_rows,
            |field_id| {
                // Get all values and filter out rows being updated
                let all_vals =
                    self.collect_row_values_for_ids(table, &all_visible_row_ids, &[field_id])?;
                let filtered: Vec<PlanValue> = all_vals
                    .into_iter()
                    .zip(all_visible_row_ids.iter())
                    .filter_map(|(row, row_id)| {
                        if !row_ids.contains(row_id) {
                            row.into_iter().next()
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(filtered)
            },
            |field_ids| {
                // Get all multi-column values and filter out rows being updated
                let all_rows =
                    self.collect_row_values_for_ids(table, &all_visible_row_ids, field_ids)?;
                let filtered: Vec<Vec<PlanValue>> = all_rows
                    .into_iter()
                    .zip(all_visible_row_ids.iter())
                    .filter_map(|(row, row_id)| {
                        if !row_ids.contains(row_id) {
                            Some(row)
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(filtered)
            },
        )?;

        if let Some(pk) = primary_key_spec {
            self.constraint_service.validate_update_primary_keys(
                &constraint_ctx.schema_field_ids,
                pk,
                &column_order,
                &new_rows,
                &original_primary_key_keys,
                |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
            )?;
        }

        // Check foreign key constraints before updating (update_filtered_rows)
        self.check_foreign_keys_on_update(
            table,
            &display_name,
            &canonical_name,
            &row_ids,
            &updated_field_ids,
            snapshot,
        )?;

        // Also validate foreign keys for the new rows BEFORE deleting old rows
        // This prevents data loss if the new values violate FK constraints
        self.check_foreign_keys_on_insert(
            table,
            &display_name,
            &new_rows,
            &column_order,
            snapshot,
        )?;

        let touches_constraints =
            self.update_touches_constraint_columns(&updated_field_ids, &constraint_ctx);
        let use_in_place =
            snapshot.txn_id == llkv_transaction::TXN_ID_AUTO_COMMIT && !touches_constraints;

        if use_in_place {
            self.update_rows_in_place(
                table,
                &display_name,
                row_ids,
                new_rows,
                updated_field_ids,
                snapshot,
            )?;
        } else {
            let _ = self.apply_delete(
                table,
                display_name.clone(),
                canonical_name.clone(),
                row_ids.clone(),
                snapshot,
                false,
            )?;

            let insert_ctx = InsertExecContext::new(
                table,
                display_name.clone(),
                canonical_name,
                column_names,
                snapshot,
                ConstraintEnforcementMode::Immediate,
            );
            let _ = self.insert_rows(insert_ctx, new_rows)?;
        }

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count as usize,
        })
    }

    /// Update all rows in a table.
    pub(super) fn update_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        assignments: Vec<ColumnAssignment>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let total_rows_usize = usize::try_from(total_rows).map_err(|_| {
            Error::InvalidArgumentError("table row count exceeds supported range".into())
        })?;
        if total_rows_usize == 0 {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let schema = table.schema.as_ref();

        // SQLite allows duplicate column assignments (e.g., SET x=3, x=4, x=5)
        // and uses the rightmost value. We'll use a map to track the last assignment.
        let mut column_assignments: FxHashMap<String, (ExecutorColumn, PreparedAssignmentValue)> =
            FxHashMap::default();
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;

            let prepared_value = match assignment.value {
                AssignmentValue::Literal(value) => PreparedAssignmentValue::Literal(value),
                AssignmentValue::Expression(expr) => {
                    let translated = translation::expression::translate_scalar_with(
                        &expr,
                        schema,
                        |name| {
                            Error::InvalidArgumentError(format!(
                                "Binder Error: does not have a column named '{}'",
                                name
                            ))
                        },
                        |name| {
                            Error::InvalidArgumentError(format!(
                                "unknown column '{}' in aggregate",
                                name
                            ))
                        },
                    )?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    PreparedAssignmentValue::Expression { expr_index }
                }
            };

            // Store in map - if column appears multiple times, last one wins (SQLite behavior)
            column_assignments.insert(normalized, (column.clone(), prepared_value));
        }

        // Convert map to vector for processing
        let prepared: Vec<(ExecutorColumn, PreparedAssignmentValue)> =
            column_assignments.into_values().collect();

        // Use ROW_ID as the anchor for scanning. This ensures we find ALL rows
        // since every row has a row_id and it's never NULL. User columns might have
        // NULL values or be indexed, and indexes typically don't include NULLs.
        use llkv_table::ROW_ID_FIELD_ID;
        let filter_expr = translation::expression::full_table_scan_filter(ROW_ID_FIELD_ID);

        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.cardinality();
        let table_id = table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count as usize];

        {
            let mut stream =
                table.stream_columns(logical_fields, &row_ids, GatherNullPolicy::IncludeNulls)?;

            let mut emitted = 0usize;
            while let Some(chunk) = stream.next_chunk()? {
                let batch = chunk.record_batch();
                let base = emitted;
                let local_len = batch.num_rows();
                for col_idx in 0..batch.num_columns() {
                    let array = batch.column(col_idx);
                    for local_idx in 0..local_len {
                        let target_index = base + local_idx;
                        debug_assert!(
                            target_index < new_rows.len(),
                            "row stream produced out-of-range row index"
                        );
                        if let Some(row) = new_rows.get_mut(target_index) {
                            let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                            row.push(value);
                        }
                    }
                }
                emitted += local_len;
            }
        }
        debug_assert!(
            new_rows
                .iter()
                .all(|row| row.len() == table.schema.columns.len())
        );

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();
        let mut original_primary_key_keys: Vec<Option<UniqueKey>> = Vec::new();
        if let Some(pk) = primary_key_spec {
            original_primary_key_keys.reserve(row_count as usize);
            for row in &new_rows {
                let mut values = Vec::with_capacity(pk.schema_indices.len());
                for &idx in &pk.schema_indices {
                    let value = row.get(idx).cloned().unwrap_or(PlanValue::Null);
                    values.push(value);
                }
                let key = build_composite_unique_key(&values, &pk.column_names)?;
                original_primary_key_keys.push(key);
            }
        }

        let column_positions: FxHashMap<FieldId, usize> = FxHashMap::from_iter(
            table
                .schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, column)| (column.field_id, idx)),
        );

        // Extract field IDs being updated for FK validation later (update_all_rows)
        let updated_field_ids: Vec<FieldId> =
            prepared.iter().map(|(column, _)| column.field_id).collect();

        for (column, value) in prepared {
            let column_index =
                column_positions
                    .get(&column.field_id)
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' missing in table schema during UPDATE",
                            column.name
                        ))
                    })?;

            let values = match value {
                PreparedAssignmentValue::Literal(lit) => vec![lit; row_count as usize],
                PreparedAssignmentValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count as usize {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };

            for (row_idx, new_value) in values.into_iter().enumerate() {
                if let Some(row) = new_rows.get_mut(row_idx) {
                    let coerced = self.coerce_plan_value_for_column(new_value, &column)?;
                    row[column_index] = coerced;
                }
            }
        }

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_order = resolve_insert_columns(&column_names, table.schema.as_ref())?;

        // Validate row-level constraints (NOT NULL, CHECK)
        self.constraint_service.validate_row_level_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &column_order,
            &new_rows,
        )?;

        // For UPDATE, validate UNIQUE constraints excluding rows being updated
        let all_visible_row_ids = {
            let first_field = table
                .schema
                .first_field_id()
                .ok_or_else(|| Error::Internal("table has no columns for validation".into()))?;
            let filter_expr = translation::expression::full_table_scan_filter(first_field);
            let all_ids = table.filter_row_ids(&filter_expr)?;
            filter_row_ids_for_snapshot(table.table.as_ref(), all_ids, &self.txn_manager, snapshot)?
        };

        self.constraint_service.validate_insert_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &constraint_ctx.unique_columns,
            &constraint_ctx.multi_column_uniques,
            primary_key_spec,
            &column_order,
            &new_rows,
            |field_id| {
                let all_vals =
                    self.collect_row_values_for_ids(table, &all_visible_row_ids, &[field_id])?;
                let filtered: Vec<PlanValue> = all_vals
                    .into_iter()
                    .zip(all_visible_row_ids.iter())
                    .filter_map(|(row, row_id)| {
                        if !row_ids.contains(row_id) {
                            row.into_iter().next()
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(filtered)
            },
            |field_ids| {
                let all_rows =
                    self.collect_row_values_for_ids(table, &all_visible_row_ids, field_ids)?;
                let filtered: Vec<Vec<PlanValue>> = all_rows
                    .into_iter()
                    .zip(all_visible_row_ids.iter())
                    .filter_map(|(row, row_id)| {
                        if !row_ids.contains(row_id) {
                            Some(row)
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(filtered)
            },
        )?;

        if let Some(pk) = primary_key_spec {
            self.constraint_service.validate_update_primary_keys(
                &constraint_ctx.schema_field_ids,
                pk,
                &column_order,
                &new_rows,
                &original_primary_key_keys,
                |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
            )?;
        }

        // Check foreign key constraints before updating (update_all_rows)
        self.check_foreign_keys_on_update(
            table,
            &display_name,
            &canonical_name,
            &row_ids,
            &updated_field_ids,
            snapshot,
        )?;

        // Also validate foreign keys for the new rows BEFORE deleting old rows
        self.check_foreign_keys_on_insert(
            table,
            &display_name,
            &new_rows,
            &column_order,
            snapshot,
        )?;

        let touches_constraints =
            self.update_touches_constraint_columns(&updated_field_ids, &constraint_ctx);
        let use_in_place =
            snapshot.txn_id == llkv_transaction::TXN_ID_AUTO_COMMIT && !touches_constraints;

        if use_in_place {
            self.update_rows_in_place(
                table,
                &display_name,
                row_ids,
                new_rows,
                updated_field_ids,
                snapshot,
            )?;
        } else {
            let _ = self.apply_delete(
                table,
                display_name.clone(),
                canonical_name.clone(),
                row_ids.clone(),
                snapshot,
                false,
            )?;

            let insert_ctx = InsertExecContext::new(
                table,
                display_name.clone(),
                canonical_name,
                column_names,
                snapshot,
                ConstraintEnforcementMode::Immediate,
            );
            let _ = self.insert_rows(insert_ctx, new_rows)?;
        }

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count as usize,
        })
    }

    /// Rewrite updated rows in-place for auto-commit transactions.
    ///
    /// This keeps MVCC churn bounded by reusing existing `row_id` values and
    /// preserving the prior `created_by`/`deleted_by` metadata. Snapshot reads
    /// continue to see the correct version, while OLTP-style auto-commit
    /// workloads avoid unbounded version growth. Conflict checks mirror the
    /// delete+insert path so transactional behavior stays identical.
    ///
    /// TODO: once `ColumnStore::vacuum_table` exists, trigger chunk compaction
    /// here for multi-statement transactions after they commit.
    fn update_rows_in_place(
        &self,
        table: &ExecutorTable<P>,
        display_name: &str,
        row_ids: Treemap,
        new_rows: Vec<Vec<PlanValue>>,
        updated_field_ids: Vec<FieldId>,
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        use arrow::array::UInt64Builder;
        use arrow::datatypes::Schema;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        debug_assert_eq!(
            snapshot.txn_id,
            llkv_transaction::TXN_ID_AUTO_COMMIT,
            "update_rows_in_place should only be called for auto-commit transactions",
        );
        debug_assert_eq!(
            row_ids.cardinality() as usize,
            new_rows.len(),
            "row_ids and new_rows must have the same length",
        );

        if row_ids.is_empty() || updated_field_ids.is_empty() {
            return Ok(());
        }

        // Preserve conflict detection semantics from delete+insert path.
        self.detect_delete_conflicts(table, display_name, &row_ids, snapshot)?;

        let row_count = row_ids.cardinality();
        tracing::debug!(
            table_id = table.table_id(),
            row_count,
            ?row_ids,
            "update_rows_in_place: rewriting rows",
        );

        let mut update_columns: Vec<(usize, ExecutorColumn)> =
            Vec::with_capacity(updated_field_ids.len());
        for field_id in updated_field_ids {
            let (schema_index, column) = table
                .schema
                .columns
                .iter()
                .enumerate()
                .find(|(_, col)| col.field_id == field_id)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "target column with field_id {} missing during in-place update",
                        field_id
                    ))
                })?;
            update_columns.push((schema_index, column.clone()));
        }

        let mut column_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_count as usize); update_columns.len()];
        for row in &new_rows {
            debug_assert_eq!(
                row.len(),
                table.schema.columns.len(),
                "in-place update row width mismatch",
            );
            for (dest_idx, (schema_index, _)) in update_columns.iter().enumerate() {
                let value = row.get(*schema_index).cloned().unwrap_or(PlanValue::Null);
                column_values[dest_idx].push(value);
            }
        }

        let mut row_id_builder = UInt64Builder::with_capacity(row_count as usize);
        for rid in row_ids.iter() {
            row_id_builder.append_value(rid);
        }
        let row_id_array = Arc::new(row_id_builder.finish()) as ArrayRef;

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(update_columns.len() + 1);
        arrays.push(row_id_array);

        let mut fields = mvcc::build_mvcc_fields();
        let mut result_fields = Vec::with_capacity(update_columns.len() + 1);
        result_fields.push(fields.remove(0));

        for ((_, column), values) in update_columns.iter().zip(column_values.into_iter()) {
            tracing::debug!(
                column = %column.name,
                values = ?values,
                "update_rows_in_place: column rewrite",
            );
            let array = build_array_for_column(&column.data_type, &values)?;
            let field = mvcc::build_field_with_metadata(
                &column.name,
                column.data_type.clone(),
                column.nullable,
                column.field_id,
            );
            arrays.push(array);
            result_fields.push(field);
        }

        let schema = Arc::new(Schema::new(result_fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| Error::Internal(format!("failed to build update batch: {}", e)))?;

        table.table.append(&batch)?;

        Ok(())
    }

    fn update_touches_constraint_columns(
        &self,
        updated_field_ids: &[FieldId],
        constraint_ctx: &TableConstraintContext,
    ) -> bool {
        if updated_field_ids.is_empty() {
            return false;
        }

        if constraint_ctx
            .unique_columns
            .iter()
            .any(|column| updated_field_ids.contains(&column.field_id))
        {
            return true;
        }

        if constraint_ctx.multi_column_uniques.iter().any(|unique| {
            unique
                .field_ids
                .iter()
                .any(|field_id| updated_field_ids.contains(field_id))
        }) {
            return true;
        }

        if let Some(pk) = &constraint_ctx.primary_key
            && pk
                .field_ids
                .iter()
                .any(|field_id| updated_field_ids.contains(field_id))
        {
            return true;
        }

        false
    }

    /// Collect row IDs and expression values for UPDATE operations.
    pub(super) fn collect_update_rows(
        &self,
        table: &ExecutorTable<P>,
        filter_expr: &LlkvExpr<'static, FieldId>,
        expressions: &[ScalarExpr<FieldId>],
        snapshot: TransactionSnapshot,
    ) -> Result<(Treemap, Vec<Vec<PlanValue>>)> {
        let row_ids = table.filter_row_ids(filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        if row_ids.is_empty() {
            return Ok((Treemap::new(), vec![Vec::new(); expressions.len()]));
        }

        if expressions.is_empty() {
            return Ok((row_ids, Vec::new()));
        }

        let mut projections: Vec<ScanProjection> = Vec::with_capacity(expressions.len());
        for (idx, expr) in expressions.iter().enumerate() {
            let alias = format!("__expr_{idx}");
            projections.push(ScanProjection::computed(expr.clone(), alias));
        }

        let mut expr_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_ids.cardinality() as usize); expressions.len()];
        let mut error: Option<Error> = None;
        let row_filter: Arc<dyn llkv_table::table::RowIdFilter<P>> = Arc::new(
            MvccRowIdFilter::new(Arc::clone(&self.txn_manager), snapshot),
        );
        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: Some(row_filter),
            include_row_ids: true,
            ranges: None,
            driving_column: None,
        };

        table
            .table
            .scan_stream_with_exprs(&projections, filter_expr, options, |batch| {
                if error.is_some() {
                    return;
                }
                if let Err(err) = Self::collect_expression_values(&mut expr_values, batch) {
                    error = Some(err);
                }
            })?;

        if let Some(err) = error {
            return Err(err);
        }

        for values in &expr_values {
            if values.len() != row_ids.cardinality() as usize {
                return Err(Error::InvalidArgumentError(
                    "expression result count did not match targeted row count".into(),
                ));
            }
        }

        Ok((row_ids, expr_values))
    }

    /// Helper to collect expression values from a batch during UPDATE.
    fn collect_expression_values(
        expr_values: &mut [Vec<PlanValue>],
        batch: RecordBatch,
    ) -> Result<()> {
        for row_idx in 0..batch.num_rows() {
            for (expr_index, values) in expr_values.iter_mut().enumerate() {
                let value = llkv_plan::plan_value_from_array(batch.column(expr_index), row_idx)?;
                values.push(value);
            }
        }

        Ok(())
    }
}
