//! UPDATE operation implementation for RuntimeContext.
//!
//! This module contains all logic for updating rows in tables, including:
//! - Filtered and full table updates
//! - Expression-based assignments
//! - Constraint validation for updates
//! - Foreign key validation for updates

use crate::{RuntimeStatementResult, canonical_table_name};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::GatherNullPolicy;
use llkv_column_map::types::LogicalFieldId;
use llkv_executor::{ExecutorColumn, ExecutorTable, resolve_insert_columns, translation};
use llkv_expr::{Expr as LlkvExpr, ScalarExpr};
use llkv_plan::{AssignmentValue, ColumnAssignment, PlanValue, UpdatePlan};
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::table::ScanProjection;
use llkv_table::table::ScanStreamOptions;
use llkv_table::{FieldId, RowId, UniqueKey, build_composite_unique_key};
use llkv_transaction::{MvccRowIdFilter, TransactionSnapshot, filter_row_ids_for_snapshot};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{PreparedAssignmentValue, RuntimeContext};

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
        if self.is_view(table.table.table_id())? {
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

        let mut seen_columns: FxHashSet<String> =
            FxHashSet::with_capacity_and_hasher(assignments.len(), Default::default());
        let mut prepared: Vec<(ExecutorColumn, PreparedAssignmentValue)> =
            Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedAssignmentValue::Literal(value)));
                }
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
                    prepared.push((
                        column.clone(),
                        PreparedAssignmentValue::Expression { expr_index },
                    ));
                }
            }
        }

        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let table_id = table.table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let mut stream = table.table.stream_columns(
            logical_fields.clone(),
            row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < new_rows.len(),
                        "column stream produced out-of-range row index"
                    );
                    if let Some(row) = new_rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
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
            original_primary_key_keys.reserve(row_count);
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
                PreparedAssignmentValue::Literal(lit) => vec![lit; row_count],
                PreparedAssignmentValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
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
        let row_ids_set: FxHashSet<RowId> = row_ids.iter().copied().collect();
        let all_visible_row_ids = {
            let first_field = table
                .schema
                .first_field_id()
                .ok_or_else(|| Error::Internal("table has no columns for validation".into()))?;
            let filter_expr = translation::expression::full_table_scan_filter(first_field);
            let all_ids = table.table.filter_row_ids(&filter_expr)?;
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
                    .zip(&all_visible_row_ids)
                    .filter_map(|(row, &row_id)| {
                        if !row_ids_set.contains(&row_id) {
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
                    .zip(&all_visible_row_ids)
                    .filter_map(|(row, &row_id)| {
                        if !row_ids_set.contains(&row_id) {
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

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            canonical_name.clone(),
            row_ids.clone(),
            snapshot,
            false,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            canonical_name,
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
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

        let mut seen_columns: FxHashSet<String> =
            FxHashSet::with_capacity_and_hasher(assignments.len(), Default::default());
        let mut prepared: Vec<(ExecutorColumn, PreparedAssignmentValue)> =
            Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();
        let mut first_field_id: Option<FieldId> = None;

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;
            if first_field_id.is_none() {
                first_field_id = Some(column.field_id);
            }

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedAssignmentValue::Literal(value)));
                }
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
                    prepared.push((
                        column.clone(),
                        PreparedAssignmentValue::Expression { expr_index },
                    ));
                }
            }
        }

        let anchor_field = first_field_id.ok_or_else(|| {
            Error::InvalidArgumentError("UPDATE requires at least one target column".into())
        })?;

        let filter_expr = translation::expression::full_table_scan_filter(anchor_field);
        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let table_id = table.table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let mut stream = table.table.stream_columns(
            logical_fields.clone(),
            row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < new_rows.len(),
                        "column stream produced out-of-range row index"
                    );
                    if let Some(row) = new_rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
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
            original_primary_key_keys.reserve(row_count);
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
                PreparedAssignmentValue::Literal(lit) => vec![lit; row_count],
                PreparedAssignmentValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
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
        let row_ids_set: FxHashSet<RowId> = row_ids.iter().copied().collect();
        let all_visible_row_ids = {
            let first_field = table
                .schema
                .first_field_id()
                .ok_or_else(|| Error::Internal("table has no columns for validation".into()))?;
            let filter_expr = translation::expression::full_table_scan_filter(first_field);
            let all_ids = table.table.filter_row_ids(&filter_expr)?;
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
                    .zip(&all_visible_row_ids)
                    .filter_map(|(row, &row_id)| {
                        if !row_ids_set.contains(&row_id) {
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
                    .zip(&all_visible_row_ids)
                    .filter_map(|(row, &row_id)| {
                        if !row_ids_set.contains(&row_id) {
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

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            canonical_name.clone(),
            row_ids.clone(),
            snapshot,
            false,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            canonical_name,
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    /// Collect row IDs and expression values for UPDATE operations.
    pub(super) fn collect_update_rows(
        &self,
        table: &ExecutorTable<P>,
        filter_expr: &LlkvExpr<'static, FieldId>,
        expressions: &[ScalarExpr<FieldId>],
        snapshot: TransactionSnapshot,
    ) -> Result<(Vec<RowId>, Vec<Vec<PlanValue>>)> {
        let row_ids = table.table.filter_row_ids(filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        if row_ids.is_empty() {
            return Ok((row_ids, vec![Vec::new(); expressions.len()]));
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
            vec![Vec::with_capacity(row_ids.len()); expressions.len()];
        let mut error: Option<Error> = None;
        let row_filter: Arc<dyn llkv_table::table::RowIdFilter<P>> = Arc::new(
            MvccRowIdFilter::new(Arc::clone(&self.txn_manager), snapshot),
        );
        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: Some(row_filter),
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
            if values.len() != row_ids.len() {
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
