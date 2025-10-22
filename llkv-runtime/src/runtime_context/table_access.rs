//! Table access and caching utilities for `RuntimeContext`.
//!
//! This module centralizes lower-level table access helpers that were previously
//! embedded directly inside `mod.rs`. Moving them here keeps the core module
//! focused on high-level orchestration while these helpers encapsulate caching,
//! lazy loading, and direct batch interactions.

use crate::{RuntimeContext, RuntimeTableHandle, canonical_table_name};
use arrow::array::{ArrayRef, RecordBatch, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_column_map::store::{GatherNullPolicy, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_executor::{
    ExecutorColumn, ExecutorMultiColumnUnique, ExecutorRowBatch, ExecutorSchema, ExecutorTable,
    translation,
};
use llkv_result::{Error, Result};
use llkv_storage::pager::Pager;
use llkv_table::resolvers::{FieldConstraints, FieldDefinition};
use llkv_table::{
    ConstraintKind, FieldId, MultiColumnUniqueEntryMeta, RowId, Table, TableConstraintSummaryView,
};
use llkv_transaction::{TransactionSnapshot, mvcc};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Exports all rows from a table as a `RowBatch` - internal storage API.
    /// Use through `RuntimeSession` or `RuntimeTableHandle` instead.
    pub(crate) fn export_table_rows(self: &Arc<Self>, name: &str) -> Result<ExecutorRowBatch> {
        let handle = RuntimeTableHandle::new(Arc::clone(self), name)?;
        handle.lazy()?.collect_rows()
    }

    /// Get raw batches from a table including row_ids - internal storage API.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    /// Use through `RuntimeSession` or transaction context instead.
    pub(crate) fn get_batches_with_row_ids(
        &self,
        table_name: &str,
        filter: Option<llkv_expr::Expr<'static, String>>,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<RecordBatch>> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let filter_expr = match filter {
            Some(expr) => {
                translation::expression::translate_predicate(expr, table.schema.as_ref(), |name| {
                    Error::InvalidArgumentError(format!(
                        "Binder Error: does not have a column named '{}'",
                        name
                    ))
                })?
            }
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                translation::expression::full_table_scan_filter(field_id)
            }
        };

        // First, get the row_ids that match the filter
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let visible_row_ids = self.filter_visible_row_ids(table.as_ref(), row_ids, snapshot)?;
        if visible_row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Scan to get the column data without materializing full columns
        let table_id = table.table.table_id();

        let mut fields: Vec<Field> = Vec::with_capacity(table.schema.columns.len() + 1);
        let mut logical_fields: Vec<LogicalFieldId> =
            Vec::with_capacity(table.schema.columns.len());

        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for column in &table.schema.columns {
            let logical_field_id = LogicalFieldId::for_user(table_id, column.field_id);
            logical_fields.push(logical_field_id);
            let field = mvcc::build_field_with_metadata(
                &column.name,
                column.data_type.clone(),
                column.nullable,
                column.field_id,
            );
            fields.push(field);
        }

        let schema = Arc::new(Schema::new(fields));

        if logical_fields.is_empty() {
            // Tables without user columns should still return row_id batches.
            let mut row_id_builder = UInt64Builder::with_capacity(visible_row_ids.len());
            for row_id in &visible_row_ids {
                row_id_builder.append_value(*row_id);
            }
            let arrays: Vec<ArrayRef> = vec![Arc::new(row_id_builder.finish()) as ArrayRef];
            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
            return Ok(vec![batch]);
        }

        let mut stream = table.table.stream_columns(
            Arc::from(logical_fields),
            visible_row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut batches = Vec::new();
        while let Some(chunk) = stream.next_batch()? {
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(chunk.batch().num_columns() + 1);

            let mut row_id_builder = UInt64Builder::with_capacity(chunk.len());
            for row_id in chunk.row_ids() {
                row_id_builder.append_value(*row_id);
            }
            arrays.push(Arc::new(row_id_builder.finish()) as ArrayRef);

            let chunk_batch = chunk.into_batch();
            for column_array in chunk_batch.columns() {
                arrays.push(column_array.clone());
            }

            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Append batches directly to a table, preserving row_ids - internal storage API.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    /// Use through `RuntimeSession` or transaction context instead.
    pub(crate) fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<usize> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let mut total_rows = 0;
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            // Verify the batch has a row_id column
            let _row_id_idx = batch.schema().index_of(ROW_ID_COLUMN_NAME).map_err(|_| {
                Error::InvalidArgumentError(
                    "batch must contain row_id column for direct append".into(),
                )
            })?;

            // Append the batch directly to the underlying table
            table.table.append(&batch)?;
            total_rows += batch.num_rows();
        }

        Ok(total_rows)
    }

    /// Looks up a table in the executor cache, lazily loading it from metadata if not already cached.
    /// This is the primary method for obtaining table references for query execution.
    pub fn lookup_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        // Fast path: check if table is already loaded
        {
            let tables = self.tables.read().unwrap();
            if let Some(table) = tables.get(canonical_name) {
                // Check if table has been dropped
                if self.dropped_tables.read().unwrap().contains(canonical_name) {
                    // Table was dropped - treat as not found
                    return Err(Error::NotFound);
                }
                tracing::trace!(
                    "=== LOOKUP_TABLE '{}' (cached) table_id={} columns={} context_pager={:p} ===",
                    canonical_name,
                    table.table.table_id(),
                    table.schema.columns.len(),
                    &*self.pager
                );
                return Ok(Arc::clone(table));
            }
        } // Release read lock

        // Slow path: load table from catalog (happens once per table)
        tracing::debug!(
            "[LAZY_LOAD] Loading table '{}' from catalog",
            canonical_name
        );

        // Check catalog first for table existence
        let catalog_table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let table_id = catalog_table_id;
        let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;
        let store = table.store();
        let mut logical_fields = store.user_field_ids_for_table(table_id);
        logical_fields.sort_by_key(|lfid| lfid.field_id());
        let field_ids: Vec<FieldId> = logical_fields.iter().map(|lfid| lfid.field_id()).collect();
        let summary = self
            .catalog_service
            .table_constraint_summary(canonical_name)?;
        let TableConstraintSummaryView {
            table_meta,
            column_metas,
            constraint_records,
            multi_column_uniques,
        } = summary;
        let _table_meta = table_meta.ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;
        let catalog_field_resolver = self.catalog.field_resolver(catalog_table_id);
        let mut metadata_primary_keys: FxHashSet<FieldId> = FxHashSet::default();
        let mut metadata_unique_fields: FxHashSet<FieldId> = FxHashSet::default();
        let mut has_primary_key_records = false;
        let mut has_single_unique_records = false;

        for record in constraint_records
            .iter()
            .filter(|record| record.is_active())
        {
            match &record.kind {
                ConstraintKind::PrimaryKey(pk) => {
                    has_primary_key_records = true;
                    for field_id in &pk.field_ids {
                        metadata_primary_keys.insert(*field_id);
                        metadata_unique_fields.insert(*field_id);
                    }
                }
                ConstraintKind::Unique(unique) => {
                    if unique.field_ids.len() == 1 {
                        has_single_unique_records = true;
                        metadata_unique_fields.insert(unique.field_ids[0]);
                    }
                }
                _ => {}
            }
        }

        // Build ExecutorSchema from metadata manager snapshots
        let mut executor_columns = Vec::new();
        let mut lookup = FxHashMap::with_capacity_and_hasher(field_ids.len(), Default::default());

        for (idx, lfid) in logical_fields.iter().enumerate() {
            let field_id = lfid.field_id();
            let normalized_index = executor_columns.len();

            let column_name = column_metas
                .get(idx)
                .and_then(|meta| meta.as_ref())
                .and_then(|meta| meta.name.clone())
                .unwrap_or_else(|| format!("col_{}", field_id));

            let normalized = column_name.to_ascii_lowercase();
            lookup.insert(normalized, normalized_index);

            let fallback_constraints: FieldConstraints = catalog_field_resolver
                .as_ref()
                .and_then(|resolver| resolver.field_constraints_by_name(&column_name))
                .unwrap_or_default();

            let metadata_primary = metadata_primary_keys.contains(&field_id);
            let primary_key = if has_primary_key_records {
                metadata_primary
            } else {
                fallback_constraints.primary_key
            };

            let metadata_unique = metadata_primary || metadata_unique_fields.contains(&field_id);
            let unique = if has_primary_key_records || has_single_unique_records {
                metadata_unique
            } else {
                fallback_constraints.primary_key || fallback_constraints.unique
            };

            let data_type = store.data_type(*lfid)?;
            let nullable = !primary_key;

            executor_columns.push(ExecutorColumn {
                name: column_name,
                data_type,
                nullable,
                primary_key,
                unique,
                field_id,
                check_expr: fallback_constraints.check_expr.clone(),
            });
        }

        let exec_schema = Arc::new(ExecutorSchema {
            columns: executor_columns,
            lookup,
        });

        // Find the maximum row_id in the table to set next_row_id correctly
        let max_row_id = {
            use arrow::array::UInt64Array;
            use llkv_column_map::store::rowid_fid;
            use llkv_column_map::store::scan::{
                PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
                PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
            };

            struct MaxRowIdVisitor {
                max: RowId,
            }

            impl PrimitiveVisitor for MaxRowIdVisitor {
                fn u64_chunk(&mut self, values: &UInt64Array) {
                    for i in 0..values.len() {
                        let val = values.value(i);
                        if val > self.max {
                            self.max = val;
                        }
                    }
                }
            }

            impl PrimitiveWithRowIdsVisitor for MaxRowIdVisitor {}
            impl PrimitiveSortedVisitor for MaxRowIdVisitor {}
            impl PrimitiveSortedWithRowIdsVisitor for MaxRowIdVisitor {}

            // Scan the row_id column for any user field in this table
            let row_id_field = rowid_fid(LogicalFieldId::for_user(table_id, 1));
            let mut visitor = MaxRowIdVisitor { max: 0 };

            match ScanBuilder::new(table.store(), row_id_field)
                .options(ScanOptions::default())
                .run(&mut visitor)
            {
                Ok(_) => visitor.max,
                Err(llkv_result::Error::NotFound) => 0,
                Err(e) => {
                    tracing::warn!(
                        "[LAZY_LOAD] Failed to scan max row_id for table '{}': {}",
                        canonical_name,
                        e
                    );
                    0
                }
            }
        };

        let next_row_id = if max_row_id > 0 {
            max_row_id.saturating_add(1)
        } else {
            0
        };

        // Get the actual persisted row count from table metadata
        // This is an O(1) catalog lookup that reads ColumnDescriptor.total_row_count
        // Fallback to 0 for truly empty tables
        let total_rows = table.total_rows().unwrap_or(0);

        let executor_table = Arc::new(ExecutorTable {
            table: Arc::new(table),
            schema: exec_schema,
            next_row_id: AtomicU64::new(next_row_id),
            total_rows: AtomicU64::new(total_rows),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

        if !multi_column_uniques.is_empty() {
            let executor_uniques =
                Self::build_executor_multi_column_uniques(&executor_table, &multi_column_uniques);
            executor_table.set_multi_column_uniques(executor_uniques);
        }

        // Cache the loaded table
        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(canonical_name.to_string(), Arc::clone(&executor_table));
        }

        // Register fields in catalog (may already be registered from RuntimeContext::new())
        if let Some(field_resolver) = self.catalog.field_resolver(catalog_table_id) {
            for col in &executor_table.schema.columns {
                let definition = FieldDefinition::new(&col.name)
                    .with_primary_key(col.primary_key)
                    .with_unique(col.unique)
                    .with_check_expr(col.check_expr.clone());
                let _ = field_resolver.register_field(definition); // Ignore "already exists" errors
            }
            tracing::debug!(
                "[CATALOG] Registered {} field(s) for lazy-loaded table '{}'",
                executor_table.schema.columns.len(),
                canonical_name
            );
        }

        tracing::debug!(
            "[LAZY_LOAD] Loaded table '{}' (id={}) with {} columns, next_row_id={}",
            canonical_name,
            table_id,
            field_ids.len(),
            next_row_id
        );

        Ok(executor_table)
    }

    pub(super) fn build_executor_multi_column_uniques(
        table: &ExecutorTable<P>,
        stored: &[MultiColumnUniqueEntryMeta],
    ) -> Vec<ExecutorMultiColumnUnique> {
        let mut results = Vec::with_capacity(stored.len());

        'outer: for entry in stored {
            if entry.column_ids.is_empty() {
                continue;
            }

            let mut column_indices = Vec::with_capacity(entry.column_ids.len());
            for field_id in &entry.column_ids {
                if let Some((idx, _)) = table
                    .schema
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, col)| &col.field_id == field_id)
                {
                    column_indices.push(idx);
                } else {
                    tracing::warn!(
                        "[CATALOG] Skipping persisted multi-column UNIQUE {:?} for table_id={} missing field_id {}",
                        entry.index_name,
                        table.table.table_id(),
                        field_id
                    );
                    continue 'outer;
                }
            }

            results.push(ExecutorMultiColumnUnique {
                index_name: entry.index_name.clone(),
                column_indices,
            });
        }

        results
    }

    pub(super) fn rebuild_executor_table_with_unique(
        table: &ExecutorTable<P>,
        field_id: FieldId,
    ) -> Option<Arc<ExecutorTable<P>>> {
        let mut columns = table.schema.columns.clone();
        let mut found = false;
        for column in &mut columns {
            if column.field_id == field_id {
                column.unique = true;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }

        let schema = Arc::new(ExecutorSchema {
            columns,
            lookup: table.schema.lookup.clone(),
        });

        let next_row_id = table.next_row_id.load(Ordering::SeqCst);
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let uniques = table.multi_column_uniques();

        Some(Arc::new(ExecutorTable {
            table: Arc::clone(&table.table),
            schema,
            next_row_id: AtomicU64::new(next_row_id),
            total_rows: AtomicU64::new(total_rows),
            multi_column_uniques: RwLock::new(uniques),
        }))
    }
}
