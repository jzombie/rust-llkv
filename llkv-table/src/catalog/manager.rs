//! High-level service for creating tables.
//!
//! Use `CatalogManager` to create tables. It coordinates metadata persistence,
//! catalog registration, and storage initialization.

#![forbid(unsafe_code)]

use std::convert::TryFrom;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::IndexKind;
use llkv_plan::{ColumnSpec, ForeignKeySpec};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use super::table_catalog::{FieldDefinition, TableCatalog};
use crate::constraints::ConstraintKind;
use crate::metadata::{MetadataManager, MultiColumnUniqueRegistration};
use crate::table::Table;
use crate::types::{FieldId, RowId, TableColumn, TableId};
use crate::{
    ForeignKeyColumn, ForeignKeyTableInfo, ForeignKeyView, TableConstraintSummaryView, TableView,
    ValidatedForeignKey,
};

/// Result of creating a table. The caller is responsible for wiring executor
/// caches and any higher-level state that depends on the table schema.
pub struct CreateTableResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub table_id: TableId,
    pub table: Arc<Table<P>>,
    pub table_columns: Vec<TableColumn>,
    pub column_lookup: FxHashMap<String, usize>,
}

/// Trait for constructing MVCC columns and Arrow metadata during batch ingestion.
///
/// Implementations can delegate to `llkv_transaction::mvcc` or provide custom logic
/// for synthetic data sources. This indirection avoids coupling the table crate to the
/// transaction crate while still centralizing MVCC helpers.
pub trait MvccColumnBuilder: Send + Sync {
    /// Build MVCC columns (row_id, created_by, deleted_by) for INSERT/CTAS operations.
    fn build_insert_columns(
        &self,
        row_count: usize,
        start_row_id: RowId,
        creator_txn_id: u64,
        deleted_marker: u64,
    ) -> (ArrayRef, ArrayRef, ArrayRef);

    /// Return the Arrow field definitions for MVCC columns.
    fn mvcc_fields(&self) -> Vec<Field>;

    /// Construct a user column field with the required metadata assigned.
    fn field_with_metadata(
        &self,
        name: &str,
        data_type: DataType,
        nullable: bool,
        field_id: FieldId,
    ) -> Field;
}

/// Service for creating tables.
///
/// Coordinates metadata persistence (`MetadataManager`), catalog registration
/// (`TableCatalog`), and storage initialization (`ColumnStore`).
#[derive(Clone)]
pub struct CatalogManager<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    metadata: Arc<MetadataManager<P>>,
    catalog: Arc<TableCatalog>,
    store: Arc<ColumnStore<P>>,
}

impl<P> CatalogManager<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Creates a new CatalogManager coordinating metadata, catalog, and storage layers.
    pub fn new(
        metadata: Arc<MetadataManager<P>>,
        catalog: Arc<TableCatalog>,
        store: Arc<ColumnStore<P>>,
    ) -> Self {
        Self {
            metadata,
            catalog,
            store,
        }
    }

    /// Create a new table using column specifications.
    ///
    /// Reserves table ID from metadata, validates columns, persists schema,
    /// registers in catalog, and returns a Table handle for data operations.
    pub(crate) fn create_table_from_columns(
        &self,
        display_name: &str,
        canonical_name: &str,
        columns: &[ColumnSpec],
    ) -> LlkvResult<CreateTableResult<P>> {
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        let mut lookup: FxHashMap<String, usize> =
            FxHashMap::with_capacity_and_hasher(columns.len(), Default::default());
        let mut table_columns: Vec<TableColumn> = Vec::with_capacity(columns.len());

        for (idx, column) in columns.iter().enumerate() {
            let normalized = column.name.to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }

            table_columns.push(TableColumn {
                field_id: field_id_for_index(idx)?,
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                unique: column.unique,
                check_expr: column.check_expr.clone(),
            });
        }

        self.create_table_inner(display_name, canonical_name, table_columns, lookup)
    }

    /// Create a new table using an Arrow schema (used by CTAS flows).
    pub fn create_table_from_schema(
        &self,
        display_name: &str,
        canonical_name: &str,
        schema: &Schema,
    ) -> LlkvResult<CreateTableResult<P>> {
        if schema.fields().is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one column".into(),
            ));
        }

        let mut lookup: FxHashMap<String, usize> =
            FxHashMap::with_capacity_and_hasher(schema.fields().len(), Default::default());
        let mut table_columns: Vec<TableColumn> = Vec::with_capacity(schema.fields().len());

        for (idx, field) in schema.fields().iter().enumerate() {
            let data_type = match field.data_type() {
                DataType::Int64
                | DataType::Float64
                | DataType::Utf8
                | DataType::Date32
                | DataType::Struct(_) => field.data_type().clone(),
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported column type in CTAS result: {other:?}"
                    )));
                }
            };

            let normalized = field.name().to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in CTAS result",
                    field.name()
                )));
            }

            table_columns.push(TableColumn {
                field_id: field_id_for_index(idx)?,
                name: field.name().to_string(),
                data_type,
                nullable: field.is_nullable(),
                primary_key: false,
                unique: false,
                check_expr: None,
            });
        }

        self.create_table_inner(display_name, canonical_name, table_columns, lookup)
    }

    fn create_table_inner(
        &self,
        display_name: &str,
        _canonical_name: &str,
        table_columns: Vec<TableColumn>,
        column_lookup: FxHashMap<String, usize>,
    ) -> LlkvResult<CreateTableResult<P>> {
        let table_id = self.metadata.reserve_table_id()?;
        let timestamp = current_time_micros();
        let table_meta = crate::sys_catalog::TableMeta {
            table_id,
            name: Some(display_name.to_string()),
            created_at_micros: timestamp,
            flags: 0,
            epoch: 0,
        };

        self.metadata.set_table_meta(table_id, table_meta)?;
        self.metadata
            .apply_column_definitions(table_id, &table_columns, timestamp)?;
        self.metadata.flush_table(table_id)?;

        let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;

        // Register table in catalog using the table_id from metadata
        if let Err(err) = self.catalog.register_table(display_name, table_id) {
            self.metadata.remove_table_state(table_id);
            return Err(err);
        }

        if let Some(field_resolver) = self.catalog.field_resolver(table_id) {
            for column in &table_columns {
                let definition = FieldDefinition::new(&column.name)
                    .with_primary_key(column.primary_key)
                    .with_unique(column.unique)
                    .with_check_expr(column.check_expr.clone());
                if let Err(err) = field_resolver.register_field(definition) {
                    self.catalog.unregister_table(table_id);
                    self.metadata.remove_table_state(table_id);
                    return Err(err);
                }
            }
        }

        Ok(CreateTableResult {
            table_id,
            table: Arc::new(table),
            table_columns,
            column_lookup,
        })
    }

    /// Prepare metadata state and unregister catalog entries for a dropped table.
    pub fn drop_table(
        &self,
        canonical_name: &str,
        table_id: TableId,
        column_field_ids: &[FieldId],
    ) -> LlkvResult<()> {
        self.metadata
            .prepare_table_drop(table_id, column_field_ids)?;
        self.metadata.flush_table(table_id)?;
        self.metadata.remove_table_state(table_id);
        if let Some(table_id_from_catalog) = self.catalog.table_id(canonical_name) {
            let _ = self.catalog.unregister_table(table_id_from_catalog);
        } else {
            let _ = self.catalog.unregister_table(table_id);
        }
        Ok(())
    }

    /// Register a single-column sort (B-tree) index. Optionally marks the field unique.
    /// Returns `true` if the index was newly created, `false` if it already existed and `if_not_exists` was true.
    #[allow(clippy::too_many_arguments)]
    pub fn register_single_column_index(
        &self,
        display_name: &str,
        canonical_name: &str,
        table: &Table<P>,
        field_id: FieldId,
        column_name: &str,
        mark_unique: bool,
        if_not_exists: bool,
    ) -> LlkvResult<bool> {
        let existing_indexes = table.list_registered_indexes(field_id)?;
        if existing_indexes.contains(&IndexKind::Sort) {
            if if_not_exists {
                return Ok(false);
            }
            return Err(Error::CatalogError(format!(
                "Index already exists on column '{}' in table '{}'",
                column_name, display_name
            )));
        }

        let table_id = table.table_id();
        self.metadata.register_sort_index(table_id, field_id)?;

        if mark_unique {
            let catalog_table_id = self.catalog.table_id(canonical_name).unwrap_or(table_id);
            if let Some(resolver) = self.catalog.field_resolver(catalog_table_id) {
                resolver.set_field_unique(column_name, true)?;
            }
        }

        self.metadata.flush_table(table_id)?;
        Ok(true)
    }

    /// Register a multi-column UNIQUE index.
    pub fn register_multi_column_unique_index(
        &self,
        table_id: TableId,
        field_ids: &[FieldId],
        index_name: Option<String>,
    ) -> LlkvResult<MultiColumnUniqueRegistration> {
        let registration = self
            .metadata
            .register_multi_column_unique(table_id, field_ids, index_name)?;

        if matches!(registration, MultiColumnUniqueRegistration::Created) {
            self.metadata.flush_table(table_id)?;
        }

        Ok(registration)
    }

    /// Append RecordBatches to a freshly created table, injecting MVCC columns.
    #[allow(clippy::too_many_arguments)]
    pub fn append_batches_with_mvcc(
        &self,
        table: &Table<P>,
        table_columns: &[TableColumn],
        batches: &[RecordBatch],
        creator_txn_id: u64,
        deleted_marker: u64,
        starting_row_id: RowId,
        mvcc_builder: &dyn MvccColumnBuilder,
    ) -> LlkvResult<(RowId, u64)> {
        let mut next_row_id = starting_row_id;
        let mut total_rows: u64 = 0;

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            if batch.num_columns() != table_columns.len() {
                return Err(Error::InvalidArgumentError(format!(
                    "CTAS query returned unexpected column count (expected {}, found {})",
                    table_columns.len(),
                    batch.num_columns()
                )));
            }

            let row_count = batch.num_rows();

            let (row_id_array, created_by_array, deleted_by_array) = mvcc_builder
                .build_insert_columns(row_count, next_row_id, creator_txn_id, deleted_marker);

            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(table_columns.len() + 3);
            arrays.push(row_id_array);
            arrays.push(created_by_array);
            arrays.push(deleted_by_array);

            let mut fields = mvcc_builder.mvcc_fields();

            for (idx, column) in table_columns.iter().enumerate() {
                let array = batch.column(idx).clone();
                let field = mvcc_builder.field_with_metadata(
                    &column.name,
                    column.data_type.clone(),
                    column.nullable,
                    column.field_id,
                );
                arrays.push(array);
                fields.push(field);
            }

            let append_schema = Arc::new(Schema::new(fields));
            let append_batch = RecordBatch::try_new(append_schema, arrays).map_err(Error::Arrow)?;
            table.append(&append_batch)?;

            next_row_id = next_row_id.saturating_add(row_count as u64);
            total_rows = total_rows.saturating_add(row_count as u64);
        }

        Ok((next_row_id, total_rows))
    }

    /// Validate and register foreign keys for a newly created table.
    #[allow(clippy::too_many_arguments)]
    pub fn register_foreign_keys_for_new_table<F>(
        &self,
        table_id: TableId,
        display_name: &str,
        canonical_name: &str,
        table_columns: &[TableColumn],
        specs: &[ForeignKeySpec],
        lookup_table: F,
        timestamp_micros: u64,
    ) -> LlkvResult<Vec<ValidatedForeignKey>>
    where
        F: FnMut(&str) -> LlkvResult<ForeignKeyTableInfo>,
    {
        if specs.is_empty() {
            return Ok(Vec::new());
        }

        let referencing_columns: Vec<ForeignKeyColumn> = table_columns
            .iter()
            .map(|column| ForeignKeyColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                unique: column.unique,
                field_id: column.field_id,
            })
            .collect();

        let referencing_table = ForeignKeyTableInfo {
            display_name: display_name.to_string(),
            canonical_name: canonical_name.to_string(),
            table_id,
            columns: referencing_columns,
        };

        self.metadata.validate_and_register_foreign_keys(
            &referencing_table,
            specs,
            lookup_table,
            timestamp_micros,
        )
    }

    /// Resolve referenced tables for the provided foreign key view definitions.
    pub fn referenced_table_info(
        &self,
        views: &[ForeignKeyView],
    ) -> LlkvResult<Vec<ForeignKeyTableInfo>> {
        let mut results = Vec::with_capacity(views.len());
        for view in views {
            let Some(table_id) = self.catalog.table_id(&view.referenced_table_canonical) else {
                return Err(Error::InvalidArgumentError(format!(
                    "referenced table '{}' does not exist",
                    view.referenced_table_display
                )));
            };

            let Some(resolver) = self.catalog.field_resolver(table_id) else {
                return Err(Error::Internal(format!(
                    "catalog resolver missing for table '{}'",
                    view.referenced_table_display
                )));
            };

            let mut columns = Vec::with_capacity(view.referenced_field_ids.len());
            for field_id in &view.referenced_field_ids {
                let info = resolver.field_info(*field_id).ok_or_else(|| {
                    Error::Internal(format!(
                        "field metadata missing for id {} in table '{}'",
                        field_id, view.referenced_table_display
                    ))
                })?;

                let data_type = self.metadata.column_data_type(table_id, *field_id)?;

                columns.push(ForeignKeyColumn {
                    name: info.display_name.to_string(),
                    data_type,
                    nullable: !info.constraints.primary_key,
                    primary_key: info.constraints.primary_key,
                    unique: info.constraints.unique,
                    field_id: *field_id,
                });
            }

            results.push(ForeignKeyTableInfo {
                display_name: view.referenced_table_display.clone(),
                canonical_name: view.referenced_table_canonical.clone(),
                table_id,
                columns,
            });
        }

        Ok(results)
    }

    /// Return the current metadata snapshot for a table, including column metadata and constraints.
    pub fn table_view(&self, canonical_name: &str) -> LlkvResult<TableView> {
        let table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let (_, field_ids) = self.sorted_user_fields(table_id);
        self.table_view_with_field_ids(table_id, &field_ids)
    }

    /// Produce a read-only view of a table's catalog, including column metadata and constraints.
    pub fn table_column_specs(&self, canonical_name: &str) -> LlkvResult<Vec<ColumnSpec>> {
        let table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let resolver = self
            .catalog
            .field_resolver(table_id)
            .ok_or_else(|| Error::Internal("missing field resolver for table".into()))?;

        let (logical_fields, field_ids) = self.sorted_user_fields(table_id);

        let table_view = self.table_view_with_field_ids(table_id, &field_ids)?;
        let column_metas = table_view.column_metas;
        let constraint_records = table_view.constraint_records;

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
                        if let Some(field_id) = unique.field_ids.first() {
                            metadata_unique_fields.insert(*field_id);
                        }
                    }
                }
                _ => {}
            }
        }

        let mut specs = Vec::with_capacity(field_ids.len());

        for (idx, lfid) in logical_fields.iter().enumerate() {
            let field_id = lfid.field_id();

            let column_name = column_metas
                .get(idx)
                .and_then(|meta| meta.as_ref())
                .and_then(|meta| meta.name.clone())
                .unwrap_or_else(|| format!("col_{}", field_id));

            let fallback_constraints = resolver
                .field_constraints_by_name(&column_name)
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

            let data_type = self.store.data_type(*lfid)?;
            let nullable = !primary_key;

            let mut spec = ColumnSpec::new(column_name.clone(), data_type, nullable)
                .with_primary_key(primary_key)
                .with_unique(unique);

            if let Some(check_expr) = fallback_constraints.check_expr.clone() {
                spec = spec.with_check(Some(check_expr));
            }

            specs.push(spec);
        }

        Ok(specs)
    }

    /// Return the foreign key metadata for the specified table.
    pub fn foreign_key_views(&self, canonical_name: &str) -> LlkvResult<Vec<ForeignKeyView>> {
        let table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        self.metadata.foreign_key_views(&self.catalog, table_id)
    }

    /// Return constraint-related catalog metadata for the specified table.
    pub fn table_constraint_summary(
        &self,
        canonical_name: &str,
    ) -> LlkvResult<TableConstraintSummaryView> {
        let table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let (_, field_ids) = self.sorted_user_fields(table_id);
        let table_meta = self.metadata.table_meta(table_id)?;
        let column_metas = self.metadata.column_metas(table_id, &field_ids)?;
        let constraint_records = self.metadata.constraint_records(table_id)?;
        let multi_column_uniques = self.metadata.multi_column_uniques(table_id)?;

        Ok(TableConstraintSummaryView {
            table_meta,
            column_metas,
            constraint_records,
            multi_column_uniques,
        })
    }

    fn sorted_user_fields(
        &self,
        table_id: TableId,
    ) -> (Vec<llkv_column_map::types::LogicalFieldId>, Vec<FieldId>) {
        let mut logical_fields = self.store.user_field_ids_for_table(table_id);
        logical_fields.sort_by_key(|lfid| lfid.field_id());
        let field_ids = logical_fields
            .iter()
            .map(|lfid| lfid.field_id())
            .collect::<Vec<_>>();

        (logical_fields, field_ids)
    }

    fn table_view_with_field_ids(
        &self,
        table_id: TableId,
        field_ids: &[FieldId],
    ) -> LlkvResult<TableView> {
        self.metadata.table_view(&self.catalog, table_id, field_ids)
    }

    // -------------------------------------------------------------------------
    // Catalog read-only views
    // -------------------------------------------------------------------------

    /// Returns all table names in the catalog.
    pub fn table_names(&self) -> Vec<String> {
        self.catalog.table_names()
    }

    /// Returns the TableId for a canonical table name.
    pub fn table_id(&self, canonical_name: &str) -> Option<TableId> {
        self.catalog.table_id(canonical_name)
    }

    /// Returns a field resolver for the given table.
    pub fn field_resolver(&self, table_id: TableId) -> Option<crate::catalog::FieldResolver> {
        self.catalog.field_resolver(table_id)
    }

    /// Returns a snapshot of the catalog for read-only access.
    pub fn catalog_snapshot(&self) -> crate::catalog::TableCatalogSnapshot {
        self.catalog.snapshot()
    }

    /// Returns a reference to the internal catalog for services that need it.
    /// Note: This is primarily for internal use by services like ConstraintService.
    pub fn catalog(&self) -> &Arc<TableCatalog> {
        &self.catalog
    }
}

fn field_id_for_index(idx: usize) -> LlkvResult<FieldId> {
    FieldId::try_from(idx + 1).map_err(|_| {
        Error::Internal(format!(
            "column index {} exceeded supported field id range",
            idx + 1
        ))
    })
}

#[allow(clippy::unnecessary_wraps)]
fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros() as u64)
        .unwrap_or(0)
}
