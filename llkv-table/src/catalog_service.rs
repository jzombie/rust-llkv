//! High-level service for creating tables.
//!
//! Use `CatalogService` to create tables. It coordinates metadata persistence,
//! catalog registration, and storage initialization.

#![forbid(unsafe_code)]

use std::convert::TryFrom;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_plan::{ColumnSpec, ForeignKeySpec};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;

use crate::mvcc;

use crate::catalog::{FieldDefinition, TableCatalog};
use crate::metadata::{MetadataManager, MultiColumnUniqueRegistration};
use crate::table::Table;
use crate::types::{FieldId, RowId, TableColumn, TableId};
use crate::{ForeignKeyColumn, ForeignKeyTableInfo, ForeignKeyView, ValidatedForeignKey};

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

/// Service for creating tables.
///
/// Coordinates metadata persistence (`MetadataManager`), catalog registration
/// (`TableCatalog`), and storage initialization (`ColumnStore`).
#[derive(Clone)]
pub struct CatalogService<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    metadata: Arc<MetadataManager<P>>,
    catalog: Arc<TableCatalog>,
    store: Arc<ColumnStore<P>>,
}

impl<P> CatalogService<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
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
    pub fn register_single_column_index(
        &self,
        canonical_name: &str,
        table_id: TableId,
        field_id: FieldId,
        column_name: &str,
        mark_unique: bool,
    ) -> LlkvResult<()> {
        self.metadata.register_sort_index(table_id, field_id)?;

        if mark_unique {
            let catalog_table_id = self.catalog.table_id(canonical_name).unwrap_or(table_id);
            if let Some(resolver) = self.catalog.field_resolver(catalog_table_id) {
                resolver.set_field_unique(column_name, true)?;
            }
        }

        self.metadata.flush_table(table_id)?;
        Ok(())
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
    pub fn append_batches_with_mvcc(
        &self,
        table: &Table<P>,
        table_columns: &[TableColumn],
        batches: &[RecordBatch],
        creator_txn_id: u64,
        deleted_marker: u64,
        starting_row_id: RowId,
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

            let (row_id_array, created_by_array, deleted_by_array) =
                mvcc::build_insert_mvcc_columns(
                    row_count,
                    next_row_id,
                    creator_txn_id,
                    deleted_marker,
                );

            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(table_columns.len() + 3);
            arrays.push(row_id_array);
            arrays.push(created_by_array);
            arrays.push(deleted_by_array);

            let mut fields = mvcc::build_mvcc_fields();

            for (idx, column) in table_columns.iter().enumerate() {
                let array = batch.column(idx).clone();
                let field = mvcc::build_field_with_metadata(
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
