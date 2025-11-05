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
use llkv_plan::{DropIndexPlan, ForeignKeySpec, PlanColumnSpec};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use super::table_catalog::{FieldDefinition, TableCatalog};
use crate::constraints::{ConstraintId, ConstraintKind};
use crate::metadata::{MetadataManager, MultiColumnUniqueRegistration, SingleColumnIndexEntry};
use crate::sys_catalog::{
    ColMeta, MultiColumnIndexEntryMeta, SysCatalog, TriggerEntryMeta, TriggerEventMeta,
    TriggerTimingMeta,
};
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

/// Result of attempting to register a single-column index definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SingleColumnIndexRegistration {
    Created { index_name: String },
    AlreadyExists { index_name: String },
}

/// Descriptor for a single-column index resolved from catalog metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleColumnIndexDescriptor {
    pub index_name: String,
    pub table_id: TableId,
    pub canonical_table_name: String,
    pub display_table_name: String,
    pub field_id: FieldId,
    pub column_name: String,
    pub was_unique: bool,
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
    type_registry: Arc<std::sync::RwLock<FxHashMap<String, sqlparser::ast::DataType>>>,
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
            type_registry: Arc::new(std::sync::RwLock::new(FxHashMap::default())),
        }
    }

    // ============================================================================
    // Type Registry Management
    // ============================================================================

    /// Load custom types from system catalog.
    /// Should be called during initialization to restore persisted types.
    pub fn load_types_from_catalog(&self) -> LlkvResult<()> {
        use crate::sys_catalog::SysCatalog;

        let sys_catalog = SysCatalog::new(&self.store);
        match sys_catalog.all_custom_type_metas() {
            Ok(type_metas) => {
                tracing::debug!(
                    "[CATALOG] Loaded {} custom type(s) from catalog",
                    type_metas.len()
                );

                let mut registry = self.type_registry.write().unwrap();
                for type_meta in type_metas {
                    // Parse the base_type_sql back to a DataType
                    if let Ok(parsed_type) = parse_data_type_from_sql(&type_meta.base_type_sql) {
                        registry.insert(type_meta.name.to_lowercase(), parsed_type);
                    } else {
                        tracing::warn!(
                            "[CATALOG] Failed to parse base type SQL for type '{}': {}",
                            type_meta.name,
                            type_meta.base_type_sql
                        );
                    }
                }

                tracing::debug!(
                    "[CATALOG] Type registry initialized with {} type(s)",
                    registry.len()
                );
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    "[CATALOG] Failed to load custom types: {}, starting with empty type registry",
                    e
                );
                Ok(()) // Non-fatal, start with empty registry
            }
        }
    }

    /// Register a custom type alias (CREATE TYPE/DOMAIN).
    pub fn register_type(&self, name: String, data_type: sqlparser::ast::DataType) {
        let mut registry = self.type_registry.write().unwrap();
        registry.insert(name.to_lowercase(), data_type);
    }

    /// Drop a custom type alias (DROP TYPE/DOMAIN).
    pub fn drop_type(&self, name: &str) -> LlkvResult<()> {
        let mut registry = self.type_registry.write().unwrap();
        if registry.remove(&name.to_lowercase()).is_none() {
            return Err(Error::InvalidArgumentError(format!(
                "Type '{}' does not exist",
                name
            )));
        }
        Ok(())
    }

    /// Resolve a type name to its base DataType, recursively following aliases.
    pub fn resolve_type(&self, data_type: &sqlparser::ast::DataType) -> sqlparser::ast::DataType {
        use sqlparser::ast::DataType;

        match data_type {
            DataType::Custom(obj_name, _) => {
                let name = obj_name.to_string().to_lowercase();
                let registry = self.type_registry.read().unwrap();
                if let Some(base_type) = registry.get(&name) {
                    // Recursively resolve in case the base type is also an alias
                    self.resolve_type(base_type)
                } else {
                    // Not a custom type, return as-is
                    data_type.clone()
                }
            }
            // For non-custom types, return as-is
            _ => data_type.clone(),
        }
    }

    // ============================================================================
    // View Management
    // ============================================================================

    /// Create a view by storing its SQL definition in the catalog.
    /// The view will be registered as a table with a view_definition.
    pub fn create_view(
        &self,
        display_name: &str,
        view_definition: String,
        column_specs: Vec<PlanColumnSpec>,
    ) -> LlkvResult<crate::types::TableId> {
        if column_specs.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE VIEW requires at least one column".into(),
            ));
        }

        use crate::sys_catalog::TableMeta;

        // Reserve a new table ID for the view
        let table_id = self.metadata.reserve_table_id()?;

        let created_at_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Create the table metadata with view_definition set
        let table_meta = TableMeta {
            table_id,
            name: Some(display_name.to_string()),
            created_at_micros,
            flags: 0,
            epoch: 0,
            view_definition: Some(view_definition),
        };

        let mut table_columns = Vec::with_capacity(column_specs.len());
        for (idx, spec) in column_specs.iter().enumerate() {
            let field_id = field_id_for_index(idx)?;
            table_columns.push(TableColumn {
                field_id,
                name: spec.name.clone(),
                data_type: spec.data_type.clone(),
                nullable: spec.nullable,
                primary_key: spec.primary_key,
                unique: spec.unique,
                check_expr: spec.check_expr.clone(),
            });
        }

        // Store the metadata and flush to disk
        self.metadata.set_table_meta(table_id, table_meta)?;
        self.metadata
            .apply_column_definitions(table_id, &table_columns, created_at_micros)?;
        self.metadata.flush_table(table_id)?;

        // Register the view in the catalog (no namespace prefix - namespacing handled at runtime session layer)
        self.catalog.register_table(display_name, table_id)?;

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

        tracing::debug!("Created view '{}' with table_id={}", display_name, table_id);
        Ok(table_id)
    }

    /// Check if a table is actually a view by looking at its metadata.
    /// Returns true if the table exists and has a view_definition.
    pub fn is_view(&self, table_id: crate::types::TableId) -> LlkvResult<bool> {
        match self.metadata.table_meta(table_id)? {
            Some(meta) => Ok(meta.view_definition.is_some()),
            None => Ok(false),
        }
    }

    /// Drop a view by removing its metadata and catalog entry.
    pub fn drop_view(&self, canonical_name: &str, table_id: TableId) -> LlkvResult<()> {
        let (_, field_ids) = self.sorted_user_fields(table_id);
        self.metadata.prepare_table_drop(table_id, &field_ids)?;
        self.metadata.flush_table(table_id)?;
        self.metadata.remove_table_state(table_id);

        if let Some(table_id_from_catalog) = self.catalog.table_id(canonical_name) {
            let _ = self.catalog.unregister_table(table_id_from_catalog);
        } else {
            let _ = self.catalog.unregister_table(table_id);
        }

        Ok(())
    }

    // ============================================================================
    // Table Creation
    // ============================================================================

    /// Create a new table using column specifications.
    ///
    /// Reserves table ID from metadata, validates columns, persists schema,
    /// registers in catalog, and returns a Table handle for data operations.
    pub(crate) fn create_table_from_columns(
        &self,
        display_name: &str,
        canonical_name: &str,
        columns: &[PlanColumnSpec],
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
            view_definition: None, // Regular table, not a view
        };

        self.metadata.set_table_meta(table_id, table_meta)?;
        self.metadata
            .apply_column_definitions(table_id, &table_columns, timestamp)?;
        self.metadata.flush_table(table_id)?;

        let table = Table::from_id_and_store(table_id, Arc::clone(&self.store))?;

        // Register table in catalog using the table_id from metadata
        tracing::debug!(
            "[CATALOG_REGISTER] Registering table '{}' (id={}) in catalog @ {:p}",
            display_name,
            table_id,
            &*self.catalog
        );
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

    /// Rename a table across metadata and catalog layers.
    pub fn rename_table(
        &self,
        table_id: TableId,
        current_name: &str,
        new_name: &str,
    ) -> LlkvResult<()> {
        if !current_name.eq_ignore_ascii_case(new_name) && self.catalog.table_id(new_name).is_some()
        {
            return Err(Error::CatalogError(format!(
                "Table '{}' already exists",
                new_name
            )));
        }

        let previous_meta = self.metadata.table_meta(table_id)?;
        let mut prior_snapshot = None;
        if let Some(mut meta) = previous_meta.clone() {
            prior_snapshot = Some(meta.clone());
            meta.name = Some(new_name.to_string());
            self.metadata.set_table_meta(table_id, meta)?;
        }

        if let Err(err) = self.catalog.rename_registered_table(current_name, new_name) {
            if let Some(prior) = prior_snapshot {
                let _ = self.metadata.set_table_meta(table_id, prior);
            }
            return Err(err);
        }

        if let Some(prior) = prior_snapshot.clone()
            && let Err(err) = self.metadata.flush_table(table_id)
        {
            let _ = self.metadata.set_table_meta(table_id, prior);
            let _ = self.catalog.rename_registered_table(new_name, current_name);
            let _ = self.metadata.flush_table(table_id);
            return Err(err);
        }

        Ok(())
    }

    /// Rename a column in a table by updating its metadata.
    pub fn rename_column(
        &self,
        table_id: TableId,
        old_column_name: &str,
        new_column_name: &str,
    ) -> LlkvResult<()> {
        // Get all column metas for this table
        let (_, field_ids) = self.sorted_user_fields(table_id);
        let column_metas = self.metadata.column_metas(table_id, &field_ids)?;

        // Find the column by old name
        let mut found_col: Option<(u32, ColMeta)> = None;
        for (idx, meta_opt) in column_metas.iter().enumerate() {
            if let Some(meta) = meta_opt
                && let Some(name) = &meta.name
                && name.eq_ignore_ascii_case(old_column_name)
            {
                found_col = Some((field_ids[idx], meta.clone()));
                break;
            }
        }

        let (_field_id, mut col_meta) = found_col.ok_or_else(|| {
            Error::InvalidArgumentError(format!("column '{}' not found in table", old_column_name))
        })?;

        // Update the column name
        col_meta.name = Some(new_column_name.to_string());

        // Save to catalog
        let catalog = SysCatalog::new(&self.store);
        catalog.put_col_meta(table_id, &col_meta);

        // Update metadata manager cache
        self.metadata.set_column_meta(table_id, col_meta)?;

        // Update field resolver mapping for this column name change.
        if let Some(resolver) = self.catalog.field_resolver(table_id) {
            resolver.rename_field(old_column_name, new_column_name)?;
        }

        self.metadata.flush_table(table_id)?;

        Ok(())
    }

    /// Alter the data type of a column.
    ///
    /// This updates both the column metadata and the storage layer's data type fingerprint.
    /// Note that actual data conversion is NOT performed - the caller must ensure that
    /// existing data is compatible with the new type (or that no data exists).
    ///
    /// # Arguments
    /// * `table_id` - The table containing the column
    /// * `column_name` - Name of the column to alter
    /// * `new_data_type` - Arrow DataType to set for this column
    pub fn alter_column_type(
        &self,
        table_id: TableId,
        column_name: &str,
        new_data_type: &DataType,
    ) -> LlkvResult<()> {
        // Get all column metas for this table
        let (logical_fields, field_ids) = self.sorted_user_fields(table_id);
        let column_metas = self.metadata.column_metas(table_id, &field_ids)?;

        // Find the column by name
        let mut found_col: Option<(usize, u32, ColMeta)> = None;
        for (idx, meta_opt) in column_metas.iter().enumerate() {
            if let Some(meta) = meta_opt
                && let Some(name) = &meta.name
                && name.eq_ignore_ascii_case(column_name)
            {
                found_col = Some((idx, field_ids[idx], meta.clone()));
                break;
            }
        }

        let (col_idx, _field_id, col_meta) = found_col.ok_or_else(|| {
            Error::InvalidArgumentError(format!("column '{}' not found in table", column_name))
        })?;

        // Update the data type in the storage layer
        let lfid = logical_fields[col_idx];
        self.store.update_data_type(lfid, new_data_type)?;

        // Save metadata to catalog
        let catalog = SysCatalog::new(&self.store);
        catalog.put_col_meta(table_id, &col_meta);

        // Update metadata manager cache
        self.metadata.set_column_meta(table_id, col_meta)?;

        Ok(())
    }

    /// Drop a column from a table by removing its metadata.
    pub fn drop_column(&self, table_id: TableId, column_name: &str) -> LlkvResult<()> {
        // Get all column metas for this table
        let (_, field_ids) = self.sorted_user_fields(table_id);
        let column_metas = self.metadata.column_metas(table_id, &field_ids)?;

        // Find the column by name
        let mut found_col_id: Option<u32> = None;
        for (idx, meta_opt) in column_metas.iter().enumerate() {
            if let Some(meta) = meta_opt
                && let Some(name) = &meta.name
                && name.eq_ignore_ascii_case(column_name)
            {
                found_col_id = Some(field_ids[idx]);
                break;
            }
        }

        let col_id = found_col_id.ok_or_else(|| {
            Error::InvalidArgumentError(format!("column '{}' not found in table", column_name))
        })?;

        // Delete from catalog
        let catalog = SysCatalog::new(&self.store);
        catalog.delete_col_meta(table_id, &[col_id])?;

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
        index_name: Option<String>,
        mark_unique: bool,
        ascending: bool,
        nulls_first: bool,
        if_not_exists: bool,
    ) -> LlkvResult<SingleColumnIndexRegistration> {
        let table_id = table.table_id();
        let existing_indexes = table.list_registered_indexes(field_id)?;
        if existing_indexes.contains(&IndexKind::Sort) {
            let existing_name = self
                .metadata
                .single_column_indexes(table_id)?
                .into_iter()
                .find(|entry| entry.column_id == field_id)
                .map(|entry| entry.index_name)
                .unwrap_or_else(|| column_name.to_string());

            if if_not_exists {
                return Ok(SingleColumnIndexRegistration::AlreadyExists {
                    index_name: existing_name,
                });
            }

            return Err(Error::CatalogError(format!(
                "Index already exists on column '{}' in table '{}'",
                column_name, display_name
            )));
        }

        let index_display_name = match index_name {
            Some(name) => name,
            None => {
                self.generate_single_column_index_name(table_id, canonical_name, column_name)?
            }
        };
        if index_display_name.is_empty() {
            return Err(Error::InvalidArgumentError(
                "Index name must not be empty".into(),
            ));
        }
        let canonical_index_name = index_display_name.to_ascii_lowercase();

        if let Some(existing) = self
            .metadata
            .single_column_index(table_id, &canonical_index_name)?
        {
            if if_not_exists {
                return Ok(SingleColumnIndexRegistration::AlreadyExists {
                    index_name: existing.index_name,
                });
            }

            return Err(Error::CatalogError(format!(
                "Index '{}' already exists on table '{}'",
                existing.index_name, display_name
            )));
        }

        let entry = SingleColumnIndexEntry {
            index_name: index_display_name.clone(),
            canonical_name: canonical_index_name,
            column_id: field_id,
            column_name: column_name.to_string(),
            unique: mark_unique,
            ascending,
            nulls_first,
        };

        self.metadata.put_single_column_index(table_id, entry)?;
        self.metadata.register_sort_index(table_id, field_id)?;

        if mark_unique {
            let catalog_table_id = self.catalog.table_id(canonical_name).unwrap_or(table_id);
            if let Some(resolver) = self.catalog.field_resolver(catalog_table_id) {
                resolver.set_field_unique(column_name, true)?;
            }
        }

        self.metadata.flush_table(table_id)?;

        Ok(SingleColumnIndexRegistration::Created {
            index_name: index_display_name,
        })
    }

    pub fn drop_single_column_index(
        &self,
        plan: DropIndexPlan,
    ) -> LlkvResult<Option<SingleColumnIndexDescriptor>> {
        let canonical_index = plan.canonical_name.to_ascii_lowercase();
        let snapshot = self.catalog.snapshot();

        for canonical_table_name in snapshot.table_names() {
            let Some(table_id) = snapshot.table_id(&canonical_table_name) else {
                continue;
            };

            if let Some(entry) = self
                .metadata
                .single_column_index(table_id, &canonical_index)?
            {
                self.metadata
                    .remove_single_column_index(table_id, &canonical_index)?;

                if entry.unique
                    && let Some(resolver) = self.catalog.field_resolver(table_id)
                {
                    resolver.set_field_unique(&entry.column_name, false)?;
                }

                self.metadata.flush_table(table_id)?;

                let display_table_name = self
                    .metadata
                    .table_meta(table_id)?
                    .and_then(|meta| meta.name)
                    .unwrap_or_else(|| canonical_table_name.clone());

                return Ok(Some(SingleColumnIndexDescriptor {
                    index_name: entry.index_name,
                    table_id,
                    canonical_table_name,
                    display_table_name,
                    field_id: entry.column_id,
                    column_name: entry.column_name,
                    was_unique: entry.unique,
                }));
            }
        }

        if plan.if_exists {
            Ok(None)
        } else {
            Err(Error::CatalogError(format!(
                "Index '{}' does not exist",
                plan.name
            )))
        }
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

    #[allow(clippy::too_many_arguments)]
    pub fn create_trigger(
        &self,
        trigger_display_name: &str,
        canonical_trigger_name: &str,
        table_display_name: &str,
        canonical_table_name: &str,
        timing: TriggerTimingMeta,
        event: TriggerEventMeta,
        for_each_row: bool,
        condition: Option<String>,
        body_sql: String,
        if_not_exists: bool,
    ) -> LlkvResult<bool> {
        let Some(table_id) = self.catalog.table_id(canonical_table_name) else {
            return Err(Error::CatalogError(format!(
                "Table '{}' does not exist",
                table_display_name
            )));
        };

        let table_meta = self.metadata.table_meta(table_id)?;
        let is_view = table_meta
            .as_ref()
            .and_then(|meta| meta.view_definition.as_ref())
            .is_some();

        match timing {
            TriggerTimingMeta::InsteadOf => {
                if !is_view {
                    return Err(Error::InvalidArgumentError(format!(
                        "INSTEAD OF trigger '{}' requires a view target",
                        trigger_display_name
                    )));
                }
            }
            _ => {
                if is_view {
                    return Err(Error::InvalidArgumentError(format!(
                        "Trigger '{}' must use INSTEAD OF when targeting a view",
                        trigger_display_name
                    )));
                }
            }
        }

        if self
            .metadata
            .trigger(table_id, canonical_trigger_name)?
            .is_some()
        {
            if if_not_exists {
                return Ok(false);
            }
            return Err(Error::CatalogError(format!(
                "Trigger '{}' already exists",
                trigger_display_name
            )));
        }

        let entry = TriggerEntryMeta {
            name: trigger_display_name.to_string(),
            canonical_name: canonical_trigger_name.to_string(),
            timing,
            event,
            for_each_row,
            condition,
            body_sql,
        };

        self.metadata.insert_trigger(table_id, entry)?;
        self.metadata.flush_table(table_id)?;
        Ok(true)
    }

    pub fn drop_trigger(
        &self,
        trigger_display_name: &str,
        canonical_trigger_name: &str,
        table_hint_display: Option<&str>,
        table_hint_canonical: Option<&str>,
        if_exists: bool,
    ) -> LlkvResult<bool> {
        let mut candidate_tables: Vec<(TableId, String)> = Vec::new();

        if let Some(canonical_table) = table_hint_canonical {
            match self.catalog.table_id(canonical_table) {
                Some(table_id) => candidate_tables.push((table_id, canonical_table.to_string())),
                None => {
                    if if_exists {
                        return Ok(false);
                    }
                    let display = table_hint_display.unwrap_or(canonical_table);
                    return Err(Error::CatalogError(format!(
                        "Table '{}' does not exist",
                        display
                    )));
                }
            }
        } else {
            let snapshot = self.catalog.snapshot();
            for canonical_table in snapshot.table_names() {
                if let Some(table_id) = snapshot.table_id(&canonical_table) {
                    candidate_tables.push((table_id, canonical_table));
                }
            }
        }

        for (table_id, canonical_table) in candidate_tables {
            if self
                .metadata
                .remove_trigger(table_id, canonical_trigger_name)?
            {
                self.metadata.flush_table(table_id)?;
                return Ok(true);
            } else if table_hint_canonical.is_some()
                && table_hint_canonical
                    .unwrap()
                    .eq_ignore_ascii_case(&canonical_table)
            {
                break;
            }
        }

        if if_exists {
            Ok(false)
        } else {
            Err(Error::CatalogError(format!(
                "Trigger '{}' does not exist",
                trigger_display_name
            )))
        }
    }

    /// Register a multi-column index (unique or non-unique).
    ///
    /// Returns true if the index was created, false if it already exists.
    pub fn register_multi_column_index(
        &self,
        table_id: TableId,
        field_ids: &[FieldId],
        index_name: String,
        unique: bool,
    ) -> LlkvResult<bool> {
        let canonical_name = index_name.to_lowercase();

        // Check if index already exists
        if let Some(_existing) = self
            .metadata
            .get_multi_column_index(table_id, &canonical_name)?
        {
            return Ok(false);
        }

        // Create new index entry
        let entry = MultiColumnIndexEntryMeta {
            index_name: Some(index_name),
            canonical_name,
            column_ids: field_ids.to_vec(),
            unique,
        };

        self.metadata.put_multi_column_index(table_id, entry)?;
        self.metadata.flush_table(table_id)?;

        Ok(true)
    }

    fn generate_single_column_index_name(
        &self,
        table_id: TableId,
        canonical_table_name: &str,
        column_name: &str,
    ) -> LlkvResult<String> {
        let table_token = if canonical_table_name.is_empty() {
            "table".to_string()
        } else {
            canonical_table_name.replace('.', "_")
        };
        let column_token = column_name.to_ascii_lowercase();

        let mut candidate = format!("{}_{}_idx", table_token, column_token);
        let mut suffix: u32 = 1;
        loop {
            let canonical = candidate.to_ascii_lowercase();
            if self
                .metadata
                .single_column_index(table_id, &canonical)?
                .is_none()
            {
                return Ok(candidate);
            }

            candidate = format!("{}_{}_idx{}", table_token, column_token, suffix);
            suffix = suffix.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted unique index name generation space".into())
            })?;
        }
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

        let multi_column_uniques = {
            let catalog = SysCatalog::new(&self.store);
            let all_indexes = catalog.get_multi_column_indexes(table_id)?;
            all_indexes.into_iter().filter(|idx| idx.unique).collect()
        };

        let referencing_table = ForeignKeyTableInfo {
            display_name: display_name.to_string(),
            canonical_name: canonical_name.to_string(),
            table_id,
            columns: referencing_columns,
            multi_column_uniques,
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
                return Err(Error::CatalogError(format!(
                    "Catalog Error: referenced table '{}' does not exist",
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

            let multi_column_uniques = {
                let catalog = SysCatalog::new(&self.store);
                let all_indexes = catalog.get_multi_column_indexes(table_id)?;
                all_indexes.into_iter().filter(|idx| idx.unique).collect()
            };

            results.push(ForeignKeyTableInfo {
                display_name: view.referenced_table_display.clone(),
                canonical_name: view.referenced_table_canonical.clone(),
                table_id,
                columns,
                multi_column_uniques,
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
    pub fn table_column_specs(&self, canonical_name: &str) -> LlkvResult<Vec<PlanColumnSpec>> {
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

            let mut spec = PlanColumnSpec::new(column_name.clone(), data_type, nullable)
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
        tracing::trace!(
            "[TABLE_CONSTRAINT_SUMMARY] Looking up table '{}' in catalog @ {:p}",
            canonical_name,
            &*self.catalog
        );
        let table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            tracing::error!(
                "[TABLE_CONSTRAINT_SUMMARY] Table '{}' NOT FOUND in catalog @ {:p}",
                canonical_name,
                &*self.catalog
            );
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;
        tracing::trace!(
            "[TABLE_CONSTRAINT_SUMMARY] Found table '{}' with id={} in catalog",
            canonical_name,
            table_id
        );

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

    /// Returns all foreign keys that reference the specified table.
    /// Returns a vector of (referencing_table_id, constraint_id) pairs.
    pub fn foreign_keys_referencing(
        &self,
        referenced_table_id: TableId,
    ) -> LlkvResult<Vec<(TableId, ConstraintId)>> {
        self.metadata.foreign_keys_referencing(referenced_table_id)
    }

    /// Returns detailed foreign key views for a specific table.
    /// This includes foreign keys where the specified table is the referencing table.
    pub fn foreign_key_views_for_table(
        &self,
        table_id: TableId,
    ) -> LlkvResult<Vec<ForeignKeyView>> {
        self.metadata.foreign_key_views(&self.catalog, table_id)
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

// TODO: Dedupe (another instance exists in llkv-executor)
#[allow(clippy::unnecessary_wraps)]
fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros() as u64)
        .unwrap_or(0)
}

/// Parse a SQL type string (e.g., "INTEGER") back into a DataType.
fn parse_data_type_from_sql(sql: &str) -> LlkvResult<sqlparser::ast::DataType> {
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    // Try to parse as a simple CREATE DOMAIN statement
    let create_sql = format!("CREATE DOMAIN dummy AS {}", sql);
    let dialect = GenericDialect {};

    match Parser::parse_sql(&dialect, &create_sql) {
        Ok(stmts) if !stmts.is_empty() => {
            if let sqlparser::ast::Statement::CreateDomain(create_domain) = &stmts[0] {
                Ok(create_domain.data_type.clone())
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "Failed to parse type from SQL: {}",
                    sql
                )))
            }
        }
        _ => Err(Error::InvalidArgumentError(format!(
            "Failed to parse type from SQL: {}",
            sql
        ))),
    }
}
