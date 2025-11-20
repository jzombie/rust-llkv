//! Persistent table catalog for LLKV storage.
//!
//! This module provides a [`TableCatalog`] that persists table metadata (schema,
//! row IDs, field mappings) alongside the columnar data in the same [`Pager`].
//! Tables are stored in a reserved system namespace and can be enumerated,
//! registered, and retrieved across process restarts.
//!
//! # Architecture
//!
//! Table metadata is serialized with [`bitcode`] and stored in system-reserved
//! field IDs within the [`ColumnStore`]. Each table's metadata includes:
//! - Table name
//! - Arrow schema
//! - Logical field IDs
//! - Row ID vector
//!
//! The catalog uses table ID `0` (otherwise reserved) for the system metadata
//! table that indexes all user tables.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use arrow::datatypes::{Schema, SchemaRef};
use arrow::ipc::{
    convert::try_schema_from_flatbuffer_bytes,
    writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions},
};
use bitcode::{Decode, Encode};
use llkv_column_map::types::{LogicalFieldId, RowId, TableId};
use llkv_result::{Error as LlkvError, Result as LlkvResult};

use crate::common::TableEventListener;
use crate::traits::{CatalogBackend, TableBuilder};
use datafusion::datasource::TableProvider;

/// Serializable table metadata for persistence.
#[derive(Debug, Clone, Encode, Decode)]
pub struct TableMetadata {
    pub table_id: TableId,
    pub table_name: String,
    /// Serialized Arrow schema as JSON bytes.
    pub schema_bytes: Vec<u8>,
    /// Logical field IDs for each column.
    pub logical_fields: Vec<u64>,
    /// Row IDs that exist in this table.
    pub row_ids: Vec<RowId>,
}

struct CatalogHook {
    table_name: String,
    catalog: std::sync::Weak<TableCatalog>,
}

impl CatalogHook {
    fn new(table_name: String, catalog: std::sync::Weak<TableCatalog>) -> Self {
        Self {
            table_name,
            catalog,
        }
    }
}

impl TableEventListener for CatalogHook {
    fn on_rows_appended(&self, row_ids: &[u64]) -> LlkvResult<()> {
        if let Some(catalog) = self.catalog.upgrade() {
            catalog.update_table_rows(&self.table_name, row_ids.to_vec())?;
        }
        Ok(())
    }
}

/// Persistent catalog for LLKV tables.
///
/// The catalog stores table metadata in the same pager as the table data,
/// allowing tables to be discovered and reconstructed across process restarts.
/// All operations are thread-safe via interior mutability.
pub struct TableCatalog {
    backend: Box<dyn CatalogBackend>,
    /// In-memory cache of table metadata, indexed by table name.
    tables: RwLock<HashMap<String, TableMetadata>>,
    /// Counter for assigning new table IDs (starts at 1, 0 is reserved).
    next_table_id: RwLock<TableId>,
}

impl TableCatalog {
    /// Create a new table catalog with the given backend.
    pub fn new(backend: Box<dyn CatalogBackend>) -> LlkvResult<Arc<Self>> {
        let mut catalog = Self {
            backend,
            tables: RwLock::new(HashMap::new()),
            next_table_id: RwLock::new(1),
        };
        catalog.load_metadata()?;
        Ok(Arc::new(catalog))
    }

    /// Register a new table with the given schema.
    pub fn create_table(
        self: &Arc<Self>,
        name: &str,
        schema: SchemaRef,
    ) -> LlkvResult<Box<dyn TableBuilder>> {
        // Check if table already exists
        {
            let tables = self.tables.read().unwrap();
            if tables.contains_key(name) {
                return Err(LlkvError::InvalidArgumentError(format!(
                    "table '{}' already exists",
                    name
                )));
            }
        }

        // Allocate a new table ID
        let table_id = {
            let mut next_id = self.next_table_id.write().unwrap();
            let id = *next_id;
            *next_id = next_id
                .checked_add(1)
                .ok_or_else(|| LlkvError::Internal("table ID space exhausted".into()))?;
            id
        };

        // Create the builder via backend
        let listener = Box::new(CatalogHook::new(name.to_string(), Arc::downgrade(self)));
        let builder = self.backend.create_table_builder(
            table_id as u64,
            name,
            schema.clone(),
            Some(listener),
        )?;

        // Store initial metadata (empty row_ids)
        let schema_bytes = Self::serialize_schema(schema.as_ref())?;
        let metadata = TableMetadata {
            table_id,
            table_name: name.to_string(),
            schema_bytes,
            logical_fields: (1..=schema.fields().len() as u32)
                .map(|fid| LogicalFieldId::for_user(table_id, fid).into())
                .collect(),
            row_ids: Vec::new(),
        };

        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(name.to_string(), metadata);
        }

        self.persist_metadata()?;

        Ok(builder)
    }

    /// Get a table provider for an existing table.
    pub fn get_table(self: &Arc<Self>, name: &str) -> LlkvResult<Option<Arc<dyn TableProvider>>> {
        let tables = self.tables.read().unwrap();
        let Some(metadata) = tables.get(name) else {
            return Ok(None);
        };

        let schema = Arc::new(Self::deserialize_schema(&metadata.schema_bytes)?);

        // We need to pass row_ids to the backend so it can construct the provider
        let provider = self.backend.get_table_provider(
            metadata.table_id as u64,
            name,
            schema,
            &metadata.row_ids,
        )?;

        Ok(Some(provider))
    }

    /// Get a builder for an existing table to append data.
    pub fn get_table_builder(self: &Arc<Self>, name: &str) -> LlkvResult<Box<dyn TableBuilder>> {
        let tables = self.tables.read().unwrap();
        let metadata = tables.get(name).ok_or_else(|| {
            LlkvError::InvalidArgumentError(format!("table '{}' not found", name))
        })?;

        let schema = Arc::new(Self::deserialize_schema(&metadata.schema_bytes)?);

        // Create a builder for the existing table
        // We attach the listener to ensure row counts are updated
        let listener = Box::new(CatalogHook::new(name.to_string(), Arc::downgrade(self)));

        self.backend
            .create_table_builder(metadata.table_id as u64, name, schema, Some(listener))
    }

    /// List all table names in the catalog.
    pub fn list_tables(&self) -> Vec<String> {
        let tables = self.tables.read().unwrap();
        tables.keys().cloned().collect()
    }

    /// Remove a table from the catalog and delete its persisted columns.
    pub fn drop_table(&self, name: &str) -> LlkvResult<bool> {
        let metadata = {
            let tables = self.tables.read().unwrap();
            tables.get(name).cloned()
        };

        let Some(metadata) = metadata else {
            return Ok(false);
        };

        self.backend
            .drop_table(metadata.table_id as u64, &metadata.logical_fields)?;

        let mut tables = self.tables.write().unwrap();
        tables.remove(name);
        drop(tables);
        self.persist_metadata()?;
        Ok(true)
    }

    /// Update table metadata after rows have been appended.
    pub fn update_table_rows(&self, name: &str, new_row_ids: Vec<RowId>) -> LlkvResult<()> {
        let mut tables = self.tables.write().unwrap();
        let metadata = tables.get_mut(name).ok_or_else(|| {
            LlkvError::InvalidArgumentError(format!("table '{}' not found", name))
        })?;

        metadata.row_ids.extend(new_row_ids);
        drop(tables);

        self.persist_metadata()
    }

    /// Create a sort index on a column.
    pub fn create_index(
        &self,
        _table_name: &str,
        _column_name: &str,
        _index_name: Option<&str>,
        _unique: bool,
        _if_not_exists: bool,
    ) -> LlkvResult<()> {
        // TODO: Add index support to CatalogBackend trait
        // For now, this is a no-op or returns error
        Err(LlkvError::Internal(
            "Index creation not yet supported in decoupled catalog".into(),
        ))
    }

    /// Create a multi-column index (metadata only).
    pub fn create_multi_column_index(
        &self,
        _table_name: &str,
        _column_names: &[String],
        _index_name: &str,
        _unique: bool,
        _if_not_exists: bool,
    ) -> LlkvResult<()> {
        Ok(())
    }

    /// Load catalog metadata from the backend.
    fn load_metadata(&mut self) -> LlkvResult<()> {
        let all_metadata = self.backend.load_metadata()?;

        let mut tables = self.tables.write().unwrap();
        let mut max_table_id: TableId = 0;

        for metadata in all_metadata {
            if metadata.table_id > max_table_id {
                max_table_id = metadata.table_id;
            }
            tables.insert(metadata.table_name.clone(), metadata);
        }

        *self.next_table_id.write().unwrap() =
            max_table_id.checked_add(1).unwrap_or_else(|| u16::MAX);

        Ok(())
    }

    /// Persist catalog metadata to the backend.
    fn persist_metadata(&self) -> LlkvResult<()> {
        let tables = self.tables.read().unwrap();
        let all_metadata: Vec<TableMetadata> = tables.values().cloned().collect();
        drop(tables);

        self.backend.persist_metadata(&all_metadata)
    }

    fn serialize_schema(schema: &Schema) -> LlkvResult<Vec<u8>> {
        let data_gen = IpcDataGenerator::default();
        let mut tracker = DictionaryTracker::new(true);
        let encoded = data_gen.schema_to_bytes_with_dictionary_tracker(
            schema,
            &mut tracker,
            &IpcWriteOptions::default(),
        );
        Ok(encoded.ipc_message)
    }

    fn deserialize_schema(bytes: &[u8]) -> LlkvResult<Schema> {
        try_schema_from_flatbuffer_bytes(bytes)
            .map_err(|e| LlkvError::Internal(format!("failed to deserialize schema metadata: {e}")))
    }

    /// Create a consistent snapshot of the catalog metadata.
    pub fn snapshot(&self) -> TableCatalogSnapshot {
        let tables = self.tables.read().unwrap();
        TableCatalogSnapshot::new(tables.clone())
    }
}

/// A read-only snapshot of the catalog state.
#[derive(Clone)]
pub struct TableCatalogSnapshot {
    tables: HashMap<String, TableMetadata>,
}

impl TableCatalogSnapshot {
    pub fn new(tables: HashMap<String, TableMetadata>) -> Self {
        Self { tables }
    }

    pub fn table_id(&self, name: &str) -> Option<TableId> {
        self.tables.get(name).map(|m| m.table_id)
    }

    pub fn table_exists(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }

    pub fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }
}
