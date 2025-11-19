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

use arrow::array::{Array, BinaryArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use bitcode::{Decode, Encode};
use llkv_column_map::store::{ColumnStore, GatherNullPolicy};
use llkv_column_map::types::{LogicalFieldId, RowId, TableId};
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::{LlkvTableBuilder, LlkvTableProvider};

/// Reserved table ID for the system catalog metadata.
const CATALOG_TABLE_ID: TableId = 0;

/// Get the field ID for the catalog metadata blob column (table 0, field 1).
#[inline]
fn catalog_metadata_field() -> LogicalFieldId {
    LogicalFieldId::for_user(CATALOG_TABLE_ID, 1)
}

/// Serializable table metadata for persistence.
#[derive(Debug, Clone, Encode, Decode)]
struct TableMetadata {
    table_id: TableId,
    table_name: String,
    /// Serialized Arrow schema fields.
    schema_fields: Vec<SerializedField>,
    /// Logical field IDs for each column.
    logical_fields: Vec<u64>,
    /// Row IDs that exist in this table.
    row_ids: Vec<RowId>,
}

/// Serializable representation of an Arrow field.
#[derive(Debug, Clone, Encode, Decode)]
struct SerializedField {
    name: String,
    data_type: String,
    nullable: bool,
}

impl SerializedField {
    fn from_arrow_field(field: &Field) -> Self {
        Self {
            name: field.name().to_string(),
            data_type: format!("{:?}", field.data_type()),
            nullable: field.is_nullable(),
        }
    }

    fn to_arrow_field(&self) -> LlkvResult<Field> {
        let data_type = Self::parse_data_type(&self.data_type)?;
        Ok(Field::new(&self.name, data_type, self.nullable))
    }

    /// Parse DataType from debug string representation.
    fn parse_data_type(s: &str) -> LlkvResult<DataType> {
        // Basic type parsing - extend as needed
        match s {
            "Utf8" => Ok(DataType::Utf8),
            "Int32" => Ok(DataType::Int32),
            "Int64" => Ok(DataType::Int64),
            "UInt64" => Ok(DataType::UInt64),
            "Float64" => Ok(DataType::Float64),
            "Boolean" => Ok(DataType::Boolean),
            _ => Err(LlkvError::Internal(format!(
                "unsupported data type for catalog persistence: {}",
                s
            ))),
        }
    }
}

/// Persistent catalog for LLKV tables.
///
/// The catalog stores table metadata in the same pager as the table data,
/// allowing tables to be discovered and reconstructed across process restarts.
/// All operations are thread-safe via interior mutability.
pub struct TableCatalog<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    /// In-memory cache of table metadata, indexed by table name.
    tables: RwLock<HashMap<String, TableMetadata>>,
    /// Counter for assigning new table IDs (starts at 1, 0 is reserved).
    next_table_id: RwLock<TableId>,
}

impl<P> TableCatalog<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Create or open a table catalog backed by the given pager.
    ///
    /// This loads existing table metadata from the catalog system table if it
    /// exists, or initializes an empty catalog if this is the first open.
    pub fn open(pager: Arc<P>) -> LlkvResult<Self> {
        let store = Arc::new(ColumnStore::open(pager)?);
        let mut catalog = Self {
            store,
            tables: RwLock::new(HashMap::new()),
            next_table_id: RwLock::new(1),
        };
        catalog.load_metadata()?;
        Ok(catalog)
    }

    /// Register a new table with the given schema.
    ///
    /// Returns a builder that can be used to append data to the table. The
    /// table metadata is persisted immediately upon registration.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique table name within this catalog
    /// * `schema` - Arrow schema defining the table's columns
    ///
    /// # Errors
    ///
    /// Returns an error if a table with this name already exists or if the
    /// schema contains unsupported data types.
    pub fn create_table(&self, name: &str, schema: SchemaRef) -> LlkvResult<LlkvTableBuilder<P>> {
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

        // Create the builder
        let builder = LlkvTableBuilder::new(Arc::clone(&self.store), table_id, schema.clone())?;

        // Store initial metadata (empty row_ids)
        let metadata = TableMetadata {
            table_id,
            table_name: name.to_string(),
            schema_fields: schema
                .fields()
                .iter()
                .map(|f| SerializedField::from_arrow_field(f.as_ref()))
                .collect(),
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
    ///
    /// Returns `None` if no table with the given name exists.
    pub fn get_table(&self, name: &str) -> LlkvResult<Option<Arc<LlkvTableProvider<P>>>> {
        let tables = self.tables.read().unwrap();
        let Some(metadata) = tables.get(name) else {
            return Ok(None);
        };

        let schema_fields: Vec<Field> = metadata
            .schema_fields
            .iter()
            .map(|f| f.to_arrow_field())
            .collect::<LlkvResult<Vec<_>>>()?;
        let schema = Arc::new(Schema::new(schema_fields));

        let logical_fields: Vec<LogicalFieldId> = metadata
            .logical_fields
            .iter()
            .map(|&id| LogicalFieldId::from(id))
            .collect();

        let provider = LlkvTableProvider::new(
            Arc::clone(&self.store),
            schema,
            logical_fields,
            Arc::new(metadata.row_ids.clone()),
        )?;

        Ok(Some(Arc::new(provider)))
    }

    /// Update table metadata after rows have been appended.
    ///
    /// This should be called after using [`LlkvTableBuilder::append_batch`] to
    /// ensure the catalog's row ID tracking stays synchronized with the actual
    /// data.
    pub fn update_table_rows(&self, name: &str, new_row_ids: Vec<RowId>) -> LlkvResult<()> {
        let mut tables = self.tables.write().unwrap();
        let metadata = tables.get_mut(name).ok_or_else(|| {
            LlkvError::InvalidArgumentError(format!("table '{}' not found", name))
        })?;

        metadata.row_ids.extend(new_row_ids);
        drop(tables);

        self.persist_metadata()
    }

    /// List all table names in the catalog.
    pub fn list_tables(&self) -> Vec<String> {
        let tables = self.tables.read().unwrap();
        tables.keys().cloned().collect()
    }

    /// Get the underlying column store.
    pub fn store(&self) -> &Arc<ColumnStore<P>> {
        &self.store
    }

    /// Load catalog metadata from the system table.
    fn load_metadata(&mut self) -> LlkvResult<()> {
        // Register the catalog metadata column if not already registered
        self.store
            .ensure_column_registered(catalog_metadata_field(), &DataType::Binary)?;

        // Try to load existing catalog data
        // Row ID 1 contains the serialized catalog metadata
        let row_ids = vec![1_u64];
        let field_ids = vec![catalog_metadata_field()];

        match self
            .store
            .gather_rows(&field_ids, &row_ids, GatherNullPolicy::IncludeNulls)
        {
            Ok(batch) => {
                if batch.num_rows() > 0 {
                    let binary_array = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<BinaryArray>()
                        .ok_or_else(|| {
                            LlkvError::Internal("catalog metadata column is not Binary".into())
                        })?;

                    // Check if we actually have data (not null and not empty)
                    if binary_array.is_valid(0) {
                        let bytes = binary_array.value(0);
                        if !bytes.is_empty() {
                            let all_metadata: Vec<TableMetadata> =
                                bitcode::decode(bytes).map_err(|e| {
                                    LlkvError::Internal(format!("failed to decode catalog: {}", e))
                                })?;

                            let mut tables = self.tables.write().unwrap();
                            let mut max_table_id: TableId = 0;

                            for metadata in all_metadata {
                                if metadata.table_id > max_table_id {
                                    max_table_id = metadata.table_id;
                                }
                                tables.insert(metadata.table_name.clone(), metadata);
                            }

                            *self.next_table_id.write().unwrap() =
                                max_table_id.checked_add(1).unwrap_or_else(|| {
                                    // If we overflow, just use max value
                                    u16::MAX
                                });
                        }
                    }
                }
            }
            Err(_) => {
                // No catalog data exists yet, start fresh
            }
        }

        Ok(())
    }

    /// Persist catalog metadata to the system table.
    fn persist_metadata(&self) -> LlkvResult<()> {
        // Serialize all table metadata
        let tables = self.tables.read().unwrap();
        let all_metadata: Vec<TableMetadata> = tables.values().cloned().collect();
        drop(tables);

        let bytes = bitcode::encode(&all_metadata);

        // Register the column if needed (this also registers the row_id shadow column)
        self.store
            .ensure_column_registered(catalog_metadata_field(), &DataType::Binary)?;

        // The batch must have row_id as the first column, followed by the data columns
        // The schema field names must match what ColumnStore expects
        let field_with_metadata = {
            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
                u64::from(catalog_metadata_field()).to_string(),
            );
            Field::new("catalog_metadata", DataType::Binary, false).with_metadata(metadata)
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("rowid", DataType::UInt64, false),
            field_with_metadata,
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::UInt64Array::from(vec![1_u64])),
                Arc::new(BinaryArray::from(vec![bytes.as_slice()])),
            ],
        )?;

        // Use LWW rewrite to update the catalog metadata (row ID 1 will be overwritten)
        self.store.append(&batch)?;

        Ok(())
    }
}
