//! Persistent catalog for managing table metadata and Parquet file mappings.

use crate::types::{ColumnStats, FileId, TableId};
use arrow::datatypes::SchemaRef;
use llkv_result::{Error, Result};
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// Reference to a Parquet file stored in the pager.
///
/// Contains metadata about the file for query planning and pruning.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct ParquetFileRef {
    /// Physical key in the pager where the Parquet blob is stored
    pub physical_key: FileId,

    /// Number of rows in this Parquet file
    pub row_count: u64,

    /// Minimum row_id in this file (for pruning)
    pub min_row_id: u64,

    /// Maximum row_id in this file (for pruning)
    pub max_row_id: u64,

    /// Optional column statistics extracted from Parquet metadata
    pub column_stats: Option<HashMap<String, ColumnStats>>,
}

/// Metadata for a single table.
///
/// Tracks the table's schema, Parquet files, and aggregate statistics.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct TableMetadata {
    /// Unique identifier for this table
    pub table_id: TableId,

    /// Arrow schema as IPC bytes (stored separately for bitcode compatibility)
    pub schema_bytes: Vec<u8>,

    /// List of Parquet files that comprise this table, ordered by write time
    pub parquet_files: Vec<ParquetFileRef>,

    /// Total row count across all files (including deleted rows)
    pub total_row_count: u64,

    /// Next row_id to assign for new inserts
    pub next_row_id: u64,
}

impl TableMetadata {
    /// Get the Arrow schema (lazily deserialized from bytes)
    pub fn schema(&self) -> Result<SchemaRef> {
        let reader =
            arrow::ipc::reader::FileReader::try_new(std::io::Cursor::new(&self.schema_bytes), None)
                .map_err(|e| Error::Internal(format!("failed to create IPC reader: {}", e)))?;
        Ok(reader.schema())
    }

    /// Create new TableMetadata with a schema
    pub fn new(table_id: TableId, schema: SchemaRef) -> Result<Self> {
        // Use try_schema_from_flatbuffer_bytes expects flatbuffer format
        // So we'll serialize using Arrow IPC format
        let schema_bytes = {
            let mut buffer = Vec::new();
            let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut buffer, schema.as_ref())
                .map_err(|e| Error::Internal(format!("failed to create IPC writer: {}", e)))?;
            writer
                .finish()
                .map_err(|e| Error::Internal(format!("failed to finish IPC writer: {}", e)))?;
            drop(writer);
            buffer
        };

        Ok(Self {
            table_id,
            schema_bytes,
            parquet_files: Vec::new(),
            total_row_count: 0,
            next_row_id: 0,
        })
    }
}

/// Catalog mapping table names to their metadata.
///
/// The catalog is persisted as a single blob in the pager at a reserved key.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub struct ParquetCatalog {
    /// Map from table name to metadata
    pub(crate) tables: FxHashMap<String, TableMetadata>,

    /// Next available table ID
    pub(crate) next_table_id: u64,
}

impl Default for ParquetCatalog {
    fn default() -> Self {
        Self {
            tables: FxHashMap::default(),
            next_table_id: 1, // Reserve 0 for system use
        }
    }
}

impl ParquetCatalog {
    /// Create a new empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize the catalog to bytes using bitcode.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(bitcode::encode(self))
    }

    /// Deserialize the catalog from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bitcode::decode(bytes).map_err(|e| Error::Internal(format!("bitcode decode failed: {}", e)))
    }

    /// Register a new table in the catalog.
    ///
    /// Returns the assigned `TableId`.
    pub fn create_table(&mut self, name: String, schema: SchemaRef) -> Result<TableId> {
        if self.tables.contains_key(&name) {
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                name
            )));
        }

        let table_id = TableId(self.next_table_id);
        self.next_table_id += 1;

        let metadata = TableMetadata::new(table_id, schema)?;

        self.tables.insert(name, metadata);
        Ok(table_id)
    }

    /// Get metadata for a table by name.
    pub fn get_table(&self, name: &str) -> Result<&TableMetadata> {
        self.tables.get(name).ok_or_else(|| Error::NotFound)
    }

    /// Get mutable metadata for a table by name.
    pub fn get_table_mut(&mut self, name: &str) -> Result<&mut TableMetadata> {
        self.tables.get_mut(name).ok_or_else(|| Error::NotFound)
    }

    /// Get metadata for a table by ID.
    pub fn get_table_by_id(&self, table_id: TableId) -> Result<(&String, &TableMetadata)> {
        self.tables
            .iter()
            .find(|(_, meta)| meta.table_id == table_id)
            .ok_or_else(|| Error::NotFound)
    }

    /// Get mutable metadata for a table by ID.
    pub fn get_table_by_id_mut(
        &mut self,
        table_id: TableId,
    ) -> Result<(&String, &mut TableMetadata)> {
        self.tables
            .iter_mut()
            .find(|(_, meta)| meta.table_id == table_id)
            .ok_or_else(|| Error::NotFound)
    }

    /// Add a Parquet file to a table.
    pub fn add_file_to_table(&mut self, table_id: TableId, file_ref: ParquetFileRef) -> Result<()> {
        let (_, metadata) = self.get_table_by_id_mut(table_id)?;
        metadata.total_row_count += file_ref.row_count;
        metadata.parquet_files.push(file_ref);
        Ok(())
    }

    /// Replace all Parquet files for a table (used during compaction).
    pub fn replace_table_files(
        &mut self,
        table_id: TableId,
        new_files: Vec<ParquetFileRef>,
    ) -> Result<()> {
        let (_, metadata) = self.get_table_by_id_mut(table_id)?;
        let new_total_rows: u64 = new_files.iter().map(|f| f.row_count).sum();
        metadata.parquet_files = new_files;
        metadata.total_row_count = new_total_rows;
        Ok(())
    }

    /// List all table names in the catalog.
    pub fn list_tables(&self) -> Vec<&String> {
        self.tables.keys().collect()
    }

    /// Drop a table from the catalog.
    ///
    /// Returns the table's metadata, including file IDs for cleanup.
    pub fn drop_table(&mut self, name: &str) -> Result<TableMetadata> {
        self.tables.remove(name).ok_or_else(|| Error::NotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_catalog_roundtrip() {
        let mut catalog = ParquetCatalog::new();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let table_id = catalog
            .create_table("users".to_string(), schema.clone())
            .unwrap();

        // Serialize and deserialize
        let bytes = catalog.to_bytes().unwrap();
        let restored = ParquetCatalog::from_bytes(&bytes).unwrap();

        // Verify
        let metadata = restored.get_table("users").unwrap();
        assert_eq!(metadata.table_id, table_id);
        assert_eq!(metadata.schema().unwrap().fields().len(), 2);
    }

    #[test]
    fn test_add_file_to_table() {
        let mut catalog = ParquetCatalog::new();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));

        let table_id = catalog.create_table("test".to_string(), schema).unwrap();

        let file_ref = ParquetFileRef {
            physical_key: 100,
            row_count: 1000,
            min_row_id: 0,
            max_row_id: 999,
            column_stats: None,
        };

        catalog.add_file_to_table(table_id, file_ref).unwrap();

        let metadata = catalog.get_table("test").unwrap();
        assert_eq!(metadata.parquet_files.len(), 1);
        assert_eq!(metadata.total_row_count, 1000);
    }
}
