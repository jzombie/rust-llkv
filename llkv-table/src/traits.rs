use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::datasource::TableProvider;
use llkv_result::Result;
use std::sync::Arc;

use crate::catalog::TableMetadata;
use crate::common::TableEventListener;

/// Trait for building a table (appending data and finishing).
pub trait TableBuilder: Send + Sync {
    /// Append a batch of data to the table being built.
    fn append_batch(&mut self, batch: &RecordBatch) -> Result<()>;

    /// Finish building and return a TableProvider.
    fn finish(self: Box<Self>) -> Result<Arc<dyn TableProvider>>;
}

/// Trait for storage backends that support the TableCatalog.
pub trait CatalogBackend: Send + Sync {
    /// Load all persisted table metadata.
    fn load_metadata(&self) -> Result<Vec<TableMetadata>>;

    /// Persist all table metadata.
    fn persist_metadata(&self, metadata: &[TableMetadata]) -> Result<()>;

    /// Create a builder for a new table.
    fn create_table_builder(
        &self,
        table_id: u64,
        name: &str,
        schema: SchemaRef,
        listener: Option<Box<dyn TableEventListener>>,
    ) -> Result<Box<dyn TableBuilder>>;

    /// Get a provider for an existing table.
    fn get_table_provider(
        &self,
        table_id: u64,
        name: &str,
        schema: SchemaRef,
        row_ids: &[u64],
    ) -> Result<Arc<dyn TableProvider>>;

    /// Drop a table from storage.
    fn drop_table(&self, table_id: u64, logical_fields: &[u64]) -> Result<()>;
}
