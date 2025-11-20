use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use llkv_parquet_store::ParquetStore;
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use super::provider::{LlkvTableBuilder, LlkvTableProvider};
use crate::catalog::TableMetadata;
use crate::common::TableEventListener;
use crate::traits::{CatalogBackend, TableBuilder};
use datafusion::datasource::TableProvider;

/// Backend implementation for ParquetStore to work with TableCatalog.
pub struct ParquetStoreBackend<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ParquetStore<P>>,
}

impl<P> ParquetStoreBackend<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(store: Arc<ParquetStore<P>>) -> Self {
        Self { store }
    }
}

impl<P> CatalogBackend for ParquetStoreBackend<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn load_metadata(&self) -> LlkvResult<Vec<TableMetadata>> {
        let table_names = self.store.list_tables();
        let mut metadata_list = Vec::new();

        for name in table_names {
            if let Some(table_id) = self.store.get_table_id(&name)? {
                if let Some(schema) = self.store.get_table_schema(&name)? {
                    // Serialize schema to bytes for TableMetadata
                    let schema_bytes = crate::common::schema_to_bytes(&schema)?;

                    // ParquetStore doesn't expose logical fields or row_ids in the same way
                    // as ColumnStore, so we leave them empty.
                    metadata_list.push(TableMetadata {
                        table_id: table_id.0 as u16,
                        table_name: name,
                        schema_bytes,
                        logical_fields: Vec::new(),
                        row_ids: Vec::new(),
                    });
                }
            }
        }

        Ok(metadata_list)
    }

    fn persist_metadata(&self, _metadata: &[TableMetadata]) -> LlkvResult<()> {
        // ParquetStore manages its own metadata persistence internally.
        // We don't need to do anything here.
        Ok(())
    }

    fn create_table_builder(
        &self,
        _table_id: u64,
        name: &str,
        schema: SchemaRef,
        _listener: Option<Box<dyn TableEventListener>>,
    ) -> LlkvResult<Box<dyn TableBuilder>> {
        // Note: We ignore the passed table_id because ParquetStore assigns its own IDs.
        // This might cause a mismatch if TableCatalog expects to control IDs.
        // However, since we reload metadata from ParquetStore on startup, it should sync up.
        let builder = LlkvTableBuilder::new(Arc::clone(&self.store), name, schema)?;
        Ok(Box::new(builder))
    }

    fn get_table_provider(
        &self,
        table_id: u64,
        schema: SchemaRef,
        _row_ids: &[u64],
    ) -> LlkvResult<Arc<dyn TableProvider>> {
        // We ignore row_ids because ParquetStore manages them internally.
        let provider = LlkvTableProvider::new(
            Arc::clone(&self.store),
            llkv_parquet_store::TableId(table_id),
            schema,
        )?;
        Ok(Arc::new(provider))
    }

    fn drop_table(&self, table_id: u64, _logical_fields: &[u64]) -> LlkvResult<()> {
        self.store
            .drop_table_by_id(llkv_parquet_store::TableId(table_id))
    }
}
