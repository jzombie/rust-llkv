use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use arrow::array::{Array, BinaryArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use bitcode;
use llkv_column_map::store::{ColumnStore, GatherNullPolicy};
use llkv_column_map::types::{LogicalFieldId, TableId};
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use super::provider::{ColumnMapTableBuilder, ColumnMapTableProvider};
use crate::catalog::TableMetadata;
use crate::common::{ROW_ID_COLUMN_NAME, TableEventListener};
use crate::traits::{CatalogBackend, TableBuilder};
use datafusion::datasource::TableProvider;

/// Reserved table ID for the system catalog metadata.
const CATALOG_TABLE_ID: TableId = 0;

/// Get the field ID for the catalog metadata blob column (table 0, field 1).
#[inline]
fn catalog_metadata_field() -> LogicalFieldId {
    LogicalFieldId::for_user(CATALOG_TABLE_ID, 1)
}

pub struct ColumnStoreBackend<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
}

impl<P> ColumnStoreBackend<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(store: Arc<ColumnStore<P>>) -> Self {
        Self { store }
    }
}

impl<P> CatalogBackend for ColumnStoreBackend<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn load_metadata(&self) -> LlkvResult<Vec<TableMetadata>> {
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
                            return Ok(all_metadata);
                        }
                    }
                }
            }
            Err(_) => {
                // No catalog data exists yet, start fresh
            }
        }

        Ok(Vec::new())
    }

    fn persist_metadata(&self, metadata: &[TableMetadata]) -> LlkvResult<()> {
        let bytes = bitcode::encode(metadata);

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
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
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

    fn create_table_builder(
        &self,
        table_id: u64,
        _name: &str,
        schema: SchemaRef,
        listener: Option<Arc<dyn TableEventListener>>,
    ) -> LlkvResult<Box<dyn TableBuilder>> {
        let mut builder = ColumnMapTableBuilder::new(Arc::clone(&self.store), table_id, schema);
        if let Some(l) = listener {
            builder = builder.with_listener(l);
        }
        Ok(Box::new(builder))
    }

    fn get_table_provider(
        &self,
        table_id: u64,
        schema: SchemaRef,
        row_ids: &[u64],
        listener: Option<Arc<dyn TableEventListener>>,
    ) -> LlkvResult<Arc<dyn TableProvider>> {
        // Add metadata to schema fields
        let fields = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut metadata = f.metadata().clone();
                let field_id = LogicalFieldId::for_user(table_id as u16, (i + 1) as u64 as u32);
                metadata.insert(
                    llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
                    u64::from(field_id).to_string(),
                );
                f.as_ref().clone().with_metadata(metadata)
            })
            .collect::<Vec<_>>();

        let schema_with_meta = Arc::new(Schema::new(fields));
        let row_ids = Arc::new(RwLock::new(row_ids.to_vec()));
        let provider = ColumnMapTableProvider::new(
            Arc::clone(&self.store),
            schema_with_meta,
            table_id,
            row_ids,
            listener,
        );
        Ok(Arc::new(provider))
    }

    fn drop_table(&self, _table_id: u64, logical_fields: &[u64]) -> LlkvResult<()> {
        for field in logical_fields {
            let field_id = LogicalFieldId::from(*field);
            self.store.remove_column(field_id)?;
        }
        Ok(())
    }
}
