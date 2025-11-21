use std::any::Any;
use std::fmt;
use std::sync::{Arc, RwLock};

use arrow::array::{ArrayRef, UInt64Array, UInt64Builder};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, SendableRecordBatchStream};
use datafusion_datasource::sink::DataSink;
use futures::StreamExt;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::RowId;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::common::{ROW_ID_COLUMN_NAME, TableEventListener};

/// DataSink implementation for LLKV ColumnStore.
pub struct ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    schema: SchemaRef,
    row_ids: Arc<RwLock<Vec<RowId>>>,
    next_row_id: Arc<RwLock<u64>>,
    listener: Option<Arc<dyn TableEventListener>>,
}

impl<P> ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(
        store: Arc<ColumnStore<P>>,
        schema: SchemaRef,
        _table_id: u64,
        row_ids: Arc<RwLock<Vec<RowId>>>,
        listener: Option<Arc<dyn TableEventListener>>,
    ) -> Self {
        // Determine the next row ID based on existing max
        let max_row_id = {
            let rids = row_ids.read().unwrap();
            rids.iter().max().copied().unwrap_or(0)
        };

        Self {
            store,
            schema,
            row_ids,
            next_row_id: Arc::new(RwLock::new(max_row_id + 1)),
            listener,
        }
    }
}

impl<P> fmt::Debug for ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnMapDataSink").finish()
    }
}

impl<P> DisplayAs for ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ColumnMapDataSink")
    }
}

#[async_trait]
impl<P> DataSink for ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn metrics(&self) -> Option<datafusion::physical_plan::metrics::MetricsSet> {
        None
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> DataFusionResult<u64> {
        let mut total_rows = 0;

        while let Some(batch_result) = data.next().await {
            let batch = batch_result?;
            if batch.num_rows() == 0 {
                continue;
            }

            // 1. Get or Generate Row IDs
            let (row_id_col, is_new_rows) = {
                let existing_col = if let Ok(idx) = batch.schema().index_of(ROW_ID_COLUMN_NAME) {
                    Some(batch.column(idx).clone())
                } else {
                    None
                };

                // Check if existing column is valid (non-null)
                let use_existing = if let Some(col) = &existing_col {
                    col.null_count() != col.len()
                } else {
                    false
                };

                if use_existing {
                    (existing_col.unwrap(), false)
                } else {
                    // Generate new row IDs (INSERT path)
                    let num_rows = batch.num_rows();
                    let mut builder = UInt64Builder::with_capacity(num_rows);
                    let mut next_id = self.next_row_id.write().unwrap();
                    // let start_id = *next_id; // Unused

                    for _ in 0..num_rows {
                        builder.append_value(*next_id);
                        *next_id += 1;
                    }
                    (Arc::new(builder.finish()) as ArrayRef, true)
                }
            };

            // 2. Update row_ids list if new
            if is_new_rows {
                let start_id = row_id_col
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0);
                let num_rows = batch.num_rows();

                let mut new_ids = Vec::with_capacity(num_rows);
                {
                    let mut rids = self.row_ids.write().unwrap();
                    for i in 0..num_rows {
                        let rid = start_id + i as u64;
                        rids.push(rid);
                        new_ids.push(rid);
                    }
                }

                if let Some(listener) = &self.listener {
                    listener.on_rows_appended(&new_ids).map_err(|e| {
                        DataFusionError::Internal(format!("failed to notify listener: {e}"))
                    })?;
                }
            }

            // 3. Construct new batch with metadata from self.schema
            let mut new_columns = Vec::with_capacity(self.schema.fields().len() + 1);
            let mut new_fields = Vec::with_capacity(self.schema.fields().len() + 1);

            // Add row_id field (required by ColumnStore)
            new_fields.push(arrow::datatypes::Field::new(
                ROW_ID_COLUMN_NAME,
                arrow::datatypes::DataType::UInt64,
                false,
            ));
            new_columns.push(row_id_col);

            // Identify user columns in the batch (skip row_id if present)
            let batch_schema = batch.schema();

            // Iterate over batch columns and match with self.schema by name
            for (i, field) in batch_schema.fields().iter().enumerate() {
                if field.name() == ROW_ID_COLUMN_NAME {
                    continue;
                }

                if let Ok(idx) = self.schema.index_of(field.name()) {
                    let field_with_meta = self.schema.field(idx);
                    let col = batch.column(i);

                    new_fields.push(field_with_meta.clone());
                    new_columns.push(col.clone());
                }
            }

            let batch_to_append = RecordBatch::try_new(
                Arc::new(arrow::datatypes::Schema::new(new_fields)),
                new_columns,
            )
            .map_err(|e| DataFusionError::Internal(format!("failed to create batch: {e}")))?;

            self.store
                .append(&batch_to_append)
                .map_err(|e| DataFusionError::Internal(format!("LLKV append failed: {e}")))?;

            total_rows += batch_to_append.num_rows() as u64;
        }

        Ok(total_rows as u64)
    }
}
