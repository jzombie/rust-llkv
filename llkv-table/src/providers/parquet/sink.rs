use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Builder};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, SendableRecordBatchStream};
use datafusion_datasource::sink::DataSink;
use futures::StreamExt;
use llkv_parquet_store::{ParquetStore, TableId};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::common::ROW_ID_COLUMN_NAME;

/// DataSink implementation for LLKV ParquetStore.
pub struct ParquetDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ParquetStore<P>>,
    table_id: TableId,
    schema: SchemaRef,
}

impl<P> ParquetDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(store: Arc<ParquetStore<P>>, table_id: TableId, schema: SchemaRef) -> Self {
        Self {
            store,
            table_id,
            schema,
        }
    }
}

impl<P> fmt::Debug for ParquetDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetDataSink")
            .field("table_id", &self.table_id)
            .finish()
    }
}

impl<P> DisplayAs for ParquetDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ParquetDataSink")
    }
}

#[async_trait]
impl<P> DataSink for ParquetDataSink<P>
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
        let mut batches = Vec::new();

        while let Some(batch_result) = data.next().await {
            let batch = batch_result?;
            if batch.num_rows() == 0 {
                continue;
            }
            batches.push(batch);
        }

        if batches.is_empty() {
            return Ok(0);
        }

        // Calculate total rows needed for allocation
        let rows_needing_ids: usize = batches
            .iter()
            .filter(|b| b.column_by_name(ROW_ID_COLUMN_NAME).is_none())
            .map(|b| b.num_rows())
            .sum();

        let mut next_row_id = if rows_needing_ids > 0 {
            self.store
                .allocate_row_ids(self.table_id, rows_needing_ids)
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to allocate row IDs: {}", e))
                })?
        } else {
            0
        };

        let mut batches_to_write = Vec::with_capacity(batches.len());

        for batch in batches {
            let batch_rows = batch.num_rows();
            total_rows += batch_rows as u64;

            if batch.column_by_name(ROW_ID_COLUMN_NAME).is_some() {
                batches_to_write.push(batch);
            } else {
                // Generate row_ids
                let mut builder = UInt64Builder::with_capacity(batch_rows);
                for _ in 0..batch_rows {
                    builder.append_value(next_row_id);
                    next_row_id += 1;
                }
                let row_id_array = Arc::new(builder.finish()) as ArrayRef;

                // Add row_id column to batch
                let mut new_columns = batch.columns().to_vec();
                let mut new_fields = batch.schema().fields().to_vec();

                new_fields.push(Arc::new(arrow::datatypes::Field::new(
                    ROW_ID_COLUMN_NAME,
                    arrow::datatypes::DataType::UInt64,
                    false,
                )));
                new_columns.push(row_id_array);

                let new_schema = Arc::new(arrow::datatypes::Schema::new(new_fields));
                let new_batch = RecordBatch::try_new(new_schema, new_columns)?;
                batches_to_write.push(new_batch);
            }
        }

        self.store
            .append_many(self.table_id, batches_to_write)
            .map_err(|e| DataFusionError::Execution(format!("Failed to append batches: {}", e)))?;

        Ok(total_rows)
    }
}
