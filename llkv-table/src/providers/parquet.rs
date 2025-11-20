//! DataFusion integration helpers for LLKV Parquet storage.
//!
//! This module wires [`llkv-parquet-store`] into a [`datafusion`] [`TableProvider`]
//! so DataFusion's query engine can operate directly on LLKV's persisted Parquet data.

use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::{ArrayRef, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::{Precision, Statistics};
use datafusion::common::{DataFusionError, Result as DataFusionResult, SchemaExt};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::display::{DisplayAs, DisplayFormatType};
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::sink::{DataSink, DataSinkExec};
use futures::StreamExt;

use llkv_parquet_store::ParquetStore;
use llkv_parquet_store::TableId;
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::common::{DEFAULT_SCAN_BATCH_SIZE, ROW_ID_COLUMN_NAME};

/// Custom [`TableProvider`] that surfaces LLKV Parquet data to DataFusion.
pub struct LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ParquetStore<P>>,
    table_id: TableId,
    schema: SchemaRef,
    ingest_schema: SchemaRef,
    scan_batch_size: usize,
}

impl<P> fmt::Debug for LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlkvTableProvider")
            .field("table_id", &self.table_id)
            .field("schema", &self.schema)
            .field("scan_batch_size", &self.scan_batch_size)
            .finish()
    }
}

impl<P> LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Construct a table provider.
    pub fn new(
        store: Arc<ParquetStore<P>>,
        table_id: TableId,
        schema: SchemaRef,
    ) -> LlkvResult<Self> {
        Self::with_batch_size(store, table_id, schema, DEFAULT_SCAN_BATCH_SIZE)
    }

    /// Construct a table provider with an explicit scan batch size.
    pub fn with_batch_size(
        store: Arc<ParquetStore<P>>,
        table_id: TableId,
        schema: SchemaRef,
        scan_batch_size: usize,
    ) -> LlkvResult<Self> {
        let ingest_schema = build_ingest_schema(&schema)?;
        Ok(Self {
            store,
            table_id,
            schema,
            ingest_schema,
            scan_batch_size: scan_batch_size.max(1),
        })
    }

    /// Get the ingest schema which includes rowid.
    pub fn ingest_schema(&self) -> SchemaRef {
        Arc::clone(&self.ingest_schema)
    }

    /// Total number of rows currently visible to DataFusion.
    pub fn row_count(&self) -> usize {
        self.store.get_row_count(self.table_id).unwrap_or(0) as usize
    }
}

#[async_trait]
impl<P> TableProvider for LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Map user projection to store projection (shift by 3 for row_id, created_by, deleted_by)
        // and ensure row_id (0) and created_by (1) are included for deduplication.
        let (store_projection, output_schema) = if let Some(indices) = projection {
            let mut store_indices = vec![0, 1]; // Always need row_id and created_by
            store_indices.extend(indices.iter().map(|&i| i + 3));

            let fields: Vec<Field> = indices
                .iter()
                .map(|&i| self.schema.field(i).clone())
                .collect();
            (Some(store_indices), Arc::new(Schema::new(fields)))
        } else {
            // If no projection, we want all user columns.
            // Store has: row_id, created_by, deleted_by, UserCols...
            // We need row_id, created_by for dedup, and then all UserCols.
            // UserCols start at index 3.
            let num_user_cols = self.schema.fields().len();
            let mut store_indices = vec![0, 1];
            store_indices.extend(3..(3 + num_user_cols));

            (Some(store_indices), self.schema.clone())
        };

        let batches = self
            .store
            .scan_parallel(self.table_id, filters, store_projection.clone(), limit)
            .map_err(map_storage_error)?;

        // Remove the extra columns (row_id, created_by) from the result batches
        // and ensure they match output_schema.
        let projected_batches = batches.into_iter().map(|batch| {
            // The batch has [row_id, created_by, requested_user_cols...]
            // We want [requested_user_cols...]
            // So we just slice off the first 2 columns.
            if batch.num_columns() < 2 {
                 return Err(DataFusionError::Internal(format!(
                    "Expected at least 2 columns (row_id, created_by) from store, got {}. Projection: {:?}",
                    batch.num_columns(), store_projection
                )));
            }
            let columns = batch.columns()[2..].to_vec();
            RecordBatch::try_new(output_schema.clone(), columns).map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
        }).collect::<Result<Vec<_>, _>>()?;

        let partitions = vec![projected_batches];
        let exec = MemorySourceConfig::try_new_exec(&partitions, output_schema, None)?;
        Ok(exec)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    fn statistics(&self) -> Option<Statistics> {
        let mut stats = Statistics::new_unknown(self.schema.as_ref());
        stats.num_rows = Precision::Exact(self.row_count());
        Some(stats)
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if insert_op != InsertOp::Append {
            return Err(DataFusionError::NotImplemented(format!(
                "{insert_op} not implemented for LLKV tables"
            )));
        }

        self.schema()
            .logically_equivalent_names_and_types(&input.schema())?;

        let next_row_id = self
            .store
            .get_next_row_id(self.table_id)
            .map_err(map_storage_error)?;

        let sink = LlkvDataSink::new(
            Arc::clone(&self.store),
            self.table_id,
            self.schema.clone(),
            self.ingest_schema(),
            next_row_id,
        );

        Ok(Arc::new(DataSinkExec::new(input, Arc::new(sink), None)))
    }
}

fn map_storage_error(err: LlkvError) -> DataFusionError {
    DataFusionError::Execution(format!("llkv storage error: {err}"))
}

fn build_ingest_schema(user_schema: &SchemaRef) -> LlkvResult<SchemaRef> {
    let mut ingest_fields = Vec::with_capacity(user_schema.fields().len() + 3);
    ingest_fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
    ingest_fields.push(Field::new("created_by_txn", DataType::UInt64, false));
    ingest_fields.push(Field::new("deleted_by_txn", DataType::UInt64, true));
    ingest_fields.extend(user_schema.fields().iter().map(|f| f.as_ref().clone()));
    Ok(Arc::new(Schema::new(ingest_fields)))
}

struct LlkvDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ParquetStore<P>>,
    table_id: TableId,
    schema: SchemaRef,
    ingest_schema: SchemaRef,
    next_row_id: AtomicU64,
}

impl<P> fmt::Debug for LlkvDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlkvDataSink")
            .field("table_id", &self.table_id)
            .finish()
    }
}

impl<P> DisplayAs for LlkvDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "LlkvDataSink")
            }
            DisplayFormatType::TreeRender => write!(f, ""),
        }
    }
}

impl<P> LlkvDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn new(
        store: Arc<ParquetStore<P>>,
        table_id: TableId,
        schema: SchemaRef,
        ingest_schema: SchemaRef,
        next_row_id: u64,
    ) -> Self {
        Self {
            store,
            table_id,
            schema,
            ingest_schema,
            next_row_id: AtomicU64::new(next_row_id),
        }
    }

    fn prepare_ingest_batch(&self, batch: &RecordBatch) -> LlkvResult<RecordBatch> {
        if batch.num_rows() == 0 {
            return Ok(RecordBatch::new_empty(self.ingest_schema.clone()));
        }

        let num_rows = batch.num_rows();
        let start = self
            .next_row_id
            .fetch_add(num_rows as u64, Ordering::SeqCst);

        let mut row_id_builder = UInt64Builder::with_capacity(num_rows);
        for offset in 0..num_rows {
            row_id_builder.append_value(start + offset as u64);
        }
        let row_id_array = Arc::new(row_id_builder.finish()) as ArrayRef;

        let mut columns = Vec::with_capacity(batch.num_columns() + 3);
        columns.push(row_id_array);

        // Add MVCC columns (txn_id=1, deleted=null)
        let created_by = Arc::new(UInt64Array::from(vec![1; num_rows]));
        let deleted_by = Arc::new(UInt64Array::from(vec![None; num_rows]));
        columns.push(created_by);
        columns.push(deleted_by);

        columns.extend(batch.columns().iter().cloned());

        let ingest_batch = RecordBatch::try_new_with_options(
            self.ingest_schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )?;

        Ok(ingest_batch)
    }
}

#[async_trait]
impl<P> DataSink for LlkvDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> DataFusionResult<u64> {
        let mut total_rows = 0;
        while let Some(batch) = data.next().await.transpose()? {
            if batch.num_rows() == 0 {
                continue;
            }
            let ingest_batch = self
                .prepare_ingest_batch(&batch)
                .map_err(map_storage_error)?;
            self.store
                .append_many(self.table_id, vec![ingest_batch])
                .map_err(map_storage_error)?;
            total_rows += batch.num_rows() as u64;
        }
        Ok(total_rows)
    }
}

/// Builder that registers LLKV columns, appends Arrow batches, and produces a
/// [`LlkvTableProvider`].
pub struct LlkvTableBuilder<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ParquetStore<P>>,
    table_id: TableId,
    user_schema: SchemaRef,
    ingest_schema: SchemaRef,
    next_row_id: u64,
}

impl<P> LlkvTableBuilder<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Create a builder for the provided schema.
    /// This creates a new table in the store.
    pub fn new(
        store: Arc<ParquetStore<P>>,
        table_name: &str,
        schema: SchemaRef,
    ) -> LlkvResult<Self> {
        let ingest_schema = build_ingest_schema(&schema)?;
        let table_id = store.create_table(table_name, ingest_schema.clone())?;

        Ok(Self {
            store,
            table_id,
            user_schema: schema,
            ingest_schema,
            next_row_id: 1,
        })
    }

    /// Append a [`RecordBatch`] whose schema matches the user-facing schema.
    pub fn append_batch(&mut self, batch: &RecordBatch) -> LlkvResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }

        self.validate_batch_schema(batch)?;

        let num_rows = batch.num_rows();
        let mut row_id_builder = UInt64Builder::with_capacity(num_rows);
        for _ in 0..num_rows {
            row_id_builder.append_value(self.next_row_id);
            self.next_row_id += 1;
        }
        let row_id_array = Arc::new(row_id_builder.finish()) as ArrayRef;

        let mut columns = Vec::with_capacity(batch.num_columns() + 3);
        columns.push(row_id_array);

        // Add MVCC columns (txn_id=1, deleted=null)
        let created_by = Arc::new(UInt64Array::from(vec![1; num_rows]));
        let deleted_by = Arc::new(UInt64Array::from(vec![None; num_rows]));
        columns.push(created_by);
        columns.push(deleted_by);

        columns.extend(batch.columns().iter().cloned());

        let ingest_batch = RecordBatch::try_new(self.ingest_schema.clone(), columns)?;
        self.store.append_many(self.table_id, vec![ingest_batch])
    }

    /// Finish ingestion and build a [`LlkvTableProvider`].
    pub fn finish(self) -> LlkvResult<LlkvTableProvider<P>> {
        LlkvTableProvider::new(self.store, self.table_id, self.user_schema)
    }

    fn validate_batch_schema(&self, batch: &RecordBatch) -> LlkvResult<()> {
        if batch.schema().fields().len() != self.user_schema.fields().len() {
            return Err(LlkvError::InvalidArgumentError(format!(
                "expected {} columns but received {}",
                self.user_schema.fields().len(),
                batch.schema().fields().len()
            )));
        }

        for (expected, actual) in self
            .user_schema
            .fields()
            .iter()
            .zip(batch.schema().fields().iter())
        {
            if expected != actual {
                return Err(LlkvError::InvalidArgumentError(format!(
                    "column mismatch: expected {expected:?}, received {actual:?}"
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray, UInt64Array};
    use arrow::util::pretty::pretty_format_batches;
    use datafusion::prelude::SessionContext;
    use llkv_storage::pager::MemPager;

    fn build_demo_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::UInt64, false),
            Field::new("score", DataType::Int32, false),
            Field::new("status", DataType::Utf8, true),
        ]));

        let user_ids = Arc::new(UInt64Array::from(vec![1_u64, 2, 3]));
        let scores = Arc::new(Int32Array::from(vec![42, 7, 99]));
        let statuses = Arc::new(StringArray::from(vec![Some("active"), None, Some("vip")]));

        RecordBatch::try_new(schema, vec![user_ids, scores, statuses]).expect("batch")
    }

    #[tokio::test]
    async fn datafusion_scans_llkv_provider() {
        let pager = Arc::new(MemPager::default());
        let store = Arc::new(ParquetStore::open(Arc::clone(&pager)).expect("store"));

        let schema = build_demo_batch().schema();
        let mut builder =
            LlkvTableBuilder::new(Arc::clone(&store), "llkv_demo", schema).expect("builder");
        builder.append_batch(&build_demo_batch()).expect("append");
        let provider = builder.finish().expect("provider");

        let ctx = SessionContext::new();
        ctx.register_table("llkv_demo", Arc::new(provider))
            .expect("register");

        let df = ctx
            .sql("SELECT user_id, score FROM llkv_demo WHERE score > 10 ORDER BY score DESC")
            .await
            .expect("sql");
        let results = df.collect().await.expect("collect");
        let formatted = pretty_format_batches(&results).expect("format").to_string();

        let expected = vec![
            "+---------+-------+",
            "| user_id | score |",
            "+---------+-------+",
            "| 3       | 99    |",
            "| 1       | 42    |",
            "+---------+-------+",
        ]
        .join("\n");

        assert_eq!(formatted.trim(), expected.trim());
    }
}
