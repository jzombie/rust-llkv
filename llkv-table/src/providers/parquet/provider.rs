//! DataFusion integration helpers for LLKV Parquet storage.
//!
//! This module wires [`llkv-parquet-store`] into a [`datafusion`] [`TableProvider`]
//! so DataFusion's query engine can operate directly on LLKV's persisted Parquet data.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::{Precision, Statistics};
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::sink::DataSinkExec;

use llkv_parquet_store::ParquetStore;
use llkv_parquet_store::TableId;
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::common::{DEFAULT_SCAN_BATCH_SIZE, ROW_ID_COLUMN_NAME};
use crate::traits::TableBuilder;

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
        // Ensure schema has row_id column
        let schema = if schema.field_with_name(ROW_ID_COLUMN_NAME).is_ok() {
            schema
        } else {
            let mut fields = schema.fields().to_vec();
            fields.push(Arc::new(Field::new(
                ROW_ID_COLUMN_NAME,
                DataType::UInt64,
                true,
            )));
            Arc::new(Schema::new(fields))
        };

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

    /// Create a data sink for writing to this table.
    pub fn create_sink(&self) -> Arc<dyn datafusion_datasource::sink::DataSink> {
        Arc::new(crate::providers::parquet::sink::ParquetDataSink::new(
            self.store.clone(),
            self.table_id,
            self.schema.clone(),
        ))
    }

    /// Create a data sink with a specific schema.
    pub fn create_sink_from_schema(
        &self,
        schema: SchemaRef,
    ) -> Arc<dyn datafusion_datasource::sink::DataSink> {
        Arc::new(crate::providers::parquet::sink::ParquetDataSink::new(
            self.store.clone(),
            self.table_id,
            schema,
        ))
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
        let row_id_idx = self.schema.index_of(ROW_ID_COLUMN_NAME).ok();
        let num_fields = self.schema.fields().len();

        let projection_indices = match projection {
            Some(indices) => indices.clone(),
            None => (0..num_fields).collect(),
        };

        let output_schema = Arc::new(self.schema.project(&projection_indices)?);

        let mut store_indices = Vec::new();
        let mut output_to_store_map = Vec::new();

        for &idx in &projection_indices {
            let store_idx = if Some(idx) == row_id_idx {
                0
            } else {
                let user_col_idx = if let Some(ridx) = row_id_idx {
                    if idx > ridx { idx - 1 } else { idx }
                } else {
                    idx
                };
                3 + user_col_idx
            };
            store_indices.push(store_idx);
            output_to_store_map.push(store_idx);
        }

        let mut fetch_indices = store_indices.clone();
        if !fetch_indices.contains(&0) {
            fetch_indices.push(0);
        }
        if !fetch_indices.contains(&1) {
            fetch_indices.push(1);
        }
        if !fetch_indices.contains(&2) {
            fetch_indices.push(2);
        }

        fetch_indices.sort();
        fetch_indices.dedup();

        let mut store_to_batch_map = std::collections::HashMap::new();
        for (i, &store_idx) in fetch_indices.iter().enumerate() {
            store_to_batch_map.insert(store_idx, i);
        }

        let batches = self
            .store
            .scan_parallel(self.table_id, filters, Some(fetch_indices), limit)
            .map_err(map_storage_error)?;

        let projected_batches = batches
            .into_iter()
            .map(|batch| {
                // Filter deleted rows
                let deleted_by_idx = store_to_batch_map[&2];
                let deleted_by_col = batch
                    .column(deleted_by_idx)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("deleted_by should be UInt64Array");

                let mut filter_builder =
                    arrow::array::BooleanBuilder::with_capacity(batch.num_rows());
                for val in deleted_by_col.iter() {
                    let is_active = val.unwrap_or(0) == 0;
                    filter_builder.append_value(is_active);
                }
                let filter_array = filter_builder.finish();

                let filtered_batch = arrow::compute::filter_record_batch(&batch, &filter_array)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

                let mut columns = Vec::with_capacity(output_to_store_map.len());
                for &store_idx in &output_to_store_map {
                    let batch_idx = store_to_batch_map[&store_idx];
                    columns.push(filtered_batch.column(batch_idx).clone());
                }
                RecordBatch::try_new(output_schema.clone(), columns)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
            })
            .collect::<Result<Vec<_>, _>>()?;

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

        // Ensure schema compatibility
        // Note: We don't strictly enforce exact schema match here because DataFusion
        // might have casted columns or reordered them. The sink will handle it.
        // But checking logical equivalence is good practice.
        // self.schema().logically_equivalent_names_and_types(&input.schema())?;

        let sink = Arc::new(crate::providers::parquet::sink::ParquetDataSink::new(
            self.store.clone(),
            self.table_id,
            self.schema.clone(),
        ));

        Ok(Arc::new(DataSinkExec::new(input, sink, None)))
    }
}

fn map_storage_error(err: LlkvError) -> DataFusionError {
    DataFusionError::Execution(format!("llkv storage error: {err}"))
}

fn build_ingest_schema(user_schema: &SchemaRef) -> LlkvResult<SchemaRef> {
    let mut ingest_fields = Vec::with_capacity(user_schema.fields().len() + 3);
    ingest_fields.push(Arc::new(Field::new(
        ROW_ID_COLUMN_NAME,
        DataType::UInt64,
        false,
    )));
    ingest_fields.push(Arc::new(Field::new(
        "created_by_txn",
        DataType::UInt64,
        false,
    )));
    ingest_fields.push(Arc::new(Field::new(
        "deleted_by_txn",
        DataType::UInt64,
        true,
    )));

    for field in user_schema.fields() {
        if field.name() != ROW_ID_COLUMN_NAME {
            ingest_fields.push(field.clone());
        }
    }

    Ok(Arc::new(Schema::new(ingest_fields)))
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

impl<P> TableBuilder for LlkvTableBuilder<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn append_batch(&mut self, batch: &RecordBatch) -> LlkvResult<()> {
        self.append_batch(batch)
    }

    fn finish(self: Box<Self>) -> LlkvResult<Arc<dyn TableProvider>> {
        Ok(Arc::new((*self).finish()?))
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
