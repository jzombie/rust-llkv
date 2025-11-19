//! DataFusion integration helpers for LLKV storage.
//!
//! This crate wires [`llkv-column-map`] and [`llkv-storage`] into a
//! [`datafusion`] [`TableProvider`] so DataFusion's query engine can operate
//! directly on LLKV's persisted columnar data. The integration is intentionally
//! minimal—writes flow through [`ColumnStore::append`], while reads rely on the
//! store's projection utilities to materialize [`RecordBatch`]es for DataFusion.
//!
//! The primary entry points are:
//! - [`LlkvTableBuilder`]: registers logical columns inside a [`ColumnStore`],
//!   appends Arrow batches (with automatic `rowid` management), and tracks the
//!   row IDs that back each insert.
//! - [`LlkvTableProvider`]: implements DataFusion's [`TableProvider`] by
//!   gathering LLKV rows in scan-sized chunks and surfacing them through a
//!   [`MemorySourceConfig`].
//! - [`LlkvQueryPlanner`]: intercepts `DELETE` statements during physical
//!   planning and executes them through LLKV's storage layer.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::{Precision, Statistics};
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion_datasource::memory::MemorySourceConfig;
use llkv_column_map::store::{
    ColumnStore, FIELD_ID_META_KEY, GatherNullPolicy, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::{LogicalFieldId, RowId, TableId};
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

const DEFAULT_SCAN_BATCH_SIZE: usize = 1024;

/// Custom [`TableProvider`] that surfaces LLKV column-map data to DataFusion.
pub struct LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    schema: SchemaRef,
    logical_fields: Vec<LogicalFieldId>,
    row_ids: Arc<Vec<RowId>>,
    scan_batch_size: usize,
}

impl<P> fmt::Debug for LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlkvTableProvider")
            .field("schema", &self.schema)
            .field("row_count", &self.row_ids.len())
            .field("scan_batch_size", &self.scan_batch_size)
            .finish()
    }
}

impl<P> LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Construct a table provider with the default scan batch size.
    pub fn new(
        store: Arc<ColumnStore<P>>,
        schema: SchemaRef,
        logical_fields: Vec<LogicalFieldId>,
        row_ids: Arc<Vec<RowId>>,
    ) -> LlkvResult<Self> {
        Self::with_batch_size(
            store,
            schema,
            logical_fields,
            row_ids,
            DEFAULT_SCAN_BATCH_SIZE,
        )
    }

    /// Construct a table provider with an explicit scan batch size.
    pub fn with_batch_size(
        store: Arc<ColumnStore<P>>,
        schema: SchemaRef,
        logical_fields: Vec<LogicalFieldId>,
        row_ids: Arc<Vec<RowId>>,
        scan_batch_size: usize,
    ) -> LlkvResult<Self> {
        if schema.fields().len() != logical_fields.len() {
            return Err(LlkvError::InvalidArgumentError(
                "schema field count must match logical field bindings".into(),
            ));
        }

        for fid in &logical_fields {
            if !store.has_field(*fid) {
                return Err(LlkvError::InvalidArgumentError(format!(
                    "logical field {fid:?} is not registered in the column store"
                )));
            }
        }

        Ok(Self {
            store,
            schema,
            logical_fields,
            row_ids,
            scan_batch_size: scan_batch_size.max(1),
        })
    }

    /// Total number of rows currently visible to DataFusion.
    pub fn row_count(&self) -> usize {
        self.row_ids.len()
    }

    fn slice_row_ids(&self, limit: Option<usize>) -> &[RowId] {
        let rows = self.row_ids.as_slice();
        match limit {
            Some(requested) => &rows[..requested.min(rows.len())],
            None => rows,
        }
    }

    fn build_projection(
        &self,
        projection: Option<&Vec<usize>>,
    ) -> DataFusionResult<(SchemaRef, Vec<LogicalFieldId>)> {
        if let Some(indices) = projection {
            let mut fields = Vec::with_capacity(indices.len());
            let mut logical = Vec::with_capacity(indices.len());
            for &idx in indices {
                if idx >= self.logical_fields.len() {
                    return Err(DataFusionError::Plan(format!(
                        "projection index {idx} is out of bounds for schema with {} columns",
                        self.logical_fields.len()
                    )));
                }
                fields.push(self.schema.field(idx).clone());
                logical.push(self.logical_fields[idx]);
            }
            Ok((Arc::new(Schema::new(fields)), logical))
        } else {
            Ok((self.schema.clone(), self.logical_fields.clone()))
        }
    }

    fn gather_batches(
        &self,
        field_ids: &[LogicalFieldId],
        schema: SchemaRef,
        row_ids: &[RowId],
    ) -> LlkvResult<Vec<RecordBatch>> {
        if row_ids.is_empty() {
            return Ok(vec![RecordBatch::new_empty(schema)]);
        }

        // Handle empty projection (no columns requested, but rows are needed for cardinality)
        if field_ids.is_empty() {
            let mut batches = Vec::new();
            for chunk in row_ids.chunks(self.scan_batch_size) {
                // Create a RecordBatch with no columns but with row_count set
                let options = RecordBatchOptions::new().with_row_count(Some(chunk.len()));
                let batch = RecordBatch::try_new_with_options(schema.clone(), vec![], &options)?;
                batches.push(batch);
            }
            return Ok(batches);
        }

        let mut batches = Vec::new();
        for chunk in row_ids.chunks(self.scan_batch_size) {
            let llkv_batch =
                self.store
                    .gather_rows(field_ids, chunk, GatherNullPolicy::IncludeNulls)?;
            let columns: Vec<ArrayRef> = llkv_batch
                .columns()
                .iter()
                .map(|array| Arc::clone(array))
                .collect();
            let batch = RecordBatch::try_new(schema.clone(), columns)?;
            batches.push(batch);
        }

        Ok(batches)
    }

    fn map_storage_error(err: LlkvError) -> DataFusionError {
        DataFusionError::Execution(format!("llkv storage error: {err}"))
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
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        let (projected_schema, field_ids) = self.build_projection(projection)?;
        let rows = self.slice_row_ids(limit);
        let batches = self
            .gather_batches(&field_ids, projected_schema.clone(), rows)
            .map_err(Self::map_storage_error)?;
        let partitions = vec![batches];
        let exec = MemorySourceConfig::try_new_exec(&partitions, projected_schema, None)?;
        Ok(exec)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }

    fn statistics(&self) -> Option<Statistics> {
        let mut stats = Statistics::new_unknown(self.schema.as_ref());
        stats.num_rows = Precision::Exact(self.row_ids.len());
        Some(stats)
    }
}

impl<P> LlkvTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Override the default scan batch size.
    pub fn with_scan_batch_size(mut self, batch_size: usize) -> Self {
        self.scan_batch_size = batch_size.max(1);
        self
    }
}

/// Builder that registers LLKV columns, appends Arrow batches, and produces a
/// [`LlkvTableProvider`].
pub struct LlkvTableBuilder<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    _table_id: TableId,
    user_schema: SchemaRef,
    ingest_schema: SchemaRef,
    logical_fields: Vec<LogicalFieldId>,
    row_ids: Vec<RowId>,
    next_row_id: RowId,
    scan_batch_size: usize,
}

impl<P> LlkvTableBuilder<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Create a builder for the provided schema. Field IDs are assigned in
    /// declaration order starting at `1`.
    pub fn new(
        store: Arc<ColumnStore<P>>,
        table_id: TableId,
        schema: SchemaRef,
    ) -> LlkvResult<Self> {
        if table_id == 0 {
            return Err(LlkvError::reserved_table_id(table_id));
        }

        let fields: Vec<Field> = schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        let mut logical_fields = Vec::with_capacity(fields.len());
        let mut ingest_fields = Vec::with_capacity(fields.len() + 1);
        ingest_fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (idx, field) in fields.iter().enumerate() {
            let fid = LogicalFieldId::for_user(table_id, (idx as u32) + 1);
            store.ensure_column_registered(fid, field.data_type())?;

            let mut metadata = field.metadata().clone();
            metadata.insert(FIELD_ID_META_KEY.to_string(), u64::from(fid).to_string());
            let field_with_meta =
                Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                    .with_metadata(metadata);

            ingest_fields.push(field_with_meta);
            logical_fields.push(fid);
        }

        Ok(Self {
            store,
            _table_id: table_id,
            user_schema: schema,
            ingest_schema: Arc::new(Schema::new(ingest_fields)),
            logical_fields,
            row_ids: Vec::new(),
            next_row_id: 1,
            scan_batch_size: DEFAULT_SCAN_BATCH_SIZE,
        })
    }

    /// Tune the chunk size used when scanning through the resulting provider.
    pub fn with_scan_batch_size(mut self, batch_size: usize) -> Self {
        self.scan_batch_size = batch_size.max(1);
        self
    }

    /// Append a [`RecordBatch`] whose schema matches the user-facing schema.
    pub fn append_batch(&mut self, batch: &RecordBatch) -> LlkvResult<()> {
        self.validate_batch_schema(batch)?;
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(());
        }

        let mut builder = UInt64Builder::with_capacity(num_rows);
        self.row_ids.reserve(num_rows);
        for _ in 0..num_rows {
            let row_id = self.next_row_id;
            builder.append_value(row_id);
            self.row_ids.push(row_id);
            self.next_row_id = self
                .next_row_id
                .checked_add(1)
                .ok_or_else(|| LlkvError::Internal("row id space exhausted".into()))?;
        }
        let row_id_array = Arc::new(builder.finish()) as ArrayRef;

        let mut columns = Vec::with_capacity(batch.num_columns() + 1);
        columns.push(row_id_array);
        columns.extend(batch.columns().iter().cloned());

        let ingest_batch = RecordBatch::try_new(self.ingest_schema.clone(), columns)?;
        self.store.append(&ingest_batch)
    }

    /// Finish ingestion and build a [`LlkvTableProvider`].
    pub fn finish(self) -> LlkvResult<LlkvTableProvider<P>> {
        LlkvTableProvider::with_batch_size(
            self.store,
            self.user_schema,
            self.logical_fields,
            Arc::new(self.row_ids),
            self.scan_batch_size,
        )
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
        let store = Arc::new(ColumnStore::open(Arc::clone(&pager)).expect("store"));

        let schema = build_demo_batch().schema();
        let mut builder = LlkvTableBuilder::new(Arc::clone(&store), 1, schema).expect("builder");
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
