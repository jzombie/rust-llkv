//! DataFusion integration helpers for LLKV Column Map storage.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::sink::{DataSink, DataSinkExec};
use futures::StreamExt;

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder,
};
use llkv_column_map::store::{ColumnStore, FIELD_ID_META_KEY, GatherNullPolicy};
use llkv_column_map::types::LogicalFieldId;
use llkv_column_map::{
    llkv_for_each_arrow_boolean, llkv_for_each_arrow_numeric, llkv_for_each_arrow_string,
};
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

/// Custom [`TableProvider`] that surfaces LLKV Column Map data to DataFusion.
pub struct ColumnMapTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    schema: SchemaRef,
    table_id: u64,
}

impl<P> fmt::Debug for ColumnMapTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnMapTableProvider")
            .field("schema", &self.schema)
            .field("table_id", &self.table_id)
            .finish()
    }
}

impl<P> ColumnMapTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(store: Arc<ColumnStore<P>>, schema: SchemaRef, table_id: u64) -> Self {
        Self {
            store,
            schema,
            table_id,
        }
    }
}

#[async_trait]
impl<P> TableProvider for ColumnMapTableProvider<P>
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
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // 1. Identify the first column to scan for row IDs.
        // We assume at least one column exists.
        if self.schema.fields().is_empty() {
            return Ok(MemorySourceConfig::try_new_exec(
                &[vec![]],
                self.schema.clone(),
                None,
            )?);
        }

        // Use the first column (index 0) to find all row IDs.
        // In a real implementation, we would pick the best column or use the row_id column if available.
        let first_field_id = LogicalFieldId::for_user(self.table_id as u16, 1); // 1-based field index

        // 2. Scan to collect row IDs.
        let mut collector = RowIdCollector::default();
        let builder = ScanBuilder::new(&self.store, first_field_id).with_row_ids(first_field_id);

        // We need to run the visitor. ScanBuilder::run is synchronous.
        // In an async context, we should probably spawn_blocking, but for now we run it directly.
        builder
            .run(&mut collector)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        let row_ids = collector.row_ids;
        if row_ids.is_empty() {
            return Ok(MemorySourceConfig::try_new_exec(
                &[vec![]],
                self.schema.clone(),
                None,
            )?);
        }

        // 3. Gather rows.
        // We need to map projection indices to LogicalFieldIds.
        let projected_indices = if let Some(proj) = projection {
            proj.clone()
        } else {
            (0..self.schema.fields().len()).collect()
        };

        let projected_field_ids: Vec<LogicalFieldId> = projected_indices
            .iter()
            .map(|&i| LogicalFieldId::for_user(self.table_id as u16, (i + 1) as u64 as u32))
            .collect();

        // Gather in chunks to avoid huge allocations (though MemorySourceConfig will hold them all anyway).
        let chunk_size = 1024;
        let mut batches = Vec::new();

        for chunk in row_ids.chunks(chunk_size) {
            let batch = self
                .store
                .gather_rows(&projected_field_ids, chunk, GatherNullPolicy::IncludeNulls)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            // The batch returned by gather_rows has columns in order of projected_field_ids.
            // We need to ensure the schema matches what DataFusion expects (names, types).
            // gather_rows returns a RecordBatch with a schema derived from the store.
            // We should cast/rename if necessary, but for now assume types match.

            // Construct the projected schema for this batch
            let projected_fields: Vec<_> = projected_indices
                .iter()
                .map(|&i| self.schema.field(i).clone())
                .collect();
            let projected_schema = Arc::new(Schema::new(projected_fields));

            let batch_with_schema =
                RecordBatch::try_new(projected_schema, batch.columns().to_vec())?;
            batches.push(batch_with_schema);
        }

        let partitions = vec![batches];
        let output_schema = if let Some(proj) = projection {
            let fields: Vec<_> = proj.iter().map(|&i| self.schema.field(i).clone()).collect();
            Arc::new(Schema::new(fields))
        } else {
            self.schema.clone()
        };

        Ok(MemorySourceConfig::try_new_exec(
            &partitions,
            output_schema,
            None,
        )?)
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if insert_op != InsertOp::Append {
            return Err(DataFusionError::NotImplemented(format!(
                "{insert_op} not implemented for ColumnMap tables"
            )));
        }

        let sink = ColumnMapDataSink::new(Arc::clone(&self.store), self.schema.clone());

        Ok(Arc::new(DataSinkExec::new(input, Arc::new(sink), None)))
    }
}

struct ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    schema: SchemaRef,
}

impl<P> fmt::Debug for ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnMapDataSink").finish()
    }
}

impl<P> datafusion::physical_plan::display::DisplayAs for ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::display::DisplayFormatType,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match t {
            datafusion::physical_plan::display::DisplayFormatType::Default
            | datafusion::physical_plan::display::DisplayFormatType::Verbose => {
                write!(f, "ColumnMapDataSink")
            }
            datafusion::physical_plan::display::DisplayFormatType::TreeRender => write!(f, ""),
        }
    }
}

impl<P> ColumnMapDataSink<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn new(store: Arc<ColumnStore<P>>, schema: SchemaRef) -> Self {
        Self { store, schema }
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
            // Ensure batch has metadata
            let batch_with_meta =
                RecordBatch::try_new(self.schema.clone(), batch.columns().to_vec())?;
            self.store
                .append(&batch_with_meta)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            total_rows += batch.num_rows() as u64;
        }
        Ok(total_rows)
    }
}

/// Builder for ColumnMap tables.
pub struct ColumnMapTableBuilder<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    store: Arc<ColumnStore<P>>,
    table_id: u64,
    schema: SchemaRef,
}

impl<P> ColumnMapTableBuilder<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(store: Arc<ColumnStore<P>>, table_id: u64, schema: SchemaRef) -> Self {
        // Add metadata to schema fields
        let fields = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut metadata = f.metadata().clone();
                let field_id = LogicalFieldId::for_user(table_id as u16, (i + 1) as u64 as u32);
                metadata.insert(
                    FIELD_ID_META_KEY.to_string(),
                    u64::from(field_id).to_string(),
                );
                f.as_ref().clone().with_metadata(metadata)
            })
            .collect::<Vec<_>>();

        let schema_with_meta = Arc::new(Schema::new(fields));

        Self {
            store,
            table_id,
            schema: schema_with_meta,
        }
    }

    pub fn append_batch(&self, batch: &RecordBatch) -> LlkvResult<()> {
        let batch_with_meta = RecordBatch::try_new(self.schema.clone(), batch.columns().to_vec())?;
        self.store.append(&batch_with_meta)?;
        Ok(())
    }

    pub fn finish(self) -> ColumnMapTableProvider<P> {
        ColumnMapTableProvider::new(self.store, self.schema, self.table_id)
    }
}

#[derive(Default)]
struct RowIdCollector {
    row_ids: Vec<u64>,
}

macro_rules! impl_primitive_with_rids {
    ($base:ident, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype:expr, $native_ty:ty, $cast_expr:expr) => {
        fn $chunk_with_rids_fn(&mut self, _v: &$array_ty, r: &UInt64Array) {
            self.row_ids.extend(r.values());
        }
    };
}

macro_rules! impl_sorted_with_rids {
    ($base:ident, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype:expr, $native_ty:ty, $cast_expr:expr) => {
        fn $run_with_rids_fn(&mut self, _v: &$array_ty, r: &UInt64Array, start: usize, len: usize) {
            self.row_ids.extend(&r.values()[start..start + len]);
        }
    };
}

macro_rules! impl_primitive_empty {
    ($base:ident, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype:expr, $native_ty:ty, $cast_expr:expr) => {
        fn $chunk_fn(&mut self, _a: &$array_ty) {}
    };
}

macro_rules! impl_sorted_empty {
    ($base:ident, $chunk_fn:ident, $chunk_with_rids_fn:ident, $run_fn:ident, $run_with_rids_fn:ident, $array_ty:ty, $physical_ty:ty, $dtype:expr, $native_ty:ty, $cast_expr:expr) => {
        fn $run_fn(&mut self, _a: &$array_ty, _start: usize, _len: usize) {}
    };
}

impl PrimitiveWithRowIdsVisitor for RowIdCollector {
    llkv_for_each_arrow_numeric!(impl_primitive_with_rids);
    llkv_for_each_arrow_boolean!(impl_primitive_with_rids);
    llkv_for_each_arrow_string!(impl_primitive_with_rids);
}

impl PrimitiveSortedWithRowIdsVisitor for RowIdCollector {
    llkv_for_each_arrow_numeric!(impl_sorted_with_rids);
    llkv_for_each_arrow_boolean!(impl_sorted_with_rids);
    llkv_for_each_arrow_string!(impl_sorted_with_rids);
}

impl PrimitiveVisitor for RowIdCollector {
    llkv_for_each_arrow_numeric!(impl_primitive_empty);
    llkv_for_each_arrow_boolean!(impl_primitive_empty);
    llkv_for_each_arrow_string!(impl_primitive_empty);
}

impl PrimitiveSortedVisitor for RowIdCollector {
    llkv_for_each_arrow_numeric!(impl_sorted_empty);
    llkv_for_each_arrow_boolean!(impl_sorted_empty);
    llkv_for_each_arrow_string!(impl_sorted_empty);
}
