use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::Schema;
use llkv_column_map::store::{ColumnStore, GatherNullPolicy, MultiGatherContext};
use llkv_column_map::types::LogicalFieldId;
use llkv_result::Result as LlkvResult;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;

use crate::types::{FieldId, RowId, TableId};

use crate::planner::{materialize_row_window, ProjectionEval};

#[allow(dead_code)]
pub(crate) trait ColumnSliceSet<'a> {
    fn len(&self) -> usize;
    fn column(&self, idx: usize) -> &'a ArrayRef;
    fn columns(&self) -> &'a [ArrayRef];
}

#[allow(dead_code)]
pub(crate) struct ColumnSlices<'a> {
    columns: &'a [ArrayRef],
}

impl<'a> ColumnSlices<'a> {
    pub fn new(columns: &'a [ArrayRef]) -> Self {
        Self { columns }
    }
}

impl<'a> ColumnSliceSet<'a> for ColumnSlices<'a> {
    fn len(&self) -> usize {
        self.columns.len()
    }

    fn column(&self, idx: usize) -> &'a ArrayRef {
        &self.columns[idx]
    }

    fn columns(&self) -> &'a [ArrayRef] {
        self.columns
    }
}

#[allow(dead_code)]
pub(crate) struct RowChunk<'a, C> {
    pub row_ids: &'a UInt64Array,
    pub columns: C,
    pub visibility: Option<&'a BooleanBuffer>,
    record_batch: &'a RecordBatch,
}

#[allow(dead_code)]
impl<'a, C> RowChunk<'a, C> {
    pub fn record_batch(&self) -> &'a RecordBatch {
        self.record_batch
    }

    pub fn to_record_batch(&self) -> RecordBatch {
        self.record_batch.clone()
    }
}

pub(crate) trait RowStream {
    type Columns<'a>: ColumnSliceSet<'a>
    where
        Self: 'a;

    fn schema(&self) -> &Arc<Schema>;

    fn next_chunk<'a>(&'a mut self) -> LlkvResult<Option<RowChunk<'a, Self::Columns<'a>>>>;
}

pub(crate) struct RowStreamBuilder<'table, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: &'table ColumnStore<P>,
    table_id: TableId,
    schema: Arc<Schema>,
    unique_lfids: Arc<Vec<LogicalFieldId>>,
    projection_evals: Arc<Vec<ProjectionEval>>,
    passthrough_fields: Arc<Vec<Option<FieldId>>>,
    unique_index: Arc<FxHashMap<LogicalFieldId, usize>>,
    numeric_fields: Arc<FxHashSet<FieldId>>,
    requires_numeric: bool,
    null_policy: GatherNullPolicy,
    row_ids: Vec<RowId>,
    chunk_size: usize,
    gather_ctx: Option<MultiGatherContext>,
}

impl<'table, P> RowStreamBuilder<'table, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(
        store: &'table ColumnStore<P>,
        table_id: TableId,
        schema: Arc<Schema>,
        unique_lfids: Arc<Vec<LogicalFieldId>>,
        projection_evals: Arc<Vec<ProjectionEval>>,
        passthrough_fields: Arc<Vec<Option<FieldId>>>,
        unique_index: Arc<FxHashMap<LogicalFieldId, usize>>,
        numeric_fields: Arc<FxHashSet<FieldId>>,
        requires_numeric: bool,
        null_policy: GatherNullPolicy,
        row_ids: Vec<RowId>,
        chunk_size: usize,
    ) -> Self {
        Self {
            store,
            table_id,
            schema,
            unique_lfids,
            projection_evals,
            passthrough_fields,
            unique_index,
            numeric_fields,
            requires_numeric,
            null_policy,
            row_ids,
            chunk_size,
            gather_ctx: None,
        }
    }

    pub fn with_gather_context(mut self, ctx: MultiGatherContext) -> Self {
        self.gather_ctx = Some(ctx);
        self
    }

    pub fn build(self) -> LlkvResult<TableRowStream<'table, P>> {
        let RowStreamBuilder {
            store,
            table_id,
            schema,
            unique_lfids,
            projection_evals,
            passthrough_fields,
            unique_index,
            numeric_fields,
            requires_numeric,
            null_policy,
            row_ids,
            chunk_size,
            mut gather_ctx,
        } = self;

        let gather_ctx = if unique_lfids.is_empty() {
            None
        } else if let Some(ctx) = gather_ctx.take() {
            Some(ctx)
        } else {
            Some(store.prepare_gather_context(unique_lfids.as_ref())?)
        };

        let chunk_ranges = if row_ids.is_empty() {
            Vec::new()
        } else {
            (0..row_ids.len())
                .step_by(chunk_size.max(1))
                .map(|start| {
                    let end = (start + chunk_size).min(row_ids.len());
                    (start, end)
                })
                .collect()
        };

        let row_id_array = Arc::new(UInt64Array::from(row_ids));

        Ok(TableRowStream {
            store,
            table_id,
            schema,
            unique_lfids,
            projection_evals,
            passthrough_fields,
            unique_index,
            numeric_fields,
            requires_numeric,
            null_policy,
            row_ids: row_id_array,
            chunk_ranges,
            range_idx: 0,
            gather_ctx,
            current_batch: None,
            current_row_ids: None,
        })
    }
}

pub(crate) struct TableRowStream<'table, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: &'table ColumnStore<P>,
    table_id: TableId,
    schema: Arc<Schema>,
    unique_lfids: Arc<Vec<LogicalFieldId>>,
    projection_evals: Arc<Vec<ProjectionEval>>,
    passthrough_fields: Arc<Vec<Option<FieldId>>>,
    unique_index: Arc<FxHashMap<LogicalFieldId, usize>>,
    numeric_fields: Arc<FxHashSet<FieldId>>,
    requires_numeric: bool,
    null_policy: GatherNullPolicy,
    row_ids: Arc<UInt64Array>,
    chunk_ranges: Vec<(usize, usize)>,
    range_idx: usize,
    gather_ctx: Option<MultiGatherContext>,
    current_batch: Option<RecordBatch>,
    current_row_ids: Option<ArrayRef>,
}

impl<'table, P> RowStream for TableRowStream<'table, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    type Columns<'a> = ColumnSlices<'a> where Self: 'a;

    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_chunk<'a>(&'a mut self) -> LlkvResult<Option<RowChunk<'a, Self::Columns<'a>>>> {
        while self.range_idx < self.chunk_ranges.len() {
            let (start, end) = self.chunk_ranges[self.range_idx];
            self.range_idx += 1;

            let values = self.row_ids.values();
            let window: &[RowId] = &values[start..end];

            let unique_lfids = Arc::clone(&self.unique_lfids);
            let projection_evals = Arc::clone(&self.projection_evals);
            let passthrough_fields = Arc::clone(&self.passthrough_fields);
            let unique_index = Arc::clone(&self.unique_index);
            let numeric_fields = Arc::clone(&self.numeric_fields);
            let requires_numeric = self.requires_numeric;
            let null_policy = self.null_policy;
            let schema = Arc::clone(&self.schema);

            let batch_opt = materialize_row_window(
                self.store,
                self.table_id,
                unique_lfids.as_ref(),
                projection_evals.as_ref(),
                passthrough_fields.as_ref(),
                unique_index.as_ref(),
                numeric_fields.as_ref(),
                requires_numeric,
                null_policy,
                &schema,
                window,
                self.gather_ctx.as_mut(),
            )?;

            let Some(batch) = batch_opt else {
                continue;
            };

            if batch.num_rows() == 0 {
                continue;
            }

            let row_id_slice = self.row_ids.slice(start, end - start);
            self.current_row_ids = Some(Arc::new(row_id_slice) as ArrayRef);
            self.current_batch = Some(batch);

            let row_ids_ref = self
                .current_row_ids
                .as_ref()
                .and_then(|arr| arr.as_any().downcast_ref::<UInt64Array>())
                .expect("row id slice must be UInt64Array");

            let batch_ref = self.current_batch.as_ref().expect("batch must be present");
            let columns = batch_ref.columns();
            let column_set = ColumnSlices::new(columns);

            return Ok(Some(RowChunk {
                row_ids: row_ids_ref,
                columns: column_set,
                visibility: None,
                record_batch: batch_ref,
            }));
        }

        Ok(None)
    }
}

impl<'table, P> TableRowStream<'table, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn into_gather_context(self) -> Option<MultiGatherContext> {
        self.gather_ctx
    }
}
