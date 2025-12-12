use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::Schema;
use croaring::Treemap;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_compute::analysis::computed_expr_requires_numeric;
use llkv_compute::eval::{NumericArrayMap as ComputeNumericArrayMap, ScalarEvaluator};
use llkv_compute::projection::{ComputedLiteralInfo, synthesize_computed_literal_array};
use llkv_expr::ScalarExpr;
use llkv_result::Result as LlkvResult;
use llkv_types::{FieldId, LogicalFieldId, RowId, TableId};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use crate::ScanStorage;

pub type NumericArrayMap = ComputeNumericArrayMap<FieldId>;

pub enum RowIdSource {
    Bitmap(Treemap),
    Vector(Vec<RowId>),
}

impl From<Treemap> for RowIdSource {
    fn from(bitmap: Treemap) -> Self {
        RowIdSource::Bitmap(bitmap)
    }
}

impl From<Vec<RowId>> for RowIdSource {
    fn from(vector: Vec<RowId>) -> Self {
        RowIdSource::Vector(vector)
    }
}

impl From<&Treemap> for RowIdSource {
    fn from(bitmap: &Treemap) -> Self {
        RowIdSource::Vector(bitmap.iter().collect())
    }
}

impl From<&[RowId]> for RowIdSource {
    fn from(slice: &[RowId]) -> Self {
        RowIdSource::Vector(slice.to_vec())
    }
}

impl From<&Vec<RowId>> for RowIdSource {
    fn from(vec: &Vec<RowId>) -> Self {
        RowIdSource::Vector(vec.clone())
    }
}

pub trait ColumnSliceSet<'a> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn column(&self, idx: usize) -> &'a ArrayRef;
    fn columns(&self) -> &'a [ArrayRef];
}

pub struct ColumnSlices<'a> {
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

    fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    fn column(&self, idx: usize) -> &'a ArrayRef {
        &self.columns[idx]
    }

    fn columns(&self) -> &'a [ArrayRef] {
        self.columns
    }
}

pub struct RowChunk<'a, C> {
    pub row_ids: Option<&'a UInt64Array>,
    pub columns: C,
    pub visibility: Option<&'a BooleanBuffer>,
    record_batch: &'a RecordBatch,
}

impl<'a, C> RowChunk<'a, C> {
    pub fn record_batch(&self) -> &'a RecordBatch {
        self.record_batch
    }

    pub fn to_record_batch(&self) -> RecordBatch {
        self.record_batch.clone()
    }

    pub fn row_ids(&self) -> Option<&'a UInt64Array> {
        self.row_ids
    }
}

pub trait RowStream {
    type Columns<'a>: ColumnSliceSet<'a>
    where
        Self: 'a;

    fn schema(&self) -> &Arc<Schema>;

    fn next_chunk<'a>(&'a mut self) -> LlkvResult<Option<RowChunk<'a, Self::Columns<'a>>>>;
}

#[derive(Clone)]
pub struct ColumnProjectionInfo {
    pub logical_field_id: LogicalFieldId,
    pub data_type: arrow::datatypes::DataType,
    pub output_name: String,
}

pub type ComputedProjectionInfo = ComputedLiteralInfo<FieldId>;

#[derive(Clone)]
pub enum ProjectionEval {
    Column(ColumnProjectionInfo),
    Computed(ComputedProjectionInfo),
}

#[derive(Clone)]
pub enum ProjectionPlan {
    Column {
        source_idx: usize,
    },
    Computed {
        eval_idx: usize,
        passthrough_idx: Option<usize>,
    },
}

pub struct RowStreamBuilder<P, S>
where
    P: llkv_storage::pager::Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    storage: S,
    table_id: TableId,
    schema: Arc<Schema>,
    unique_lfids: Arc<Vec<LogicalFieldId>>,
    projection_evals: Arc<Vec<ProjectionEval>>,
    _passthrough_fields: Arc<Vec<Option<FieldId>>>,
    unique_index: Arc<FxHashMap<LogicalFieldId, usize>>,
    numeric_fields: Arc<FxHashSet<FieldId>>,
    requires_numeric: bool,
    null_policy: GatherNullPolicy,
    row_ids: RowIdSource,
    chunk_size: usize,
    gather_ctx: Option<MultiGatherContext>,
    phantom: PhantomData<P>,
    include_row_ids: bool,
}

impl<P, S> RowStreamBuilder<P, S>
where
    P: llkv_storage::pager::Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        storage: S,
        table_id: TableId,
        schema: Arc<Schema>,
        unique_lfids: Arc<Vec<LogicalFieldId>>,
        projection_evals: Arc<Vec<ProjectionEval>>,
        passthrough_fields: Arc<Vec<Option<FieldId>>>,
        unique_index: Arc<FxHashMap<LogicalFieldId, usize>>,
        numeric_fields: Arc<FxHashSet<FieldId>>,
        requires_numeric: bool,
        null_policy: GatherNullPolicy,
        row_ids: impl Into<RowIdSource>,
        chunk_size: usize,
        include_row_ids: bool,
    ) -> Self {
        Self {
            storage,
            table_id,
            schema,
            unique_lfids,
            projection_evals,
            _passthrough_fields: passthrough_fields,
            unique_index,
            numeric_fields,
            requires_numeric,
            null_policy,
            row_ids: row_ids.into(),
            chunk_size,
            gather_ctx: None,
            phantom: PhantomData,
            include_row_ids,
        }
    }

    pub fn with_gather_context(mut self, ctx: MultiGatherContext) -> Self {
        self.gather_ctx = Some(ctx);
        self
    }

    pub fn build(self) -> LlkvResult<ScanRowStream<P, S>> {
        let RowStreamBuilder {
            storage,
            table_id,
            schema,
            unique_lfids,
            projection_evals,
            _passthrough_fields: passthrough_fields,
            unique_index,
            numeric_fields,
            requires_numeric,
            null_policy,
            row_ids,
            chunk_size,
            mut gather_ctx,
            include_row_ids,
            ..
        } = self;

        let gather_ctx = if unique_lfids.is_empty() {
            None
        } else if let Some(ctx) = gather_ctx.take() {
            Some(ctx)
        } else {
            Some(storage.prepare_gather_context(unique_lfids.as_ref())?)
        };

        let (row_id_array, total_rows) = match row_ids {
            RowIdSource::Bitmap(bitmap) => {
                let len = bitmap.cardinality();
                let array = Arc::new(UInt64Array::from_iter_values(bitmap.iter()));
                (array, len as usize)
            }
            RowIdSource::Vector(vector) => {
                let len = vector.len();
                let array = Arc::new(UInt64Array::from(vector));
                (array, len)
            }
        };

        let columns_buf = Vec::with_capacity(projection_evals.len());
        let numeric_arrays_cache = if requires_numeric {
            Some(FxHashMap::default())
        } else {
            None
        };

        let mut projection_plan = Vec::with_capacity(projection_evals.len());
        for (idx, eval) in projection_evals.iter().enumerate() {
            match eval {
                ProjectionEval::Column(info) => {
                    let arr_idx = *unique_index
                        .get(&info.logical_field_id)
                        .expect("logical field id missing from index");
                    projection_plan.push(ProjectionPlan::Column {
                        source_idx: arr_idx,
                    });
                }
                ProjectionEval::Computed(_) => {
                    let passthrough_idx = passthrough_fields[idx].as_ref().map(|fid| {
                        let lfid = LogicalFieldId::for_user(table_id, *fid);
                        *unique_index
                            .get(&lfid)
                            .expect("passthrough field missing from index")
                    });
                    projection_plan.push(ProjectionPlan::Computed {
                        eval_idx: idx,
                        passthrough_idx,
                    });
                }
            }
        }

        let chunk_ranges = if total_rows == 0 {
            Vec::new()
        } else {
            (0..total_rows)
                .step_by(chunk_size.max(1))
                .map(|start| {
                    let end = (start + chunk_size).min(total_rows);
                    (start, end)
                })
                .collect()
        };

        Ok(ScanRowStream {
            storage,
            table_id,
            schema,
            unique_lfids,
            projection_evals,
            _passthrough_fields: passthrough_fields,
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
            columns_buf,
            numeric_arrays_cache,
            phantom: PhantomData,
            emit_row_ids: include_row_ids,
            projection_plan,
        })
    }
}

pub struct ScanRowStream<P, S>
where
    P: llkv_storage::pager::Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    storage: S,
    table_id: TableId,
    schema: Arc<Schema>,
    unique_lfids: Arc<Vec<LogicalFieldId>>,
    projection_evals: Arc<Vec<ProjectionEval>>,
    _passthrough_fields: Arc<Vec<Option<FieldId>>>,
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
    columns_buf: Vec<ArrayRef>,
    numeric_arrays_cache: Option<NumericArrayMap>,
    phantom: PhantomData<P>,
    emit_row_ids: bool,
    projection_plan: Vec<ProjectionPlan>,
}

impl<P, S> RowStream for ScanRowStream<P, S>
where
    P: llkv_storage::pager::Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    type Columns<'a>
        = ColumnSlices<'a>
    where
        Self: 'a;

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
            let unique_index = Arc::clone(&self.unique_index);
            let numeric_fields = Arc::clone(&self.numeric_fields);
            let requires_numeric = self.requires_numeric;
            let null_policy = self.null_policy;
            let schema = Arc::clone(&self.schema);
            let projection_plan = self.projection_plan.clone();

            let numeric_cache = self.numeric_arrays_cache.as_mut();
            let batch_opt = materialize_row_window(
                &self.storage,
                self.table_id,
                unique_lfids.as_ref(),
                projection_evals.as_ref(),
                &projection_plan,
                unique_index.as_ref(),
                numeric_fields.as_ref(),
                requires_numeric,
                null_policy,
                &schema,
                window,
                self.gather_ctx.as_mut(),
                numeric_cache,
                &mut self.columns_buf,
            )?;

            let Some(batch) = batch_opt else {
                continue;
            };

            if batch.num_rows() == 0 {
                continue;
            }

            let row_ids_ref = if self.emit_row_ids {
                let row_id_slice = self.row_ids.slice(start, end - start);
                self.current_row_ids = Some(Arc::new(row_id_slice) as ArrayRef);
                self.current_row_ids
                    .as_ref()
                    .and_then(|arr| arr.as_any().downcast_ref::<UInt64Array>())
            } else {
                self.current_row_ids = None;
                None
            };
            self.current_batch = Some(batch);

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

impl<P, S> ScanRowStream<P, S>
where
    P: llkv_storage::pager::Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    pub fn into_gather_context(self) -> Option<MultiGatherContext> {
        self.gather_ctx
    }
}

#[allow(clippy::too_many_arguments)]
pub fn materialize_row_window<P, S>(
    storage: &S,
    table_id: TableId,
    unique_lfids: &[LogicalFieldId],
    projection_evals: &[ProjectionEval],
    projection_plan: &[ProjectionPlan],
    unique_index: &FxHashMap<LogicalFieldId, usize>,
    numeric_fields: &FxHashSet<FieldId>,
    requires_numeric: bool,
    null_policy: GatherNullPolicy,
    out_schema: &Arc<Schema>,
    window: &[RowId],
    gather_ctx: Option<&mut MultiGatherContext>,
    numeric_cache: Option<&mut NumericArrayMap>,
    columns: &mut Vec<ArrayRef>,
) -> LlkvResult<Option<RecordBatch>>
where
    P: llkv_storage::pager::Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    if window.is_empty() {
        return Ok(None);
    }

    let mut gathered_batch: Option<RecordBatch> = None;
    let mut numeric_arrays_holder: Option<&mut NumericArrayMap> = None;
    let batch_len = if unique_lfids.is_empty() {
        if requires_numeric {
            let map = numeric_cache.expect("numeric cache missing for computed projections");
            map.clear();
            numeric_arrays_holder = Some(map);
        }
        window.len()
    } else {
        let batch = storage.gather_row_window_with_context(
            unique_lfids,
            window,
            null_policy,
            gather_ctx,
        )?;
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let batch_len = batch.num_rows();
        if requires_numeric {
            let map = numeric_cache.expect("numeric cache missing for computed projections");
            map.clear();
            for (lfid, array) in unique_lfids.iter().zip(batch.columns().iter()) {
                let fid = lfid.field_id();
                if numeric_fields.contains(&fid) {
                    map.insert(fid, array.clone());
                }
            }
            numeric_arrays_holder = Some(map);
        }
        gathered_batch = Some(batch);
        batch_len
    };

    if batch_len == 0 {
        return Ok(None);
    }

    let gathered_columns: &[ArrayRef] = if let Some(batch) = gathered_batch.as_ref() {
        batch.columns()
    } else {
        &[]
    };

    columns.clear();
    columns.reserve(projection_evals.len());
    for (idx, plan) in projection_plan.iter().enumerate() {
        match plan {
            ProjectionPlan::Column { source_idx } => {
                columns.push(Arc::clone(&gathered_columns[*source_idx]));
            }
            ProjectionPlan::Computed {
                eval_idx,
                passthrough_idx,
            } => {
                if let Some(arr_idx) = passthrough_idx {
                    columns.push(Arc::clone(&gathered_columns[*arr_idx]));
                    continue;
                }
                let info = match &projection_evals[*eval_idx] {
                    ProjectionEval::Computed(info) => info,
                    ProjectionEval::Column(_) => unreachable!("plan mismatch"),
                };
                let mut temp_numeric_map = NumericArrayMap::default();
                let array: ArrayRef = match &info.expr {
                    ScalarExpr::Literal(_) => synthesize_computed_literal_array(
                        info,
                        out_schema.field(idx).data_type(),
                        batch_len,
                    )?,
                    ScalarExpr::Cast { .. } if !computed_expr_requires_numeric(&info.expr) => {
                        synthesize_computed_literal_array(
                            info,
                            out_schema.field(idx).data_type(),
                            batch_len,
                        )?
                    }
                    ScalarExpr::GetField { base, field_name } => {
                        fn eval_get_field(
                            expr: &ScalarExpr<FieldId>,
                            field_name: &str,
                            gathered_columns: &[ArrayRef],
                            unique_index: &FxHashMap<LogicalFieldId, usize>,
                            table_id: TableId,
                        ) -> LlkvResult<ArrayRef> {
                            let base_array = match expr {
                                ScalarExpr::Column(fid) => {
                                    let lfid = LogicalFieldId::for_user(table_id, *fid);
                                    let arr_idx = *unique_index.get(&lfid).ok_or_else(|| {
                                        llkv_result::Error::Internal(
                                            "field missing from unique arrays".into(),
                                        )
                                    })?;
                                    Arc::clone(&gathered_columns[arr_idx])
                                }
                                ScalarExpr::GetField {
                                    base: inner_base,
                                    field_name: inner_field,
                                } => eval_get_field(
                                    inner_base,
                                    inner_field,
                                    gathered_columns,
                                    unique_index,
                                    table_id,
                                )?,
                                _ => {
                                    return Err(llkv_result::Error::InvalidArgumentError(
                                        "GetField base must be a column or another GetField".into(),
                                    ));
                                }
                            };

                            let struct_array = base_array
                                .as_any()
                                .downcast_ref::<arrow::array::StructArray>()
                                .ok_or_else(|| {
                                    llkv_result::Error::InvalidArgumentError(
                                        "GetField can only be applied to struct types".into(),
                                    )
                                })?;

                            struct_array
                                .column_by_name(field_name)
                                .ok_or_else(|| {
                                    llkv_result::Error::InvalidArgumentError(format!(
                                        "Field '{}' not found in struct",
                                        field_name
                                    ))
                                })
                                .map(Arc::clone)
                        }

                        eval_get_field(base, field_name, gathered_columns, unique_index, table_id)?
                    }
                    _ => {
                        let numeric_arrays = if let Some(arrays) = numeric_arrays_holder.as_deref()
                        {
                            arrays
                        } else {
                            // Build a minimal numeric map on demand when the caller did not
                            // precompute one (e.g., when `requires_numeric` was false but
                            // the expression still needs evaluation).
                            temp_numeric_map.clear();
                            for (lfid, array) in unique_lfids.iter().zip(gathered_columns.iter()) {
                                temp_numeric_map.insert(lfid.field_id(), array.clone());
                            }
                            &temp_numeric_map
                        };
                        ScalarEvaluator::evaluate_batch(&info.expr, batch_len, numeric_arrays)?
                    }
                };

                // Cast if needed to match output schema
                let expected_type = out_schema.field(columns.len()).data_type();
                let array = if array.data_type() != expected_type {
                    arrow::compute::cast(&array, expected_type)
                        .map_err(|e| llkv_result::Error::Internal(e.to_string()))?
                } else {
                    array
                };

                columns.push(array);
            }
        }
    }

    if columns.is_empty() {
        let options =
            arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(batch_len));
        let batch =
            RecordBatch::try_new_with_options(Arc::clone(out_schema), columns.clone(), &options)?;
        Ok(Some(batch))
    } else {
        let batch = RecordBatch::try_new(Arc::clone(out_schema), columns.clone())?;
        Ok(Some(batch))
    }
}
