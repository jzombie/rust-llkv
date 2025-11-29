mod row_stream;

use std::sync::Arc;

use arrow::array::RecordBatch;
use croaring::Treemap;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_expr::Expr;
use llkv_plan::{PlanGraph, ProgramSet};
use llkv_result::{Error, Result as ExecutorResult};
use llkv_table::constants::STREAM_BATCH_ROWS;
use llkv_table::table::{ScanProjection, ScanStreamOptions};
use llkv_types::{FieldId, LogicalFieldId, TableId};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_storage::pager::Pager;
use row_stream::{RowIdSource, RowStream, RowStreamBuilder};
use rustc_hash::{FxHashMap, FxHashSet};


/// Capabilities the scan executor needs from storage, kept minimal to avoid
/// leaking execution details into the storage crate.
pub trait ScanStorage<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId;
    fn field_data_type(&self, fid: LogicalFieldId) -> ExecutorResult<arrow::datatypes::DataType>;
    fn total_rows(&self) -> ExecutorResult<u64>;
    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> ExecutorResult<MultiGatherContext>;
    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> ExecutorResult<RecordBatch>;
    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> ExecutorResult<Treemap>;
    fn stream_row_ids(
        &self,
        chunk_size: usize,
        on_chunk: &mut dyn FnMut(Vec<u64>) -> ExecutorResult<()>,
    ) -> ExecutorResult<()>;
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<P> ScanStorage<P> for crate::types::TableStorageAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId {
        self.table().table_id()
    }

    fn field_data_type(&self, fid: LogicalFieldId) -> ExecutorResult<arrow::datatypes::DataType> {
        self.table()
            .store()
            .data_type(fid)
            .map_err(Error::from)
    }

    fn total_rows(&self) -> ExecutorResult<u64> {
        self.table().total_rows().map_err(Error::from)
    }

    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> ExecutorResult<MultiGatherContext> {
        self.table()
            .store()
            .prepare_gather_context(logical_fields)
            .map_err(Error::from)
    }

    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> ExecutorResult<RecordBatch> {
        self.table()
            .store()
            .gather_row_window_with_context(logical_fields, row_ids, null_policy, ctx)
            .map_err(Error::from)
    }

    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> ExecutorResult<Treemap> {
        self.table().filter_row_ids(filter_expr).map_err(Error::from)
    }

    fn stream_row_ids(
        &self,
        chunk_size: usize,
        on_chunk: &mut dyn FnMut(Vec<u64>) -> ExecutorResult<()>,
    ) -> ExecutorResult<()> {
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        let ids = self.table().filter_row_ids(&Expr::Pred(Filter {
            field_id: llkv_table::ROW_ID_FIELD_ID,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        }))?;
        let mut buffer = Vec::new();
        for chunk in ids.iter().collect::<Vec<_>>().chunks(chunk_size.max(1)) {
            buffer.clear();
            buffer.extend_from_slice(chunk);
            on_chunk(buffer.clone())?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Thin wrapper capturing what the executor needs to run a scan.
pub struct ScanExecutor<'a, P, S>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    storage: &'a S,
    _phantom: std::marker::PhantomData<P>,
}

impl<'a, P, S> ScanExecutor<'a, P, S>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
{
    pub fn new(storage: &'a S) -> Self {
        Self {
            storage,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn table_id(&self) -> TableId {
        self.storage.table_id()
    }

    /// Execute a scan using the executor's storage abstraction.
    pub fn execute<'expr, F>(
        &self,
        _plan_graph: PlanGraph,
        _programs: ProgramSet<'expr>,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: &mut F,
    ) -> ExecutorResult<()>
    where
        F: FnMut(RecordBatch),
    {
        // Features still routed through llkv-table's scan_stream for correctness.
        if options.order.is_some() || options.row_id_filter.is_some() {
            if let Some(adapter) = self.storage.as_any().downcast_ref::<crate::types::TableStorageAdapter<P>>() {
                return adapter
                    .table()
                    .scan_stream_with_exprs(projections, filter_expr, options, on_batch)
                    .map_err(Error::from);
            }
            return Err(Error::InvalidArgumentError(
                "order or row_id_filter requires table-backed storage".into(),
            ));
        }

        if projections.is_empty() {
            return Err(Error::InvalidArgumentError(
                "scan requires at least one projection".into(),
            ));
        }

        // Collect row ids via storage filter.
        let row_ids = self.storage.filter_row_ids(filter_expr)?;
        let row_source = if row_ids.is_empty() {
            RowIdSource::Bitmap(row_ids)
        } else {
            RowIdSource::Bitmap(row_ids)
        };

        // Determine projection evaluation plan and output schema.
        let mut projection_evals = Vec::with_capacity(projections.len());
        let mut unique_index = FxHashMap::default();
        let mut unique_lfids = Vec::new();
        let mut numeric_fields = FxHashSet::default();
        let mut lfid_dtypes = FxHashMap::default();

        for proj in projections {
            match proj {
                ScanProjection::Column(p) => {
                    let lfid = p.logical_field_id;
                    let dtype = self.storage.field_data_type(lfid)?;
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        unique_index.entry(lfid)
                    {
                        entry.insert(unique_lfids.len());
                        unique_lfids.push(lfid);
                    }
                    let fallback = lfid.field_id().to_string();
                    let output_name = p.alias.clone().unwrap_or(fallback);
                    projection_evals.push(row_stream::ProjectionEval::Column(
                        row_stream::ColumnProjectionInfo {
                            logical_field_id: lfid,
                            data_type: dtype,
                            output_name,
                        },
                    ));
                }
                ScanProjection::Computed { expr, alias } => {
                    let simplified = llkv_table::NumericKernels::simplify(expr);
                    let mut fields_set: FxHashSet<llkv_table::types::FieldId> =
                        FxHashSet::default();
                    llkv_table::NumericKernels::collect_fields(&simplified, &mut fields_set);
                    for fid in fields_set.iter().copied() {
                        numeric_fields.insert(fid);
                        let lfid = LogicalFieldId::for_user(self.table_id(), fid);
                        let dtype = self.storage.field_data_type(lfid)?;
                        lfid_dtypes.entry(lfid).or_insert_with(|| dtype.clone());
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            unique_index.entry(lfid)
                        {
                            entry.insert(unique_lfids.len());
                            unique_lfids.push(lfid);
                        }
                    }
                    projection_evals.push(row_stream::ProjectionEval::Computed(
                        row_stream::ComputedProjectionInfo {
                            expr: simplified,
                            alias: alias.clone(),
                        },
                    ));
                }
            }
        }

        let passthrough_fields: Vec<Option<llkv_table::types::FieldId>> = projection_evals
            .iter()
            .map(|eval| match eval {
                row_stream::ProjectionEval::Computed(info) => {
                    llkv_table::NumericKernels::passthrough_column(&info.expr)
                }
                _ => None,
            })
            .collect();

        let null_policy = if options.include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::DropNulls
        };

        let requires_numeric = projection_evals.iter().enumerate().any(|(idx, eval)| {
            matches!(
                eval,
                row_stream::ProjectionEval::Computed(info)
                if passthrough_fields[idx].is_none()
                    && llkv_compute::analysis::computed_expr_requires_numeric(&info.expr)
            )
        });

        let mut schema_fields: Vec<arrow::datatypes::Field> =
            Vec::with_capacity(projection_evals.len());
        for (idx, eval) in projection_evals.iter().enumerate() {
            match eval {
                row_stream::ProjectionEval::Column(info) => schema_fields.push(
                    arrow::datatypes::Field::new(
                        info.output_name.clone(),
                        info.data_type.clone(),
                        true,
                    ),
                ),
                row_stream::ProjectionEval::Computed(info) => {
                    if let Some(fid) = passthrough_fields[idx] {
                        let lfid = LogicalFieldId::for_user(self.table_id(), fid);
                        let dtype = lfid_dtypes.get(&lfid).cloned().ok_or_else(|| {
                            Error::Internal("missing dtype for passthrough".into())
                        })?;
                        schema_fields.push(arrow::datatypes::Field::new(
                            info.alias.clone(),
                            dtype,
                            true,
                        ));
                    } else {
                        let dtype = llkv_compute::projection::infer_computed_dtype(
                            &info.expr,
                            self.table_id(),
                            &lfid_dtypes,
                        )?;
                        schema_fields.push(arrow::datatypes::Field::new(
                            info.alias.clone(),
                            dtype,
                            true,
                        ));
                    }
                }
            }
        }
        let out_schema = Arc::new(arrow::datatypes::Schema::new(schema_fields));

        let mut row_stream = RowStreamBuilder::new(
            self.storage,
            self.table_id(),
            Arc::clone(&out_schema),
            Arc::new(unique_lfids),
            Arc::new(projection_evals),
            Arc::new(passthrough_fields),
            Arc::new(unique_index),
            Arc::new(numeric_fields),
            requires_numeric,
            null_policy,
            row_source,
            STREAM_BATCH_ROWS,
        )
        .build()?;

        let expected_columns = row_stream.schema().fields().len();
        while let Some(chunk) = row_stream.next_chunk()? {
            let batch = chunk.to_record_batch();
            debug_assert_eq!(batch.num_columns(), expected_columns);
            if batch.num_rows() > 0 {
                on_batch(batch);
            }
        }

        Ok(())
    }
}
