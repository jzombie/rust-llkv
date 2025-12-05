use arrow::array::RecordBatch;
use croaring::Treemap;
use llkv_expr::Expr;
use llkv_plan::{PlanGraph, ProgramSet};
use llkv_result::{Error, Result as ExecutorResult};
use llkv_table::table::Table as LlkvTable;
use llkv_types::{FieldId, LogicalFieldId, RowId, TableId};
use simd_r_drive_entry_handle::EntryHandle;

use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_compute::analysis::PredicateFusionCache;
use llkv_compute::program::OwnedFilter;
use llkv_scan::{RowIdSource, ScanProjection, ScanStorage, ScanStreamOptions};
use llkv_storage::pager::Pager;

impl<P> ScanStorage<P> for crate::types::TableStorageAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId {
        self.table().table_id()
    }

    fn field_data_type(&self, fid: LogicalFieldId) -> ExecutorResult<arrow::datatypes::DataType> {
        self.table().store().data_type(fid)
    }

    fn total_rows(&self) -> ExecutorResult<u64> {
        self.table().total_rows()
    }

    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> ExecutorResult<MultiGatherContext> {
        self.table().store().prepare_gather_context(logical_fields)
    }

    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> ExecutorResult<RecordBatch> {
        self.table().store().gather_row_window_with_context(
            logical_fields,
            row_ids,
            null_policy,
            ctx,
        )
    }

    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> ExecutorResult<Treemap> {
        self.table().filter_row_ids(filter_expr)
    }

    fn all_row_ids(&self) -> ExecutorResult<Treemap> {
        self.table().all_row_ids()
    }

    fn sorted_row_ids_full_table(
        &self,
        order_spec: llkv_scan::ScanOrderSpec,
    ) -> ExecutorResult<Option<Vec<u64>>> {
        use llkv_scan::ScanStorage;
        <LlkvTable<P> as ScanStorage<P>>::sorted_row_ids_full_table(self.table(), order_spec)
    }

    fn filter_leaf(&self, filter: &OwnedFilter) -> ExecutorResult<Treemap> {
        self.table().filter_leaf(filter)
    }

    fn filter_fused(
        &self,
        field_id: FieldId,
        filters: &[OwnedFilter],
        cache: &PredicateFusionCache,
    ) -> ExecutorResult<RowIdSource> {
        self.table().filter_fused(field_id, filters, cache)
    }

    fn stream_row_ids(
        &self,
        chunk_size: usize,
        ranges: Option<llkv_column_map::store::scan::ranges::IntRanges>,
        driving_column: Option<LogicalFieldId>,
        on_chunk: &mut dyn FnMut(&[RowId]) -> ExecutorResult<()>,
    ) -> ExecutorResult<()> {
        self.table()
            .stream_row_ids(chunk_size, ranges, driving_column, on_chunk)
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
        let table = self
            .storage
            .as_any()
            .downcast_ref::<LlkvTable<P>>()
            .ok_or_else(|| {
                Error::InvalidArgumentError(
                    "scan executor requires table-backed storage for now".into(),
                )
            })?;

        table.scan_stream_with_exprs(projections, filter_expr, options, on_batch)
    }
}
