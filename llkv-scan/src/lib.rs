//! Shared scan interfaces and streaming helpers for LLKV.
//!
//! This crate is intended to host the storage-agnostic scan surface used by
//! both the table layer and the executor. It currently contains the core scan
//! types and storage abstraction; execution wiring will migrate here over time.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use croaring::Treemap;
use llkv_column_map::store::scan::ranges::IntRanges;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext, Projection};
use llkv_expr::{Expr, ScalarExpr};
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::{MemPager, Pager};
use llkv_types::{FieldId, LogicalFieldId, RowId, TableId};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
pub mod row_stream;
pub use row_stream::ScanRowStream;
pub use row_stream::{
    ColumnProjectionInfo, ComputedProjectionInfo, ProjectionEval, RowChunk, RowIdSource, RowStream,
    RowStreamBuilder, materialize_row_window,
};

pub mod execute;
pub mod ordering;
pub mod predicate;
pub use ordering::sort_row_ids_with_order;

/// Sort direction for scan ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanOrderDirection {
    Ascending,
    Descending,
}

/// Value transformation to apply before sorting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanOrderTransform {
    IdentityInt64,
    IdentityInt32,
    IdentityUtf8,
    CastUtf8ToInteger,
}

/// Specification for ordering scan results.
#[derive(Clone, Copy, Debug)]
pub struct ScanOrderSpec {
    pub field_id: FieldId,
    pub direction: ScanOrderDirection,
    pub nulls_first: bool,
    pub transform: ScanOrderTransform,
}

/// A column or computed expression to include in scan results.
#[derive(Clone, Debug)]
pub enum ScanProjection {
    Column(Projection),
    Computed {
        expr: ScalarExpr<FieldId>,
        alias: String,
    },
}

impl ScanProjection {
    pub fn column<P: Into<Projection>>(proj: P) -> Self {
        Self::Column(proj.into())
    }

    pub fn computed<S: Into<String>>(expr: ScalarExpr<FieldId>, alias: S) -> Self {
        Self::Computed {
            expr,
            alias: alias.into(),
        }
    }
}

impl From<Projection> for ScanProjection {
    fn from(value: Projection) -> Self {
        ScanProjection::Column(value)
    }
}

impl From<&Projection> for ScanProjection {
    fn from(value: &Projection) -> Self {
        ScanProjection::Column(value.clone())
    }
}

impl From<&ScanProjection> for ScanProjection {
    fn from(value: &ScanProjection) -> Self {
        value.clone()
    }
}

/// Options for configuring table scans.
pub struct ScanStreamOptions<P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub include_nulls: bool,
    pub order: Option<ScanOrderSpec>,
    pub row_id_filter: Option<Arc<dyn RowIdFilter<P>>>,
    pub include_row_ids: bool,
    pub ranges: Option<IntRanges>,
    pub driving_column: Option<LogicalFieldId>,
}

impl<P> Clone for ScanStreamOptions<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            include_nulls: self.include_nulls,
            order: self.order,
            row_id_filter: self.row_id_filter.clone(),
            include_row_ids: self.include_row_ids,
            ranges: self.ranges,
            driving_column: self.driving_column,
        }
    }
}

impl<P> Default for ScanStreamOptions<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn default() -> Self {
        Self {
            include_nulls: false,
            order: None,
            row_id_filter: None,
            include_row_ids: false,
            ranges: None,
            driving_column: None,
        }
    }
}

impl<P> std::fmt::Debug for ScanStreamOptions<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanStreamOptions")
            .field("include_nulls", &self.include_nulls)
            .field("order", &self.order)
            .field(
                "row_id_filter",
                &self.row_id_filter.as_ref().map(|_| "<RowIdFilter>"),
            )
            .field("include_row_ids", &self.include_row_ids)
            .field("ranges", &self.ranges)
            .field("driving_column", &self.driving_column)
            .finish()
    }
}

/// Filter row IDs before they are materialized into batches.
pub trait RowIdFilter<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn filter(
        &self,
        table_id: TableId,
        storage: &dyn ScanStorage<P>,
        row_ids: Treemap,
    ) -> LlkvResult<Treemap>;
}

/// Capabilities the scan executor needs from storage.
pub trait ScanStorage<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId;
    fn field_data_type(&self, fid: LogicalFieldId) -> LlkvResult<DataType>;
    fn total_rows(&self) -> LlkvResult<u64>;
    fn all_row_ids(&self) -> LlkvResult<croaring::Treemap>;
    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> LlkvResult<MultiGatherContext>;
    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> LlkvResult<RecordBatch>;
    fn filter_row_ids<'expr>(
        &self,
        filter_expr: &Expr<'expr, FieldId>,
    ) -> LlkvResult<croaring::Treemap>;

    /// Evaluate a leaf predicate (single column filter) against the storage.
    fn filter_leaf(
        &self,
        filter: &llkv_compute::program::OwnedFilter,
    ) -> LlkvResult<croaring::Treemap>;

    /// Evaluate fused predicates against a single column.
    fn filter_fused(
        &self,
        field_id: FieldId,
        filters: &[llkv_compute::program::OwnedFilter],
        cache: &llkv_compute::analysis::PredicateFusionCache,
    ) -> LlkvResult<RowIdSource>;

    /// Optionally return row IDs ordered by a column's sorted permutation when
    /// the caller is scanning the entire table without additional filtering.
    ///
    /// Implementations should return `Ok(None)` when the storage backend cannot
    /// satisfy the request for the given [`ScanOrderSpec`].
    fn sorted_row_ids_full_table(&self, order_spec: ScanOrderSpec) -> LlkvResult<Option<Vec<u64>>>;

    fn stream_row_ids(
        &self,
        chunk_size: usize,
        ranges: Option<IntRanges>,
        driving_column: Option<LogicalFieldId>,
        on_chunk: &mut dyn FnMut(&[RowId]) -> LlkvResult<()>,
    ) -> LlkvResult<()>;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Utility alias for tracked numeric arrays during computed projection evaluation.
pub type NumericArrayMap = FxHashMap<FieldId, arrow_array::ArrayRef>;

impl<P, T> ScanStorage<P> for std::sync::Arc<T>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    T: ScanStorage<P> + ?Sized,
{
    fn table_id(&self) -> TableId {
        (**self).table_id()
    }
    fn field_data_type(&self, fid: LogicalFieldId) -> LlkvResult<DataType> {
        (**self).field_data_type(fid)
    }
    fn total_rows(&self) -> LlkvResult<u64> {
        (**self).total_rows()
    }
    fn all_row_ids(&self) -> LlkvResult<croaring::Treemap> {
        (**self).all_row_ids()
    }
    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> LlkvResult<MultiGatherContext> {
        (**self).prepare_gather_context(logical_fields)
    }
    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> LlkvResult<RecordBatch> {
        (**self).gather_row_window_with_context(logical_fields, row_ids, null_policy, ctx)
    }
    fn filter_row_ids<'expr>(
        &self,
        filter_expr: &Expr<'expr, FieldId>,
    ) -> LlkvResult<croaring::Treemap> {
        (**self).filter_row_ids(filter_expr)
    }
    fn filter_leaf(
        &self,
        filter: &llkv_compute::program::OwnedFilter,
    ) -> LlkvResult<croaring::Treemap> {
        (**self).filter_leaf(filter)
    }
    fn filter_fused(
        &self,
        field_id: FieldId,
        filters: &[llkv_compute::program::OwnedFilter],
        cache: &llkv_compute::analysis::PredicateFusionCache,
    ) -> LlkvResult<RowIdSource> {
        (**self).filter_fused(field_id, filters, cache)
    }
    fn sorted_row_ids_full_table(&self, order_spec: ScanOrderSpec) -> LlkvResult<Option<Vec<u64>>> {
        (**self).sorted_row_ids_full_table(order_spec)
    }
    fn stream_row_ids(
        &self,
        chunk_size: usize,
        ranges: Option<IntRanges>,
        driving_column: Option<LogicalFieldId>,
        on_chunk: &mut dyn FnMut(&[RowId]) -> LlkvResult<()>,
    ) -> LlkvResult<()> {
        (**self).stream_row_ids(chunk_size, ranges, driving_column, on_chunk)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        (**self).as_any()
    }
}

impl<P, T> ScanStorage<P> for &T
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    T: ScanStorage<P> + ?Sized,
{
    fn table_id(&self) -> TableId {
        (**self).table_id()
    }
    fn field_data_type(&self, fid: LogicalFieldId) -> LlkvResult<DataType> {
        (**self).field_data_type(fid)
    }
    fn total_rows(&self) -> LlkvResult<u64> {
        (**self).total_rows()
    }
    fn all_row_ids(&self) -> LlkvResult<croaring::Treemap> {
        (**self).all_row_ids()
    }
    fn prepare_gather_context(
        &self,
        logical_fields: &[LogicalFieldId],
    ) -> LlkvResult<MultiGatherContext> {
        (**self).prepare_gather_context(logical_fields)
    }
    fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> LlkvResult<RecordBatch> {
        (**self).gather_row_window_with_context(logical_fields, row_ids, null_policy, ctx)
    }
    fn filter_row_ids<'expr>(
        &self,
        filter_expr: &Expr<'expr, FieldId>,
    ) -> LlkvResult<croaring::Treemap> {
        (**self).filter_row_ids(filter_expr)
    }
    fn filter_leaf(
        &self,
        filter: &llkv_compute::program::OwnedFilter,
    ) -> LlkvResult<croaring::Treemap> {
        (**self).filter_leaf(filter)
    }
    fn filter_fused(
        &self,
        field_id: FieldId,
        filters: &[llkv_compute::program::OwnedFilter],
        cache: &llkv_compute::analysis::PredicateFusionCache,
    ) -> LlkvResult<RowIdSource> {
        (**self).filter_fused(field_id, filters, cache)
    }
    fn sorted_row_ids_full_table(&self, order_spec: ScanOrderSpec) -> LlkvResult<Option<Vec<u64>>> {
        (**self).sorted_row_ids_full_table(order_spec)
    }
    fn stream_row_ids(
        &self,
        chunk_size: usize,
        ranges: Option<IntRanges>,
        driving_column: Option<LogicalFieldId>,
        on_chunk: &mut dyn FnMut(&[RowId]) -> LlkvResult<()>,
    ) -> LlkvResult<()> {
        (**self).stream_row_ids(chunk_size, ranges, driving_column, on_chunk)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        (**self).as_any()
    }
}
