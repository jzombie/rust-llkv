//! Shared scan interfaces and streaming helpers for LLKV.
//!
//! This crate is intended to host the storage-agnostic scan surface used by
//! both the table layer and the executor. It currently contains the core scan
//! types and storage abstraction; execution wiring will migrate here over time.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext, Projection};
use llkv_expr::{Expr, ScalarExpr};
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::{MemPager, Pager};
use llkv_types::{FieldId, LogicalFieldId, TableId};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use croaring::Treemap;
pub mod row_stream;
pub use row_stream::{
    ColumnProjectionInfo, ComputedProjectionInfo, ProjectionEval, RowChunk, RowIdSource, RowStream,
    RowStreamBuilder, materialize_row_window,
};

pub mod execute;

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

/// Options for configuring table scans.
#[derive(Clone)]
pub struct ScanStreamOptions<P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub include_nulls: bool,
    pub order: Option<ScanOrderSpec>,
    pub row_id_filter: Option<Arc<dyn RowIdFilter<P>>>,
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
    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> LlkvResult<croaring::Treemap>;
    fn stream_row_ids(
        &self,
        chunk_size: usize,
        on_chunk: &mut dyn FnMut(Vec<u64>) -> LlkvResult<()>,
    ) -> LlkvResult<()>;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Utility alias for tracked numeric arrays during computed projection evaluation.
pub type NumericArrayMap = FxHashMap<FieldId, arrow_array::ArrayRef>;
