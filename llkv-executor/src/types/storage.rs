use std::any::Any;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use croaring::Treemap;
use llkv_expr::Expr;
use llkv_join::{JoinKey, JoinOptions, TableJoinExt};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_table::table::{ScanProjection, ScanStreamOptions, Table};
use llkv_types::FieldId;
use llkv_types::ids::TableId;
use simd_r_drive_entry_handle::EntryHandle;

/// Minimal storage surface needed by the executor.
pub trait StorageTable<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId;

    fn scan_stream<'expr>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: &mut dyn FnMut(RecordBatch),
    ) -> LlkvResult<()>;

    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> LlkvResult<Treemap>;

    fn join_stream(
        &self,
        right: &dyn StorageTable<P>,
        keys: &[JoinKey],
        options: &JoinOptions,
        on_batch: &mut dyn FnMut(RecordBatch),
    ) -> LlkvResult<()>;

    fn as_any(&self) -> &dyn Any;
}

/// Adapter over `llkv_table::table::Table` implementing `StorageTable`.
#[derive(Clone)]
pub struct TableStorageAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: Arc<Table<P>>,
}

impl<P> TableStorageAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(table: Arc<Table<P>>) -> Self {
        Self { table }
    }

    pub fn table(&self) -> &Table<P> {
        &self.table
    }
}

impl<P> StorageTable<P> for TableStorageAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn table_id(&self) -> TableId {
        self.table.table_id()
    }

    fn scan_stream<'expr>(
        &self,
        projections: &[ScanProjection],
        filter_expr: &Expr<'expr, FieldId>,
        options: ScanStreamOptions<P>,
        on_batch: &mut dyn FnMut(RecordBatch),
    ) -> LlkvResult<()> {
        self.table.scan_stream(projections, filter_expr, options, on_batch)
    }

    fn filter_row_ids<'expr>(&self, filter_expr: &Expr<'expr, FieldId>) -> LlkvResult<Treemap> {
        self.table.filter_row_ids(filter_expr)
    }

    fn join_stream(
        &self,
        right: &dyn StorageTable<P>,
        keys: &[JoinKey],
        options: &JoinOptions,
        on_batch: &mut dyn FnMut(RecordBatch),
    ) -> LlkvResult<()> {
        let Some(rhs) = right.as_any().downcast_ref::<TableStorageAdapter<P>>() else {
            return Err(Error::InvalidArgumentError(
                "join_stream requires compatible storage adapter".into(),
            ));
        };
        self.table.join_stream(rhs.table(), keys, options, on_batch)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
