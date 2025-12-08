use std::sync::Arc;
use llkv_storage::pager::Pager;
use llkv_types::{TableId, FieldId};
use simd_r_drive_entry_handle::EntryHandle;
use llkv_result::Result;
use crate::schema::PlanSchema;
use llkv_scan::{ScanProjection, ScanStreamOptions};
use llkv_expr::Expr;
use arrow::record_batch::RecordBatch;

/// Trait for table providers that can resolve table names to table instances.
pub trait TableProvider<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, name: &str) -> Option<Arc<dyn ExecutionTable<P>>>;
}

/// Trait for table instances that can be used in execution.
pub trait ExecutionTable<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn as_any(&self) -> &dyn std::any::Any;
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync>;
    fn table_id(&self) -> TableId;
    fn schema(&self) -> &PlanSchema;

    fn scan_stream(
        &self,
        projections: &[ScanProjection],
        predicate: &Expr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        callback: &mut dyn FnMut(RecordBatch),
    ) -> Result<()>;
}
