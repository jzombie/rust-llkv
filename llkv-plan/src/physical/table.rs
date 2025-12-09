use arrow::record_batch::RecordBatch;
use llkv_expr::Expr;
use llkv_result::Result;
use llkv_scan::{ScanProjection, ScanStreamOptions};
use llkv_storage::pager::Pager;
use llkv_types::FieldId;
use llkv_types::ids::TableId;
use simd_r_drive_entry_handle::EntryHandle;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::schema::PlanSchema;

// TODO: Dedupe (another one exists inside of llkv-plan)
pub trait ExecutionTable<P>: Debug + Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> Arc<PlanSchema>;
    fn table_id(&self) -> TableId;

    fn scan_stream(
        &self,
        projections: &[ScanProjection],
        predicate: &Expr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        callback: &mut dyn FnMut(RecordBatch),
    ) -> Result<()>;

    fn as_any(&self) -> &dyn Any;
}

// TODO: Dedupe (another one exists inside of llkv-plan)
pub trait TableProvider<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, name: &str) -> Result<Arc<dyn ExecutionTable<P>>>;
}
