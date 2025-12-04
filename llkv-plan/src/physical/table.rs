use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;
use arrow::record_batch::RecordBatch;
use llkv_scan::{ScanProjection, ScanStreamOptions};
use llkv_expr::Expr;
use llkv_types::FieldId;
use llkv_types::ids::TableId;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::schema::PlanSchema;

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
    ) -> Result<(), String>;

    fn as_any(&self) -> &dyn Any;
}

pub trait TableProvider<P>: Send + Sync 
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, name: &str) -> Result<Arc<dyn ExecutionTable<P>>, String>;
}
