use crate::schema::PlanSchema;
use arrow::record_batch::RecordBatch;
use llkv_expr::Expr;
use llkv_result::Result;
use llkv_scan::{ScanProjection, ScanStreamOptions};
use llkv_storage::pager::Pager;
use llkv_types::{FieldId, TableId};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Trait for table providers that can resolve table names to table instances.
pub trait TableProvider<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, name: &str) -> Option<Arc<dyn ExecutionTable<P>>>;

    /// Get approximate row counts for multiple tables in a batch.
    ///
    /// The default implementation calls `get_table` and `approximate_row_count` sequentially.
    /// Implementations can override this to provide optimized batch lookups.
    fn batch_approximate_row_counts(&self, tables: &[&str]) -> Vec<Option<usize>> {
        tables
            .iter()
            .map(|name| self.get_table(name).and_then(|t| t.approximate_row_count()))
            .collect()
    }
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

    fn approximate_row_count(&self) -> Option<usize>;

    fn scan_stream(
        &self,
        projections: &[ScanProjection],
        predicate: &Expr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        callback: &mut dyn FnMut(RecordBatch),
    ) -> Result<()>;
}
