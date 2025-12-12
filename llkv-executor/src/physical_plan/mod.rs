pub mod aggregate;
pub mod filter;
pub mod join;
pub mod projection;
pub mod ranges;
pub mod scan;
pub mod sort;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use llkv_result::Result;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::fmt::Debug;
use std::sync::Arc;

pub type BatchIter = Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

pub trait PhysicalPlan<P>: Debug + Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Returns the schema of the output of this plan
    fn schema(&self) -> SchemaRef;

    /// Execute the plan and return an iterator of record batches
    fn execute(&self) -> Result<BatchIter>;

    /// Returns the children of this plan
    fn children(&self) -> Arc<[Arc<dyn PhysicalPlan<P>>]>;

    /// Returns a new plan with the specified children
    fn with_new_children(
        self: Arc<Self>,
        children: Arc<[Arc<dyn PhysicalPlan<P>>]>,
    ) -> Result<Arc<dyn PhysicalPlan<P>>>;

    fn as_any(&self) -> &dyn std::any::Any;
}
