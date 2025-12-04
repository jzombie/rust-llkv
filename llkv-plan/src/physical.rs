use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

pub mod scan;
pub mod table;
pub mod ranges;

/// A physical plan represents an executable query plan.
///
/// It is similar to a logical plan but contains all necessary details for execution,
/// such as specific algorithms (e.g. HashJoin vs SortMergeJoin) and physical
/// properties (e.g. partitioning).
pub trait PhysicalPlan: Debug + Send + Sync {
    /// Returns the schema of the output of this plan
    fn schema(&self) -> SchemaRef;

    /// Execute the plan and return an iterator of record batches
    fn execute(
        &self,
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, String>> + Send>, String>;

    /// Returns the children of this plan
    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>>;

    /// Returns a new plan with the specified children
    fn with_new_children(
        &self,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Result<Arc<dyn PhysicalPlan>, String>;

    fn as_any(&self) -> &dyn Any;
}
