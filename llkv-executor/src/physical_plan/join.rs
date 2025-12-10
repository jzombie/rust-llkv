use crate::physical_plan::{BatchIter, PhysicalPlan};
use arrow::datatypes::SchemaRef;
use llkv_plan::plans::JoinPlan;
use llkv_result::Result;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub struct HashJoinExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub left: Arc<dyn PhysicalPlan<P>>,
    pub right: Arc<dyn PhysicalPlan<P>>,
    pub on: Vec<(usize, usize)>, // (left_col_idx, right_col_idx)
    pub join_type: JoinPlan,
    pub schema: SchemaRef,
}

impl<P> HashJoinExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(
        left: Arc<dyn PhysicalPlan<P>>,
        right: Arc<dyn PhysicalPlan<P>>,
        on: Vec<(usize, usize)>,
        join_type: JoinPlan,
        schema: SchemaRef,
    ) -> Self {
        Self {
            left,
            right,
            on,
            join_type,
            schema,
        }
    }
}

impl<P> fmt::Debug for HashJoinExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoinExec")
            .field("on", &self.on)
            .field("join_type", &self.join_type)
            .field("schema", &self.schema)
            .finish()
    }
}

impl<P> PhysicalPlan<P> for HashJoinExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Result<BatchIter> {
        Err(llkv_result::Error::Internal(
            "Execution of HashJoinExec must be handled by the executor crate".to_string(),
        ))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan<P>>> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan<P>>>,
    ) -> Result<Arc<dyn PhysicalPlan<P>>> {
        if children.len() != 2 {
            return Err(llkv_result::Error::Internal(
                "HashJoinExec expects exactly 2 children".to_string(),
            ));
        }
        Ok(Arc::new(HashJoinExec::new(
            children[0].clone(),
            children[1].clone(),
            self.on.clone(),
            self.join_type,
            self.schema.clone(),
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
