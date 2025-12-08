use crate::physical_plan::{PhysicalPlan, BatchIter};
use arrow::datatypes::SchemaRef;
use llkv_expr::expr::Expr;
use llkv_types::FieldId;
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use llkv_result::Result;

pub struct FilterExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub input: Arc<dyn PhysicalPlan<P>>,
    pub predicate: Expr<'static, FieldId>,
    pub schema: SchemaRef,
}

impl<P> FilterExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(
        input: Arc<dyn PhysicalPlan<P>>,
        predicate: Expr<'static, FieldId>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            input,
            predicate,
            schema,
        }
    }
}

impl<P> fmt::Debug for FilterExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterExec")
            .field("predicate", &self.predicate)
            .field("schema", &self.schema)
            .finish()
    }
}

impl<P> PhysicalPlan<P> for FilterExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Result<BatchIter> {
        Err(llkv_result::Error::Internal("Execution of FilterExec must be handled by the executor crate".to_string()))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan<P>>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan<P>>>,
    ) -> Result<Arc<dyn PhysicalPlan<P>>> {
        if children.len() != 1 {
            return Err(llkv_result::Error::Internal("FilterExec expects exactly 1 child".to_string()));
        }
        Ok(Arc::new(FilterExec::new(
            children[0].clone(),
            self.predicate.clone(),
            self.schema.clone(),
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

