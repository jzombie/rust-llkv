use crate::physical_plan::{BatchIter, PhysicalPlan};
use arrow::datatypes::SchemaRef;
use llkv_plan::plans::AggregateExpr;
use llkv_result::Result;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub struct AggregateExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub input: Arc<dyn PhysicalPlan<P>>,
    pub group_expr: Arc<[usize]>, // Indices in input schema
    pub aggr_expr: Arc<[AggregateExpr]>,
    pub schema: SchemaRef,
}

impl<P> AggregateExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(
        input: Arc<dyn PhysicalPlan<P>>,
        group_expr: impl Into<Arc<[usize]>>,
        aggr_expr: impl Into<Arc<[AggregateExpr]>>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            input,
            group_expr: group_expr.into(),
            aggr_expr: aggr_expr.into(),
            schema,
        }
    }
}

impl<P> fmt::Debug for AggregateExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AggregateExec")
            .field("group_expr", &self.group_expr)
            .field("aggr_expr", &self.aggr_expr)
            .field("schema", &self.schema)
            .finish()
    }
}

impl<P> PhysicalPlan<P> for AggregateExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Result<BatchIter> {
        Err(llkv_result::Error::Internal(
            "Execution of AggregateExec must be handled by the executor crate".to_string(),
        ))
    }

    fn children(&self) -> Arc<[Arc<dyn PhysicalPlan<P>>]> {
        Arc::from([self.input.clone()])
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Arc<[Arc<dyn PhysicalPlan<P>>]>,
    ) -> Result<Arc<dyn PhysicalPlan<P>>> {
        if children.len() != 1 {
            return Err(llkv_result::Error::Internal(
                "AggregateExec expects exactly 1 child".to_string(),
            ));
        }
        Ok(Arc::new(AggregateExec::new(
            children[0].clone(),
            self.group_expr.clone(),
            self.aggr_expr.clone(),
            self.schema.clone(),
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
