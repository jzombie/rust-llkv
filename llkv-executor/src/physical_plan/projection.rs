use crate::physical_plan::{PhysicalPlan, BatchIter};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use llkv_expr::expr::ScalarExpr;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use llkv_result::Result;
use llkv_compute::eval::ScalarEvaluator;
use rustc_hash::FxHashMap;

pub struct ProjectionExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub input: Arc<dyn PhysicalPlan<P>>,
    pub schema: SchemaRef,
    pub expr: Vec<(ScalarExpr<usize>, String)>, // Expression, output name
}

impl<P> ProjectionExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(
        input: Arc<dyn PhysicalPlan<P>>,
        schema: SchemaRef,
        expr: Vec<(ScalarExpr<usize>, String)>,
    ) -> Self {
        Self {
            input,
            schema,
            expr,
        }
    }
}

impl<P> fmt::Debug for ProjectionExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectionExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl<P> PhysicalPlan<P> for ProjectionExec<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Result<BatchIter> {
        let input_stream = self.input.execute()?;
        let schema = self.schema.clone();
        let expr = self.expr.clone();

        Ok(Box::new(input_stream.map(move |batch_result| {
            let batch = batch_result?;
            
            let mut field_arrays = FxHashMap::default();
            for (i, col) in batch.columns().iter().enumerate() {
                field_arrays.insert(i, col.clone());
            }

            let mut columns = Vec::new();
            for (i, (e, _)) in expr.iter().enumerate() {
                let col = ScalarEvaluator::evaluate_batch_simplified(
                    e,
                    batch.num_rows(),
                    &field_arrays,
                ).map_err(|e| llkv_result::Error::Internal(e.to_string()))?;

                // Cast if needed
                let expected_type = schema.field(i).data_type();
                let col = if col.data_type() != expected_type {
                    // panic!("DEBUG: ProjectionExec mismatch! Col: {:?}, Expected: {:?}", col.data_type(), expected_type);
                    
                    arrow::compute::cast(&col, expected_type)
                        .map_err(|e| llkv_result::Error::Internal(e.to_string()))?
                    
                } else {
                    col
                };

                columns.push(col);
            }

            if columns.is_empty() {
                let options = arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                RecordBatch::try_new_with_options(schema.clone(), columns, &options).map_err(|e| llkv_result::Error::Internal(e.to_string()))
            } else {
                RecordBatch::try_new(schema.clone(), columns).map_err(|e| llkv_result::Error::Internal(e.to_string()))
            }
        })))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan<P>>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan<P>>>,
    ) -> Result<Arc<dyn PhysicalPlan<P>>> {
        if children.len() != 1 {
            return Err(llkv_result::Error::Internal("ProjectionExec expects exactly one child".to_string()));
        }
        Ok(Arc::new(ProjectionExec::new(
            children[0].clone(),
            self.schema.clone(),
            self.expr.clone(),
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

