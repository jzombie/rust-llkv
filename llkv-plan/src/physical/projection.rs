use std::any::Any;
use std::sync::Arc;
use std::fmt;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use crate::physical::PhysicalPlan;

pub struct ProjectionExec {
    pub input: Arc<dyn PhysicalPlan>,
    pub schema: SchemaRef,
    pub expr: Vec<(usize, String)>, // Index in input schema, output name
}

impl ProjectionExec {
    pub fn new(input: Arc<dyn PhysicalPlan>, schema: SchemaRef, expr: Vec<(usize, String)>) -> Self {
        Self { input, schema, expr }
    }
}

impl fmt::Debug for ProjectionExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectionExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PhysicalPlan for ProjectionExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn execute(&self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, String>> + Send>, String> {
        let input_stream = self.input.execute()?;
        let schema = self.schema.clone();
        let expr = self.expr.clone();

        Ok(Box::new(input_stream.map(move |batch_result| {
            let batch = batch_result?;
            let columns: Vec<_> = expr.iter()
                .map(|(idx, _)| batch.column(*idx).clone())
                .collect();
            
            RecordBatch::try_new(schema.clone(), columns).map_err(|e| e.to_string())
        })))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Result<Arc<dyn PhysicalPlan>, String> {
        if children.len() != 1 {
            return Err("ProjectionExec expects exactly one child".to_string());
        }
        Ok(Arc::new(ProjectionExec::new(children[0].clone(), self.schema.clone(), self.expr.clone())))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
