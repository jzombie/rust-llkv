use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::Result as DataFusionResult;
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::stream::StreamExt;
use llkv_table::catalog::TableCatalog;

/// Physical plan for creating a table in the LLKV catalog.
pub struct CreateTableExec {
    catalog: Arc<TableCatalog>,
    table_name: String,
    schema: SchemaRef,
    input: Option<Arc<dyn ExecutionPlan>>,
    if_not_exists: bool,
    backend_name: Option<String>,
    properties: PlanProperties,
}

impl fmt::Debug for CreateTableExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CreateTableExec")
            .field("table_name", &self.table_name)
            .field("if_not_exists", &self.if_not_exists)
            .field("backend_name", &self.backend_name)
            .finish_non_exhaustive()
    }
}

impl CreateTableExec {
    pub fn new(
        catalog: Arc<TableCatalog>,
        table_name: String,
        schema: SchemaRef,
        input: Option<Arc<dyn ExecutionPlan>>,
        if_not_exists: bool,
        backend_name: Option<String>,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            catalog,
            table_name,
            schema,
            input,
            if_not_exists,
            backend_name,
            properties,
        }
    }
}

impl DisplayAs for CreateTableExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "CreateTableExec: name={}, if_not_exists={}",
            self.table_name, self.if_not_exists
        )
    }
}

impl ExecutionPlan for CreateTableExec {
    fn name(&self) -> &str {
        "CreateTableExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        if let Some(input) = &self.input {
            vec![input]
        } else {
            vec![]
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let input = if children.is_empty() {
            None
        } else {
            Some(Arc::clone(&children[0]))
        };
        Ok(Arc::new(CreateTableExec::new(
            Arc::clone(&self.catalog),
            self.table_name.clone(),
            Arc::clone(&self.schema),
            input,
            self.if_not_exists,
            self.backend_name.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<datafusion::physical_plan::SendableRecordBatchStream> {
        let catalog = Arc::clone(&self.catalog);
        let table_name = self.table_name.clone();
        let schema = Arc::clone(&self.schema);
        let schema_adapter = Arc::clone(&self.schema);
        let input = self.input.clone();
        let if_not_exists = self.if_not_exists;
        let backend_name = self.backend_name.clone();

        let stream = futures::stream::once(async move {
            println!("CreateTableExec: executing for table {}", table_name);
            // 1. Check if table exists
            if if_not_exists
                && catalog
                    .get_table(&table_name)
                    .map_err(|e| datafusion::error::DataFusionError::Internal(e.to_string()))?
                    .is_some()
            {
                return Ok(RecordBatch::new_empty(Arc::clone(&schema)));
            }

            // 2. Create table
            let mut builder = catalog
                .create_table(&table_name, Arc::clone(&schema), backend_name.as_deref())
                .map_err(|e| datafusion::error::DataFusionError::Internal(e.to_string()))?;

            // 3. If CTAS, execute input and append
            if let Some(input_plan) = input {
                let mut stream = input_plan.execute(partition, context)?;
                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    builder
                        .append_batch(&batch)
                        .map_err(|e| datafusion::error::DataFusionError::Internal(e.to_string()))?;
                }
            }

            // 4. Finish builder
            let _provider = builder
                .finish()
                .map_err(|e| datafusion::error::DataFusionError::Internal(e.to_string()))?;

            Ok(RecordBatch::new_empty(Arc::clone(&schema)))
        });

        Ok(Box::pin(
            datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
                schema_adapter,
                Box::pin(stream),
            ),
        ))
    }
}
