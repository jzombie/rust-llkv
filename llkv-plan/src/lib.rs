use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::TableProvider;
use datafusion::execution::context::{QueryPlanner, SessionState};
use datafusion::logical_expr::{DdlStatement, DmlStatement, LogicalPlan, WriteOp};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion::scalar::ScalarValue;
use datafusion_datasource::sink::DataSinkExec;
use llkv_storage::pager::BoxedPager;
use llkv_table::catalog::TableCatalog;
use llkv_table::common::ROW_ID_COLUMN_NAME;
use llkv_table::providers::column_map::ColumnMapTableProvider;

/// Custom physical planner that intercepts DDL and DML operations.
pub struct LlkvQueryPlanner {
    catalog: Arc<TableCatalog>,
    fallback: Arc<dyn PhysicalPlanner>,
}

impl fmt::Debug for LlkvQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlkvQueryPlanner")
            .field("catalog", &"<TableCatalog>")
            .finish()
    }
}

impl LlkvQueryPlanner {
    /// Construct a new planner with a catalog reference.
    pub fn new(catalog: Arc<TableCatalog>) -> Self {
        Self {
            catalog,
            fallback: Arc::new(DefaultPhysicalPlanner::default()),
        }
    }
}

#[async_trait]
impl QueryPlanner for LlkvQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        println!("LlkvQueryPlanner received plan: {:?}", logical_plan);
        if let LogicalPlan::Ddl(_) = logical_plan {
             println!("LlkvQueryPlanner DDL: {:?}", logical_plan);
        }
        match logical_plan {
            LogicalPlan::Dml(DmlStatement {
                table_name,
                op: WriteOp::Update,
                input,
                ..
            }) => {
                let name = table_name.table();
                let provider = self
                    .catalog
                    .get_table(name)
                    .map_err(|e| DataFusionError::Internal(e.to_string()))?
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!("Table {} not found", name))
                    })?;

                if let Some(cmp) = provider
                    .as_any()
                    .downcast_ref::<ColumnMapTableProvider<BoxedPager>>()
                {
                    let input_exec = self
                        .fallback
                        .create_physical_plan(input, session_state)
                        .await?;
                    let sink = cmp.create_sink();
                    return Ok(Arc::new(DataSinkExec::new(input_exec, sink, None)));
                }
            }
            LogicalPlan::Dml(DmlStatement {
                table_name,
                op: WriteOp::Delete,
                input,
                ..
            }) => {
                let name = table_name.table();
                let provider = self
                    .catalog
                    .get_table(name)
                    .map_err(|e| DataFusionError::Internal(e.to_string()))?
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!("Table {} not found", name))
                    })?;

                if let Some(cmp) = provider
                    .as_any()
                    .downcast_ref::<ColumnMapTableProvider<BoxedPager>>()
                {
                    // 1. Create physical plan for the input
                    let input_exec = self
                        .fallback
                        .create_physical_plan(input, session_state)
                        .await?;

                    // 2. Create a projection that maps:
                    //    row_id -> row_id
                    //    other_col -> NULL
                    let schema = cmp.schema();
                    let input_schema = input_exec.schema();

                    // Find row_id index in input
                    let row_id_idx = input_schema.index_of(ROW_ID_COLUMN_NAME).map_err(|_| {
                        DataFusionError::Execution(format!(
                            "DELETE requires {} column to be present in the plan. \
                             This might require an optimizer rule to force its selection.",
                            ROW_ID_COLUMN_NAME
                        ))
                    })?;

                    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
                        Vec::with_capacity(schema.fields().len());

                    for field in schema.fields() {
                        if field.name() == ROW_ID_COLUMN_NAME {
                            exprs.push((
                                Arc::new(Column::new(ROW_ID_COLUMN_NAME, row_id_idx)),
                                ROW_ID_COLUMN_NAME.to_string(),
                            ));
                        } else {
                            let null_value = ScalarValue::try_from(field.data_type())?;
                            exprs.push((
                                Arc::new(Literal::new(null_value)),
                                field.name().to_string(),
                            ));
                        }
                    }

                    let projection_plan = Arc::new(ProjectionExec::try_new(exprs, input_exec)?);
                    let sink = cmp.create_sink();

                    // 3. Sink the projected NULLs (which triggers LWW delete)
                    return Ok(Arc::new(DataSinkExec::new(projection_plan, sink, None)));
                }
            }
            LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(cmd)) => {
                println!("Intercepted CreateMemoryTable for {}", cmd.name);
                let table_name = cmd.name.table();

                let input = &cmd.input;

                let schema = Arc::new(input.schema().inner().clone());

                let input_plan = self
                    .fallback
                    .create_physical_plan(input, session_state)
                    .await?;

                return Ok(Arc::new(CreateTableExec::new(
                    Arc::clone(&self.catalog),
                    table_name.to_string(),
                    Arc::clone(&schema),
                    Some(input_plan),
                    cmd.if_not_exists,
                )));
            }
            _ => {}
        }

        self.fallback
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

pub mod ddl;
use ddl::CreateTableExec;
