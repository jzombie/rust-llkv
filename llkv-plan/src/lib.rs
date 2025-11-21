use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::datasource::TableProvider;
use datafusion::execution::context::QueryPlanner;
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
use llkv_table::common::{CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, ROW_ID_COLUMN_NAME};
use llkv_table::providers::column_map::ColumnMapTableProvider;
use llkv_table::providers::parquet::LlkvTableProvider;

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
        session_state: &datafusion::execution::context::SessionState,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        println!("LlkvQueryPlanner: {:?}", logical_plan);
        match logical_plan {
            LogicalPlan::Dml(DmlStatement {
                table_name,
                op: WriteOp::Update,
                input,
                ..
            }) => {
                let provider = self
                    .catalog
                    .get_table(table_name.table())
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Table {} not found",
                            table_name.table()
                        ))
                    })?;

                if let Some(ptp) = provider
                    .as_any()
                    .downcast_ref::<LlkvTableProvider<BoxedPager>>()
                {
                    // 1. Create physical plan for the input
                    let input_exec = self
                        .fallback
                        .create_physical_plan(input, session_state)
                        .await?;

                    // 2. Ensure row_id is present
                    let input_schema = input_exec.schema();
                    if input_schema.index_of(ROW_ID_COLUMN_NAME).is_err() {
                        return Err(DataFusionError::Execution(format!(
                            "UPDATE requires {} column to be present in the plan.",
                            ROW_ID_COLUMN_NAME
                        )));
                    }

                    // 3. Create a projection to match ingest schema
                    let ingest_schema = ptp.ingest_schema();
                    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
                        Vec::with_capacity(ingest_schema.fields().len());

                    for field in ingest_schema.fields() {
                        if field.name() == ROW_ID_COLUMN_NAME {
                            let idx = input_schema.index_of(ROW_ID_COLUMN_NAME).unwrap();
                            exprs.push((
                                Arc::new(Column::new(ROW_ID_COLUMN_NAME, idx)),
                                ROW_ID_COLUMN_NAME.to_string(),
                            ));
                        } else if field.name() == CREATED_BY_COLUMN_NAME {
                            exprs.push((
                                Arc::new(Literal::new(ScalarValue::UInt64(Some(0)))),
                                CREATED_BY_COLUMN_NAME.to_string(),
                            ));
                        } else if field.name() == DELETED_BY_COLUMN_NAME {
                            exprs.push((
                                Arc::new(Literal::new(ScalarValue::UInt64(Some(0)))),
                                DELETED_BY_COLUMN_NAME.to_string(),
                            ));
                        } else {
                            // User column
                            if let Ok(idx) = input_schema.index_of(field.name()) {
                                exprs.push((
                                    Arc::new(Column::new(field.name(), idx)),
                                    field.name().to_string(),
                                ));
                            } else {
                                return Err(DataFusionError::Execution(format!(
                                    "Column {} not found in UPDATE input plan",
                                    field.name()
                                )));
                            }
                        }
                    }

                    let projection_plan = Arc::new(ProjectionExec::try_new(exprs, input_exec)?);
                    let sink = ptp.create_sink_from_schema(ingest_schema);

                    return Ok(Arc::new(DataSinkExec::new(projection_plan, sink, None)));
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
                } else if let Some(ptp) = provider
                    .as_any()
                    .downcast_ref::<LlkvTableProvider<BoxedPager>>()
                {
                    // 1. Create physical plan for the input
                    let input_exec = self
                        .fallback
                        .create_physical_plan(input, session_state)
                        .await?;

                    // 2. Create a projection that maps:
                    //    row_id -> row_id
                    //    created_by -> 0 (or from input)
                    //    deleted_by -> 1
                    //    other_col -> NULL
                    let schema = ptp.ingest_schema();
                    let input_schema = input_exec.schema();

                    // Find row_id index in input
                    let row_id_idx = input_schema.index_of(ROW_ID_COLUMN_NAME).map_err(|_| {
                        DataFusionError::Execution(format!(
                            "DELETE requires {} column to be present in the plan.",
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
                        } else if field.name() == CREATED_BY_COLUMN_NAME {
                            if let Ok(idx) = input_schema.index_of(CREATED_BY_COLUMN_NAME) {
                                exprs.push((
                                    Arc::new(Column::new(CREATED_BY_COLUMN_NAME, idx)),
                                    CREATED_BY_COLUMN_NAME.to_string(),
                                ));
                            } else {
                                exprs.push((
                                    Arc::new(Literal::new(ScalarValue::UInt64(Some(0)))),
                                    CREATED_BY_COLUMN_NAME.to_string(),
                                ));
                            }
                        } else if field.name() == DELETED_BY_COLUMN_NAME {
                            exprs.push((
                                Arc::new(Literal::new(ScalarValue::UInt64(Some(1)))),
                                DELETED_BY_COLUMN_NAME.to_string(),
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
                    let sink = ptp.create_sink_from_schema(schema.clone());

                    return Ok(Arc::new(DataSinkExec::new(projection_plan, sink, None)));
                }
            }
            LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(cmd)) => {
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
                    None, // TODO: Extract backend from options if available
                )));
            }
            LogicalPlan::Ddl(DdlStatement::CreateExternalTable(cmd)) => {
                let table_name = cmd.name.table();
                let schema = Arc::new(cmd.schema.as_ref().inner().clone());
                let if_not_exists = cmd.if_not_exists;

                // Determine backend from file_type or options or location
                let backend_name = if let Some(backend) = cmd.options.get("backend") {
                    Some(backend.clone())
                } else if cmd.location.contains("backend=columnstore") {
                    Some("columnstore".to_string())
                } else if cmd.location.contains("backend=parquet") {
                    Some("parquet".to_string())
                } else if cmd.file_type == "PARQUET" {
                    Some("parquet".to_string())
                } else {
                    // Default or infer from other options
                    None
                };

                return Ok(Arc::new(CreateTableExec::new(
                    Arc::clone(&self.catalog),
                    table_name.to_string(),
                    Arc::clone(&schema),
                    None, // External tables usually don't have input plan in this context?
                    if_not_exists,
                    backend_name,
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
