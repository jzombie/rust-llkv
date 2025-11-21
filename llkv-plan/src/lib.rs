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
        match logical_plan {
            LogicalPlan::Extension(extension) => {
                let maybe_node = match extension.node.as_any() {
                    node if node.is::<LlkvCreateTable>() => {
                        Some(node.downcast_ref::<LlkvCreateTable>().unwrap().clone())
                    }
                    _ => None,
                };

                match maybe_node {
                    Some(node) => {
                        let input_plan = if let Some(input) = &node.input {
                            Some(
                                self.fallback
                                    .create_physical_plan(input, session_state)
                                    .await?,
                            )
                        } else {
                            None
                        };

                        let schema = node.schema.inner().clone();

                        return Ok(Arc::new(CreateTableExec::new(
                            Arc::clone(&self.catalog),
                            node.name.clone(),
                            schema,
                            input_plan,
                            node.if_not_exists,
                            node.backend.clone(),
                        )));
                    }
                    _ => {}
                }
            }
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

                if let Some(cmp) = provider
                    .as_any()
                    .downcast_ref::<ColumnMapTableProvider<BoxedPager>>()
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

                    // 3. Create a projection to match table schema
                    let schema = cmp.schema();
                    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
                        Vec::with_capacity(schema.fields().len());

                    for field in schema.fields() {
                        if field.name() == ROW_ID_COLUMN_NAME {
                            let idx = input_schema.index_of(ROW_ID_COLUMN_NAME).unwrap();
                            exprs.push((
                                Arc::new(Column::new(ROW_ID_COLUMN_NAME, idx)),
                                ROW_ID_COLUMN_NAME.to_string(),
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
                    let sink = cmp.create_sink();

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
            _ => {}
        }

        self.fallback
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

pub mod ddl;
use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::UserDefinedLogicalNodeCore;
use ddl::CreateTableExec;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LlkvCreateTable {
    pub name: String,
    pub schema: DFSchemaRef,
    pub input: Option<LogicalPlan>,
    pub if_not_exists: bool,
    pub backend: Option<String>,
}

impl PartialOrd for LlkvCreateTable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.name.partial_cmp(&other.name)
    }
}

impl UserDefinedLogicalNodeCore for LlkvCreateTable {
    fn name(&self) -> &str {
        "LlkvCreateTable"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        match &self.input {
            Some(input) => vec![input],
            None => vec![],
        }
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<datafusion::logical_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "LlkvCreateTable: name={}, backend={:?}",
            self.name, self.backend
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DataFusionResult<Self> {
        Ok(LlkvCreateTable {
            name: self.name.clone(),
            schema: self.schema.clone(),
            input: inputs.into_iter().next(),
            if_not_exists: self.if_not_exists,
            backend: self.backend.clone(),
        })
    }
}
