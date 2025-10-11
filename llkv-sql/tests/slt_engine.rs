use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::Array as ArrowArray;
use llkv_dsl::{Context, StatementResult};
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};

pub struct EngineHarness {
    engine: SqlEngine<MemPager>,
}

impl EngineHarness {
    pub fn new(engine: SqlEngine<MemPager>) -> Self {
        Self { engine }
    }
}

#[derive(Clone)]
pub struct SharedContext {
    context: Arc<Context<MemPager>>,
}

impl Default for SharedContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedContext {
    pub fn new() -> Self {
        let pager = Arc::new(MemPager::default());
        let context = Arc::new(Context::new(pager));
        Self { context }
    }

    pub fn make_engine(&self) -> SqlEngine<MemPager> {
        SqlEngine::with_context(Arc::clone(&self.context), false)
    }
}

#[async_trait::async_trait]
impl AsyncDB for EngineHarness {
    type Error = llkv_result::Error;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        tracing::trace!(
            "[HARNESS] run() called with sql (length={}, lines={}):",
            sql.len(),
            sql.lines().count()
        );
        for (i, line) in sql.lines().enumerate() {
            tracing::trace!("[HARNESS]   line {}: {:?}", i, line);
        }
        match self.engine.execute(sql) {
            Ok(mut results) => {
                tracing::trace!(
                    "[HARNESS] execute() returned Ok with {} results",
                    results.len()
                );
                if results.is_empty() {
                    return Ok(DBOutput::StatementComplete(0));
                }
                let result = results.remove(0);
                match result {
                    StatementResult::Select { execution, .. } => {
                        let batches = execution.collect()?;
                        let mut rows: Vec<Vec<String>> = Vec::new();
                        for batch in &batches {
                            for row_idx in 0..batch.num_rows() {
                                let mut row: Vec<String> = Vec::new();
                                for col in 0..batch.num_columns() {
                                    let array = batch.column(col);
                                    let val = match array.data_type() {
                                        arrow::datatypes::DataType::Int64 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::Int64Array>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        arrow::datatypes::DataType::UInt64 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::UInt64Array>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        arrow::datatypes::DataType::Float64 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::Float64Array>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        arrow::datatypes::DataType::Utf8 => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<arrow::array::StringArray>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                a.value(row_idx).to_string()
                                            }
                                        }
                                        _ => "".to_string(),
                                    };
                                    row.push(val);
                                }
                                rows.push(row);
                            }
                        }

                        let types = if let Some(first) = batches.first() {
                            (0..first.num_columns())
                                .map(|col| match first.column(col).data_type() {
                                    arrow::datatypes::DataType::Int64
                                    | arrow::datatypes::DataType::UInt64 => {
                                        DefaultColumnType::Integer
                                    }
                                    arrow::datatypes::DataType::Float64 => {
                                        DefaultColumnType::FloatingPoint
                                    }
                                    arrow::datatypes::DataType::Utf8 => DefaultColumnType::Text,
                                    _ => DefaultColumnType::Any,
                                })
                                .collect()
                        } else {
                            vec![]
                        };

                        Ok(DBOutput::Rows { types, rows })
                    }
                    StatementResult::Insert { rows_inserted, .. } => {
                        // Return as a single-row result for compatibility with query directives
                        Ok(DBOutput::Rows {
                            types: vec![DefaultColumnType::Integer],
                            rows: vec![vec![rows_inserted.to_string()]],
                        })
                    }
                    StatementResult::Update { rows_updated, .. } => {
                        // Return as a single-row result for compatibility with query directives
                        Ok(DBOutput::Rows {
                            types: vec![DefaultColumnType::Integer],
                            rows: vec![vec![rows_updated.to_string()]],
                        })
                    }
                    StatementResult::Delete { rows_deleted, .. } => {
                        // Return as a single-row result for compatibility with query directives
                        Ok(DBOutput::Rows {
                            types: vec![DefaultColumnType::Integer],
                            rows: vec![vec![rows_deleted.to_string()]],
                        })
                    }
                    StatementResult::CreateTable { .. } => Ok(DBOutput::StatementComplete(0)),
                    StatementResult::Transaction { .. } => Ok(DBOutput::StatementComplete(0)),
                    StatementResult::NoOp => Ok(DBOutput::StatementComplete(0)),
                }
            }
            Err(e) => {
                tracing::trace!("[HARNESS] execute() returned Err: {:?}", e);
                Err(e)
            }
        }
    }

    async fn shutdown(&mut self) {}
}

pub type HarnessFuture = Pin<Box<dyn Future<Output = Result<EngineHarness, ()>> + Send + 'static>>;
pub type HarnessFactory = Box<dyn Fn() -> HarnessFuture + Send + Sync + 'static>;

pub fn make_factory_factory() -> impl Fn() -> HarnessFactory + Clone {
    || {
        tracing::trace!("[FACTORY] make_factory_factory: Creating SharedContext");
        let shared = SharedContext::new();
        let factory: HarnessFactory = Box::new(move || {
            tracing::trace!("[FACTORY] Factory called: Creating new EngineHarness");
            let shared_clone = shared.clone();
            Box::pin(async move {
                let engine = shared_clone.make_engine();
                tracing::trace!("[FACTORY] Created SqlEngine with new Session");
                Ok::<_, ()>(EngineHarness::new(engine))
            })
        });
        factory
    }
}
