use arrow::array::Array as ArrowArray;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray, UInt64Array};
use llkv_result::Error as LlkvError;
use llkv_sql::{SqlEngine, StatementResult};
use llkv_storage::pager::MemPager;
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType, Runner};

pub struct EngineHarness {
    engine: SqlEngine<MemPager>,
}

impl EngineHarness {
    pub fn new() -> Self {
        let pager = Arc::new(MemPager::default());
        Self {
            engine: SqlEngine::new(pager),
        }
    }
}

impl Default for EngineHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AsyncDB for EngineHarness {
    type Error = LlkvError;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        match self.engine.execute(sql) {
            Ok(mut results) => {
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
                                                .downcast_ref::<Int64Array>()
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
                                                .downcast_ref::<UInt64Array>()
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
                                                .downcast_ref::<Float64Array>()
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
                                                .downcast_ref::<StringArray>()
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
                        Ok(DBOutput::StatementComplete(rows_inserted as u64))
                    }
                    StatementResult::Update { rows_updated, .. } => {
                        Ok(DBOutput::StatementComplete(rows_updated as u64))
                    }
                    StatementResult::CreateTable { .. } => Ok(DBOutput::StatementComplete(0)),
                    StatementResult::NoOp => Ok(DBOutput::StatementComplete(0)),
                    StatementResult::Transaction { .. } => Ok(DBOutput::StatementComplete(0)),
                }
            }
            Err(e) => Err(e),
        }
    }

    async fn shutdown(&mut self) {}
}

#[tokio::test]
async fn run_slt_basic() {
    let mut runner = Runner::new(|| async { Ok(EngineHarness::new()) });

    // Create a tiny inline script in the system temp dir and run it via the runner.
    let script = "";
    let mut path = std::env::temp_dir();
    path.push(format!("llkv_basic_{}.slt", std::process::id()));
    std::fs::write(&path, script).expect("write slt");

    // use the async variant
    runner
        .run_file_async(&path)
        .await
        .expect("slt runner failed");
}

// TODO: Improve test
#[test]
fn validator_space_vs_tab() {
    use sqllogictest::runner::{default_normalizer, default_validator};

    let actual = vec![
        vec!["1".to_string(), "3.14".to_string()],
        vec!["2".to_string(), "2.71".to_string()],
    ];
    let expected = vec!["1 3.14".to_string(), "2 2.71".to_string()];

    assert!(default_validator(default_normalizer, &actual, &expected));
}
