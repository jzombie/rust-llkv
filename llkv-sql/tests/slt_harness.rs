use llkv_test_utils::slt;

#[test]
fn slt_harness() {
    // Create an engine factory closure that will be passed to the shared
    // harness. Keep this file minimal — the real harness lives in
    // `llkv-test-utils`.
    use llkv_dsl::DslContext;
    use llkv_sql::SqlEngine;
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    // Build the same engine adapter that the previous harness used. We keep
    // this crate-specific adapter here so `llkv-test-utils` remains engine-
    // agnostic.
    use arrow::array::Array as ArrowArray;
    use llkv_sql::StatementResult;
    use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};

    struct EngineHarness {
        engine: SqlEngine<MemPager>,
    }
    impl EngineHarness {
        pub fn new(engine: SqlEngine<MemPager>) -> Self {
            Self { engine }
        }
    }

    struct EngineFactory {
        context: Arc<DslContext<MemPager>>,
        default_nulls_first: bool,
    }

    impl EngineFactory {
        fn new() -> Self {
            let pager = Arc::new(MemPager::default());
            let context = Arc::new(DslContext::new(pager));
            Self {
                context,
                default_nulls_first: false,
            }
        }

        fn make_engine(&self) -> SqlEngine<MemPager> {
            SqlEngine::with_context(Arc::clone(&self.context), self.default_nulls_first)
        }
    }

    impl Clone for EngineFactory {
        fn clone(&self) -> Self {
            EngineFactory::new()
        }
    }

    #[async_trait::async_trait]
    impl AsyncDB for EngineHarness {
        type Error = llkv_result::Error;
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
                            Ok(DBOutput::StatementComplete(rows_inserted as u64))
                        }
                        StatementResult::Update { rows_updated, .. } => {
                            Ok(DBOutput::StatementComplete(rows_updated as u64))
                        }
                        StatementResult::CreateTable { .. } => Ok(DBOutput::StatementComplete(0)),
                        StatementResult::Transaction { .. } => Ok(DBOutput::StatementComplete(0)),
                        StatementResult::NoOp => Ok(DBOutput::StatementComplete(0)),
                    }
                }
                Err(e) => Err(e),
            }
        }

        async fn shutdown(&mut self) {}
    }

    // Construct and pass a factory closure to the shared harness.
    let factory = EngineFactory::new();
    slt::run_slt_harness("tests/slt", move || {
        let engine = factory.make_engine();
        async move { Ok::<_, ()>(EngineHarness::new(engine)) }
    });
}

// Unit tests for the mapping/expansion helpers live in `llkv-test-utils`.
