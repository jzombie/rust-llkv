// TODO: Move to `llkv-test-utils`

use libtest_mimic::{Arguments, Trial};
use std::path::PathBuf;

// Discover all .slt files under tests/slt
fn find_slt_files(dir: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let base = std::path::Path::new(dir);
    if !base.exists() {
        return out;
    }
    let mut stack = vec![base.to_path_buf()];
    while let Some(p) = stack.pop() {
        if p.is_dir() {
            if let Ok(read) = std::fs::read_dir(&p) {
                for entry in read.flatten() {
                    stack.push(entry.path());
                }
            }
        } else if let Some(ext) = p.extension()
            && ext == "slt"
        {
            out.push(p);
        }
    }
    out.sort();
    out
}

use llkv_test_utils::slt::run_slt_file_with_factory;

async fn run_single_slt(path: &std::path::Path) -> Result<(), llkv_result::Error> {
    // Delegate to the shared test-utils implementation. Provide a factory
    // that constructs a new `SqlEngine<MemPager>` instance for each run.
    use llkv_sql::SqlEngine;
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    // Small wrapper type that implements sqllogictest::AsyncDB by calling
    // into `SqlEngine`. We construct and return this from the factory.
    use arrow::array::Array as ArrowArray;
    use llkv_sql::StatementResult;
    use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};

    struct EngineHarness {
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

    run_slt_file_with_factory(path, || async move { Ok::<_, ()>(EngineHarness::new()) }).await
}

#[test]
fn slt_harness() {
    let files = find_slt_files("tests/slt");
    let mut trials: Vec<Trial> = Vec::new();
    for f in files {
        let name = f
            .strip_prefix("tests")
            .unwrap_or(&f)
            .to_string_lossy()
            .trim_start_matches('/')
            .to_string();
        let path_clone = f.clone();
        trials.push(Trial::test(name, move || {
            let p = path_clone.clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio rt");
            let res: Result<(), llkv_result::Error> =
                rt.block_on(async move { run_single_slt(&p).await });
            // If the slt runner returns an error, panic with the error message so
            // libtest_mimic prints the failing line and context. Returning a
            // libtest_mimic::Failed doesn't accept a message in this version of
            // the crate, so we use panic to surface the text.
            match res {
                Ok(()) => Ok(()),
                Err(e) => panic!("slt runner error: {}", e),
            }
        }));
    }

    let args = Arguments::from_args();
    let _ = libtest_mimic::run(&args, trials);
}

// Unit tests for the mapping/expansion helpers live in `llkv-test-utils`.
