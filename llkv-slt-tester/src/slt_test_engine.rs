use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use arrow::array::{
    Array as ArrowArray, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray,
    StructArray, UInt64Array,
};
use llkv_result::Error;
use llkv_runtime::{RuntimeContext, RuntimeStatementResult};
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};

/// Thread-local storage for expected column types from sqllogictest directives.
mod expectations {
    use sqllogictest::DefaultColumnType;
    use std::cell::RefCell;

    thread_local! {
        /// Caches the column types declared by the current sqllogictest statement so we can format
        /// the next result batch with the exact textual shape the harness expects.
        static EXPECTED_COLUMN_TYPES: RefCell<Option<Vec<DefaultColumnType>>> = const { RefCell::new(None) };
    }

    /// Records the sqllogictest column type expectations for the next query result.
    pub(crate) fn set(types: Vec<DefaultColumnType>) {
        EXPECTED_COLUMN_TYPES.with(|cell| {
            *cell.borrow_mut() = Some(types);
        });
    }

    /// Removes any pending column type overrides so future statements fall back to inference.
    pub(crate) fn clear() {
        EXPECTED_COLUMN_TYPES.with(|cell| {
            cell.borrow_mut().take();
        });
    }

    /// Check if column types are currently set without consuming them.
    pub(super) fn is_set() -> bool {
        EXPECTED_COLUMN_TYPES.with(|cell| cell.borrow().is_some())
    }

    /// Drains the cached sqllogictest column types, returning ownership to the caller if present.
    pub(super) fn take() -> Option<Vec<DefaultColumnType>> {
        EXPECTED_COLUMN_TYPES.with(|cell| cell.borrow_mut().take())
    }
}

// Re-export the functions that the runner needs
pub(crate) use expectations::{
    clear as clear_expected_column_types, set as set_expected_column_types,
};

/// Statistics tracking for SLT query execution.
#[derive(Debug, Clone)]
pub struct QueryStats {
    /// Total number of queries executed
    pub total_queries: usize,
    /// Total number of statements executed
    pub total_statements: usize,
    /// Execution times for each query (sql, duration, result_type)
    pub query_timings: Vec<(String, Duration, String)>,
}

impl QueryStats {
    fn new() -> Self {
        Self {
            total_queries: 0,
            total_statements: 0,
            query_timings: Vec::new(),
        }
    }

    /// Get the top N slowest queries
    pub fn slowest_queries(&self, n: usize) -> Vec<(String, Duration, String)> {
        let mut sorted = self.query_timings.clone();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(n).collect()
    }

    /// Get total execution time
    pub fn total_duration(&self) -> Duration {
        self.query_timings.iter().map(|(_, d, _)| *d).sum()
    }

    /// Get average query time
    pub fn average_duration(&self) -> Duration {
        if self.total_queries == 0 {
            Duration::ZERO
        } else {
            self.total_duration() / self.total_queries as u32
        }
    }

    /// Print statistics as a formatted Arrow table
    pub fn print_summary(&self) {
        if self.total_queries == 0 && self.total_statements == 0 {
            println!("\n📊 No queries or statements executed");
            return;
        }

        // Determine build profile
        let profile = if cfg!(debug_assertions) {
            "DEBUG"
        } else {
            "RELEASE"
        };

        println!("\n📊 SLT Execution Statistics [{}]", profile);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Total Queries:     {}", self.total_queries);
        println!("  Total Statements:  {}", self.total_statements);
        println!("  Total Duration:    {:?}", self.total_duration());
        println!("  Average Query:     {:?}", self.average_duration());
        println!();

        if !self.query_timings.is_empty() {
            self.print_slowest_queries(10);
        }
    }

    /// Print top N slowest queries in Arrow table format
    pub fn print_slowest_queries(&self, n: usize) {
        let slowest = self.slowest_queries(n);
        if slowest.is_empty() {
            return;
        }

        use arrow::array::{Float64Array, StringArray, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;

        println!("🐌 Top {} Slowest Queries:", slowest.len().min(n));
        println!();

        let ranks: Vec<u64> = (1..=slowest.len() as u64).collect();
        let durations_ms: Vec<f64> = slowest
            .iter()
            .map(|(_, d, _)| d.as_secs_f64() * 1000.0)
            .collect();
        let types: Vec<String> = slowest.iter().map(|(_, _, t)| t.clone()).collect();
        let queries: Vec<String> = slowest
            .iter()
            .map(|(sql, _, _)| {
                if sql.len() > 80 {
                    format!("{}...", &sql[..77])
                } else {
                    sql.clone()
                }
            })
            .collect();

        let schema = Schema::new(vec![
            Field::new("Rank", DataType::UInt64, false),
            Field::new("Duration (ms)", DataType::Float64, false),
            Field::new("Type", DataType::Utf8, false),
            Field::new("Query", DataType::Utf8, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt64Array::from(ranks)),
                Arc::new(Float64Array::from(durations_ms)),
                Arc::new(StringArray::from(types)),
                Arc::new(StringArray::from(queries)),
            ],
        )
        .expect("failed to create record batch");

        // Pretty print using arrow's built-in utilities
        let formatted =
            arrow::util::pretty::pretty_format_batches(&[batch]).expect("failed to format table");
        println!("{}", formatted);
    }
}

/// Global statistics tracker
static STATS: LazyLock<Arc<Mutex<Option<QueryStats>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(None)));

/// Enable statistics collection
pub fn enable_stats() {
    let mut stats = STATS.lock().unwrap();
    *stats = Some(QueryStats::new());
}

/// Disable statistics collection and return collected stats
pub fn take_stats() -> Option<QueryStats> {
    let mut stats = STATS.lock().unwrap();
    stats.take()
}

/// Record a query execution
fn record_query(sql: &str, duration: Duration, result_type: &str) {
    if let Some(stats) = STATS.lock().unwrap().as_mut() {
        stats.total_queries += 1;
        stats
            .query_timings
            .push((sql.to_string(), duration, result_type.to_string()));
    }
}

/// Record a statement execution
fn record_statement(sql: &str, duration: Duration, result_type: &str) {
    if let Some(stats) = STATS.lock().unwrap().as_mut() {
        stats.total_statements += 1;
        stats
            .query_timings
            .push((sql.to_string(), duration, result_type.to_string()));
    }
}

/// Tokio-agnostic harness that adapts `SqlEngine` to the `sqllogictest` runner.
pub struct EngineHarness {
    engine: SqlEngine<MemPager>,
}

impl EngineHarness {
    pub fn new(engine: SqlEngine<MemPager>) -> Self {
        tracing::debug!("[HARNESS] new() created harness at {:p}", &engine);
        // The SLT workload streams thousands of literal INSERTs. Enable cross-statement
        // batching so we exercise the optimized ingestion path while keeping single-engine
        // unit tests on the default immediate execution path.
        engine
            .set_insert_buffering(true)
            .expect("enable insert buffering");
        Self { engine }
    }
}

#[derive(Clone)]
pub struct SharedContext {
    context: Arc<RuntimeContext<MemPager>>,
}

impl Default for SharedContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedContext {
    pub fn new() -> Self {
        let pager = Arc::new(MemPager::default());
        let context = Arc::new(RuntimeContext::new(pager));
        Self { context }
    }

    pub fn make_engine(&self) -> SqlEngine<MemPager> {
        SqlEngine::with_context(Arc::clone(&self.context), false)
    }
}

fn format_struct_value(struct_array: &StructArray, row_idx: usize) -> String {
    let mut parts = Vec::new();
    let field_names = struct_array.column_names();
    for (field_idx, column) in struct_array.columns().iter().enumerate() {
        let field_name = field_names[field_idx];
        let value_str = match column.data_type() {
            arrow::datatypes::DataType::Int64 => {
                let a = column.as_any().downcast_ref::<Int64Array>().unwrap();
                if a.is_null(row_idx) {
                    "NULL".to_string()
                } else {
                    a.value(row_idx).to_string()
                }
            }
            arrow::datatypes::DataType::Int32 => {
                let a = column.as_any().downcast_ref::<Int32Array>().unwrap();
                if a.is_null(row_idx) {
                    "NULL".to_string()
                } else {
                    a.value(row_idx).to_string()
                }
            }
            arrow::datatypes::DataType::Utf8 => {
                let a = column.as_any().downcast_ref::<StringArray>().unwrap();
                if a.is_null(row_idx) {
                    "NULL".to_string()
                } else {
                    format!("'{}'", a.value(row_idx))
                }
            }
            arrow::datatypes::DataType::Float64 => {
                let a = column.as_any().downcast_ref::<Float64Array>().unwrap();
                if a.is_null(row_idx) {
                    "NULL".to_string()
                } else {
                    // SQLite-compatible float formatting - try to match expected format
                    let val = a.value(row_idx);
                    val.to_string()
                }
            }
            arrow::datatypes::DataType::Boolean => {
                let a = column.as_any().downcast_ref::<BooleanArray>().unwrap();
                if a.is_null(row_idx) {
                    "NULL".to_string()
                } else if a.value(row_idx) {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            arrow::datatypes::DataType::Struct(_) => {
                let a = column.as_any().downcast_ref::<StructArray>().unwrap();
                if a.is_null(row_idx) {
                    "NULL".to_string()
                } else {
                    format_struct_value(a, row_idx)
                }
            }
            _ => "NULL".to_string(),
        };
        parts.push(format!("'{}': {}", field_name, value_str));
    }
    format!("{{{}}}", parts.join(", "))
}

#[async_trait::async_trait]
impl AsyncDB for EngineHarness {
    type Error = Error;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        tracing::debug!("[HARNESS] run() called, sql=\"{}\"", sql.trim());
        let start = std::time::Instant::now();
        match self.engine.execute(sql) {
            Ok(mut results) => {
                if results.is_empty() {
                    let duration = start.elapsed();
                    record_statement(sql, duration, "EMPTY");
                    return Ok(DBOutput::StatementComplete(0));
                }
                let mut result = results.remove(0);
                let in_query_context = expectations::is_set();
                if in_query_context
                    && let RuntimeStatementResult::Insert { rows_inserted, .. } = &result
                    && *rows_inserted == 0
                    && let Ok(flushed) = self.engine.flush_pending_inserts()
                    && let Some(first) = flushed.into_iter().next()
                {
                    // When the current INSERT buffered zero rows we need to surface the first
                    // newly flushed result so sqllogictest observes the expected row count.
                    result = first;
                }
                match result {
                    RuntimeStatementResult::Select { execution, .. } => {
                        let batches = execution.collect()?;
                        let mut expected_types = expectations::take();
                        let mut rows: Vec<Vec<String>> = Vec::new();
                        for batch in &batches {
                            for row_idx in 0..batch.num_rows() {
                                let mut row: Vec<String> = Vec::new();
                                for col in 0..batch.num_columns() {
                                    let array = batch.column(col);
                                    let expected_type = expected_types
                                        .as_ref()
                                        .and_then(|types| types.get(col))
                                        .cloned();
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
                                            } else if matches!(
                                                expected_type,
                                                Some(DefaultColumnType::Integer)
                                            ) {
                                                let value = a.value(row_idx).trunc();
                                                if !value.is_finite()
                                                    || value < i64::MIN as f64
                                                    || value > i64::MAX as f64
                                                {
                                                    value.to_string()
                                                } else {
                                                    (value as i64).to_string()
                                                }
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
                                                let text = a.value(row_idx);
                                                // SQLite-style coercion: when test expects INTEGER output, parse text as number
                                                // This only applies to SELECT output, not intermediate arithmetic operations
                                                if matches!(
                                                    expected_type,
                                                    Some(DefaultColumnType::Integer)
                                                ) {
                                                    text.trim()
                                                        .parse::<i64>()
                                                        .unwrap_or(0)
                                                        .to_string()
                                                } else {
                                                    text.to_string()
                                                }
                                            }
                                        }
                                        arrow::datatypes::DataType::Boolean => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<BooleanArray>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else if a.value(row_idx) {
                                                "1".to_string()
                                            } else {
                                                "0".to_string()
                                            }
                                        }
                                        arrow::datatypes::DataType::Struct(_) => {
                                            let a = array
                                                .as_any()
                                                .downcast_ref::<StructArray>()
                                                .unwrap();
                                            if a.is_null(row_idx) {
                                                "NULL".to_string()
                                            } else {
                                                format_struct_value(a, row_idx)
                                            }
                                        }
                                        arrow::datatypes::DataType::Null => "NULL".to_string(),
                                        _ => "".to_string(),
                                    };
                                    row.push(val);
                                }
                                rows.push(row);
                            }
                        }

                        let types = if let Some(expected) = expected_types.take() {
                            expected
                        } else if let Some(first) = batches.first() {
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

                        tracing::debug!(
                            "[HARNESS] Returning {} rows with types {:?}",
                            rows.len(),
                            types
                        );
                        tracing::debug!(
                            "[HARNESS] Total values: {}",
                            rows.len() * rows.first().map(|r| r.len()).unwrap_or(0)
                        );
                        for (i, row) in rows.iter().enumerate() {
                            tracing::debug!(
                                "[HARNESS] Row {}: {} columns = {:?}",
                                i,
                                row.len(),
                                row
                            );
                        }

                        let duration = start.elapsed();
                        record_query(sql, duration, "SELECT");
                        Ok(DBOutput::Rows { types, rows })
                    }
                    RuntimeStatementResult::Insert { rows_inserted, .. } => {
                        let duration = start.elapsed();
                        record_statement(sql, duration, "INSERT");
                        if in_query_context {
                            let types = expectations::take()
                                .unwrap_or_else(|| vec![DefaultColumnType::Integer]);
                            Ok(DBOutput::Rows {
                                types,
                                rows: vec![vec![rows_inserted.to_string()]],
                            })
                        } else {
                            Ok(DBOutput::StatementComplete(rows_inserted as u64))
                        }
                    }
                    RuntimeStatementResult::Update { rows_updated, .. } => {
                        let duration = start.elapsed();
                        record_statement(sql, duration, "UPDATE");
                        if in_query_context {
                            let types = expectations::take()
                                .unwrap_or_else(|| vec![DefaultColumnType::Integer]);
                            Ok(DBOutput::Rows {
                                types,
                                rows: vec![vec![rows_updated.to_string()]],
                            })
                        } else {
                            Ok(DBOutput::StatementComplete(rows_updated as u64))
                        }
                    }
                    RuntimeStatementResult::Delete { rows_deleted, .. } => {
                        let duration = start.elapsed();
                        record_statement(sql, duration, "DELETE");
                        if in_query_context {
                            let types = expectations::take()
                                .unwrap_or_else(|| vec![DefaultColumnType::Integer]);
                            Ok(DBOutput::Rows {
                                types,
                                rows: vec![vec![rows_deleted.to_string()]],
                            })
                        } else {
                            Ok(DBOutput::StatementComplete(rows_deleted as u64))
                        }
                    }
                    RuntimeStatementResult::CreateTable { .. }
                    | RuntimeStatementResult::CreateIndex { .. }
                    | RuntimeStatementResult::Transaction { .. }
                    | RuntimeStatementResult::NoOp => {
                        let duration = start.elapsed();
                        record_statement(sql, duration, "DDL/TXN");
                        Ok(DBOutput::StatementComplete(0))
                    }
                }
            }
            Err(e) => {
                let duration = start.elapsed();
                record_statement(sql, duration, "ERROR");
                Err(e)
            }
        }
    }

    async fn shutdown(&mut self) {}
}

pub type HarnessFuture = Pin<Box<dyn Future<Output = Result<EngineHarness, ()>> + Send + 'static>>;
pub type HarnessFactory = Box<dyn Fn() -> HarnessFuture + Send + Sync + 'static>;

/// Create a factory factory that shares a runtime context across harnesses.
pub fn make_in_memory_factory_factory() -> impl Fn() -> HarnessFactory + Clone {
    || {
        tracing::trace!("[FACTORY] make_factory_factory: Creating SharedContext");
        let shared = SharedContext::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let factory: HarnessFactory = Box::new(move || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            tracing::debug!(
                "[FACTORY] Factory called #{}: Creating new EngineHarness",
                n
            );
            let shared_clone = shared.clone();
            Box::pin(async move {
                let engine = shared_clone.make_engine();
                tracing::debug!(
                    "[FACTORY] Factory #{}: Created SqlEngine with new Session",
                    n
                );
                Ok::<_, ()>(EngineHarness::new(engine))
            })
        });
        factory
    }
}
