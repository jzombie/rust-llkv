use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

// Re-export DataFusion harness for backward compatibility
pub use crate::datafusion_harness::DataFusionHarness as EngineHarness;

/// Thread-local storage for expected column types from sqllogictest directives.
pub(crate) mod expectations {
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
    pub(crate) fn is_set() -> bool {
        EXPECTED_COLUMN_TYPES.with(|cell| cell.borrow().is_some())
    }

    /// Drains the cached sqllogictest column types, returning ownership to the caller if present.
    pub(crate) fn take() -> Option<Vec<DefaultColumnType>> {
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
    #[allow(clippy::print_stdout)]
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
    #[allow(clippy::print_stdout)]
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
pub(crate) fn record_query(sql: &str, duration: Duration, result_type: &str) {
    if let Some(stats) = STATS.lock().unwrap().as_mut() {
        stats.total_queries += 1;
        stats
            .query_timings
            .push((sql.to_string(), duration, result_type.to_string()));
    }
}

/// Record a statement execution
pub(crate) fn record_statement(sql: &str, duration: Duration, result_type: &str) {
    if let Some(stats) = STATS.lock().unwrap().as_mut() {
        stats.total_statements += 1;
        stats
            .query_timings
            .push((sql.to_string(), duration, result_type.to_string()));
    }
}

pub type HarnessFuture = Pin<Box<dyn Future<Output = Result<EngineHarness, ()>> + Send + 'static>>;
pub type HarnessFactory = Box<dyn Fn() -> HarnessFuture + Send + Sync + 'static>;

/// Create a factory factory that produces DataFusion-backed harnesses.
pub fn make_in_memory_factory_factory() -> impl Fn() -> HarnessFactory + Clone {
    || {
        tracing::trace!("[FACTORY] make_factory_factory: Creating DataFusion factory");
        let counter = Arc::new(AtomicUsize::new(0));
        let factory: HarnessFactory = Box::new(move || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            tracing::debug!(
                "[FACTORY] Factory called #{}: Creating new DataFusionHarness",
                n
            );
            Box::pin(async move {
                tracing::debug!(
                    "[FACTORY] Factory #{}: Created DataFusion harness",
                    n
                );
                EngineHarness::new().map_err(|_| ())
            })
        });
        factory
    }
}
