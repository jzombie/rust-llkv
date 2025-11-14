//! Table-level pager diagnostics built on top of pager statistics.
//!
//! - [`llkv_storage::pager::PagerDiagnostics`] captures raw [`IoStatsSnapshot`] values.
//! - This module wraps that handle so table loaders can bracket ingest work with
//!   [`TablePagerIngestionDiagnostics::begin_table`] / [`TablePagerIngestionDiagnostics::finish_table`] and receive
//!   a [`TablePagerIngestionSample`] (table-scoped snapshot delta).
//! - Consumers format `TablePagerIngestionSample` for reporting, while [`TablePagerIngestionDiagnostics`]
//!   keeps the mutex bookkeeping out of the application layer.

use llkv_storage::pager::{IoStatsSnapshot, PagerDiagnostics};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Captures pager I/O metrics for a single table ingest span.
#[derive(Clone, Debug)]
pub struct TablePagerIngestionSample {
    /// Logical table name for the recorded span.
    pub table: String,
    /// Rows written while the table ingest was active.
    pub rows: usize,
    /// Wall-clock duration of the ingest span.
    pub elapsed: Duration,
    /// Pager I/O deltas collected during the ingest span.
    pub delta: IoStatsSnapshot,
}

impl TablePagerIngestionSample {
    /// Overwrite percentage (0-100) derived from [`IoStatsSnapshot::overwrite_put_bytes`].
    pub fn overwrite_pct(&self) -> f64 {
        self.delta.overwrite_pct()
    }

    /// Average physical put operations per batch for this table ingest.
    pub fn puts_per_batch(&self) -> f64 {
        self.delta.puts_per_batch()
    }

    /// Average physical get operations per batch for this table ingest.
    pub fn gets_per_batch(&self) -> f64 {
        self.delta.gets_per_batch()
    }

    /// Fresh bytes written during this ingest converted to mebibytes.
    pub fn fresh_mib(&self) -> f64 {
        self.delta.fresh_mib()
    }

    /// Overwrite bytes written during this ingest converted to mebibytes.
    pub fn overwrite_mib(&self) -> f64 {
        self.delta.overwrite_mib()
    }
}

/// Tracks per-table pager spans over the course of a loader run.
#[derive(Debug)]
pub struct TablePagerIngestionDiagnostics {
    pager: Arc<PagerDiagnostics>,
    table_starts: Mutex<HashMap<String, IoStatsSnapshot>>,
    completed: Mutex<Vec<TablePagerIngestionSample>>,
}

impl TablePagerIngestionDiagnostics {
    /// Create diagnostics bound to the provided pager statistics helper.
    pub fn new(pager: Arc<PagerDiagnostics>) -> Self {
        Self {
            pager,
            table_starts: Mutex::new(HashMap::new()),
            completed: Mutex::new(Vec::new()),
        }
    }

    /// Record the start snapshot for a table ingest span.
    pub fn begin_table(&self, table: impl Into<String>) {
        let mut starts = self.table_starts.lock().unwrap();
        starts.insert(table.into(), self.pager.snapshot());
    }

    /// Complete a table ingest span and store the resulting diagnostics entry.
    pub fn finish_table(
        &self,
        table: &str,
        rows: usize,
        elapsed: Duration,
    ) -> TablePagerIngestionSample {
        let start_snapshot = {
            let mut starts = self.table_starts.lock().unwrap();
            starts
                .remove(table)
                .unwrap_or_else(|| self.pager.snapshot())
        };
        let delta = self.pager.delta_since(&start_snapshot);
        let entry = TablePagerIngestionSample {
            table: table.to_string(),
            rows,
            elapsed,
            delta,
        };
        self.completed.lock().unwrap().push(entry.clone());
        entry
    }

    /// Inspect the completed per-table diagnostics.
    pub fn completed_tables(&self) -> Vec<TablePagerIngestionSample> {
        self.completed.lock().unwrap().clone()
    }

    /// Return the cumulative pager totals captured during the run.
    pub fn totals(&self) -> IoStatsSnapshot {
        self.pager.totals()
    }
}
