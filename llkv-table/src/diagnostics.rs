//! Table-level pager diagnostics built on top of pager statistics.
//!
//! Storage exposes raw pager counters via [`PagerDiagnostics`], but tables sit several
//! layers above the pager and need to correlate ingest events with those counters.
//! This module tracks per-table spans so higher-level tools (such as loaders) can
//! reason about overwrite rates and batch behavior.

use llkv_storage::pager::{IoStatsSnapshot, PagerDiagnostics};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Captures pager I/O metrics for a single table ingest span.
#[derive(Clone, Debug)]
pub struct TablePagerDiagnostic {
    /// Logical table name for the recorded span.
    pub table: String,
    /// Rows written while the table ingest was active.
    pub rows: usize,
    /// Wall-clock duration of the ingest span.
    pub elapsed: Duration,
    /// Pager I/O deltas collected during the ingest span.
    pub delta: IoStatsSnapshot,
}

/// Tracks per-table pager spans over the course of a loader run.
#[derive(Debug)]
pub struct TablePagerDiagnostics {
    pager: Arc<PagerDiagnostics>,
    table_starts: Mutex<HashMap<String, IoStatsSnapshot>>,
    completed: Mutex<Vec<TablePagerDiagnostic>>,
}

impl TablePagerDiagnostics {
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
    pub fn complete_table(
        &self,
        table: &str,
        rows: usize,
        elapsed: Duration,
    ) -> TablePagerDiagnostic {
        let start_snapshot = {
            let mut starts = self.table_starts.lock().unwrap();
            starts
                .remove(table)
                .unwrap_or_else(|| self.pager.snapshot())
        };
        let delta = self.pager.delta_since(&start_snapshot);
        let entry = TablePagerDiagnostic {
            table: table.to_string(),
            rows,
            elapsed,
            delta,
        };
        self.completed.lock().unwrap().push(entry.clone());
        entry
    }

    /// Inspect the completed per-table diagnostics.
    pub fn completed_tables(&self) -> Vec<TablePagerDiagnostic> {
        self.completed.lock().unwrap().clone()
    }

    /// Return the cumulative pager totals captured during the run.
    pub fn totals(&self) -> IoStatsSnapshot {
        self.pager.totals()
    }
}
