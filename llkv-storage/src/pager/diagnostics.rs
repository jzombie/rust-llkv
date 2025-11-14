use super::{IoStats, IoStatsSnapshot};
use std::sync::Arc;

/// Tracks pager-level statistics snapshots so higher layers can build diagnostics.
#[derive(Debug)]
pub struct PagerDiagnostics {
    stats: Arc<IoStats>,
    run_start: IoStatsSnapshot,
}

impl PagerDiagnostics {
    /// Create a diagnostics helper using the provided statistics handle.
    pub fn new(stats: Arc<IoStats>) -> Self {
        let run_start = stats.snapshot();
        Self { stats, run_start }
    }

    /// Capture the latest pager snapshot.
    pub fn snapshot(&self) -> IoStatsSnapshot {
        self.stats.snapshot()
    }

    /// Compute the cumulative totals since diagnostics began.
    pub fn totals(&self) -> IoStatsSnapshot {
        self.snapshot().delta_since(&self.run_start)
    }

    /// Compute the delta between now and the supplied snapshot.
    pub fn delta_since(&self, older: &IoStatsSnapshot) -> IoStatsSnapshot {
        self.snapshot().delta_since(older)
    }

    /// Access the underlying statistics handle.
    pub fn stats(&self) -> Arc<IoStats> {
        Arc::clone(&self.stats)
    }
}
