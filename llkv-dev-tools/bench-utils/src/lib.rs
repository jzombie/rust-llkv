//! Small helper to standardize Criterion group settings across the workspace.

use criterion::BenchmarkGroup;
use criterion::measurement::WallTime;
use std::time::Duration;

/// Configure a Criterion benchmark group with project-wide sensible defaults.
///
/// - sample_size: default 50
/// - warm_up_time: default 2s
/// - measurement_time: default 5s
///
/// Callers can still override these settings after calling this helper.
pub fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
}

// Helper notes: callers should create the group with `let mut group = c.benchmark_group("name")`
// and then call `configure_group(&mut group)` to apply workspace defaults.
// with `let mut group = c.benchmark_group("name")` and then call `configure_group(&mut group)`.
