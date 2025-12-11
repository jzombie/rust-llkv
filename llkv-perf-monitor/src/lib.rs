//! Performance monitoring helpers used throughout the query pipeline.

use line_ending::LineEnding;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Lightweight container for per-query metrics.
///
/// When disabled, calls are no-ops so callers can always pass a reference
/// without branching on feature flags.
#[derive(Clone, Debug, Default)]
pub struct PerfContext {
    inner: Option<Arc<PerfInner>>,
}

#[derive(Debug)]
struct PerfInner {
    root_label: String,
    started_at: Instant,
    measurements: Mutex<Vec<Measurement>>,
    stack: Mutex<Vec<String>>,
}

#[derive(Debug, Clone)]
struct Measurement {
    label: String,
    duration: Duration,
}

#[derive(Debug, Default)]
struct ReportNode {
    label: String,
    duration: Duration,
    hits: usize,
    children: BTreeMap<String, ReportNode>,
}

impl ReportNode {
    fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            duration: Duration::ZERO,
            hits: 0,
            children: BTreeMap::new(),
        }
    }

    fn insert(&mut self, path: &[&str], duration: Duration) {
        // Repeated measurements of the same path aggregate: bump the hit count and
        // accumulate the duration so callers see totals, not just the last sample.
        self.hits = self.hits.saturating_add(1);
        if let Some((head, tail)) = path.split_first() {
            let child = self
                .children
                .entry((*head).to_string())
                .or_insert_with(|| ReportNode::new(*head));
            child.insert(tail, duration);
        } else {
            self.duration = self.duration.saturating_add(duration);
        }
    }

    fn total_duration(&self) -> Duration {
        let children_sum = self
            .children
            .values()
            .map(|c| c.total_duration())
            .fold(Duration::ZERO, |acc, d| acc.saturating_add(d));
        std::cmp::max(self.duration, children_sum)
    }
}

impl PerfContext {
    /// Construct a disabled context that ignores all recordings.
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// Create a new context anchored to a top-level label (usually the SQL).
    pub fn new(root_label: impl Into<String>) -> Self {
        // TODO: Remove cfg! check here and init using macro
        if cfg!(feature = "perf-mon") {
            let inner = PerfInner {
                root_label: root_label.into(),
                started_at: Instant::now(),
                measurements: Mutex::new(Vec::new()),
                stack: Mutex::new(Vec::new()),
            };
            Self {
                inner: Some(Arc::new(inner)),
            }
        } else {
            Self::disabled()
        }
    }

    /// Returns true when the context is actively recording measurements.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Record a new measurement under the provided label.
    pub fn record(&self, label: impl Into<String>, duration: Duration) {
        // TODO: Remove cfg! check here and gate using macro
        if !cfg!(feature = "perf-mon") {
            return;
        }

        if let Some(inner) = &self.inner {
            let mut guard = inner
                .measurements
                .lock()
                .expect("perf context mutex poisoned");
            guard.push(Measurement {
                label: label.into(),
                duration,
            });
        }
    }

    /// Record a measurement with an explicit label path (e.g. `parent:child`).
    pub fn record_path(&self, label_path: String, duration: Duration) {
        // TODO: Remove cfg! check here and gate using macro
        if !cfg!(feature = "perf-mon") {
            return;
        }

        if let Some(inner) = &self.inner {
            let mut guard = inner
                .measurements
                .lock()
                .expect("perf context mutex poisoned");
            guard.push(Measurement {
                label: label_path,
                duration,
            });
        }
    }

    /// Push a label onto the active stack, returning the joined path.
    pub fn push_label(&self, label: &str) -> Option<String> {
        let inner = self.inner.as_ref()?;
        let mut stack = inner.stack.lock().expect("perf context mutex poisoned");
        let effective_label = if stack.last().map(|l| l.as_str() == label).unwrap_or(false) {
            format!("nested_{label}")
        } else {
            label.to_string()
        };
        stack.push(effective_label);
        Some(stack.join(":"))
    }

    /// Pop the most recent label from the active stack.
    pub fn pop_label(&self) {
        if let Some(inner) = &self.inner {
            let mut stack = inner.stack.lock().expect("perf context mutex poisoned");
            stack.pop();
        }
    }

    /// Return a snapshot of all recorded measurements.
    pub fn measurements(&self) -> Vec<(String, Duration)> {
        self.inner
            .as_ref()
            .map(|inner| {
                inner
                    .measurements
                    .lock()
                    .expect("perf context mutex poisoned")
                    .iter()
                    .map(|m| (m.label.clone(), m.duration))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn total_elapsed(&self) -> Option<Duration> {
        self.inner.as_ref().map(|inner| inner.started_at.elapsed())
    }

    fn root_label(&self) -> Option<&str> {
        self.inner.as_ref().map(|inner| inner.root_label.as_str())
    }

    /// Return a formatted textual report of the recorded measurements.
    ///
    /// When performance monitoring is disabled or no measurements are present,
    /// this returns `None`.
    pub fn render_report(&self) -> Option<String> {
        // TODO: Remove cfg! check here and gate using macro
        if !cfg!(feature = "perf-mon") {
            return None;
        }
        let measurements = self.measurements();
        if measurements.is_empty() {
            return None;
        }

        let total_duration = measurements
            .iter()
            .find(|(label, _)| label == "query_total")
            .map(|(_, duration)| *duration)
            .or_else(|| self.total_elapsed())
            .unwrap_or(Duration::ZERO);

        // Build a simple tree by splitting labels on ':' to preserve nesting.
        let mut root = ReportNode::new(self.root_label().unwrap_or("query"));
        root.duration = total_duration;

        for (label, duration) in measurements
            .into_iter()
            .filter(|(label, _)| label != "query_total")
        {
            let label_parts: Vec<_> = label.split(':').collect();
            root.insert(&label_parts, duration);
        }

        let root_total = root.total_duration();
        let line_sep = LineEnding::from_current_platform().as_str();
        let mut output = String::new();
        write_node(&mut output, line_sep, &root, "", true, root_total, true);
        Some(output)
    }
}

/// Guard that captures the total query duration when dropped.
pub struct PerfGuard {
    ctx: PerfContext,
}

impl Drop for PerfGuard {
    fn drop(&mut self) {
        if let Some(total) = self.ctx.total_elapsed() {
            self.ctx.record("query_total", total);
        }
    }
}

impl PerfGuard {
    /// Construct a guard that records total duration for the provided context.
    pub fn new(ctx: PerfContext) -> Self {
        Self { ctx }
    }
}

impl AsRef<PerfContext> for PerfContext {
    fn as_ref(&self) -> &PerfContext {
        self
    }
}

/// Measure the provided expression and record the duration into the supplied [`PerfContext`].
///
/// Optional feature list: `maybe_record!( ["perf-mon"], ctx, label, { body })`. When the list is empty
/// (default overload), measurement is enabled. When non-empty, measurement is enabled if **any**
/// listed feature is active. This helper returns only the expression result; use
/// `measure_and_maybe_record!` when the caller also needs the elapsed time. When measurement is not
/// enabled, this macro executes the body without timing overhead.
#[macro_export]
macro_rules! maybe_record {
    ([$($feat:literal),* $(,)?], $ctx:expr, $label:expr, $body:expr) => {{
        let __ctx_ref: &llkv_perf_monitor::PerfContext =
            ::core::convert::AsRef::<llkv_perf_monitor::PerfContext>::as_ref(&$ctx);
        let __measure_enabled = {
            let mut __enabled = false;
            let mut __count = 0;
            $(
                __count += 1;
                #[allow(unexpected_cfgs)]
                if cfg!(feature = $feat) {
                    __enabled = true;
                }
            )*
            if __count == 0 {
                __enabled = true;
            }
            __enabled
        };

        if __measure_enabled {
            let __path = __ctx_ref.push_label($label);
            let __perf_start = std::time::Instant::now();
            let __perf_result = { $body };
            let __perf_duration = __perf_start.elapsed();
            if let Some(__path) = __path {
                __ctx_ref.record_path(__path, __perf_duration);
                __ctx_ref.pop_label();
            }
            __perf_result
        } else {
            { $body }
        }
    }};
    ($ctx:expr, $label:expr, $body:expr) => {{
        $crate::maybe_record!([], $ctx, $label, { $body })
    }};
}
/// Measure the provided expression, record the duration into the supplied [`PerfContext`], and
/// return both the expression result and the elapsed time.
///
/// Optional feature list behaves like `maybe_record!`: empty enables recording; non-empty records when
/// at least one listed feature is active. Timing always runs; recording to the [`PerfContext`] is
/// controlled by the feature list. When no listed features are active, the duration is still
/// returned but no measurement is recorded.
#[macro_export]
macro_rules! measure_and_maybe_record {
    ([$($feat:literal),* $(,)?], $ctx:expr, $label:expr, $body:expr) => {{
        let __ctx_ref: &llkv_perf_monitor::PerfContext =
            ::core::convert::AsRef::<llkv_perf_monitor::PerfContext>::as_ref(&$ctx);
        let __measure_enabled = {
            let mut __enabled = false;
            let mut __count = 0;
            $(
                __count += 1;
                #[allow(unexpected_cfgs)]
                if cfg!(feature = $feat) {
                    __enabled = true;
                }
            )*
            if __count == 0 {
                __enabled = true;
            }
            __enabled
        };
        // Always time; record is gated by feature list
        // This defers from maybe_record! which skips timing when disabled
        let __perf_start = std::time::Instant::now();
        let __perf_result = { $body };
        let __perf_duration = __perf_start.elapsed();
        if __measure_enabled {
            let __path = __ctx_ref.push_label($label);
            if let Some(__path) = __path {
                __ctx_ref.record_path(__path, __perf_duration);
                __ctx_ref.pop_label();
            }
        }
        (__perf_result, __perf_duration)
    }};
    ($ctx:expr, $label:expr, $body:expr) => {{
        $crate::measure_and_maybe_record!([], $ctx, $label, { $body })
    }};
}

fn write_node(
    output: &mut String,
    line_sep: &str,
    node: &ReportNode,
    prefix: &str,
    is_last: bool,
    parent_total: Duration,
    is_root: bool,
) {
    let node_total = node.total_duration();
    let connector = if is_root {
        String::new()
    } else if is_last {
        "└── ".to_string()
    } else {
        "├── ".to_string()
    };

    let duration_ms = node_total.as_secs_f64() * 1000.0;
    let pct = if parent_total.is_zero() {
        0.0
    } else {
        (node_total.as_secs_f64() / parent_total.as_secs_f64()) * 100.0
    };

    if is_root {
        output.push_str(&format!(
            "Slow query report (total={duration_ms:.3}ms):{line_sep}",
        ));
        output.push_str(line_sep);
        output.push_str(node.label.as_str());
        output.push_str(line_sep);
        output.push_str(line_sep);
    } else {
        let hits_suffix = if node.hits > 1 {
            format!(" [hits={}]", node.hits)
        } else {
            String::new()
        };
        output.push_str(&format!(
            "{prefix}{connector}{}{}: {duration_ms:.3}ms ({pct:.1}%){}",
            node.label, hits_suffix, line_sep
        ));
    }

    let mut children: Vec<&ReportNode> = node.children.values().collect();
    children.sort_by(|a, b| b.duration.cmp(&a.duration));

    let next_prefix = if is_root {
        String::new()
    } else if is_last {
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
    };

    for (idx, child) in children.iter().enumerate() {
        let child_is_last = idx + 1 == children.len();
        write_node(
            output,
            line_sep,
            child,
            &next_prefix,
            child_is_last,
            node_total,
            false,
        );
    }
}
