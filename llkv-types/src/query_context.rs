use llkv_perf_monitor::{PerfContext, PerfGuard};

/// Per-query context threaded through planning and execution.
#[derive(Clone, Debug)]
pub struct QueryContext {
    perf: PerfContext,
}

impl Default for QueryContext {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryContext {
    /// Construct a context with performance monitoring disabled.
    pub fn new() -> Self {
        Self {
            perf: PerfContext::disabled(),
        }
    }

    /// Attach a specific performance context.
    pub fn with_perf(perf: PerfContext) -> Self {
        Self { perf }
    }

    /// Access the performance context for instrumentation.
    pub fn perf(&self) -> &PerfContext {
        &self.perf
    }

    /// Render a formatted performance report when monitoring is enabled.
    pub fn render_report(&self) -> Option<String> {
        self.perf.render_report()
    }
}

impl AsRef<PerfContext> for QueryContext {
    fn as_ref(&self) -> &PerfContext {
        &self.perf
    }
}

// TODO: Why isn't this just part of the context constructor?  Is `PerfGuard` needed to be used as a tuple here?
/// Start a new query context and guard that records total elapsed time.
pub fn begin_query(label: impl Into<String>) -> (QueryContext, PerfGuard) {
    let perf = PerfContext::new(label);
    let ctx = QueryContext::with_perf(perf.clone());
    let guard = PerfGuard::new(perf);
    (ctx, guard)
}
