use llkv_perf_monitor::{PerfContext, PerfGuard};

/// Per-query context threaded through planning and execution.
#[derive(Clone, Debug)]
pub struct QueryContext {
    perf: PerfContext,
}

/// RAII wrapper that owns a query context and its guard so callers do not have to track both.
pub struct QueryContextHandle {
    ctx: QueryContext,
    guard: Option<PerfGuard>,
}

impl std::fmt::Debug for QueryContextHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryContextHandle")
            .field("ctx", &self.ctx)
            .field("guard", &"<perf guard>")
            .finish()
    }
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

    /// Construct a context with an attached guard that records total elapsed time.
    pub fn begin(label: impl Into<String>) -> QueryContextHandle {
        QueryContextHandle::new(label)
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

impl QueryContextHandle {
    /// Create a new handle that owns both the context and its guard.
    pub fn new(label: impl Into<String>) -> Self {
        let perf = PerfContext::new(label);
        let ctx = QueryContext::with_perf(perf.clone());
        let guard = PerfGuard::new(perf);
        Self {
            ctx,
            guard: Some(guard),
        }
    }

    /// Alternate constructor matching the `begin` terminology used by callers.
    pub fn begin(label: impl Into<String>) -> Self {
        Self::new(label)
    }

    /// Access the underlying query context.
    pub fn context(&self) -> &QueryContext {
        &self.ctx
    }

    /// Clone the underlying context; useful when callers need to pass it by value.
    pub fn clone_context(&self) -> QueryContext {
        self.ctx.clone()
    }

    /// Consume and return the guard and context if callers need to split ownership.
    pub fn into_parts(self) -> (QueryContext, PerfGuard) {
        let Self { ctx, mut guard } = self;
        let guard = guard.take().expect("guard already taken");
        (ctx, guard)
    }

    /// Drop the guard early to force `query_total` to be recorded before rendering reports.
    pub fn drop_guard(&mut self) {
        if let Some(guard) = self.guard.take() {
            drop(guard);
        }
    }

    /// Render a formatted performance report when monitoring is enabled.
    pub fn render_report(&self) -> Option<String> {
        self.ctx.render_report()
    }
}

impl AsRef<QueryContext> for QueryContextHandle {
    fn as_ref(&self) -> &QueryContext {
        &self.ctx
    }
}

impl AsRef<PerfContext> for QueryContextHandle {
    fn as_ref(&self) -> &PerfContext {
        &self.ctx.perf
    }
}

impl std::ops::Deref for QueryContextHandle {
    type Target = QueryContext;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

