//! SLT runner utilities for LLKV.

pub mod engine;
mod parser;
mod runner;

pub use parser::{expand_loops_with_mapping, map_temp_error_message, normalize_inline_connections};
pub use runner::{run_slt_file_blocking, run_slt_file_with_factory, run_slt_harness,
    run_slt_harness_with_args};

use std::path::Path;

use llkv_result::Error;

/// Convenience runner that owns the resources required to execute SLT test suites.
///
/// This wrapper creates a shared runtime context and internally manages
/// a Tokio runtime so callers do not need to be async-aware.
#[derive(Clone)]
pub struct LlkvSltRunner {
    factory_factory: std::sync::Arc<dyn Fn() -> engine::HarnessFactory + Send + Sync>,
    runtime_kind: RuntimeKind,
}

#[derive(Clone, Copy)]
pub enum RuntimeKind {
    /// Single-threaded runtime suitable for most CLI use-cases.
    CurrentThread,
    /// Multi-threaded runtime for workloads that benefit from thread-pooling.
    MultiThread,
}

impl LlkvSltRunner {
    /// Create a runner that executes against an in-memory `MemPager` backend.
    pub fn in_memory() -> Self {
        Self::with_factory_factory(engine::make_in_memory_factory_factory())
    }

    /// Create a runner that executes against a user-supplied factory factory.
    pub fn with_factory_factory<F>(factory_factory: F) -> Self
    where
        F: Fn() -> engine::HarnessFactory + Send + Sync + 'static,
    {
        Self {
            factory_factory: std::sync::Arc::new(factory_factory),
            runtime_kind: RuntimeKind::CurrentThread,
        }
    }

    /// Override runtime configuration.
    pub fn with_runtime_kind(mut self, kind: RuntimeKind) -> Self {
        self.runtime_kind = kind;
        self
    }

    /// Run the provided `.slt` file synchronously, returning the first error if any.
    pub fn run_file(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let factory = (self.factory_factory)();
        runner::run_slt_file_blocking_with_runtime(path.as_ref(), factory, self.runtime_kind)
    }

    /// Discover and execute all `.slt` files under the given directory.
    pub fn run_directory(&self, dir: &str) -> Result<(), Error> {
        let factory_factory = self.factory_factory.clone();
        runner::run_slt_dir_blocking(dir, move || (factory_factory)(), self.runtime_kind)
    }

    // TODO: Add ability to run scripts from strings or readers.
    // TODO: Add ability to run scripts from URLs.
}
