//! SLT runner utilities for LLKV.

pub mod engine;
mod parser;
mod runner;

pub use parser::{expand_loops_with_mapping, map_temp_error_message, normalize_inline_connections};
pub use runner::{
    run_slt_file_blocking, run_slt_file_blocking_with_runtime, run_slt_file_with_factory,
    run_slt_harness, run_slt_harness_with_args, run_slt_text_blocking,
    run_slt_text_blocking_with_runtime, run_slt_text_with_factory,
};

use std::io::Read;
use std::path::{Path, PathBuf};

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

    /// Run the provided `.slt` or `.slturl` file synchronously, returning the first error if any.
    /// If the file has a `.slturl` extension, it will be treated as a pointer file containing
    /// a URL to the actual test content, which will be fetched and executed.
    pub fn run_file(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        
        // Check if this is a .slturl pointer file
        if path.extension().is_some_and(|ext| ext == "slturl") {
            let url = std::fs::read_to_string(path)
                .map_err(|e| Error::Internal(format!("failed to read .slturl file: {e}")))?
                .trim()
                .to_string();
            return self.run_url(&url);
        }
        
        // Otherwise, run as a normal .slt file
        let factory = (self.factory_factory)();
        runner::run_slt_file_blocking_with_runtime(path, factory, self.runtime_kind)
    }

    /// Discover and execute all `.slt` files under the given directory.
    pub fn run_directory(&self, dir: &str) -> Result<(), Error> {
        let factory_factory = self.factory_factory.clone();
        runner::run_slt_dir_blocking(dir, move || (factory_factory)(), self.runtime_kind)
    }

    /// Execute the provided SLT script contents, tagging diagnostics with `name` for context.
    pub fn run_script(&self, name: &str, script: &str) -> Result<(), Error> {
        let display_name = if name.trim().is_empty() {
            "<memory>"
        } else {
            name
        };
        let origin = PathBuf::from(display_name);
        let factory = (self.factory_factory)();
        runner::run_slt_text_blocking_with_runtime(
            origin.as_path(),
            script,
            factory,
            self.runtime_kind,
        )
    }

    /// Execute SLT content read from an arbitrary reader.
    pub fn run_reader<R: Read>(&self, name: &str, mut reader: R) -> Result<(), Error> {
        let mut buf = String::new();
        reader
            .read_to_string(&mut buf)
            .map_err(|e| Error::Internal(format!("failed to read SLT stream: {e}")))?;
        self.run_script(name, &buf)
    }

    /// Fetch an SLT script from `url` and execute it.
    pub fn run_url(&self, url: &str) -> Result<(), Error> {
        let response = reqwest::blocking::get(url)
            .map_err(|e| Error::Internal(format!("failed to fetch SLT URL {url}: {e}")))?;
        let script = response.text().map_err(|e| {
            Error::Internal(format!("failed to read SLT response body for {url}: {e}"))
        })?;
        let name = format!("url:{url}");
        self.run_script(&name, &script)
    }
}
