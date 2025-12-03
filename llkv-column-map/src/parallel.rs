//! Helper utilities for Rayon thread-pool management.
//!
//! The column map crate centralizes Rayon configuration so callers reuse a single
//! bounded pool. This avoids opportunistic global overrides and keeps worker naming
//! consistent across crates. The helpers below expose a `with_thread_pool` entry point
//! and a cheap accessor for the current thread count.

use std::env;
use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

const ENV_MAX_THREADS: &str = "LLKV_MAX_THREADS";

fn detected_thread_count() -> usize {
    match std::thread::available_parallelism() {
        Ok(nz) => nz.get(),
        Err(_) => 1,
    }
}

fn configured_thread_count() -> usize {
    let default = detected_thread_count();
    match env::var(ENV_MAX_THREADS) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(value) if value > 0 => value,
            _ => default,
        },
        Err(_) => default,
    }
}

fn build_pool() -> ThreadPool {
    let limit = configured_thread_count();
    ThreadPoolBuilder::new()
        .num_threads(limit)
        .thread_name(|idx| format!("llkv-worker-{idx}"))
        .build()
        .unwrap_or_else(|_| {
            ThreadPoolBuilder::new()
                .build()
                .expect("failed to build rayon thread pool")
        })
}

fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(build_pool)
}

fn log_pool_size_once() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        let count = pool().current_num_threads();
        tracing::debug!(
            "[llkv-column-map] Rayon pool initialized with {count} threads (LLKV_MAX_THREADS={})",
            env::var(ENV_MAX_THREADS).unwrap_or_else(|_| "<unset>".into())
        );
    });
}

/// Execute the provided closure within the shared Rayon thread pool.
///
/// This helper ensures all parallel work within the crate shares the same
/// worker configuration (thread count, naming). The pool size can be capped via
/// the `LLKV_MAX_THREADS` environment variable; non-positive values fall back to
/// the detected hardware parallelism.
///
/// # Arguments
/// - `f`: Closure to run inside the pool.
///
/// # Panics
/// Panics only if the global pool fails to initialize, which indicates a bug in
/// the underlying Rayon builder.
pub fn with_thread_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    log_pool_size_once();
    pool().install(f)
}

/// Return the number of worker threads currently active in the shared pool.
///
/// This value reflects either hardware parallelism or the override supplied via
/// `LLKV_MAX_THREADS`. The count may change if the environment variable differs
/// between runs, but remains stable for the lifetime of the process.
pub fn current_thread_count() -> usize {
    pool().current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override_zero_defaults() {
        let prev = env::var(ENV_MAX_THREADS).ok();
        unsafe {
            env::set_var(ENV_MAX_THREADS, "0");
        }
        let count = configured_thread_count();
        if let Some(prev) = prev {
            unsafe {
                env::set_var(ENV_MAX_THREADS, prev);
            }
        } else {
            unsafe {
                env::remove_var(ENV_MAX_THREADS);
            }
        }
        assert!(count >= 1);
    }
}
