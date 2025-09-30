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

pub fn with_thread_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    pool().install(f)
}

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
