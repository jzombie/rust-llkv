use std::time::Duration;
use std::cell::RefCell;

thread_local! {
    static CONTEXT: RefCell<Option<String>> = RefCell::new(None);
    static CONTEXT_PRINTED: RefCell<bool> = RefCell::new(false);
    static NESTING: RefCell<usize> = RefCell::new(0);
}

pub struct ScopeGuard;

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        NESTING.with(|n| {
            let mut n = n.borrow_mut();
            if *n > 0 {
                *n -= 1;
            }
        });
    }
}

#[cfg(feature = "perf-mon")]
pub fn enter_scope(_label: Option<&'static str>) -> ScopeGuard {
    NESTING.with(|n| *n.borrow_mut() += 1);
    ScopeGuard
}

#[cfg(not(feature = "perf-mon"))]
pub fn enter_scope(_label: Option<&'static str>) -> ScopeGuard {
    ScopeGuard
}

#[cfg(feature = "perf-mon")]
pub fn set_context(ctx: &str) {
    CONTEXT.with(|c| *c.borrow_mut() = Some(ctx.to_string()));
    CONTEXT_PRINTED.with(|b| *b.borrow_mut() = false);
}

#[cfg(not(feature = "perf-mon"))]
pub fn set_context(_ctx: &str) {}

#[cfg(feature = "perf-mon")]
pub fn clear_context() {
    CONTEXT.with(|c| *c.borrow_mut() = None);
}

#[cfg(not(feature = "perf-mon"))]
pub fn clear_context() {}

#[cfg(feature = "perf-mon")]
fn ensure_context_printed() {
    CONTEXT_PRINTED.with(|b| {
        if !*b.borrow() {
            CONTEXT.with(|c| {
                if let Some(ctx) = &*c.borrow() {
                    eprintln!("---------------------------------------------------");
                    eprintln!("Slow Query: {}", ctx.trim());
                    *b.borrow_mut() = true;
                }
            });
        }
    });
}

/// Measures the execution time of the provided expression or block.
///
/// Returns a tuple `(result, duration)`.
///
/// If the `perf-mon` feature is disabled, the duration is always `Duration::ZERO`
/// and the timing overhead is eliminated.
#[macro_export]
macro_rules! measure {
    ($label:literal, $e:expr) => {{
        #[cfg(feature = "perf-mon")]
        {
            let _guard = $crate::enter_scope(Some($label));
            let start = std::time::Instant::now();
            let result = $e;
            (result, start.elapsed())
        }
        #[cfg(not(feature = "perf-mon"))]
        {
            let result = $e;
            (result, std::time::Duration::ZERO)
        }
    }};
    ($e:expr) => {{
        #[cfg(feature = "perf-mon")]
        {
            let _guard = $crate::enter_scope(None);
            let start = std::time::Instant::now();
            let result = $e;
            (result, start.elapsed())
        }
        #[cfg(not(feature = "perf-mon"))]
        {
            let result = $e;
            (result, std::time::Duration::ZERO)
        }
    }};
}

/// Checks if the provided duration exceeds the configured threshold.
#[cfg(feature = "perf-mon")]
pub fn is_slow(duration: Duration) -> bool {
    use std::sync::atomic::{AtomicU64, Ordering};
    static THRESHOLD_MS: AtomicU64 = AtomicU64::new(1); // Default 1ms
    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        if let Ok(val) = std::env::var("LLKV_PERF_THRESHOLD_MS") {
            if let Ok(parsed) = val.parse::<u64>() {
                THRESHOLD_MS.store(parsed, Ordering::Relaxed);
            }
        }
    });

    let threshold = Duration::from_millis(THRESHOLD_MS.load(Ordering::Relaxed));
    duration > threshold
}

#[cfg(not(feature = "perf-mon"))]
#[inline(always)]
pub fn is_slow(_duration: Duration) -> bool {
    false
}

/// Logs the durations if any of them exceed the threshold configured via `LLKV_PERF_THRESHOLD_MS`.
/// Defaults to 1ms if not set.
///
/// When `perf-mon` is disabled, this function is a no-op and should be optimized away.
#[cfg(feature = "perf-mon")]
pub fn log_if_slow(label: &str, parts: &[(&str, Duration)]) -> bool {
    let mut any_slow = false;
    for (_, dur) in parts {
        if is_slow(*dur) {
            any_slow = true;
            break;
        }
    }

    if any_slow {
        ensure_context_printed();
        let depth = NESTING.with(|n| *n.borrow());
        
        let mut prefix = "│   ".repeat(depth);
        prefix.push_str("├── ");
        
        let mut msg = format!("{}Slow {}: ", prefix, label);
        for (i, (name, dur)) in parts.iter().enumerate() {
            if i > 0 {
                msg.push_str(", ");
            }
            msg.push_str(&format!("{}: {:?}", name, dur));
        }
        eprintln!("{}", msg);
        true
    } else {
        false
    }
}

#[cfg(not(feature = "perf-mon"))]
#[inline(always)]
pub fn log_if_slow(_label: &str, _parts: &[(&str, Duration)]) -> bool {
    false
}
