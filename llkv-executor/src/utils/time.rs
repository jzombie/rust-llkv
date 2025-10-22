//! Time utility functions for the executor.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current time in microseconds since the Unix epoch.
///
/// This is commonly used for timestamp columns and time-based operations.
/// Falls back to 0 if the system clock is before the Unix epoch (should never happen).
pub fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
