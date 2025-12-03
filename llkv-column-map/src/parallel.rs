//! Thread pool helpers have moved to the `llkv-threading` crate.
//! This module re-exports them for backward compatibility.

pub use llkv_threading::{current_thread_count, with_thread_pool};
