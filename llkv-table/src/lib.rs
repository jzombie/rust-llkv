#![recursion_limit = "65536"]

//! DataFusion integration helpers for LLKV storage.
//!
//! This crate wires LLKV storage engines into [`datafusion`] [`TableProvider`]s.
//! It supports two backends:
//! - [`parquet`]: Row-group based Parquet storage (via `llkv-parquet-store`).
//! - [`column_map`]: Native columnar storage (via `llkv-column-map`).

pub mod column_map;
pub mod parquet;

// Re-export the Parquet implementation as the default for backward compatibility
pub use parquet::{LlkvTableBuilder, LlkvTableProvider};
