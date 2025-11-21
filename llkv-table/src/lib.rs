//! DataFusion integration helpers for LLKV storage.
//!
//! This crate wires LLKV storage engines into [`datafusion`] [`TableProvider`]s.
//! It supports two backends:
//! - [`parquet`]: Row-group based Parquet storage (via `llkv-parquet-store`).
//! - [`column_map`]: Native columnar storage (via `llkv-column-map`).

pub mod catalog;
pub mod common;
pub mod providers;
pub mod traits;

// Re-export the Parquet implementation as the default for backward compatibility
pub use common::*;
pub use providers::column_map::{ColumnMapTableBuilder, ColumnMapTableProvider};
pub use providers::parquet::{LlkvTableBuilder, LlkvTableProvider};
