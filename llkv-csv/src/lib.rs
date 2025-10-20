//! CSV ingestion and export helpers for LLKV tables.
//!
//! The crate wraps Arrow's CSV reader/writer to stream data into LLKV tables and export query
//! results. High-level entry points live in [`csv_ingest`] and [`csv_export`], while internal
//! modules provide schema inference and streaming utilities.

use std::error::Error;

/// Result type alias used by CSV ingestion components.
pub type CsvResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub mod csv_export;
pub mod csv_ingest;
pub(crate) mod inference;

mod reader;
mod writer;

pub use reader::{CsvReadOptions, CsvReadSession, CsvReader};

pub use writer::{CsvExportColumn, CsvWriteOptions, CsvWriter};

pub use csv_export::{
    export_csv_from_table, export_csv_from_table_with_filter,
    export_csv_from_table_with_projections, export_csv_to_writer_with_filter,
    export_csv_to_writer_with_projections,
};

pub use csv_ingest::{append_csv_into_table, append_csv_into_table_with_mapping};
