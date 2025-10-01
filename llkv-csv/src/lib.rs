use std::error::Error;

pub type CsvResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub(crate) mod inference;
pub mod csv_ingest;
pub mod csv_export;

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
