//! CSV ingest for llkv-table (FS-agnostic).
//!
//! Parses CSV rows from any `Read` and batches `Table::insert_many`.
//! The mapping closure returns an optional `RowPatch<'static>`.
//! Returning `None` skips that row. Only the example binary should
//! touch the filesystem.

#![forbid(unsafe_code)]

use csv::{ReaderBuilder, StringRecord};
use llkv_table::Table;
use llkv_table::types::RowPatch;
use std::io::Read;

/// Simple error type for this crate.
#[derive(Debug)]
pub enum CsvError {
    /// CSV parsing failed.
    Parse,
}

impl std::fmt::Display for CsvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CsvError::Parse => write!(f, "csv parse error"),
        }
    }
}

impl std::error::Error for CsvError {}

/// CSV parser configuration.
#[derive(Clone, Debug)]
pub struct CsvConfig {
    /// Field delimiter (default: b',').
    pub delimiter: u8,
    /// Whether the first row is headers.
    pub has_headers: bool,
    /// Batch size for grouped inserts to `insert_many`.
    pub insert_batch: usize,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_headers: true,
            insert_batch: 8192,
        }
    }
}

/// CSV stream bound to a generic reader.
pub struct CsvSource<R: Read> {
    rdr: csv::Reader<R>,
}

impl<R: Read> CsvSource<R> {
    /// Build a CSV source from any `Read` using `CsvConfig`.
    pub fn from_reader(reader: R, cfg: &CsvConfig) -> Self {
        let rdr = ReaderBuilder::new()
            .has_headers(cfg.has_headers)
            .delimiter(cfg.delimiter)
            .flexible(true)
            .from_reader(reader);
        Self { rdr }
    }

    /// Stream `StringRecord`s from the underlying reader.
    pub fn records(&mut self) -> csv::StringRecordsIter<'_, R> {
        self.rdr.records()
    }
}

/// Stream CSV rows into a `Table` using a mapping closure.
///
/// The `map` closure converts each CSV record into an optional
/// `RowPatch<'static>`. Return `None` to skip malformed rows.
/// Insertions are batched; the last partial batch is flushed at end.
pub fn load_into_table<R, F>(
    reader: R,
    cfg: &CsvConfig,
    table: &Table,
    mut map: F,
) -> Result<(), CsvError>
where
    R: Read,
    F: FnMut(StringRecord) -> Option<RowPatch<'static>>,
{
    let mut src = CsvSource::from_reader(reader, cfg);

    // Only batch-sized buffering. Never loads the whole file.
    let mut buf: Vec<RowPatch<'static>> = Vec::with_capacity(cfg.insert_batch);

    for rec_res in src.records() {
        let rec = match rec_res {
            Ok(r) => r,
            Err(_e) => return Err(CsvError::Parse),
        };

        if let Some(row_patch) = map(rec) {
            buf.push(row_patch);
            if buf.len() >= cfg.insert_batch {
                table.insert_many(&buf);
                buf.clear();
            }
        }
    }

    if !buf.is_empty() {
        table.insert_many(&buf);
    }

    Ok(())
}
