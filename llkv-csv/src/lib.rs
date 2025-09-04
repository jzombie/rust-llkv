//! CSV ingest for llkv-table (FS-agnostic).
//!
//! Parses CSV rows from any `Read` and batches `Table::insert_many`.
//! The mapping closure returns an optional `RowPatch`. Returning `None`
//! skips that row. Only the example binary should touch the filesystem.
//!
//! All lines <= 80 chars.

#![forbid(unsafe_code)]

use csv::{ReaderBuilder, StringRecord};
use llkv_table::types::{RootId, RowPatch};
use llkv_table::{
    Table,
    btree::{errors::Error, pager::Pager},
};
use std::io::Read;

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
/// `RowPatch`. Return `None` to skip a malformed or unwanted row.
///
/// CSV parse errors are mapped to `Error::Corrupt("csv parse error")`.
/// Insertions are batched; the last partial batch is flushed at end.
pub fn load_into_table<R, P, F>(
    reader: R,
    cfg: &CsvConfig,
    table: &Table<P>,
    mut map: F,
) -> Result<(), Error>
where
    R: Read,
    P: Pager<Id = RootId> + Clone + Send + Sync + 'static,
    <P as Pager>::Page: Send + Sync + 'static,
    // NOTE: take the record by value, return an OWNED RowPatch
    F: FnMut(StringRecord) -> Option<RowPatch<'static>>,
{
    let mut src = CsvSource::from_reader(reader, cfg);
    let mut buf: Vec<RowPatch> = Vec::with_capacity(cfg.insert_batch);

    // TODO: Don't buffer entire thing at once!
    for rec_res in src.records() {
        let rec = match rec_res {
            Ok(r) => r,
            Err(_) => return Err(Error::Corrupt("csv parse error")),
        };

        // Move `rec` into the mapper; mapper must build an owned RowPatch.
        if let Some(row_patch) = map(rec) {
            buf.push(row_patch);
            if buf.len() == cfg.insert_batch {
                table.insert_many(&buf)?;
                buf.clear();
            }
        }
    }

    if !buf.is_empty() {
        table.insert_many(&buf)?;
    }
    Ok(())
}
