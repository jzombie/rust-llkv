//! Load a CSV into a Table using filename, page size, headers flag,
//! and delimiter. No header inference.
//!
//! Usage:
//!   cargo run --release --example csv_to_table -- \
//!     <file.csv> <has_headers:0|1> <delimiter>

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::ops::Bound;
use std::time::Instant;

use csv::ReaderBuilder;
use llkv_csv::{CsvConfig, load_into_table};
use llkv_table::types::{ColumnInput, FieldId, IndexKey, RowId};
use llkv_table::{Table, TableCfg};

fn parse_args() -> Result<(String, bool, u8), String> {
    let mut it = env::args().skip(1);

    let file = it.next().ok_or("missing <file.csv>")?;

    let has_headers = match it.next().ok_or("missing <has_headers> (0|1)")?.as_str() {
        "0" => false,
        "1" => true,
        _ => return Err("has_headers must be 0 or 1".into()),
    };

    let delim_str = it.next().ok_or("missing <delimiter>")?;
    let b = delim_str.as_bytes();
    if b.len() != 1 {
        return Err("delimiter must be a single byte".into());
    }
    Ok((file, has_headers, b[0]))
}

/// Read one record (no headers) only to learn the width, then rewind.
fn sniff_width(f: &mut File, delimiter: u8) -> Result<usize, Box<dyn std::error::Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delimiter)
        .flexible(true)
        .from_reader(BufReader::new(f));

    let width = match rdr.records().next() {
        Some(res) => res?.len(),
        None => 0,
    };
    Ok(width)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (path, has_headers, delimiter) = parse_args().map_err(|e| format!("arg error: {}", e))?;

    // Open, sniff width, rewind.
    let mut file = File::open(&path)?;
    let width = sniff_width(&mut file, delimiter)?;
    file.seek(SeekFrom::Start(0))?;
    if width < 1 {
        return Err("csv must have at least 1 column".into());
    }

    // Build the new non-generic Table.
    let table = Table::new(TableCfg::default());

    // Reader and config for ingest.
    let reader = BufReader::new(file);
    let cfg = CsvConfig {
        delimiter,
        has_headers,
        insert_batch: 8192, // tune freely; unrelated to <page_size>
    };

    // Synthesize RowId sequentially starting at 1.
    let mut next_row_id: RowId = 1;

    // Ingest
    let t0 = Instant::now();
    load_into_table(reader, &cfg, &table, move |rec| {
        let rid = {
            let r = next_row_id;
            next_row_id += 1;
            r
        };

        // (RowId, HashMap<FieldId, (IndexKey, ColumnInput)>)
        // IndexKey is ignored by the new column-map table; set 0.
        let mut patch: HashMap<FieldId, (IndexKey, ColumnInput<'static>)> =
            HashMap::with_capacity(width);

        for col in 0..width {
            let val = rec.get(col)?;
            let field_id = (col + 1) as FieldId;
            patch.insert(
                field_id,
                (0u64, ColumnInput::from(Cow::Owned(val.as_bytes().to_vec()))),
            );
        }
        Some((rid, patch))
    })?;
    eprintln!("ingest: {} ms", t0.elapsed().as_millis());

    // ---- Stream the table contents back out (row-id order) ----
    let projection: Vec<FieldId> = (1..=width as FieldId).collect();

    // Large buffered stdout to avoid syscall thrash.
    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(8 << 20, stdout.lock());

    let t1 = Instant::now();
    writeln!(out, "--- table dump start ---")?;

    // Use column 1 as the driver; scan all row ids in order.
    let mut total_rows = 0usize;
    table.scan_by_row_id_range(
        1,                // driver_fid (first CSV column)
        Bound::Unbounded, // lo row_id
        Bound::Unbounded, // hi row_id
        &projection,      // project all columns
        8192,             // page size for scan
        |row_id, cols| {
            total_rows += 1;

            // Build one line per row and write once.
            let mut line = String::with_capacity(64 * (cols.len().max(1)));
            line.push_str("row_id=");
            line.push_str(&row_id.to_string());
            line.push(':');

            for (i, cell) in cols.iter().enumerate() {
                if i == 0 {
                    line.push(' ');
                } else {
                    line.push_str(" | ");
                }
                match cell {
                    Some(bytes) => {
                        line.push('[');
                        line.push_str(&String::from_utf8_lossy(bytes));
                        line.push(']');
                    }
                    None => line.push_str("[NULL]"),
                }
            }
            line.push('\n');
            let _ = out.write_all(line.as_bytes());
        },
    );
    writeln!(out, "--- table dump end ---")?;
    out.flush()?;
    eprintln!(
        "dump:   {} ms ({} rows)",
        t1.elapsed().as_millis(),
        total_rows
    );

    Ok(())
}
