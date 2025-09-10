//! Load a CSV into a Table using filename, page size, headers flag,
//! and delimiter. No header inference.
//!
//! Usage:
//!   cargo run --example csv_to_table -- \
//!     <file.csv> <page_size> <has_headers:0|1> <delimiter>

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::ops::Bound;

use csv::ReaderBuilder;
use llkv_csv::{CsvConfig, load_into_table};
use llkv_table::types::{ColumnInput, FieldId, IndexKey, RowId};
use llkv_table::{Table, TableCfg};

fn parse_args() -> Result<(String, usize, bool, u8), String> {
    let mut it = env::args().skip(1);

    let file = it.next().ok_or("missing <file.csv>")?;
    let page_sz = it
        .next()
        .ok_or("missing <page_size>")?
        .parse::<usize>()
        .map_err(|_| "invalid <page_size>")?;

    let has_headers = match it.next().ok_or("missing <has_headers>")?.as_str() {
        "0" => false,
        "1" => true,
        _ => return Err("has_headers must be 0 or 1".into()),
    };

    let delim_str = it.next().ok_or("missing <delimiter>")?;
    let b = delim_str.as_bytes();
    if b.len() != 1 {
        return Err("delimiter must be a single byte".into());
    }
    Ok((file, page_sz, has_headers, b[0]))
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
    let (path, _page_sz, has_headers, delimiter) =
        parse_args().map_err(|e| format!("arg error: {}", e))?;

    // Open, sniff width, rewind.
    let mut file = File::open(&path)?;
    let width = sniff_width(&mut file, delimiter)?;
    file.seek(SeekFrom::Start(0))?;

    if width < 1 {
        return Err("csv must have at least 1 column".into());
    }

    // Build the new non-generic Table.
    let mut table = Table::new(TableCfg::default());

    // Create columns for FieldId 1..=width (all CSV columns become fields).
    for fid in 1..=width as FieldId {
        table.add_column(fid);
    }

    // Reader and config for ingest.
    let reader = BufReader::new(file);
    let cfg = CsvConfig {
        delimiter,
        has_headers,
        insert_batch: 8192,
    };

    // Synthesize RowId sequentially starting at 1.
    let mut next_row_id: RowId = 1;

    // Map each record to:
    // (RowId, HashMap<FieldId, (IndexKey, ColumnInput)>)
    // Note: IndexKey is unused by the new table backend; we pass 0.
    let res = load_into_table(reader, &cfg, &table, move |rec| {
        let rid = {
            let r = next_row_id;
            next_row_id += 1;
            r
        };

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
    });

    match res {
        Ok(_) => println!("CSV ingest complete."),
        Err(e) => {
            eprintln!("ingest failed: {:?}", e);
            return Err(Box::<dyn std::error::Error>::from(e));
        }
    }

    // ---- Stream the table contents back out (row-id order) ----
    let projection: Vec<FieldId> = (1..=width as FieldId).collect();

    println!("--- table dump start ---");
    // We scan by row-id range. Use column 1 as the driver (present on every row).
    // Bounds are unbounded to dump everything; choose a page size that’s reasonable.
    table.scan_by_row_id_range(
        1,                // driver_fid
        Bound::Unbounded, // lo row_id
        Bound::Unbounded, // hi row_id
        &projection,      // project all CSV columns
        4096,             // page size
        |row_id, cols| {
            print!("row_id={}:", row_id);
            for (i, cell) in cols.iter().enumerate() {
                match cell {
                    Some(bytes) => {
                        let s = String::from_utf8_lossy(bytes);
                        if i == 0 {
                            print!(" [{}]", s);
                        } else {
                            print!(" | [{}]", s);
                        }
                    }
                    None => {
                        if i == 0 {
                            print!(" [NULL]");
                        } else {
                            print!(" | [NULL]");
                        }
                    }
                }
            }
            println!();
        },
    );
    println!("--- table dump end ---");

    Ok(())
}
