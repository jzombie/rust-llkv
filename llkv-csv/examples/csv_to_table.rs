//! Load a CSV into a Table using filename, page size, headers flag,
//! and delimiter. No header inference.
//!
//! Usage:
//!   cargo run --example csv_to_table_auto -- \
//!     <file.csv> <page_size> <has_headers:0|1> <delimiter>

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};

use csv::ReaderBuilder;
use llkv_csv::{CsvConfig, load_into_table};
use llkv_table::Table;
use llkv_table::btree::pager::{SharedPager, define_mem_pager};
use llkv_table::types::{ColumnInput, FieldId, IndexKey, RowId};

define_mem_pager! {
    name: MemPager64,
    id: u64,
    default_page_size: 256
}

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
    let (path, page_sz, has_headers, delimiter) =
        parse_args().map_err(|e| format!("arg error: {}", e))?;

    // Open, sniff width, rewind.
    let mut file = File::open(&path)?;
    let width = sniff_width(&mut file, delimiter)?;
    file.seek(SeekFrom::Start(0))?;

    if width < 2 {
        return Err("csv must have at least 2 columns".into());
    }

    // Build pager and table.
    let base = MemPager64::new(page_sz);
    let pager = SharedPager::new(base);
    let mut table: Table<_> = Table::with_pager(pager);

    // Create columns for FieldId 1..=width-1.
    for fid in 1..width as FieldId {
        table.add_column(fid);
        // If you want an index, uncomment:
        // table.add_index(fs.field_id);
    }

    // Reader and config for ingest.
    let reader = BufReader::new(file);
    let cfg = CsvConfig {
        delimiter,
        has_headers,
        insert_batch: 8192,
    };

    // Map each record to:
    // (RowId, HashMap<FieldId, (IndexKey, ColumnInput)>)
    let res = load_into_table(reader, &cfg, &table, move |rec| {
        // Column 0 is RowId.
        let rid = rec.get(0)?.parse::<RowId>().ok()?;

        let mut patch: HashMap<FieldId, (IndexKey, ColumnInput<'static>)> =
            HashMap::with_capacity(width.saturating_sub(1));

        // Columns 1..=width-1 -> FieldId 1..=width-1, IndexKey = 0.
        for col in 1..width {
            let val = rec.get(col)?;
            patch.insert(
                col as FieldId,
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
    Ok(())
}
