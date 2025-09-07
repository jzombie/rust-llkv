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
use llkv_table::Table;
use llkv_table::btree::pager::{SharedPager, define_mem_pager};
use llkv_table::expr::{Expr, Filter, Operator};
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

    if width < 1 {
        return Err("csv must have at least 1 column".into());
    }

    // Build pager and table.
    let base = MemPager64::new(page_sz);
    let pager = SharedPager::new(base);
    let mut table: Table<_> = Table::with_pager(pager);

    // Create columns for FieldId 1..=width (all CSV columns become fields).
    for fid in 1..=width as FieldId {
        table.add_column(fid);
    }
    // Add an index on FieldId 1 so we can scan everything later.
    table.add_index(1);

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

            // Use the RowId as the IndexKey for the indexed field (FieldId 1).
            // For other non-indexed fields, the key can remain 0.
            let index_key = if field_id == 1 { rid } else { 0 };

            patch.insert(
                field_id,
                (
                    index_key,
                    ColumnInput::from(Cow::Owned(val.as_bytes().to_vec())),
                ),
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

    // ---- Stream the table contents back out (in sorted order) ----
    let projection: Vec<FieldId> = (1..=width as FieldId).collect();

    // Use a full range scan on the index to get all rows in sorted order.
    let expr = Expr::Pred(Filter {
        field_id: 1,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    println!("--- table dump start ---");
    table
        .scan(&expr, &projection, &mut |row_id, values_iter| {
            print!("row_id={}:", row_id);
            for i in 0..projection.len() {
                if let Some(v) = values_iter.next() {
                    let s = String::from_utf8_lossy(v.as_slice());
                    if i == 0 {
                        print!(" [{}]", s);
                    } else {
                        print!(" | [{}]", s);
                    }
                } else if i == 0 {
                    print!(" [NULL]");
                } else {
                    print!(" | [NULL]");
                }
            }
            println!();
        })
        .map_err(|e| std::io::Error::other(format!("{:?}", e)))?;
    println!("--- table dump end ---");

    Ok(())
}
