//! Load a CSV file into an in-memory `Table` using `insert_many`.
//!
//! Usage:
//!   cargo run --example csv_to_table -- \
//!     <file.csv> <page_size> <has_headers:0|1> <delimiter> \
//!     <row_id_col> <n_fields> \
//!     <field_id_1> <index_key_col_1> <value_col_1> \
//!     ... repeated n_fields times ...
//!
//! Example:
//!   cargo run --example csv_to_table -- data.csv \
//!     4096 1 , 0 2  1 1 2  2 3 4
//!
//! Notes:
//!   - This example only uses `llkv-table` APIs.
//!   - It adds columns for each `field_id` mentioned.
//!   - Add indexes if you want; here we keep it minimal for ingest.

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::BufReader;

use llkv_csv::{CsvConfig, load_into_table};

use llkv_table::Table;
use llkv_table::btree::pager::{SharedPager, define_mem_pager};
use llkv_table::types::{ColumnInput, FieldId, IndexKey, RowId};

define_mem_pager! {
    /// In-memory pager with u64 page IDs.
    name: MemPager64,
    id: u64,
    default_page_size: 256
}

struct FieldSpec {
    field_id: FieldId,
    idx_col: usize,
    val_col: usize,
}

fn parse_args() -> Result<(String, usize, bool, u8, usize, Vec<FieldSpec>), String> {
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
    let delim_bytes = delim_str.as_bytes();
    if delim_bytes.len() != 1 {
        return Err("delimiter must be a single byte".into());
    }
    let delimiter = delim_bytes[0];

    let row_id_col = it
        .next()
        .ok_or("missing <row_id_col>")?
        .parse::<usize>()
        .map_err(|_| "invalid <row_id_col>")?;

    let n_fields = it
        .next()
        .ok_or("missing <n_fields>")?
        .parse::<usize>()
        .map_err(|_| "invalid <n_fields>")?;

    let mut fields = Vec::with_capacity(n_fields);
    for i in 0..n_fields {
        let field_id = it
            .next()
            .ok_or(format!("missing field_id #{}", i + 1))?
            .parse::<FieldId>()
            .map_err(|_| "invalid field_id")?;
        let idx_col = it
            .next()
            .ok_or(format!("missing index_key_col #{}", i + 1))?
            .parse::<usize>()
            .map_err(|_| "invalid index_key_col")?;
        let val_col = it
            .next()
            .ok_or(format!("missing value_col #{}", i + 1))?
            .parse::<usize>()
            .map_err(|_| "invalid value_col")?;

        fields.push(FieldSpec {
            field_id,
            idx_col,
            val_col,
        });
    }

    Ok((file, page_sz, has_headers, delimiter, row_id_col, fields))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (file, page_sz, has_headers, delimiter, row_id_col, fields) =
        parse_args().map_err(|e| format!("arg error: {}", e))?;

    // Build the pager and table without touching llkv-btree in our code.
    let base = MemPager64::new(page_sz);
    let pager = SharedPager::new(base);
    let mut table: Table<_> = Table::with_pager(pager);

    // Ensure columns exist for every field in the spec.
    // Index creation is optional; keep ingest minimal here.
    for fs in &fields {
        table.add_column(fs.field_id);
        // If you want an index, uncomment:
        // table.add_index(fs.field_id);
    }

    // Open the file (FS usage is confined to this example).
    let f = File::open(&file)?;
    let reader = BufReader::new(f);

    let cfg = CsvConfig {
        delimiter,
        has_headers,
        insert_batch: 8192,
    };

    // Each CSV row -> RowPatch<'static>:
    // (RowId, HashMap<FieldId, (IndexKey, ColumnInput<'static>)>)
    let res = load_into_table(reader, &cfg, &table, |rec| {
        let rid_str = rec.get(row_id_col)?;
        let rid = rid_str.parse::<RowId>().ok()?;

        // RowPatch<'static> = (RowId, HashMap<FieldId,(IndexKey,ColumnInput<'static>)>)
        let mut patch: HashMap<FieldId, (IndexKey, ColumnInput<'static>)> =
            HashMap::with_capacity(fields.len());

        for fs in &fields {
            let idx_s = rec.get(fs.idx_col)?;
            let val_s = rec.get(fs.val_col)?;
            let idx = idx_s.parse::<u64>().ok()?; // IndexKey is u64 in your code
            patch.insert(
                fs.field_id,
                (
                    idx,
                    ColumnInput::from(Cow::Owned(val_s.as_bytes().to_vec())),
                ),
            );
        }

        Some((rid, patch))
    });

    match res {
        Ok(_) => {
            println!("CSV ingest complete.");
        }
        Err(e) => {
            eprintln!("ingest failed: {:?}", e);
            return Err(Box::<dyn std::error::Error>::from(e));
        }
    }

    Ok(())
}
