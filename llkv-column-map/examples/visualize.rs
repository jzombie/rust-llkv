//! This example ingests data in 4 batches using the crate's high-level API
//! (ColumnStore), then writes ONE final `.dot` file where nodes are color-
//! coded by the batch in which they were created.
//!
//! --- How to View the Output ---
//!
//! Online: https://dreampuf.github.io/GraphvizOnline/?engine=dot
//!
//! or:
//!   1) Install Graphviz
//!      - macOS (Homebrew):      `brew install graphviz`
//!      - Ubuntu/Debian:         `sudo apt-get update && sudo apt-get install graphviz`
//!      - Windows (Chocolatey):  `choco install graphviz`
//!        or (Scoop):            `scoop install graphviz`
//!
//!   2) Run this example
//!      `cargo run --example visualize`
//!
//!      This creates one `storage_layout.dot` in the project root and prints
//!      an ASCII summary to stdout. Nodes are colored by batch of creation:
//!      - batch 0 (bootstrap/manifest): white
//!      - batch 1: lightskyblue
//!      - batch 2: palegreen
//!      - batch 3: khaki
//!      - batch 4: lightpink
//!
//!   3) Generate an image
//!      - PNG:  `dot -Tpng storage_layout.dot -o storage_layout.png`
//!      - SVG:  `dot -Tsvg storage_layout.dot -o storage_layout.svg`

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use llkv_column_map::{
    ColumnStore,
    storage::pager::MemPager,
    store::debug::{ColumnStoreDebug, discover_all_pks},
    types::{LogicalFieldId, Namespace, PhysicalKey},
};

// ---------------- Workload config (small, but shows batching clearly) --------

const BATCHES: usize = 4;

// Column 1: 500 rows, fixed width (4 bytes, auto-detected)
// Column 2: 500 rows, variable length (1..=21 bytes, auto-detected)
// Column 3: 5000 rows, fixed width (8 bytes, auto-detected)
const C1_ROWS: usize = 500;
const C2_ROWS: usize = 500;
const C3_ROWS: usize = 5_000;

use arrow::array::{Array, ArrayRef, BinaryBuilder, UInt32Array, UInt64Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;

// Global counter to ensure unique row_ids across batches.
static NEXT_ROW_ID: AtomicU64 = AtomicU64::new(0);

/// Helper to create a standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn build_put_for_col1(start: usize, end: usize) -> Option<(LogicalFieldId, ArrayRef)> {
    let s = start.min(C1_ROWS);
    let e = end.min(C1_ROWS);
    if s >= e {
        return None;
    }
    let vals: Vec<u32> = (s..e).map(|i| i as u32).collect();
    Some((fid(1), Arc::new(UInt32Array::from(vals)) as ArrayRef))
}

fn build_put_for_col2(start: usize, end: usize) -> Option<(LogicalFieldId, ArrayRef)> {
    let s = start.min(C2_ROWS);
    let e = end.min(C2_ROWS);
    if s >= e {
        return None;
    }
    // NOTE: arrow 56 BinaryBuilder::new() takes no capacity arg.
    let mut b = BinaryBuilder::new();
    for i in s..e {
        let len = i % 21 + 1; // 1..21
        b.append_value(&vec![b'A' + (i % 26) as u8; len]);
    }
    Some((fid(2), Arc::new(b.finish()) as ArrayRef))
}

fn build_put_for_col3(start: usize, end: usize) -> Option<(LogicalFieldId, ArrayRef)> {
    let s = start.min(C3_ROWS);
    let e = end.min(C3_ROWS);
    if s >= e {
        return None;
    }
    let vals: Vec<u64> = (s..e).map(|_| 0x55u64).collect(); // width=8
    Some((fid(3), Arc::new(UInt64Array::from(vals)) as ArrayRef))
}

/// Build a RecordBatch from per-column arrays (same row count).
fn batch_from_pairs(pairs: &[(LogicalFieldId, ArrayRef)]) -> RecordBatch {
    let fields: Vec<Field> = pairs
        .iter()
        .enumerate()
        .map(|(i, (fid, arr))| {
            let mut md = std::collections::HashMap::new();
            md.insert("field_id".to_string(), u64::from(*fid).to_string());
            Field::new(&format!("c{}", i), arr.data_type().clone(), false).with_metadata(md)
        })
        .collect();

    // The store's append logic requires a `row_id` column.
    let num_rows = if pairs.is_empty() {
        0
    } else {
        pairs[0].1.len()
    };
    let row_id_field = Field::new("row_id", arrow::datatypes::DataType::UInt64, false);
    let start_row_id = NEXT_ROW_ID.fetch_add(num_rows as u64, Ordering::Relaxed);
    let end_row_id = start_row_id + num_rows as u64;
    let row_id_array =
        Arc::new(UInt64Array::from_iter_values(start_row_id..end_row_id)) as ArrayRef;

    let mut final_fields = vec![row_id_field];
    final_fields.extend(fields);
    let mut final_arrays = vec![row_id_array];
    final_arrays.extend(pairs.iter().map(|(_, a)| Arc::clone(a)));

    let schema = Arc::new(Schema::new(final_fields));
    RecordBatch::try_new(schema, final_arrays).unwrap()
}

// ---------------- Main: multi-batch ingest then ONE colored DOT -------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(MemPager::default());
    // Keep a handle to the pager so we can walk storage for visualization.
    let store = ColumnStore::open(Arc::clone(&pager)).unwrap();

    // record creation batch for each physical key we ever see
    // batch 0 = pre-existing (bootstrap, manifest)
    let mut created_in_batch: HashMap<PhysicalKey, usize> = HashMap::new();
    for pk in discover_all_pks(pager.as_ref()) {
        created_in_batch.insert(pk, 0);
    }

    let rows_per_batch = C3_ROWS.div_ceil(BATCHES);

    println!(
        "Ingesting col1={} rows, col2={} rows, col3={} rows in {} batches \
         ({} rows/batch on col3)...",
        C1_ROWS, C2_ROWS, C3_ROWS, BATCHES, rows_per_batch
    );

    let t_total = Instant::now();
    for b in 0..BATCHES {
        let start = b * rows_per_batch;
        let end = (start + rows_per_batch).min(C3_ROWS);

        // Build per-column puts for this window; some will be empty.
        let mut pairs: Vec<(LogicalFieldId, ArrayRef)> = Vec::new();
        if let Some(p) = build_put_for_col1(start, end) {
            pairs.push(p);
        }
        if let Some(p) = build_put_for_col2(start, end) {
            pairs.push(p);
        }
        if let Some(p) = build_put_for_col3(start, end) {
            pairs.push(p);
        }
        if pairs.is_empty() {
            continue;
        }

        // Group columns by equal row count; make one RecordBatch per group.
        let mut by_len: HashMap<usize, Vec<(LogicalFieldId, ArrayRef)>> = HashMap::new();
        for (fid, arr) in pairs {
            by_len.entry(arr.len()).or_default().push((fid, arr));
        }

        let t_batch = Instant::now();
        for (_len, group) in by_len {
            let batch = batch_from_pairs(&group);
            store.append(&batch).unwrap();
        }
        let dt = t_batch.elapsed();
        println!("  batch {}: rows [{}..{}) in {:?}", b + 1, start, end, dt);

        // mark new physical keys as created in this batch (b+1)
        for pk in discover_all_pks(pager.as_ref()) {
            created_in_batch.entry(pk).or_insert(b + 1);
        }
    }
    println!("Total ingest time: {:?}", t_total.elapsed());

    // Simple probes by counting rows per column via scan() for primitive integer columns.
    use llkv_column_map::store::scan::{
        PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
        PrimitiveWithRowIdsVisitor, ScanOptions,
    };
    struct Count; // sums lengths for any primitive typed chunk
    impl PrimitiveVisitor for Count {
        fn u64_chunk(&mut self, a: &UInt64Array) {
            ROWS.fetch_add(a.len() as u64, Ordering::Relaxed);
        }
        fn u32_chunk(&mut self, a: &UInt32Array) {
            ROWS.fetch_add(a.len() as u64, Ordering::Relaxed);
        }
        // fn u16_chunk(&mut self, a: &arrow::array::UInt16Array) { ROWS.fetch_add(a.len() as u64, Ordering::Relaxed); }
        // fn u8_chunk(&mut self, a: &arrow::array::UInt8Array) { ROWS.fetch_add(a.len() as u64, Ordering::Relaxed); }
        // fn i64_chunk(&mut self, a: &arrow::array::Int64Array) { ROWS.fetch_add(a.len() as u64, Ordering::Relaxed); }
        // fn i32_chunk(&mut self, a: &arrow::array::Int32Array) { ROWS.fetch_add(a.len() as u64, Ordering::Relaxed); }
        // fn i16_chunk(&mut self, a: &arrow::array::Int16Array) { ROWS.fetch_add(a.len() as u64, Ordering::Relaxed); }
        // fn i8_chunk(&mut self, a: &arrow::array::Int8Array) { ROWS.fetch_add(a.len() as u64, Ordering::Relaxed); }
    }
    impl PrimitiveSortedVisitor for Count {}
    impl PrimitiveWithRowIdsVisitor for Count {}
    impl PrimitiveSortedWithRowIdsVisitor for Count {}
    static ROWS: AtomicU64 = AtomicU64::new(0);
    for id in [1u32, 2, 3] {
        let field_id = fid(id);
        ROWS.store(0, Ordering::Relaxed);
        let mut v = Count;
        match store.scan(field_id, ScanOptions::default(), &mut v) {
            Ok(()) => {
                let rows = ROWS.load(Ordering::Relaxed) as usize;
                println!(
                    "col={:?} -> total primitive rows scanned: {}",
                    field_id, rows
                );
            }
            Err(_) => {
                println!(
                    "col={:?} -> scan not supported for this dtype in this example",
                    field_id
                );
            }
        }
    }

    // ASCII summary of final layout using the new trait method.
    let summary_table = store.render_storage_as_formatted_string();
    println!("\n==== STORAGE LAYOUT ====\n{}", summary_table);

    // ONE final DOT with batch-colored nodes using the new trait method.
    let dot = store.render_storage_as_dot(&created_in_batch);
    std::fs::write("storage_layout.dot", dot)?;
    println!("Wrote storage_layout.dot (single graph, nodes colored by batch)");

    Ok(())
}
