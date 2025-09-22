//! Example: phase-by-phase I/O stats + clear “how much data we wrote” lines.
//!
//! For each phase we print ONLY the I/O done in that phase by snapshotting
//! cumulative stats before/after and printing the difference. No delta API
//! is required.
//!
//! Output per phase now includes a data summary like:
//!   data: 2 columns × 10,000 rows/col = 20,000 cells
//! or, when counts differ across columns:
//!   data: 2 columns; cells per column = {100:3, 999:10} (total=13)
//!
//! “cells” = number of (key,value) entries written, i.e., len(items) per
//! column.

use llkv_column_map::{
    ColumnStore,
    debug::ColumnStoreDebug,
    storage::pager::{InstrumentedPager, IoStats, MemPager},
    types::{LogicalFieldId, Namespace},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// -------- simple key/value generators ---------------------------------------

use arrow::array::{ArrayRef, BinaryBuilder, UInt64Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;

/// Helper to create a standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

/// Build a fixed-width u64 column [0..rows) for `field_id`.
fn build_fixed_u64(field_id: LogicalFieldId, rows: usize) -> (LogicalFieldId, ArrayRef) {
    let vals: Vec<u64> = (0..rows as u64).collect();
    (field_id, Arc::new(UInt64Array::from(vals)) as ArrayRef)
}

fn build_var_binary(
    field_id: LogicalFieldId,
    rows: usize,
    min: usize,
    max: usize,
) -> (LogicalFieldId, ArrayRef) {
    let mut b = BinaryBuilder::new();
    let fid_u64 = u64::from(field_id);
    for i in 0..rows {
        let span = (max - min + 1) as u64;
        let mix = (i as u64)
            .wrapping_mul(1103515245)
            .wrapping_add(fid_u64)
            .rotate_left(13);
        let len = (min as u64 + (mix % span)) as usize;
        let byte = (((i as u64).wrapping_add(fid_u64)) & 0xFF) as u8;
        b.append_value(vec![byte; len]);
    }
    (field_id, Arc::new(b.finish()) as ArrayRef)
}

fn batch_from_columns(cols: &[(LogicalFieldId, ArrayRef)]) -> RecordBatch {
    let fields: Vec<Field> = cols
        .iter()
        .enumerate()
        .map(|(i, (fid, arr))| {
            let mut md = std::collections::HashMap::new();
            md.insert("field_id".to_string(), u64::from(*fid).to_string());
            Field::new(format!("c{i}"), arr.data_type().clone(), false).with_metadata(md)
        })
        .collect();

    // The store's append logic requires a `row_id` column.
    let num_rows = if cols.is_empty() { 0 } else { cols[0].1.len() };
    let row_id_field = Field::new("row_id", arrow::datatypes::DataType::UInt64, false);
    // Use unique row IDs for each batch to avoid LWW updates in this example.
    let start_row_id = NEXT_ROW_ID.fetch_add(num_rows as u64, Ordering::Relaxed);
    let end_row_id = start_row_id + num_rows as u64;
    let row_id_array =
        Arc::new(UInt64Array::from_iter_values(start_row_id..end_row_id)) as ArrayRef;

    let mut final_fields = vec![row_id_field];
    final_fields.extend(fields);
    let mut final_arrays = vec![row_id_array];
    final_arrays.extend(cols.iter().map(|(_, a)| Arc::clone(a)));

    let schema = Arc::new(Schema::new(final_fields));
    RecordBatch::try_new(schema, final_arrays).unwrap()
}

// Global counter to ensure unique row_ids across batches.
static NEXT_ROW_ID: AtomicU64 = AtomicU64::new(0);

// -------- I/O metric helpers -------------------------------------------------

#[derive(Clone, Copy, Default, Debug)]
struct CountsSnapshot {
    get_batches: u64,
    put_batches: u64,
    free_batches: u64,
    alloc_batches: u64,
    physical_puts: u64,
    physical_gets: u64,
    physical_frees: u64,
    physical_allocs: u64,
}

impl From<&Arc<IoStats>> for CountsSnapshot {
    fn from(stats: &Arc<IoStats>) -> Self {
        Self {
            get_batches: stats.get_batches.load(Ordering::Relaxed),
            put_batches: stats.put_batches.load(Ordering::Relaxed),
            free_batches: stats.free_batches.load(Ordering::Relaxed),
            alloc_batches: stats.alloc_batches.load(Ordering::Relaxed),
            physical_puts: stats.physical_puts.load(Ordering::Relaxed),
            physical_gets: stats.physical_gets.load(Ordering::Relaxed),
            physical_frees: stats.physical_frees.load(Ordering::Relaxed),
            physical_allocs: stats.physical_allocs.load(Ordering::Relaxed),
        }
    }
}

impl core::ops::Sub for CountsSnapshot {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            get_batches: self.get_batches.saturating_sub(rhs.get_batches),
            put_batches: self.put_batches.saturating_sub(rhs.put_batches),
            free_batches: self.free_batches.saturating_sub(rhs.free_batches),
            alloc_batches: self.alloc_batches.saturating_sub(rhs.alloc_batches),
            physical_puts: self.physical_puts.saturating_sub(rhs.physical_puts),
            physical_gets: self.physical_gets.saturating_sub(rhs.physical_gets),
            physical_frees: self.physical_frees.saturating_sub(rhs.physical_frees),
            physical_allocs: self.physical_allocs.saturating_sub(rhs.physical_allocs),
        }
    }
}

// -------- “how much data did we write” summary -------------------------------

struct AppendSummary {
    num_cols: usize,
    total_cells: usize,
    uniform_rows_per_col: Option<usize>,
    per_col: Vec<(LogicalFieldId, usize)>,
}

fn summarize_pairs(pairs: &[(LogicalFieldId, usize)]) -> AppendSummary {
    let per_col = pairs.to_vec();
    let num_cols = per_col.len();
    let total_cells: usize = per_col.iter().map(|(_, n)| *n).sum();
    let uniform_rows_per_col = if num_cols > 0 && per_col.iter().all(|(_, n)| *n == per_col[0].1) {
        Some(per_col[0].1)
    } else {
        None
    };
    AppendSummary {
        num_cols,
        total_cells,
        uniform_rows_per_col,
        per_col,
    }
}

fn print_data_summary(label: &str, a: &AppendSummary) {
    println!("== {} ==", label);
    match a.uniform_rows_per_col {
        Some(rows) => {
            println!(
                "  data: {} columns × {} rows/col = {} cells",
                a.num_cols, rows, a.total_cells
            );
        }
        None => {
            print!("  data: {} columns; cells per column = {{", a.num_cols);
            for (i, (fid, n)) in a.per_col.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:?}:{}", fid, n);
            }
            println!("}} (total={})", a.total_cells);
        }
    }
}

fn show_phase_with_data(
    label: &str,
    stats: &Arc<IoStats>,
    prev: &mut CountsSnapshot,
    summary: &AppendSummary,
) {
    print_data_summary(label, summary);
    let now = CountsSnapshot::from(stats);
    let d = now - *prev;
    println!(
        "  batch ops: gets={}, puts={}, frees={}, allocs={}",
        d.get_batches, d.put_batches, d.free_batches, d.alloc_batches
    );
    println!(
        "  phys ops:  gets={}, puts={}, frees={}, allocs={}",
        d.physical_gets, d.physical_puts, d.physical_frees, d.physical_allocs
    );
    println!();
    *prev = now;
}

fn show_phase(label: &str, stats: &Arc<IoStats>, prev: &mut CountsSnapshot) {
    println!("== {} ==", label);
    let now = CountsSnapshot::from(stats);
    let d = now - *prev;
    println!(
        "  batch ops: gets={}, puts={}, frees={}, allocs={}",
        d.get_batches, d.put_batches, d.free_batches, d.alloc_batches
    );
    println!(
        "  phys ops:  gets={}, puts={}, frees={}, allocs={}",
        d.physical_gets, d.physical_puts, d.physical_frees, d.physical_allocs
    );
    println!();
    *prev = now;
}

// -------- sample read report + ASCII storage summary ------------------------

fn print_read_report_scan(store: &ColumnStore<InstrumentedPager<MemPager>>) {
    use arrow::array::UInt64Array;
    use llkv_column_map::store::scan::{PrimitiveVisitor, ScanOptions};

    println!("-- Read report (scan sample) --");
    for &id in &[100u32, 200, 201, 300, 301, 999] {
        let field_id = fid(id);
        // Visitor to count up to a small budget and capture last u64 value if applicable.
        struct Sample {
            seen: usize,
            last_u64: Option<u64>,
            budget: usize,
        }
        impl PrimitiveVisitor for Sample {
            fn u64_chunk(&mut self, a: &UInt64Array) {
                if a.is_empty() {
                    self.last_u64 = Some(a.value(a.len() - 1));
                }
                self.seen = (self.seen + a.len()).min(self.budget);
            }
            // fn u32_chunk(&mut self, a: &arrow::array::UInt32Array) {
            //     self.seen = (self.seen + a.len()).min(self.budget);
            // }
            // fn u16_chunk(&mut self, a: &arrow::array::UInt16Array) { self.seen = (self.seen + a.len()).min(self.budget); }
            // fn u8_chunk(&mut self, a: &arrow::array::UInt8Array) { self.seen = (self.seen + a.len()).min(self.budget); }
            // fn i64_chunk(&mut self, a: &arrow::array::Int64Array) { self.seen = (self.seen + a.len()).min(self.budget); }
            // fn i32_chunk(&mut self, a: &arrow::array::Int32Array) { self.seen = (self.seen + a.len()).min(self.budget); }
            // fn i16_chunk(&mut self, a: &arrow::array::Int16Array) { self.seen = (self.seen + a.len()).min(self.budget); }
            // fn i8_chunk(&mut self, a: &arrow::array::Int8Array) { self.seen = (self.seen + a.len()).min(self.budget); }
        }
        impl llkv_column_map::store::scan::PrimitiveSortedVisitor for Sample {}
        impl llkv_column_map::store::scan::PrimitiveWithRowIdsVisitor for Sample {}
        impl llkv_column_map::store::scan::PrimitiveSortedWithRowIdsVisitor for Sample {}
        let mut v = Sample {
            seen: 0,
            last_u64: None,
            budget: 16,
        };
        match store.scan(field_id, ScanOptions::default(), &mut v) {
            Ok(()) => {
                println!(
                    "col {:?}: scanned primitive ints (~{} rows); last_u64={:?}",
                    field_id, v.seen, v.last_u64
                );
            }
            Err(_) => {
                // Non-integer columns are not handled by the primitive scan API here.
                println!(
                    "col {:?}: scan not supported for this dtype in this example",
                    field_id
                );
            }
        }
    }
    println!();
}

// -------- main walkthrough ---------------------------------------------------

fn main() {
    let (pager, stats) = InstrumentedPager::new(MemPager::default());
    let pager_arc = Arc::new(pager);
    let store = ColumnStore::open(Arc::clone(&pager_arc)).unwrap();

    let mut prev = CountsSnapshot::from(&stats);

    show_phase("Phase 0: init (bootstrap + manifest)", &stats, &mut prev);

    {
        let c100 = build_fixed_u64(fid(100), 10_000);
        let c101 = build_fixed_u64(fid(101), 10_000);
        let batch = batch_from_columns(&[c100.clone(), c101.clone()]);
        store.append(&batch).unwrap();

        let summary = summarize_pairs(&[(c100.0, c100.1.len()), (c101.0, c101.1.len())]);
        show_phase_with_data(
            "Phase 1: append fixed-width cols 100 & 101",
            &stats,
            &mut prev,
            &summary,
        );
    }

    {
        let c200 = build_var_binary(fid(200), 12_345, 6, 18);
        let c201 = build_var_binary(fid(201), 12_345, 6, 18);
        let batch = batch_from_columns(&[c200.clone(), c201.clone()]);
        store.append(&batch).unwrap();

        let summary = summarize_pairs(&[(c200.0, c200.1.len()), (c201.0, c201.1.len())]);
        show_phase_with_data(
            "Phase 2: append variable-width cols 200 & 201",
            &stats,
            &mut prev,
            &summary,
        );
    }

    {
        let c300 = build_fixed_u64(fid(300), 2_000);
        let c301 = build_var_binary(fid(301), 2_000, 10, 30);
        let batch = batch_from_columns(&[c300.clone(), c301.clone()]);
        store.append(&batch).unwrap();

        let summary = summarize_pairs(&[(c300.0, c300.1.len()), (c301.0, c301.1.len())]);
        show_phase_with_data(
            "Phase 3: append new cols 300 (fixed) & 301 (var) — updates Manifest",
            &stats,
            &mut prev,
            &summary,
        );
    }

    {
        let c100 = (
            fid(100),
            Arc::new(UInt64Array::from(vec![5u64, 7, 9])) as ArrayRef,
        );
        let b100 = batch_from_columns(std::slice::from_ref(&c100));
        store.append(&b100).unwrap();

        let rows_999 = 10usize;
        let mut bb = BinaryBuilder::new();
        for r in 1_000..1_000 + rows_999 as u64 {
            bb.append_value(vec![0xAB; (r % 17 + 12) as usize]);
        }
        let c999 = (fid(999), Arc::new(bb.finish()) as ArrayRef);
        let b999 = batch_from_columns(std::slice::from_ref(&c999));
        store.append(&b999).unwrap();

        let summary = summarize_pairs(&[(fid(100), 3), (fid(999), rows_999)]);
        show_phase_with_data(
            "Phase 4: mixed append (existing col 100 + new col 999)",
            &stats,
            &mut prev,
            &summary,
        );
    }

    {
        print_read_report_scan(&store);

        let layout_table_str = store.render_storage_as_formatted_string();
        println!("\n==== STORAGE LAYOUT ====\n{}", layout_table_str);

        show_phase("Phase 5: describe_storage + read report", &stats, &mut prev);
    }
}
