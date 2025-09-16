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
    storage::pager::{InstrumentedPager, IoStats, MemPager, Pager},
    store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorPageHeader},
    types::{LogicalFieldId, Namespace, PhysicalKey},
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
        b.append_value(&vec![byte; len]);
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
            Field::new(&format!("c{}", i), arr.data_type().clone(), false).with_metadata(md)
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
    batches: u64,
    puts: u64,
    gets: u64,
    frees: u64,
    allocs: u64,
}

impl From<&Arc<IoStats>> for CountsSnapshot {
    fn from(stats: &Arc<IoStats>) -> Self {
        Self {
            batches: stats.get_batches.load(Ordering::Relaxed)
                + stats.put_batches.load(Ordering::Relaxed)
                + stats.free_batches.load(Ordering::Relaxed)
                + stats.alloc_batches.load(Ordering::Relaxed),
            puts: stats.physical_puts.load(Ordering::Relaxed),
            gets: stats.physical_gets.load(Ordering::Relaxed),
            frees: stats.physical_frees.load(Ordering::Relaxed),
            allocs: stats.physical_allocs.load(Ordering::Relaxed),
        }
    }
}

impl core::ops::Sub for CountsSnapshot {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            batches: self.batches.saturating_sub(rhs.batches),
            puts: self.puts.saturating_sub(rhs.puts),
            gets: self.gets.saturating_sub(rhs.gets),
            frees: self.frees.saturating_sub(rhs.frees),
            allocs: self.allocs.saturating_sub(rhs.allocs),
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
    println!("  batches: {}", d.batches);
    println!("  puts:    {}", d.puts);
    println!("  gets:    {}", d.gets);
    println!("  frees:   {}", d.frees);
    println!("  allocs:  {}", d.allocs);
    println!();
    *prev = now;
}

fn show_phase(label: &str, stats: &Arc<IoStats>, prev: &mut CountsSnapshot) {
    println!("== {} ==", label);
    let now = CountsSnapshot::from(stats);
    let d = now - *prev;
    println!("  batches: {}", d.batches);
    println!("  puts:    {}", d.puts);
    println!("  gets:    {}", d.gets);
    println!("  frees:   {}", d.frees);
    println!("  allocs:  {}", d.allocs);
    println!();
    *prev = now;
}

// -------- sample read report + ASCII storage summary ------------------------

fn print_read_report_scan(store: &ColumnStore<InstrumentedPager<MemPager>>) {
    use arrow::array::{Array, UInt64Array};

    println!("-- Read report (scan sample) --");
    for &id in &[100u32, 200, 201, 300, 301, 999] {
        let field_id = fid(id);
        let mut seen = 0usize;
        let mut last_val: Option<u64> = None;
        if let Ok(iter) = store.scan(field_id) {
            for arr in iter.flatten() {
                if let Some(u) = arr.as_any().downcast_ref::<UInt64Array>() {
                    if !u.is_empty() {
                        last_val = Some(u.value(u.len() - 1));
                    }
                }
                seen += arr.len();
                if seen >= 16 {
                    break;
                }
            }
            println!(
                "col {:?}: first ~{} rows scanned; last_u64={:?}",
                field_id, seen, last_val
            );
        }
    }
    println!();
}

const ROOT_PK: PhysicalKey = 0;

fn render_storage_ascii<P: Pager>(pager: &P) -> String {
    use llkv_column_map::store::catalog::ColumnCatalog;

    let mut s = String::new();
    let Some(cat_blob) = pager.get_raw(ROOT_PK).unwrap() else {
        return "Empty catalog".to_string();
    };
    let catalog = ColumnCatalog::from_bytes(cat_blob.as_ref()).unwrap();
    use std::fmt::Write;
    writeln!(&mut s, "Catalog entries: {}", catalog.map.len()).unwrap();

    for (fid, desc_pk) in catalog.map.iter() {
        let desc_blob = pager.get_raw(*desc_pk).unwrap().unwrap();
        let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        writeln!(
            &mut s,
            "  Field {:?}: desc_pk={} rows={} chunks={}",
            fid, desc_pk, desc.total_row_count, desc.total_chunk_count
        )
        .unwrap();

        let mut page_pk = desc.head_page_pk;
        let mut page_idx = 0usize;
        while page_pk != 0 {
            let page_blob = pager.get_raw(page_pk).unwrap().unwrap();
            let bytes = page_blob.as_ref();
            let hdr_sz = DescriptorPageHeader::DISK_SIZE;
            let hd = DescriptorPageHeader::from_le_bytes(&bytes[..hdr_sz]);
            writeln!(
                &mut s,
                "    page[{}] pk={} entries={}",
                page_idx, page_pk, hd.entry_count
            )
            .unwrap();

            for i in 0..(hd.entry_count as usize) {
                let off = hdr_sz + i * ChunkMetadata::DISK_SIZE;
                let end = off + ChunkMetadata::DISK_SIZE;
                let meta = ChunkMetadata::from_le_bytes(&bytes[off..end]);

                let data_len = pager
                    .get_raw(meta.chunk_pk)
                    .unwrap()
                    .as_ref()
                    .map(|b| b.as_ref().len())
                    .unwrap_or(0);

                let perm = if meta.value_order_perm_pk != 0 {
                    pager
                        .get_raw(meta.value_order_perm_pk)
                        .unwrap()
                        .as_ref()
                        .map(|b| b.as_ref().len())
                        .unwrap_or(0)
                } else {
                    0
                };

                writeln!(
                    &mut s,
                    "      chunk pk={} rows={} data={}B perm={}B",
                    meta.chunk_pk, meta.row_count, data_len, perm
                )
                .unwrap();
            }

            page_pk = hd.next_page_pk;
            page_idx += 1;
        }
    }

    s
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
        let b100 = batch_from_columns(&[c100.clone()]);
        store.append(&b100).unwrap();

        let rows_999 = 10usize;
        let mut bb = BinaryBuilder::new();
        for r in 1_000..1_000 + rows_999 as u64 {
            bb.append_value(&vec![0xAB; (r % 17 + 12) as usize]);
        }
        let c999 = (fid(999), Arc::new(bb.finish()) as ArrayRef);
        let b999 = batch_from_columns(&[c999.clone()]);
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

        // FIX: Pass a reference to the pager *inside* the Arc, not the Arc itself.
        let ascii = render_storage_ascii(pager_arc.as_ref());
        println!("\n==== STORAGE ASCII ====\n{}", ascii);

        show_phase("Phase 5: describe_storage + read report", &stats, &mut prev);
    }
}
