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
    ColumnStore, Result as LlResult,
    storage::pager::{BatchGet, BatchPut, MemPager, Pager},
    store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorPageHeader},
    types::{LogicalFieldId, PhysicalKey},
};
use std::sync::Arc;

// -------- simple key/value generators ---------------------------------------
//
// The legacy example used explicit key/value pairs and a KV append path.
// The new storage layout ingests Arrow arrays in RecordBatches, so we
// synthesize per-column arrays here (fixed and variable width) while
// preserving the same structure and phase breakdown.

use arrow::array::{ArrayRef, BinaryBuilder, UInt64Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;

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
    for i in 0..rows {
        let span = (max - min + 1) as u64;
        let mix = (i as u64)
            .wrapping_mul(1103515245)
            .wrapping_add(field_id)
            .rotate_left(13);
        let len = (min as u64 + (mix % span)) as usize;
        let byte = (((i as LogicalFieldId).wrapping_add(field_id)) & 0xFF) as u8;
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
            md.insert("field_id".to_string(), fid.to_string());
            Field::new(&format!("c{}", i), arr.data_type().clone(), false).with_metadata(md)
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = cols.iter().map(|(_, a)| Arc::clone(a)).collect();
    RecordBatch::try_new(schema, arrays).unwrap()
}

// -------- I/O metric helpers (count wrapper over Pager) ----------------------

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
struct IoStats {
    batches: AtomicU64,
    put_raw_ops: AtomicU64,
    get_raw_ops: AtomicU64,
    free_ops: AtomicU64,
}

#[derive(Clone)]
struct CountingPager<P: Pager> {
    inner: Arc<P>,
    stats: Arc<IoStats>,
}

impl<P: Pager> CountingPager<P> {
    fn new(inner: Arc<P>) -> Self {
        Self {
            inner,
            stats: Arc::new(IoStats::default()),
        }
    }
    fn snapshot(&self) -> Counts {
        Counts {
            batches: self.stats.batches.load(Ordering::Relaxed),
            put_raw: self.stats.put_raw_ops.load(Ordering::Relaxed),
            get_raw: self.stats.get_raw_ops.load(Ordering::Relaxed),
            frees: self.stats.free_ops.load(Ordering::Relaxed),
        }
    }
}

impl<P: Pager> Pager for CountingPager<P> {
    type Blob = P::Blob;

    fn alloc_many(&self, n: usize) -> LlResult<Vec<PhysicalKey>> {
        self.inner.alloc_many(n)
    }

    fn get_raw(&self, key: PhysicalKey) -> LlResult<Option<Self::Blob>> {
        self.inner.get_raw(key)
    }

    fn batch_get(
        &self,
        gets: &[BatchGet],
    ) -> LlResult<Vec<llkv_column_map::storage::pager::GetResult<Self::Blob>>> {
        self.stats.batches.fetch_add(1, Ordering::Relaxed);
        self.stats
            .get_raw_ops
            .fetch_add(gets.len() as u64, Ordering::Relaxed);
        self.inner.batch_get(gets)
    }

    fn batch_put(&self, puts: &[BatchPut]) -> LlResult<()> {
        self.stats.batches.fetch_add(1, Ordering::Relaxed);
        self.stats
            .put_raw_ops
            .fetch_add(puts.len() as u64, Ordering::Relaxed);
        self.inner.batch_put(puts)
    }

    fn free_many(&self, keys: &[PhysicalKey]) -> LlResult<()> {
        self.stats.batches.fetch_add(1, Ordering::Relaxed);
        self.stats
            .free_ops
            .fetch_add(keys.len() as u64, Ordering::Relaxed);
        self.inner.free_many(keys)
    }
}

#[derive(Clone, Copy, Default)]
struct Counts {
    batches: u64,
    put_raw: u64,
    get_raw: u64,
    frees: u64,
}

impl core::ops::Sub for Counts {
    type Output = Counts;
    fn sub(self, rhs: Counts) -> Counts {
        Counts {
            batches: self.batches.saturating_sub(rhs.batches),
            put_raw: self.put_raw.saturating_sub(rhs.put_raw),
            get_raw: self.get_raw.saturating_sub(rhs.get_raw),
            frees: self.frees.saturating_sub(rhs.frees),
        }
    }
}

fn read_counts<P: Pager>(pager: &CountingPager<P>) -> Counts {
    pager.snapshot()
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
                print!("{}:{}", fid, n);
            }
            println!("}} (total={})", a.total_cells);
        }
    }
}

fn show_phase_with_data<P: Pager>(
    label: &str,
    pager: &CountingPager<P>,
    prev: &mut Counts,
    summary: &AppendSummary,
) {
    print_data_summary(label, summary);
    let now = read_counts(pager);
    let d = now - *prev;
    println!("  batches: {}", d.batches);
    println!("  puts:   raw={}", d.put_raw);
    println!("  gets:   raw={}", d.get_raw);
    println!("  frees:  {}", d.frees);
    println!();
    *prev = now;
}

fn show_phase<P: Pager>(label: &str, pager: &CountingPager<P>, prev: &mut Counts) {
    println!("== {} ==", label);
    let now = read_counts(pager);
    let d = now - *prev;
    println!("  batches: {}", d.batches);
    println!("  puts:   raw={}", d.put_raw);
    println!("  gets:   raw={}", d.get_raw);
    println!("  frees:  {}", d.frees);
    println!();
    *prev = now;
}

// -------- sample read report + ASCII storage summary ------------------------

fn print_read_report_scan(store: &ColumnStore<CountingPager<MemPager>>) {
    use arrow::array::{Array, UInt64Array};

    println!("-- Read report (scan sample) --");
    for &fid in &[100u64, 200u64, 201u64, 300u64, 301u64, 999u64] {
        let mut seen = 0usize;
        let mut last_val: Option<u64> = None;
        if let Ok(iter) = store.scan(fid) {
            for arr in iter.flatten() {
                if let Some(u) = arr.as_any().downcast_ref::<UInt64Array>() {
                    if u.len() > 0 {
                        last_val = Some(u.value(u.len() - 1));
                    }
                }
                seen += arr.len();
                if seen >= 16 {
                    break;
                }
            }
            println!(
                "col {}: first ~{} rows scanned; last_u64={:?}",
                fid, seen, last_val
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
            "  Field {}: desc_pk={} rows={} chunks={}",
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

                let tomb = if meta.tombstone_pk != 0 {
                    pager
                        .get_raw(meta.tombstone_pk)
                        .unwrap()
                        .as_ref()
                        .map(|b| b.as_ref().len())
                        .unwrap_or(0)
                } else {
                    0
                };

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
                    "      chunk pk={} rows={} data={}B tomb={}B perm={}B",
                    meta.chunk_pk, meta.row_count, data_len, tomb, perm
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
    let inner = Arc::new(MemPager::default());
    let pager = Arc::new(CountingPager::new(inner));
    let store = ColumnStore::open(Arc::clone(&pager)).unwrap();

    let mut prev = read_counts(&pager);

    show_phase("Phase 0: init (bootstrap + manifest)", &pager, &mut prev);

    {
        let c100 = build_fixed_u64(100, 10_000);
        let c101 = build_fixed_u64(101, 10_000);
        let batch = batch_from_columns(&[c100.clone(), c101.clone()]);
        store.append(&batch).unwrap();

        let summary = summarize_pairs(&[(c100.0, batch.num_rows()), (c101.0, batch.num_rows())]);
        show_phase_with_data(
            "Phase 1: append fixed-width cols 100 & 101",
            &pager,
            &mut prev,
            &summary,
        );
    }

    {
        let c200 = build_var_binary(200, 12_345, 6, 18);
        let c201 = build_var_binary(201, 12_345, 6, 18);
        let batch = batch_from_columns(&[c200.clone(), c201.clone()]);
        store.append(&batch).unwrap();

        let summary = summarize_pairs(&[(c200.0, batch.num_rows()), (c201.0, batch.num_rows())]);
        show_phase_with_data(
            "Phase 2: append variable-width cols 200 & 201",
            &pager,
            &mut prev,
            &summary,
        );
    }

    {
        let c300 = build_fixed_u64(300, 2_000);
        let c301 = build_var_binary(301, 2_000, 10, 30);
        let batch = batch_from_columns(&[c300.clone(), c301.clone()]);
        store.append(&batch).unwrap();

        let summary = summarize_pairs(&[(c300.0, batch.num_rows()), (c301.0, batch.num_rows())]);
        show_phase_with_data(
            "Phase 3: append new cols 300 (fixed) & 301 (var) — updates Manifest",
            &pager,
            &mut prev,
            &summary,
        );
    }

    {
        let c100 = (
            100,
            Arc::new(UInt64Array::from(vec![5u64, 7, 9])) as ArrayRef,
        );
        let b100 = batch_from_columns(&[c100.clone()]);
        store.append(&b100).unwrap();

        let rows_999 = 10usize;
        let mut bb = BinaryBuilder::new();
        for r in 1_000..1_000 + rows_999 as u64 {
            bb.append_value(&vec![0xAB; (r % 17 + 12) as usize]);
        }
        let c999 = (999, Arc::new(bb.finish()) as ArrayRef);
        let b999 = batch_from_columns(&[c999.clone()]);
        store.append(&b999).unwrap();

        let summary = summarize_pairs(&[(100, 3), (999, rows_999)]);
        show_phase_with_data(
            "Phase 4: mixed append (existing col 100 + new col 999)",
            &pager,
            &mut prev,
            &summary,
        );
    }

    {
        print_read_report_scan(&store);

        let ascii = render_storage_ascii(&*pager);
        println!("\n==== STORAGE ASCII ====\n{}", ascii);

        show_phase("Phase 5: describe_storage + read report", &pager, &mut prev);
    }
}
