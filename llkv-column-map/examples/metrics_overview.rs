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
//! “cells” = number of (key,value) entries written, i.e., len(items) per column.

use llkv_column_map::pager::BlobLike;
use llkv_column_map::{
    AppendOptions, ColumnStore, Put, ValueMode, ValueSlice, pager::MemPager, types::LogicalFieldId,
};

// -------- simple key/value generators ---------------------------------------

fn be_key(v: u64) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

fn fixed_value(row: u64, field: u32, width: usize) -> Vec<u8> {
    // Stable 8-byte seed then repeat/truncate to requested width
    let seed = (row ^ field as u64).to_le_bytes();
    if width <= 8 {
        seed[..width].to_vec()
    } else {
        let mut out = Vec::with_capacity(width);
        while out.len() < width {
            let take = core::cmp::min(8, width - out.len());
            out.extend_from_slice(&seed[..take]);
        }
        out
    }
}

fn build_put_fixed(field_id: LogicalFieldId, start: u64, end: u64, width: usize) -> Put {
    let mut items = Vec::with_capacity((end - start) as usize);
    for r in start..end {
        items.push((be_key(r), fixed_value(r, field_id, width)));
    }
    Put { field_id, items }
}

fn build_put_var(field_id: LogicalFieldId, start: u64, end: u64, min: usize, max: usize) -> Put {
    let mut items = Vec::with_capacity((end - start) as usize);
    for r in start..end {
        // pseudo-var length that depends on (row, field)
        let span = (max - min + 1) as u64;
        let mix = r
            .wrapping_mul(1103515245)
            .wrapping_add(field_id as u64)
            .rotate_left(13);
        let len = (min as u64 + (mix % span)) as usize;
        let byte = (((r as u32).wrapping_add(field_id)) & 0xFF) as u8;
        items.push((be_key(r), vec![byte; len]));
    }
    Put { field_id, items }
}

// -------- I/O metric helpers (compute per-phase delta locally) ---------------

#[derive(Clone, Copy, Default)]
struct Counts {
    batches: u64,
    put_raw: u64,
    put_typed: u64,
    get_raw: u64,
    get_typed: u64,
    frees: u64, // number of physical keys freed
}

impl core::ops::Sub for Counts {
    type Output = Counts;
    fn sub(self, rhs: Counts) -> Counts {
        Counts {
            batches: self.batches.saturating_sub(rhs.batches),
            put_raw: self.put_raw.saturating_sub(rhs.put_raw),
            put_typed: self.put_typed.saturating_sub(rhs.put_typed),
            get_raw: self.get_raw.saturating_sub(rhs.get_raw),
            get_typed: self.get_typed.saturating_sub(rhs.get_typed),
            frees: self.frees.saturating_sub(rhs.frees),
        }
    }
}

// Uses your actual IoStats field names.
fn read_counts<P: llkv_column_map::pager::Pager>(store: &ColumnStore<'_, P>) -> Counts {
    let s = store.io_stats(); // cumulative since process start
    Counts {
        batches: s.batches as u64,
        put_raw: s.put_raw_ops as u64,
        put_typed: s.put_typed_ops as u64,
        get_raw: s.get_raw_ops as u64,
        get_typed: s.get_typed_ops as u64,
        frees: s.free_ops as u64,
    }
}

// -------- “how much data did we write” summary -------------------------------

struct AppendSummary {
    num_cols: usize,
    total_cells: usize,
    // If every column wrote the same number of rows, we fill this.
    uniform_rows_per_col: Option<usize>,
    // Always available for the per-column fallback line.
    per_col: Vec<(LogicalFieldId, usize)>,
}

fn summarize_puts(puts: &[Put]) -> AppendSummary {
    let per_col: Vec<(LogicalFieldId, usize)> =
        puts.iter().map(|p| (p.field_id, p.items.len())).collect();
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
            // Compact per-column list
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

// Prints the phase’s data summary and then the per-phase I/O deltas.
fn show_phase_with_data<P: llkv_column_map::pager::Pager>(
    label: &str,
    store: &ColumnStore<'_, P>,
    prev: &mut Counts,
    summary: &AppendSummary,
) {
    print_data_summary(label, summary);
    let now = read_counts(store);
    let d = now - *prev;
    println!("  batches: {}", d.batches);
    println!("  puts:   raw={} typed={}", d.put_raw, d.put_typed);
    println!("  gets:   raw={} typed={}", d.get_raw, d.get_typed);
    println!("  frees:  {}", d.frees);
    println!();
    *prev = now;
}

// Simple version (no data line), used for phases that don’t write (e.g. init, describe)
fn show_phase<P: llkv_column_map::pager::Pager>(
    label: &str,
    store: &ColumnStore<'_, P>,
    prev: &mut Counts,
) {
    println!("== {} ==", label);
    let now = read_counts(store);
    let d = now - *prev;
    println!("  batches: {}", d.batches);
    println!("  puts:   raw={} typed={}", d.put_raw, d.put_typed);
    println!("  gets:   raw={} typed={}", d.get_raw, d.get_typed);
    println!("  frees:  {}", d.frees);
    println!();
    *prev = now;
}

// -------- pretty read report -------------------------------------------------

use llkv_column_map::types::LogicalFieldId as Fid;

fn fmt_key(k: &[u8]) -> String {
    if k.len() == 8 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(k);
        format!("u64({})", u64::from_be_bytes(buf))
    } else {
        // short hex
        let mut s = String::with_capacity(k.len() * 2);
        for b in k {
            use core::fmt::Write;
            let _ = write!(s, "{:02x}", b);
        }
        format!("hex({})", s)
    }
}

fn print_read_report<B>(
    heading: &str,
    queries: &[(Fid, Vec<Vec<u8>>)],
    results: &[Vec<Option<ValueSlice<B>>>],
) where
    B: BlobLike,
{
    println!("-- {} --", heading);
    for (i, (fid, ks)) in queries.iter().enumerate() {
        print!("col {}: ", fid);
        for (j, k) in ks.iter().enumerate() {
            match &results[i][j] {
                Some(bytes) => print!("{} → HIT {}B;  ", fmt_key(k), bytes.as_ref().len()),
                None => print!("{} → MISSING;  ", fmt_key(k)),
            }
        }
        println!();
    }
    println!();
}

// -------- main walkthrough ---------------------------------------------------

fn main() {
    let pager = MemPager::default();
    let store = ColumnStore::init_empty(&pager);

    // We'll accumulate a previous snapshot here and compute deltas per phase.
    let mut prev = read_counts(&store);

    // Phase 0: init (writes manifest+bootstrap before ColumnStore is counting).
    show_phase("Phase 0: init (bootstrap + manifest)", &store, &mut prev);

    let mut opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 16_384,
        segment_max_bytes: 2 * 1024 * 1024,
        last_write_wins_in_batch: true,
    };

    // ---------------- Phase 1: fixed-width cols 100 & 101 -------------------
    {
        let put_100 = build_put_fixed(100, 0, 10_000, 8);
        let put_101 = build_put_fixed(101, 0, 10_000, 8);
        let puts = vec![put_100, put_101];
        let summary = summarize_puts(&puts);
        store.append_many(puts, opts.clone());
        show_phase_with_data(
            "Phase 1: append fixed-width cols 100 & 101",
            &store,
            &mut prev,
            &summary,
        );
    }

    // ------------- Phase 2: variable-width cols 200 & 201 (chunked) ---------
    {
        let put_200 = build_put_var(200, 0, 12_345, 6, 18);
        let put_201 = build_put_var(201, 0, 12_345, 6, 18);
        let puts = vec![put_200, put_201];
        let summary = summarize_puts(&puts);
        store.append_many(puts, opts.clone());
        show_phase_with_data(
            "Phase 2: append variable-width cols 200 & 201",
            &store,
            &mut prev,
            &summary,
        );
    }

    // ------- Phase 3: add new columns later (forces Manifest update) --------
    {
        let put_300 = build_put_fixed(300, 0, 2_000, 8);
        let put_301 = build_put_var(301, 0, 2_000, 10, 30);
        let puts = vec![put_300, put_301];
        let summary = summarize_puts(&puts);
        store.append_many(puts, opts.clone());
        show_phase_with_data(
            "Phase 3: append new cols 300 (fixed) & 301 (var) — updates Manifest",
            &store,
            &mut prev,
            &summary,
        );
    }

    // ---- Phase 4: mixed append: existing (100) + brand-new (999) -----------
    {
        // Existing fixed-width col 100: write 3 specific keys (5,7,9).
        let put_100 = Put {
            field_id: 100,
            items: vec![
                (be_key(5), fixed_value(5, 100, 8)),
                (be_key(7), fixed_value(7, 100, 8)),
                (be_key(9), fixed_value(9, 100, 8)),
            ],
        };
        // Brand-new variable-width col 999: write 10 keys [1000..1010).
        let put_999 = build_put_var(999, 1_000, 1_010, 12, 40);

        opts.mode = ValueMode::Auto; // allow mixed fixed/var in one append_many
        let puts = vec![put_100, put_999];
        let summary = summarize_puts(&puts);
        store.append_many(puts, opts.clone());
        show_phase_with_data(
            "Phase 4: mixed append (existing col 100 + new col 999)",
            &store,
            &mut prev,
            &summary,
        );
    }

    // ---------------- Phase 5: multi-column read + describe ------------------
    {
        let queries = vec![
            (100, vec![be_key(5), be_key(7), be_key(9), be_key(255)]), // col 100 has 10k rows → all HIT 8B
            (200, vec![be_key(1), be_key(2), be_key(3)]),              // var
            (201, vec![be_key(123), be_key(456), be_key(9_999)]),      // var
            (999, vec![be_key(1_003), be_key(1_005), be_key(1_007)]), // var; all HIT (we wrote [1000..1010))
        ];
        let results = store.get_many(queries.iter().map(|(fid, ks)| (*fid, ks.clone())).collect());
        print_read_report("Read report (get_many)", &queries, &results);

        let ascii = store.render_storage_ascii(); // triggers raw gets to size nodes
        println!("\n==== STORAGE ASCII ====\n{}", ascii);

        show_phase("Phase 5: describe_storage + read report", &store, &mut prev);
    }
}
