//! # Benchmark: Ingest 21 Columns × 150,000 Rows (batching strategies)
//!
//! **Purpose**
//! Measure end-to-end ingest throughput of the high-level `ColumnStore::append_many` API
//! while varying how much data we pack into each append call. This highlights the cost of
//! segmentation, index construction, and pager round-trips as a function of batch size.
//!
//! **What it does**
//! - Synthesizes a dataset of **21 columns** per row:
//! - 8 columns with fixed width 8 bytes
//!   - 4 columns with fixed width 4 bytes
//! - 5 columns with variable length (5..=25 bytes)
//!   - 4 columns with variable length (50..=200 bytes)
//! - Generates **150,000 rows** and ingests them through `ColumnStore::append_many`,
//!   using two batching patterns, each as a separate Criterion benchmark group:
//! 1. **`bench_ingest_by_batches`**: sweep the **number of batches** (e.g., 1, 2, 5, 10, …).
//!    More batches ⇒ smaller per-append chunk size.
//! 2. **`bench_ingest_by_rows_per_batch`**: sweep the **rows per batch** directly
//!    (e.g., 150k, 75k, 50k, …).
//!
//! **Why two patterns?**
//! They provide the same control from two angles: *“how many appends?”* vs
//! *“how big is each append?”*. Both reveal how your segment thresholds and pager usage
//! respond to different batching choices.
//!
//! **How it measures**
//! - Each benchmark creates a fresh in-memory `MemPager` and `ColumnStore`.
//! - It runs one or more `append_many` calls until all 150k rows × 21 columns are ingested.
//! - We call `store.describe_storage()` at the end of each iteration to prevent the optimizer
//!   from eliding the work and to force at least one read pass over the storage state.
//! - Criterion reports time per iteration; we also set `Throughput::Elements(total_rows * 21)`
//!   so you can interpret results as *logical cells per second* if you want.
//!
//! **Knobs to tweak**
//! - `segment_max_entries`, `segment_max_bytes`: affect how often segments roll over.
//! - `last_write_wins_in_batch`: whether duplicate keys in the same batch are deduped.
//! - The set and shape of columns in `col_spec_21()`.
//!
//! **How to read the output**
//! - **Fewer, larger batches** generally reduce overhead (fewer pager calls, fewer index objects),
//!   but increase memory pressure and produce larger segments.
//! - **More, smaller batches** give you smaller segments and lower peaks in memory usage,
//!   but higher fixed overhead per batch.
//!
//! **Caveats**
//! - Uses an in-memory pager; absolute timings won’t match a real backend, but **relative trends**
//!   are very telling.
//! - Values are synthetic.
//!
//! If your real payloads compress/branch differently, expect shifts.
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llkv_column_map::{
    ColumnStore,
    codecs::big_endian::u64_be_vec,
    storage::pager::MemPager,
    types::{AppendOptions, LogicalFieldId, Put, ValueMode},
};

use std::borrow::Cow;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
// ----------------- dataset shape: 21 columns x 150_000 rows ---------------
#[derive(Clone, Copy)]
enum ColKind {
    Fixed(u32),
    Var { min: usize, max: usize },
}

fn col_spec_21() -> Vec<(LogicalFieldId, ColKind)> {
    let mut spec = Vec::new();
    for i in 0..8 {
        spec.push((100 + i, ColKind::Fixed(8)));
    } // 8 x fixed(8)
    for i in 0..4 {
        spec.push((200 + i, ColKind::Fixed(4)));
    } // 4 x fixed(4)
    for i in 0..5 {
        spec.push((300 + i, ColKind::Var { min: 5, max: 25 }));
    } // 5 x var short
    for i in 0..4 {
        spec.push((400 + i, ColKind::Var { min: 50, max: 200 }));
    } // 4 x var med
    spec
}

#[inline]
fn row_key(row: u64) -> Vec<u8> {
    u64_be_vec(row)
}

#[inline]
fn var_len_for(row: u64, field: LogicalFieldId, min: usize, max: usize) -> usize {
    let span = (max - min + 1) as u64;
    let mix = row
        .wrapping_mul(1103515245)
        .wrapping_add(field)
        .rotate_left(13);
    (min as u64 + (mix % span)) as usize
}

fn build_puts_for_range<'a>(
    start: u64,
    end: u64,
    spec: &[(LogicalFieldId, ColKind)],
) -> Vec<Put<'a>> {
    let mut puts = Vec::with_capacity(spec.len());
    for (field_id, kind) in spec.iter().copied() {
        let mut items = Vec::with_capacity((end - start) as usize);
        for r in start..end {
            let k = row_key(r);
            let v = match kind {
                // Handle ANY fixed width (4, 8, or whatever you put in ColKind::Fixed)
                ColKind::Fixed(w) => {
                    let w = w as usize;
                    // derive a stable 8-byte seed from row & field, then repeat/truncate to w bytes
                    let seed = (r ^ field_id).to_le_bytes();
                    if w <= 8 {
                        seed[..w].to_vec()
                    } else {
                        let mut buf = Vec::with_capacity(w);
                        while buf.len() < w {
                            let take = std::cmp::min(8, w - buf.len());
                            buf.extend_from_slice(&seed[..take]);
                        }
                        buf
                    }
                }
                ColKind::Var { min, max } => {
                    let len = var_len_for(r, field_id, min, max);
                    let byte = (((r as LogicalFieldId).wrapping_add(field_id)) & 0xFF) as u8;
                    vec![byte; len]
                }
            };
            // Explicitly create Cow::Owned to match the Put struct definition.
            items.push((Cow::Owned(k), Cow::Owned(v)));
        }
        puts.push(Put { field_id, items });
    }
    puts
}

fn bench_ingest_by_batches(c: &mut Criterion) {
    let total_rows: u64 = 150_000;
    let spec = col_spec_21();

    let opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 16_384,
        segment_max_bytes: 2 * 1024 * 1024,
        last_write_wins_in_batch: true,
        value_order: None,
    };
    // num_batches: 1 means "ALL rows in ONE append_many call"
    let num_batches_options = [1usize, 2, 5, 10, 20, 50, 100];
    let mut group = c.benchmark_group("ingest_21x150k_by_num_batches");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(total_rows * 21));
    for &batches in &num_batches_options {
        let rows_per_batch = (total_rows as usize).div_ceil(batches);
        let label = format!(
            "num_batches={} (rows_per_batch≈{})",
            batches, rows_per_batch
        );
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let pager = Arc::new(MemPager::default());
                let store = ColumnStore::open(pager);

                let mut start = 0u64;
                for _ in 0..batches {
                    let end = std::cmp::min(total_rows, start + rows_per_batch as u64);
                    if start >= end {
                        break;
                    }
                    let puts = build_puts_for_range(start, end, &spec);
                    store.append_many(puts, opts.clone());
                    start = end;
                }

                // Prevent optimizer from discarding all work; also touches the storage via batch gets.
                let n_nodes = store.describe_storage().len();
                black_box(n_nodes);
            });
        });
    }
    group.finish();
}

fn bench_ingest_by_rows_per_batch(c: &mut Criterion) {
    let total_rows: u64 = 150_000;
    let spec = col_spec_21();

    let opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 16_384,
        segment_max_bytes: 2 * 1024 * 1024,
        last_write_wins_in_batch: true,
        value_order: None,
    };
    // Explicit rows-per-batch. 150_000 means ALL AT ONCE.
    let rows_per_batch_options = [150_000usize, 75_000, 50_000, 10_000, 2_000, 1_000];
    let mut group = c.benchmark_group("ingest_21x150k_by_rows_per_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(total_rows * 21));
    for &rows_per_batch in &rows_per_batch_options {
        let label = format!("rows_per_batch={}", rows_per_batch);
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let pager = Arc::new(MemPager::default());
                let store = ColumnStore::open(pager);

                let mut start = 0u64;
                while start < total_rows {
                    let end = std::cmp::min(total_rows, start + rows_per_batch as u64);
                    let puts = build_puts_for_range(start, end, &spec);
                    store.append_many(puts, opts.clone());
                    start = end;
                }

                // Prevent optimizer from discarding all work; also touches the storage via batch gets.
                let n_nodes = store.describe_storage().len();
                black_box(n_nodes);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ingest_by_batches,
    bench_ingest_by_rows_per_batch
);
criterion_main!(benches);
