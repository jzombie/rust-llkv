//! # Benchmark: Query throughput on 1M rows × 6 columns (uniform keys)
//!
//! This benchmark:
//! - Builds a table with **1,000,000 rows** and **6 columns** (4 fixed-width, 2 variable).
//! - Ingests in manageable chunks (to avoid huge peak memory).
//! - Measures `ColumnStore::get_many` with **uniform random key lookups** using `rand`.
//!
//! What you can tweak easily:
//! - `N_ROWS`, `QUERY_SIZES`, and the column set in `COLS`.
//! - `ingest_chunk_rows`, segment limits in `AppendOptions`.
//!
//! No “hot set”, no custom RNG — just straightforward uniform queries.

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use llkv_column_map::{
    ColumnStore,
    column_store::write::{AppendOptions, Put, ValueMode},
    storage::pager::MemPager,
    types::LogicalFieldId,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::cmp::min;
use std::hint::black_box;

// --------------------------- config ---------------------------

const N_ROWS: u64 = 1_000_000;
const QUERY_SIZES: &[usize] = &[1_000, 10_000, 100_000]; // per-iteration keys
// 6 columns: 2×8B fixed, 2×4B fixed, 2×variable (5..25)
const COLS: &[LogicalFieldId] = &[10, 11, 20, 21, 30, 31];

// ---------------------- value generators ----------------------

#[inline]
fn be_key_u64(v: u64) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

#[inline]
fn fixed_value(width: usize, row: u64, fid: LogicalFieldId) -> Vec<u8> {
    // deterministic but cheap content
    let seed = (row ^ (fid as u64)).to_le_bytes();
    if width <= 8 {
        seed[..width].to_vec()
    } else {
        let mut out = Vec::with_capacity(width);
        while out.len() < width {
            let take = std::cmp::min(8, width - out.len());
            out.extend_from_slice(&seed[..take]);
        }
        out
    }
}

#[inline]
fn var_value(row: u64, fid: LogicalFieldId, min_len: usize, max_len: usize) -> Vec<u8> {
    let span = (max_len - min_len + 1) as u64;
    let len = min_len as u64 + ((row.wrapping_mul(1103515245) ^ fid as u64) % span);
    let byte = (((row as u32).wrapping_add(fid)) & 0xFF) as u8;
    vec![byte; len as usize]
}

// ------------------------- ingest -----------------------------

fn ingest_1m_rows(store: &ColumnStore<'static, MemPager>) {
    // Reasonable segment knobs; adjust to taste.
    let opts = AppendOptions {
        mode: ValueMode::Auto,       // per-column auto fixed/variable
        segment_max_entries: 65_536, // roll segments by entry count
        segment_max_bytes: 8 * 1024 * 1024,
        last_write_wins_in_batch: true,
    };

    // Ingest in chunks so we don't build a gigantic Vec per column.
    let ingest_chunk_rows: u64 = 100_000;

    let mut start = 0u64;
    while start < N_ROWS {
        let end = min(N_ROWS, start + ingest_chunk_rows);

        // Build puts for our 6 columns over [start, end)
        let mut puts: Vec<Put> = Vec::with_capacity(COLS.len());

        // 10,11: fixed 8B
        for &fid in &[COLS[0], COLS[1]] {
            let mut items = Vec::with_capacity((end - start) as usize);
            for r in start..end {
                items.push((be_key_u64(r), fixed_value(8, r, fid)));
            }
            puts.push(Put {
                field_id: fid,
                items,
            });
        }

        // 20,21: fixed 4B
        for &fid in &[COLS[2], COLS[3]] {
            let mut items = Vec::with_capacity((end - start) as usize);
            for r in start..end {
                items.push((be_key_u64(r), fixed_value(4, r, fid)));
            }
            puts.push(Put {
                field_id: fid,
                items,
            });
        }

        // 30,31: variable 5..=25
        for &fid in &[COLS[4], COLS[5]] {
            let mut items = Vec::with_capacity((end - start) as usize);
            for r in start..end {
                items.push((be_key_u64(r), var_value(r, fid, 5, 25)));
            }
            puts.push(Put {
                field_id: fid,
                items,
            });
        }

        store.append_many(puts, opts.clone());
        start = end;
    }

    // one pass to make sure everything is resident/consistent prior to measuring
    black_box(store.describe_storage());
}

// ------------------------- bench ------------------------------

fn bench_query_uniform(c: &mut Criterion) {
    // Leak the pager to give the store a 'static lifetime (so we can move it into the closure).
    let pager: &'static MemPager = Box::leak(Box::new(MemPager::default()));
    let store: ColumnStore<'static, MemPager> = ColumnStore::init_empty(pager);

    // Ingest once (not timed).
    ingest_1m_rows(&store);

    let mut group = c.benchmark_group("read_uniform_1M*6");

    for &q in QUERY_SIZES {
        group.throughput(Throughput::Elements((q * COLS.len()) as u64));
        group.bench_function(format!("get_many_{q}_keys"), |b| {
            // one RNG for the whole benchmark run; deterministic but advances each iter
            let mut rng = StdRng::seed_from_u64(0xC0FFEE);

            b.iter_batched(
                // setup: build the random key set and per-column items
                || {
                    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(q);
                    for _ in 0..q {
                        let row = rng.random_range(0..N_ROWS);
                        keys.push(be_key_u64(row));
                    }
                    // Query all columns with the same key set
                    COLS.iter()
                        .copied()
                        .map(|fid| (fid, keys.clone()))
                        .collect::<Vec<(LogicalFieldId, Vec<Vec<u8>>)>>()
                },
                // measurement: only time get_many
                |items| {
                    let res = store.get_many(items);
                    std::hint::black_box(res);
                },
                BatchSize::SmallInput, // setup is relatively small vs the query
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_query_uniform);
criterion_main!(benches);
