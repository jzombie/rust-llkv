//! Random point-lookups on a 1M-row UInt64 column using the store API.
//!
//! Benchmarks:
//!   - scan_unsorted_multiset: unsorted probe (dense counts)
//!   - scan_sorted_stream_join: two-pointer over scan_sorted()
//!
//! Build once per-iter: append one shuffled 1M-row chunk, then create the
//! sort index. Timed regions reflect how the calls are actually used.
//!
//! Run:
//!   cargo bench --bench random_probe_bench
//!
//! Dev-deps (Cargo.toml):
//!   [dev-dependencies]
//!   criterion = "0.5"
//!   rand = "0.9"

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const N_ROWS: usize = 1_000_000;
const N_QUERIES: usize = 10_000;
const SEED: u64 = 0xCBF2_A1B1_D3E4_F905;

/// Standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

/// Schema: row_id (u64, non-null) + data (u64 with field_id).
fn schema_with_row_id(field_id: LogicalFieldId) -> Arc<Schema> {
    let rid = Field::new("row_id", DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data_f]))
}

fn seed_store_1m() -> (ColumnStore<MemPager>, LogicalFieldId) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let field_id = fid(42);
    let schema = schema_with_row_id(field_id);

    // row_id 0..N-1; values 0..N-1 shuffled at ingest
    let rid: Vec<u64> = (0..N_ROWS as u64).collect();
    let mut vals: Vec<u64> = (0..N_ROWS as u64).collect();
    let mut rng2 = StdRng::seed_from_u64(SEED ^ 0x5a5a_5a5a_5a5a_5a5a);
    vals.as_mut_slice().shuffle(&mut rng2);

    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();

    store.append(&batch).unwrap();
    store.create_sort_index(field_id).unwrap();

    (store, field_id)
}

fn make_queries() -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let bound = (N_ROWS as u64) * 2;
    let mut q = Vec::with_capacity(N_QUERIES);
    for _ in 0..N_QUERIES {
        q.push(rng.random_range(0..bound));
    }
    q
}

/// Unsorted probe: multiset membership via dense counts (small domain).
/// Domain is 0..2*N_ROWS in this bench.
fn count_hits_multiset_scan(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    qs: &[u64],
) -> usize {
    let mut freq = vec![0u32; 2 * N_ROWS];
    for &q in qs {
        let idx = q as usize;
        if idx < freq.len() { freq[idx] = freq[idx].saturating_add(1); }
    }

    use std::cell::Cell;
    let hits = Cell::new(0usize);
    store
        .scan_visit_fn(
            fid,
            |u: &UInt64Array| {
                let mut h = hits.get();
                for i in 0..u.len() {
                    let v = u.value(i) as usize;
                    if v < freq.len() {
                        let k = freq[v];
                        if k != 0 { h += k as usize; freq[v] = 0; }
                    }
                }
                hits.set(h);
            },
            |_i: &arrow::array::Int32Array| {},
        )
        .unwrap();
    hits.get()
}

/// Sorted probe: two-pointer merge over scan_sorted().
fn count_hits_stream_join(store: &ColumnStore<MemPager>, fid: LogicalFieldId, qs: &[u64]) -> usize {
    let mut queries = qs.to_vec();
    queries.sort_unstable();

    let mut qi = 0usize;
    let mut hits = 0usize;

    // Sorted scan via visitor closures (no type knowledge at callsite).
    store
        .scan_sorted_visit_fn(
            fid,
            |arr: &UInt64Array, start, len| {
                let end = start + len;
                for i in start..end {
                    let v = arr.value(i);
                    while qi < queries.len() && queries[qi] < v { qi += 1; }
                    while qi < queries.len() && queries[qi] == v { hits += 1; qi += 1; }
                    if qi >= queries.len() { break; }
                }
            },
            |_arr_i32: &arrow::array::Int32Array, _start, _len| {
                // This bench expects a UInt64 column; i32 not used here.
            },
        )
        .unwrap();

    hits
}

fn bench_random_probe(c: &mut Criterion) {
    let queries = make_queries();
    let expected = queries.iter().filter(|&&x| x < N_ROWS as u64).count();

    let mut g = c.benchmark_group("random_probe_u64");
    g.sample_size(10);
    g.throughput(Throughput::Elements(N_QUERIES as u64));

    g.bench_function("scan_unsorted_multiset", |b| {
        b.iter_batched(
            || seed_store_1m(),
            |(store, fid)| {
                let hits = count_hits_multiset_scan(&store, fid, &queries);
                assert_eq!(hits, expected);
                black_box(hits);
            },
            BatchSize::SmallInput,
        );
    });

    g.bench_function("scan_sorted_stream_join", |b| {
        b.iter_batched(
            || seed_store_1m(),
            |(store, fid)| {
                let hits = count_hits_stream_join(&store, fid, &queries);
                assert_eq!(hits, expected);
                black_box(hits);
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

criterion_group!(benches, bench_random_probe);
criterion_main!(benches);
