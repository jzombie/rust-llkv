//! Bench: Compare sorted scans (with prebuilt sort index) vs. unsorted scans
//! (collect + in-memory sort) on 1M integer rows, across fragmentation and
//! selectivity scenarios.
//!
//! What this file does
//! - Seeds an in-memory ColumnStore (MemPager) with 1M rows of `row_id` (u64)
//!   and a single integer data column.
//! - Uses two seeding shapes:
//!   - "lowfrag": a single large RecordBatch append (the store still slices
//!     inputs into ~1 MiB internal chunks).
//!   - "highfrag": the same 1M values appended as many smaller batches to
//!     simulate fragmentation (128 appends by default).
//! - Benchmarks a 2×2×2 matrix:
//!   - Fragmentation: lowfrag vs highfrag
//!   - Selectivity: full scan vs a 5% range window
//!   - Index: with prebuilt sort index vs no index (collect + Vec::sort)
//!
//! Notes on what you are measuring
//! - "With index" uses the store’s sorted scan path which currently merges
//!   per-chunk sorted permutations (k-way merge). To satisfy the visitor API,
//!   it materializes each emitted run with `compute::take` before calling the
//!   visitor. This is intentionally representative of today’s API; it includes
//!   merge + per-run materialization cost, not just “sum a slice”.
//! - "Without index" scans unsorted chunks, collects values into a Vec and
//!   sorts once in-memory, then black-boxes the result length to keep the work
//!   alive. This isolates a fast path for one-off scans without an index.
//! - Range variants (5% slice) show the benefit of pre-windowing by permutation
//!   (two binary searches per chunk) and merging only the needed windows.
//!
//! Nothing from these benches is persisted to disk; all storage uses MemPager.
//! The sorted scan path’s `compute::take` materializes run slices transiently
//! for the visitor — they are not written back to the store.
//!
//! Run:
//!   cargo bench --bench sort_index_bench

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

use llkv_column_map::store::scan::ScanOptions;
use llkv_column_map::store::{ColumnStore, IndexKind, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

const N_ROWS: usize = 1_000_000;
// 64-bit seed (previous literal overflowed u64)
const SEED: u64 = 0xC0FF_EE00_DEAD_BEEF;

fn fid_user(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn seed_store_1m() -> (ColumnStore<MemPager>, LogicalFieldId, LogicalFieldId) {
    // One-time ingest: a single RecordBatch append. The store will slice
    // inputs into ~1 MiB chunks internally during append.
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Column 1: u64
    let fid_u64 = fid_user(1);
    let mut md1 = HashMap::new();
    md1.insert("field_id".to_string(), u64::from(fid_u64).to_string());
    let schema1 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md1),
    ]));
    let rid: Vec<u64> = (0..N_ROWS as u64).collect();
    let mut vals_u64: Vec<u64> = (0..N_ROWS as u64).collect();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xA55A_A55A_A55A_A55A);
    vals_u64.as_mut_slice().shuffle(&mut rng);
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr_u64 = Arc::new(UInt64Array::from(vals_u64));
    let batch1 = RecordBatch::try_new(schema1, vec![rid_arr, val_arr_u64]).unwrap();
    store.append(&batch1).unwrap();

    // Column 2: i32 (not used in this bench, but mirrors workload)
    let fid_i32 = fid_user(2);
    let mut md2 = HashMap::new();
    md2.insert("field_id".to_string(), u64::from(fid_i32).to_string());
    let schema2 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::Int32, false).with_metadata(md2),
    ]));
    let rid2: Vec<u64> = (0..N_ROWS as u64).collect();
    let mut vals_i32: Vec<i32> = (0..N_ROWS as i32).collect();
    let mut rng2 = StdRng::seed_from_u64(SEED ^ 0x5A5A_5A5A_5A5A_5A5A);
    vals_i32.as_mut_slice().shuffle(&mut rng2);
    let rid2_arr = Arc::new(UInt64Array::from(rid2));
    let val_arr_i32 = Arc::new(Int32Array::from(vals_i32));
    let batch2 = RecordBatch::try_new(schema2, vec![rid2_arr, val_arr_i32]).unwrap();
    store.append(&batch2).unwrap();

    (store, fid_u64, fid_i32)
}

// ---------------- Helpers -----------------

fn seed_store_1m_fragmented_random(chunks: usize) -> (ColumnStore<MemPager>, LogicalFieldId) {
    // Multi-append ingest: write the 1M values across `chunks` batches to
    // introduce fragmentation in the descriptor chain and chunk layout.
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let fid_u64 = fid_user(101);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(fid_u64).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));

    // Global random permutation, then append in many slices (fragmentation).
    let rid: Vec<u64> = (0..N_ROWS as u64).collect();
    let mut vals: Vec<u64> = (0..N_ROWS as u64).collect();
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xDEAD_BEEF_DEAD_BEEF);
    vals.as_mut_slice().shuffle(&mut rng);

    let chunk_rows = N_ROWS.div_ceil(chunks);
    let mut off = 0usize;
    while off < N_ROWS {
        let take = (N_ROWS - off).min(chunk_rows);
        let rid_arr = Arc::new(UInt64Array::from(rid[off..off + take].to_vec()));
        let val_arr = Arc::new(UInt64Array::from(vals[off..off + take].to_vec()));
        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
        store.append(&batch).unwrap();
        off += take;
    }

    (store, fid_u64)
}

// ---------------- Organized 2x2x2 matrix -----------------

fn bench_index_matrix_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_matrix_1M");
    group.sample_size(10);
    // Range: ~5% of domain
    const LO: u64 = 200_000;
    const HI: u64 = 250_000;

    // Visitors
    use llkv_column_map::store::scan::{
        PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
        PrimitiveWithRowIdsVisitor,
    };
    struct SortedSum {
        s: u128,
    }
    impl PrimitiveSortedVisitor for SortedSum {
        fn u64_run(&mut self, a: &UInt64Array, s0: usize, l: usize) {
            let e = s0 + l;
            for i in s0..e {
                self.s += a.value(i) as u128;
            }
        }
    }
    impl PrimitiveVisitor for SortedSum {}
    impl PrimitiveWithRowIdsVisitor for SortedSum {}
    impl PrimitiveSortedWithRowIdsVisitor for SortedSum {}
    struct Collect {
        v: Vec<u64>,
    }
    impl PrimitiveVisitor for Collect {
        fn u64_chunk(&mut self, a: &UInt64Array) {
            self.v.extend((0..a.len()).map(|i| a.value(i)));
        }
    }
    impl PrimitiveSortedVisitor for Collect {}
    impl PrimitiveWithRowIdsVisitor for Collect {}
    impl PrimitiveSortedWithRowIdsVisitor for Collect {}

    // Low fragmentation (single append), Full scan
    group.bench_function("lowfrag/full/with_index", |b| {
        b.iter_batched(
            || {
                let (s, fid, _) = seed_store_1m();
                s.register_index(fid, IndexKind::Sort).unwrap();
                (s, fid)
            },
            |(store, fid)| {
                let mut v = SortedSum { s: 0 };
                store
                    .scan(
                        fid,
                        ScanOptions {
                            sorted: true,
                            reverse: false,
                            with_row_ids: false,

                            limit: None,
                            offset: 0,
                            include_nulls: false,
                            nulls_first: false,
                            anchor_row_id_field: None,
                        },
                        &mut v,
                    )
                    .unwrap();
                black_box(v.s);
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("lowfrag/full/without_index", |b| {
        b.iter_batched(
            seed_store_1m,
            |(store, fid, _)| {
                let mut c = Collect {
                    v: Vec::with_capacity(N_ROWS),
                };
                store
                    .scan(
                        fid,
                        ScanOptions {
                            sorted: false,
                            reverse: false,
                            with_row_ids: false,

                            limit: None,
                            offset: 0,
                            include_nulls: false,
                            nulls_first: false,
                            anchor_row_id_field: None,
                        },
                        &mut c,
                    )
                    .unwrap();
                c.v.sort_unstable();
                black_box(c.v.len());
            },
            BatchSize::SmallInput,
        )
    });

    // Low fragmentation, Range scan
    group.bench_function("lowfrag/range/with_index", |b| {
        use llkv_column_map::store::scan::ScanBuilder;
        b.iter_batched(
            || {
                let (s, fid, _) = seed_store_1m();
                s.register_index(fid, IndexKind::Sort).unwrap();
                (s, fid)
            },
            |(store, fid)| {
                let mut v = SortedSum { s: 0 };
                ScanBuilder::new(&store, fid)
                    .sorted(true)
                    .with_range::<u64, _>(LO..=HI)
                    .run(&mut v)
                    .unwrap();
                black_box(v.s);
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("lowfrag/range/without_index", |b| {
        b.iter_batched(
            seed_store_1m,
            |(store, fid, _)| {
                let mut c = Collect {
                    v: Vec::with_capacity(60_000),
                };
                store
                    .scan(
                        fid,
                        ScanOptions {
                            sorted: false,
                            reverse: false,
                            with_row_ids: false,

                            limit: None,
                            offset: 0,
                            include_nulls: false,
                            nulls_first: false,
                            anchor_row_id_field: None,
                        },
                        &mut c,
                    )
                    .unwrap();
                c.v.retain(|&x| (LO..=HI).contains(&x));
                c.v.sort_unstable();
                black_box(c.v.len());
            },
            BatchSize::SmallInput,
        )
    });

    // High fragmentation, Full scan
    group.bench_function("highfrag/full/with_index", |b| {
        b.iter_batched(
            || {
                let (s, fid) = seed_store_1m_fragmented_random(128);
                s.register_index(fid, IndexKind::Sort).unwrap();
                (s, fid)
            },
            |(store, fid)| {
                let mut v = SortedSum { s: 0 };
                store
                    .scan(
                        fid,
                        ScanOptions {
                            sorted: true,
                            reverse: false,
                            with_row_ids: false,

                            limit: None,
                            offset: 0,
                            include_nulls: false,
                            nulls_first: false,
                            anchor_row_id_field: None,
                        },
                        &mut v,
                    )
                    .unwrap();
                black_box(v.s);
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("highfrag/full/without_index", |b| {
        b.iter_batched(
            || seed_store_1m_fragmented_random(128),
            |(store, fid)| {
                let mut c = Collect {
                    v: Vec::with_capacity(N_ROWS),
                };
                store
                    .scan(
                        fid,
                        ScanOptions {
                            sorted: false,
                            reverse: false,
                            with_row_ids: false,

                            limit: None,
                            offset: 0,
                            include_nulls: false,
                            nulls_first: false,
                            anchor_row_id_field: None,
                        },
                        &mut c,
                    )
                    .unwrap();
                c.v.sort_unstable();
                black_box(c.v.len());
            },
            BatchSize::SmallInput,
        )
    });

    // High fragmentation, Range scan
    group.bench_function("highfrag/range/with_index", |b| {
        use llkv_column_map::store::scan::ScanBuilder;
        b.iter_batched(
            || {
                let (s, fid) = seed_store_1m_fragmented_random(128);
                s.register_index(fid, IndexKind::Sort).unwrap();
                (s, fid)
            },
            |(store, fid)| {
                let mut v = SortedSum { s: 0 };
                ScanBuilder::new(&store, fid)
                    .sorted(true)
                    .with_range::<u64, _>(LO..=HI)
                    .run(&mut v)
                    .unwrap();
                black_box(v.s);
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("highfrag/range/without_index", |b| {
        b.iter_batched(
            || seed_store_1m_fragmented_random(128),
            |(store, fid)| {
                let mut c = Collect {
                    v: Vec::with_capacity(60_000),
                };
                store
                    .scan(
                        fid,
                        ScanOptions {
                            sorted: false,
                            reverse: false,
                            with_row_ids: false,

                            limit: None,
                            offset: 0,
                            include_nulls: false,
                            nulls_first: false,
                            anchor_row_id_field: None,
                        },
                        &mut c,
                    )
                    .unwrap();
                c.v.retain(|&x| (LO..=HI).contains(&x));
                c.v.sort_unstable();
                black_box(c.v.len());
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_index_matrix_1m);
criterion_main!(benches);
