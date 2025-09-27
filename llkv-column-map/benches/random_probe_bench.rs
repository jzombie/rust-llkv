//! Random point-lookups on a 1M-row UInt64 column using the new visitor API.
//!
//! Benchmarks:
//!   - scan_unsorted_multiset: unsorted probe (dense multiset counts)
//!   - scan_sorted_stream_join: two-pointer over sorted scan (requires index)
//!
//! Setup per-iteration: seed one shuffled 1M-row column and build the sort index.
//! Storage uses MemPager; nothing is persisted to disk.
//!
//! Run:
//!   cargo bench --bench random_probe_bench

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind};
use llkv_column_map::types::LogicalFieldId;
use llkv_storage::pager::MemPager;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const N_ROWS: usize = 1_000_000;
const N_QUERIES: usize = 10_000;
const SEED: u64 = 0xCBF2_A1B1_D3E4_F905;

fn schema_with_row_id(field_id: LogicalFieldId) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data_f]))
}

fn seed_store_1m() -> (ColumnStore<MemPager>, LogicalFieldId) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let field_id = LogicalFieldId::for_default_user(42);
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
    store.register_index(field_id, IndexKind::Sort).unwrap();

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
fn count_hits_multiset_scan(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    qs: &[u64],
) -> usize {
    let mut freq = vec![0u32; 2 * N_ROWS];
    for &q in qs {
        let idx = q as usize;
        if idx < freq.len() {
            freq[idx] = freq[idx].saturating_add(1);
        }
    }

    struct Probe<'a> {
        freq: &'a mut [u32],
        hits: usize,
    }
    impl<'a> PrimitiveVisitor for Probe<'a> {
        fn u64_chunk(&mut self, a: &UInt64Array) {
            let f = &mut self.freq;
            let mut h = self.hits;
            for i in 0..a.len() {
                let v = a.value(i) as usize;
                if v < f.len() {
                    let k = f[v];
                    if k != 0 {
                        h += k as usize;
                        f[v] = 0;
                    }
                }
            }
            self.hits = h;
        }
    }
    impl<'a> PrimitiveSortedVisitor for Probe<'a> {}
    impl<'a> PrimitiveWithRowIdsVisitor for Probe<'a> {}
    impl<'a> PrimitiveSortedWithRowIdsVisitor for Probe<'a> {}

    let mut v = Probe {
        freq: &mut freq,
        hits: 0,
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
            &mut v,
        )
        .unwrap();
    v.hits
}

/// Sorted probe: two-pointer merge over sorted scan.
fn count_hits_stream_join(store: &ColumnStore<MemPager>, fid: LogicalFieldId, qs: &[u64]) -> usize {
    let mut queries = qs.to_vec();
    queries.sort_unstable();
    struct Join<'a> {
        q: &'a [u64],
        qi: usize,
        hits: usize,
    }
    impl<'a> PrimitiveSortedVisitor for Join<'a> {
        fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            let q = self.q;
            let mut qi = self.qi;
            let mut h = self.hits;
            for i in s..e {
                let v = a.value(i);
                while qi < q.len() && q[qi] < v {
                    qi += 1;
                }
                while qi < q.len() && q[qi] == v {
                    h += 1;
                    qi += 1;
                }
                if qi >= q.len() {
                    break;
                }
            }
            self.qi = qi;
            self.hits = h;
        }
    }
    impl<'a> PrimitiveVisitor for Join<'a> {}
    impl<'a> PrimitiveWithRowIdsVisitor for Join<'a> {}
    impl<'a> PrimitiveSortedWithRowIdsVisitor for Join<'a> {}

    let mut v = Join {
        q: &queries,
        qi: 0,
        hits: 0,
    };
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
    v.hits
}

fn bench_random_probe(c: &mut Criterion) {
    let queries = make_queries();
    let expected = queries.iter().filter(|&&x| x < N_ROWS as u64).count();

    let mut g = c.benchmark_group("random_probe_u64");
    g.sample_size(10);
    g.throughput(Throughput::Elements(N_QUERIES as u64));

    g.bench_function("scan_unsorted_multiset", |b| {
        b.iter_batched(
            seed_store_1m,
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
            seed_store_1m,
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
