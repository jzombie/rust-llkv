//! Compare baseline scan APIs vs visitor-based APIs for u64/i32 columns.
//!
//! Run:
//!   cargo bench --bench visitor_dispatch_bench

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::{self, ColumnStore};
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const N_ROWS: usize = 1_000_000;
const SEED: u64 = 0xCBF2_A1B1_D3E4_F905;

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let rid = Field::new("row_id", DataType::UInt64, false);
    Arc::new(Schema::new(vec![rid, field]))
}

fn seed_store_1m_u64() -> (ColumnStore<MemPager>, LogicalFieldId) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(777_001);

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    let schema = schema_with_row_id(data_f);

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

// -------------------- Visitors (u64) --------------------

struct SumU64Unsorted(u128);
impl store::iter::PrimitiveVisitor for SumU64Unsorted {
    fn u64_chunk(&mut self, a: &UInt64Array) {
        let mut s = self.0;
        for i in 0..a.len() {
            s += a.value(i) as u128;
        }
        self.0 = s;
    }
}

struct SumU64UnsortedWithRids(u128, u128);
impl store::iter::PrimitiveWithRowIdsVisitor for SumU64UnsortedWithRids {
    fn u64_chunk(&mut self, vals: &UInt64Array, rids: &UInt64Array) {
        let mut sv = self.0;
        let mut sr = self.1;
        for i in 0..vals.len() {
            sv += vals.value(i) as u128;
            sr += rids.value(i) as u128;
        }
        self.0 = sv;
        self.1 = sr;
    }
}

struct SumU64Sorted(u128);
impl store::iter::PrimitiveSortedVisitor for SumU64Sorted {
    fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
        let mut acc = self.0;
        let e = s + l;
        for i in s..e {
            acc += a.value(i) as u128;
        }
        self.0 = acc;
    }
}

struct SumU64SortedWithRids(u128, u128);
impl store::iter::PrimitiveSortedWithRowIdsVisitor for SumU64SortedWithRids {
    fn u64_run_with_rids(&mut self, vals: &UInt64Array, rids: &UInt64Array, s: usize, l: usize) {
        let mut sv = self.0;
        let mut sr = self.1;
        let e = s + l;
        for i in s..e {
            sv += vals.value(i) as u128;
            sr += rids.value(i) as u128;
        }
        self.0 = sv;
        self.1 = sr;
    }
}

fn bench_dispatch_overhead(c: &mut Criterion) {
    let mut g = c.benchmark_group("dispatch_overhead_u64_1m");
    g.sample_size(10);
    g.throughput(Throughput::Elements(N_ROWS as u64));

    // Baseline unsorted sum (scan + downcast)
    g.bench_function("baseline_unsorted_sum", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let mut acc: u128 = 0;
                for arr_res in store.scan(fid).unwrap() {
                    let arr = arr_res.unwrap();
                    let u = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
                    for i in 0..u.len() { acc += u.value(i) as u128; }
                }
                black_box(acc);
            },
            BatchSize::SmallInput,
        )
    });

    // Visitor unsorted sum
    g.bench_function("visitor_unsorted_sum", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let mut v = SumU64Unsorted(0);
                store.scan_visit(fid, &mut v).unwrap();
                black_box(v.0);
            },
            BatchSize::SmallInput,
        )
    });

    // Baseline unsorted (values + row ids)
    g.bench_function("baseline_unsorted_vals_rids", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let rid_fid = fid.with_namespace(Namespace::RowIdShadow);
                let mut sv: u128 = 0; let mut sr: u128 = 0;
                for res in store.scan_with_row_ids(fid, rid_fid).unwrap() {
                    let (vals_any, rids) = res.unwrap();
                    let vals = vals_any.as_any().downcast_ref::<UInt64Array>().unwrap();
                    for i in 0..vals.len() { sv += vals.value(i) as u128; sr += rids.value(i) as u128; }
                }
                black_box(sv ^ sr);
            },
            BatchSize::SmallInput,
        )
    });

    // Visitor unsorted (values + row ids)
    g.bench_function("visitor_unsorted_vals_rids", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let rid_fid = fid.with_namespace(Namespace::RowIdShadow);
                let mut v = SumU64UnsortedWithRids(0, 0);
                store.scan_with_row_ids_visit(fid, rid_fid, &mut v).unwrap();
                black_box(v.0 ^ v.1);
            },
            BatchSize::SmallInput,
        )
    });

    // Baseline sorted sum
    g.bench_function("baseline_sorted_sum", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let mut acc: u128 = 0;
                let mut m = store.scan_sorted(fid).unwrap();
                while let Some((arr_dyn, start, len)) = m.next_run() {
                    let arr = arr_dyn.as_any().downcast_ref::<UInt64Array>().unwrap();
                    let e = start + len; for i in start..e { acc += arr.value(i) as u128; }
                }
                black_box(acc);
            },
            BatchSize::SmallInput,
        )
    });

    // Visitor sorted sum
    g.bench_function("visitor_sorted_sum", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let mut v = SumU64Sorted(0);
                store.scan_sorted_visit(fid, &mut v).unwrap();
                black_box(v.0);
            },
            BatchSize::SmallInput,
        )
    });

    // Baseline sorted (values + row ids)
    g.bench_function("baseline_sorted_vals_rids", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let rid_fid = fid.with_namespace(Namespace::RowIdShadow);
                let mut sv: u128 = 0; let mut sr: u128 = 0;
                let mut m = store.scan_sorted_with_row_ids(fid, rid_fid).unwrap();
                while let Some((vals_dyn, rids, start, len)) = m.next_run() {
                    let vals = vals_dyn.as_any().downcast_ref::<UInt64Array>().unwrap();
                    let e = start + len; for i in start..e { sv += vals.value(i) as u128; sr += rids.value(i) as u128; }
                }
                black_box(sv ^ sr);
            },
            BatchSize::SmallInput,
        )
    });

    // Visitor sorted (values + row ids)
    g.bench_function("visitor_sorted_vals_rids", |b| {
        b.iter_batched(
            || seed_store_1m_u64(),
            |(store, fid)| {
                let rid_fid = fid.with_namespace(Namespace::RowIdShadow);
                let mut v = SumU64SortedWithRids(0, 0);
                store
                    .scan_sorted_with_row_ids_visit(fid, rid_fid, &mut v)
                    .unwrap();
                black_box(v.0 ^ v.1);
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

criterion_group!(benches, bench_dispatch_overhead);
criterion_main!(benches);
