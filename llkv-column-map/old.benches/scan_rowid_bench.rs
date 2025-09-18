//! Scan performance with/without row ids, sorted and unsorted, over 1M u64s.
//!
//! Builds a single 1M-row column where values are 0..N-1 shuffled at ingest,
//! then evaluates:
//!   - scan_unsorted_values
//!   - scan_unsorted_with_row_ids
//!   - scan_sorted_values
//!   - scan_sorted_with_row_ids
//!
//! Run:
//!   cargo bench --bench scan_rowid_bench

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::iter::{PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor};
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const N_ROWS: usize = 1_000_000;
const SEED: u64 = 0xCBF2_A1B1_D3E4_F905;

/// Standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

/// Schema: row_id (u64, non-null) + data (u64 with field_id tag).
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

fn bench_scans(c: &mut Criterion) {
    let mut g = c.benchmark_group("scan_variants_u64_1m");
    g.sample_size(10);
    g.throughput(Throughput::Elements(N_ROWS as u64));

    // Unsorted scan of values only.
    g.bench_function("scan_unsorted_values", |b| {
        b.iter_batched(
            || seed_store_1m(),
            |(store, fid)| {
                let mut acc: u128 = 0;
                for arr_res in store.scan(fid).unwrap() {
                    let arr = arr_res.unwrap();
                    let u = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
                    // Simple sum so compiler can't elide work.
                    for i in 0..u.len() {
                        acc += u.value(i) as u128;
                    }
                }
                black_box(acc);
            },
            BatchSize::SmallInput,
        );
    });

    // Unsorted scan returning values + row_ids.
    g.bench_function("scan_unsorted_with_row_ids", |b| {
        b.iter_batched(
            || seed_store_1m(),
            |(store, fid)| {
                use llkv_column_map::store::iter::PrimitiveWithRowIdsVisitor;
                let rid_fid = fid.with_namespace(Namespace::RowIdShadow);
                struct Acc(u128, u128);
                impl PrimitiveWithRowIdsVisitor for Acc {
                    fn u64_chunk(&mut self, vals: &UInt64Array, rids: &UInt64Array) {
                        for i in 0..vals.len() { self.0 += vals.value(i) as u128; self.1 += rids.value(i) as u128; }
                    }
                    fn i32_chunk(&mut self, vals: &arrow::array::Int32Array, rids: &UInt64Array) {
                        for i in 0..vals.len() { self.0 += vals.value(i) as i128 as u128; self.1 += rids.value(i) as u128; }
                    }
                }
                let mut acc = Acc(0, 0);
                store.scan_with_row_ids_visit(fid, rid_fid, &mut acc).unwrap();
                black_box(acc.0 ^ acc.1);
            },
            BatchSize::SmallInput,
        );
    });

    // Sorted scan of values only (coalesced runs over k-way merge).
    g.bench_function("scan_sorted_values", |b| {
        b.iter_batched(
            || seed_store_1m(),
            |(store, fid)| {
                use std::cell::Cell;
                let acc = Cell::new(0u128);
                store
                    .scan_sorted_visit_fn(
                        fid,
                        |a: &UInt64Array, s, l| {
                            let e = s + l; let mut sum = acc.get();
                            for i in s..e { sum += a.value(i) as u128; }
                            acc.set(sum);
                        },
                        |a: &arrow::array::Int32Array, s, l| {
                            let e = s + l; let mut sum = acc.get();
                            for i in s..e { sum += a.value(i) as i128 as u128; }
                            acc.set(sum);
                        },
                    )
                    .unwrap();
                black_box(acc.get());
            },
            BatchSize::SmallInput,
        );
    });

    // Sorted scan returning values + row_ids.
    g.bench_function("scan_sorted_with_row_ids", |b| {
        b.iter_batched(
            || seed_store_1m(),
            |(store, fid)| {
                let rid_fid = fid.with_namespace(Namespace::RowIdShadow);
                use std::cell::Cell;
                let acc = Cell::new(0u128);
                let acc_r = Cell::new(0u128);
                store
                    .scan_sorted_with_row_ids_visit_fn(
                        fid,
                        rid_fid,
                        |v: &UInt64Array, r: &UInt64Array, s, l| {
                            let e = s + l; let mut sv = acc.get(); let mut sr = acc_r.get();
                            for i in s..e { sv += v.value(i) as u128; sr += r.value(i) as u128; }
                            acc.set(sv); acc_r.set(sr);
                        },
                        |v: &arrow::array::Int32Array, r: &UInt64Array, s, l| {
                            let e = s + l; let mut sv = acc.get(); let mut sr = acc_r.get();
                            for i in s..e { sv += v.value(i) as i128 as u128; sr += r.value(i) as u128; }
                            acc.set(sv); acc_r.set(sr);
                        },
                    )
                    .unwrap();
                black_box(acc.get() ^ acc_r.get());
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

criterion_group!(benches, bench_scans);
criterion_main!(benches);
