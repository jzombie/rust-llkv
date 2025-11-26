//! Benchmarks for the ScanBuilder API over multiple 1M-row columns.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, UInt64Array};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use llkv_column_map::ROW_ID_COLUMN_NAME;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind};
use llkv_types::LogicalFieldId;
use llkv_storage::pager::MemPager;

const N_ROWS: usize = 1_000_000;
const SEED: u64 = 0xC0FF_EEF0_0DD1_5EA5;

fn seed_store_1m() -> (ColumnStore<MemPager>, LogicalFieldId, LogicalFieldId) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Column 1: u64
    let fid_u64 = LogicalFieldId::for_user_table_0(1);
    let mut md1 = HashMap::new();
    md1.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(fid_u64).to_string(),
    );
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

    // Column 2: i32
    let fid_i32 = LogicalFieldId::for_user_table_0(2);
    let mut md2 = HashMap::new();
    md2.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(fid_i32).to_string(),
    );
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

    store.register_index(fid_u64, IndexKind::Sort).unwrap();
    store.register_index(fid_i32, IndexKind::Sort).unwrap();
    (store, fid_u64, fid_i32)
}

fn bench_scan_builder(c: &mut Criterion) {
    let mut g = c.benchmark_group("scan_builder_1M");
    g.sample_size(12);
    g.throughput(Throughput::Elements(N_ROWS as u64));

    // Unsorted sum u64 via builder
    g.bench_function("unsorted_sum_u64", |b| {
        b.iter_batched(
            seed_store_1m,
            |(store, fid_u64, _)| {
                struct SumU64<'a> {
                    acc: &'a std::cell::Cell<u128>,
                }
                impl<'a> PrimitiveVisitor for SumU64<'a> {
                    fn u64_chunk(&mut self, a: &UInt64Array) {
                        if let Some(s) = compute::sum(a) {
                            self.acc.set(self.acc.get() + s as u128);
                        }
                    }
                }
                impl<'a> PrimitiveSortedVisitor for SumU64<'a> {}
                impl<'a> PrimitiveWithRowIdsVisitor for SumU64<'a> {}
                impl<'a> PrimitiveSortedWithRowIdsVisitor for SumU64<'a> {}
                let acc = std::cell::Cell::new(0u128);
                let mut v = SumU64 { acc: &acc };
                ScanBuilder::new(&store, fid_u64)
                    .options(ScanOptions {
                        sorted: false,
                        reverse: false,
                        with_row_ids: false,

                        limit: None,
                        offset: 0,
                        include_nulls: false,
                        nulls_first: false,
                        anchor_row_id_field: None,
                    })
                    .run(&mut v)
                    .unwrap();
                black_box(acc.get());
            },
            BatchSize::SmallInput,
        );
    });

    // Sorted range sum u64 via builder
    g.bench_function("sorted_range_u64", |b| {
        b.iter_batched(
            seed_store_1m,
            |(store, fid_u64, _)| {
                struct SumU64<'a> {
                    acc: &'a std::cell::Cell<u128>,
                }
                impl<'a> PrimitiveSortedVisitor for SumU64<'a> {
                    fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
                        let sl = a.slice(s, l);
                        let arr = sl.as_any().downcast_ref::<UInt64Array>().unwrap();
                        if let Some(sv) = compute::sum(arr) {
                            self.acc.set(self.acc.get() + sv as u128);
                        }
                    }
                }
                impl<'a> PrimitiveVisitor for SumU64<'a> {}
                impl<'a> PrimitiveWithRowIdsVisitor for SumU64<'a> {}
                impl<'a> PrimitiveSortedWithRowIdsVisitor for SumU64<'a> {}
                let acc = std::cell::Cell::new(0u128);
                let mut v = SumU64 { acc: &acc };
                ScanBuilder::new(&store, fid_u64)
                    .options(ScanOptions {
                        sorted: true,
                        reverse: false,
                        with_row_ids: false,

                        limit: None,
                        offset: 0,
                        include_nulls: false,
                        nulls_first: false,
                        anchor_row_id_field: None,
                    })
                    .with_range::<u64, _>(100_000..=900_000)
                    .run(&mut v)
                    .unwrap();
                black_box(acc.get());
            },
            BatchSize::SmallInput,
        );
    });

    // Sorted with row ids: just sum rids to exercise path
    g.bench_function("sorted_with_row_ids_u64", |b| {
        b.iter_batched(
            seed_store_1m,
            |(store, fid_u64, _)| {
                struct SumRids<'a> {
                    acc: &'a std::cell::Cell<u128>,
                }
                impl<'a> PrimitiveSortedWithRowIdsVisitor for SumRids<'a> {
                    fn u64_run_with_rids(
                        &mut self,
                        _v: &UInt64Array,
                        r: &UInt64Array,
                        s: usize,
                        l: usize,
                    ) {
                        let e = s + l;
                        let mut sum = 0u128;
                        for i in s..e {
                            sum += r.value(i) as u128;
                        }
                        self.acc.set(self.acc.get() + sum);
                    }
                }
                impl<'a> PrimitiveVisitor for SumRids<'a> {}
                impl<'a> PrimitiveSortedVisitor for SumRids<'a> {}
                impl<'a> PrimitiveWithRowIdsVisitor for SumRids<'a> {}
                let acc = std::cell::Cell::new(0u128);
                let mut v = SumRids { acc: &acc };
                ScanBuilder::new(&store, fid_u64)
                    .options(ScanOptions {
                        sorted: true,
                        reverse: false,
                        with_row_ids: true,

                        limit: None,
                        offset: 0,
                        include_nulls: false,
                        nulls_first: false,
                        anchor_row_id_field: None,
                    })
                    .with_range::<u64, _>(100_000..=900_000)
                    .run(&mut v)
                    .unwrap();
                black_box(acc.get());
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

criterion_group!(benches, bench_scan_builder);
criterion_main!(benches);
