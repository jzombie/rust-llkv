//! Simple and fragmented sum benches using the ColumnStore API.
//!
//! Notes for the new API:
//! - Every RecordBatch must include a non-nullable UInt64 column named by `ROW_ID_COLUMN_NAME`.
//! - Only data columns carry "field_id" metadata.
//!
//! Run:
//!   cargo bench --bench column_sum_bench

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

use llkv_column_map::store::scan::ScanOptions;
use llkv_column_map::store::{ColumnStore, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

use roaring::RoaringTreemap;

const NUM_ROWS_SIMPLE: usize = 1_000_000;
const NUM_ROWS_FRAGMENTED: u64 = 1_000_000;
const NUM_CHUNKS_FRAGMENTED: u64 = 1_000;
const CHUNK_SIZE_FRAGMENTED: u64 = NUM_ROWS_FRAGMENTED / NUM_CHUNKS_FRAGMENTED;

/// Test helper to create a standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

/// Build a 2-field schema: row_id (u64, non-null) + one data field.
fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![rid, field]))
}

/// Benchmarks for simple, non-fragmented summation.
fn bench_column_store_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_store_sum_1M");
    group.sample_size(20);

    // --- u64 column ---
    group.bench_function("sum_u64", |b| {
        b.iter_batched(
            || {
                let pager = Arc::new(MemPager::new());
                let store = ColumnStore::open(pager).unwrap();
                let field_id = fid(7777);

                let mut md = HashMap::new();
                md.insert("field_id".to_string(), u64::from(field_id).to_string());
                let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
                let schema = schema_with_row_id(data_f);

                // row_id 0..N-1, values 0..N-1 as u64
                let rid: Vec<u64> = (0..NUM_ROWS_SIMPLE as u64).collect();
                let vals: Vec<u64> = (0..NUM_ROWS_SIMPLE as u64).collect();

                let rid_arr = Arc::new(UInt64Array::from(rid));
                let val_arr = Arc::new(UInt64Array::from(vals));
                let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();

                store.append(&batch).unwrap();
                (store, field_id)
            },
            |(store, fid)| {
                use llkv_column_map::store::scan::{
                    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
                    PrimitiveWithRowIdsVisitor,
                };
                struct SumU64<'a> {
                    out: &'a std::cell::Cell<u128>,
                }
                impl<'a> PrimitiveVisitor for SumU64<'a> {
                    fn u64_chunk(&mut self, a: &UInt64Array) {
                        if let Some(s) = compute::sum(a) {
                            self.out.set(self.out.get() + s as u128);
                        }
                    }
                }
                impl<'a> PrimitiveSortedVisitor for SumU64<'a> {}
                impl<'a> PrimitiveWithRowIdsVisitor for SumU64<'a> {}
                impl<'a> PrimitiveSortedWithRowIdsVisitor for SumU64<'a> {}
                let acc = std::cell::Cell::new(0u128);
                let mut v = SumU64 { out: &acc };
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
                black_box(acc.get());
            },
            BatchSize::SmallInput,
        );
    });

    // --- i32 column ---
    group.bench_function("sum_i32", |b| {
        b.iter_batched(
            || {
                let pager = Arc::new(MemPager::new());
                let store = ColumnStore::open(pager).unwrap();
                let field_id = fid(8888);

                let mut md = HashMap::new();
                md.insert("field_id".to_string(), u64::from(field_id).to_string());
                let data_f = Field::new("data", DataType::Int32, false).with_metadata(md);
                let schema = schema_with_row_id(data_f);

                // row_id 0..N-1, values 0..N-1 as i32
                let rid: Vec<u64> = (0..NUM_ROWS_SIMPLE as u64).collect();
                let vals: Vec<i32> = (0..NUM_ROWS_SIMPLE as i32).collect();

                let rid_arr = Arc::new(UInt64Array::from(rid));
                let val_arr = Arc::new(Int32Array::from(vals));
                let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();

                store.append(&batch).unwrap();
                (store, field_id)
            },
            |(store, fid)| {
                use llkv_column_map::store::scan::{
                    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
                    PrimitiveWithRowIdsVisitor,
                };
                struct SumI32<'a> {
                    out: &'a std::cell::Cell<i128>,
                }
                impl<'a> PrimitiveVisitor for SumI32<'a> {
                    fn i32_chunk(&mut self, a: &Int32Array) {
                        if let Some(s) = compute::sum(a) {
                            self.out.set(self.out.get() + s as i128);
                        }
                    }
                }
                impl<'a> PrimitiveSortedVisitor for SumI32<'a> {}
                impl<'a> PrimitiveWithRowIdsVisitor for SumI32<'a> {}
                impl<'a> PrimitiveSortedWithRowIdsVisitor for SumI32<'a> {}
                let acc = std::cell::Cell::new(0i128);
                let mut v = SumI32 { out: &acc };
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
                black_box(acc.get());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmarks for fragmented data with deletes and updates.
fn bench_fragmented_deletes_and_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_store_fragmented_1M");
    group.sample_size(10); // slower

    group.bench_function("sum_u64_fragmented_with_deletes", |b| {
        b.iter_batched(
            || {
                let field_id = fid(9001);
                let pager = Arc::new(MemPager::new());
                let store = ColumnStore::open(pager).unwrap();

                let mut md = HashMap::new();
                md.insert("field_id".to_string(), u64::from(field_id).to_string());
                let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
                let schema = schema_with_row_id(data_f);

                // 1) Ingest in many small, fragmented chunks.
                // row_id is global 0..N-1 to keep absolute indices stable.
                for i in 0..NUM_CHUNKS_FRAGMENTED {
                    let start = i * CHUNK_SIZE_FRAGMENTED;
                    let end = start + CHUNK_SIZE_FRAGMENTED;

                    let rid: Vec<u64> = (start..end).collect();
                    let vals: Vec<u64> = (start..end).collect();

                    let rid_arr = Arc::new(UInt64Array::from(rid));
                    let val_arr = Arc::new(UInt64Array::from(vals));
                    let batch =
                        RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
                    store.append(&batch).unwrap();
                }

                // 2) Delete every 10th row (absolute row index).
                let rows_to_delete: RoaringTreemap = (0..NUM_ROWS_FRAGMENTED)
                    .step_by(10)
                    // .map(|i| i as u32)
                    .collect();
                store.delete_rows(field_id, &rows_to_delete).unwrap();

                // 3) Append one more chunk after deletions.
                let start = NUM_ROWS_FRAGMENTED;
                let end = start + CHUNK_SIZE_FRAGMENTED;

                let rid_new: Vec<u64> = (start..end).collect();
                let vals_new: Vec<u64> = (start..end).collect();

                let rid_arr_new = Arc::new(UInt64Array::from(rid_new.clone()));
                let val_arr_new = Arc::new(UInt64Array::from(vals_new.clone()));
                let batch_new =
                    RecordBatch::try_new(schema.clone(), vec![rid_arr_new, val_arr_new]).unwrap();
                store.append(&batch_new).unwrap();

                // 4) Expected final sum.
                let initial_sum: u128 = (0u64..NUM_ROWS_FRAGMENTED).map(|x| x as u128).sum();

                let deleted_sum: u128 = rows_to_delete.iter().map(|x| x as u128).sum();

                let new_sum: u128 = (start..end).map(|x| x as u128).sum();

                let expected_final_sum = initial_sum - deleted_sum + new_sum;

                (store, field_id, expected_final_sum)
            },
            |(store, fid, expected_sum)| {
                use llkv_column_map::store::scan::{
                    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
                    PrimitiveWithRowIdsVisitor,
                };
                struct SumU64<'a> {
                    out: &'a std::cell::Cell<u128>,
                }
                impl<'a> PrimitiveVisitor for SumU64<'a> {
                    fn u64_chunk(&mut self, a: &UInt64Array) {
                        if let Some(s) = compute::sum(a) {
                            self.out.set(self.out.get() + s as u128);
                        }
                    }
                }
                impl<'a> PrimitiveSortedVisitor for SumU64<'a> {}
                impl<'a> PrimitiveWithRowIdsVisitor for SumU64<'a> {}
                impl<'a> PrimitiveSortedWithRowIdsVisitor for SumU64<'a> {}
                let acc = std::cell::Cell::new(0u128);
                let mut v = SumU64 { out: &acc };
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
                assert_eq!(acc.get(), expected_sum);
                black_box(acc.get());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_column_store_sum,
    bench_fragmented_deletes_and_updates
);
criterion_main!(benches);
