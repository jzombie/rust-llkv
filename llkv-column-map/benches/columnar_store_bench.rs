use arrow::array::{Array, Int32Array, UInt64Array};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

// --- Constants for Benchmark Parameters ---
const NUM_ROWS_SIMPLE: usize = 1_000_000;
const NUM_ROWS_FRAGMENTED: u64 = 1_000_000;
const NUM_CHUNKS_FRAGMENTED: u64 = 1_000;
const CHUNK_SIZE_FRAGMENTED: u64 = NUM_ROWS_FRAGMENTED / NUM_CHUNKS_FRAGMENTED;

/// Benchmarks for simple, non-fragmented summation.
fn bench_arrow_store_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_store_sum_1M");
    group.sample_size(20);

    // --- Benchmark for u64 column ---
    group.bench_function("sum_u64", |b| {
        b.iter_batched(
            // Setup: Create a store and append one chunk of 1M u64 values.
            || {
                let pager = Arc::new(MemPager::new());
                let store = ColumnStore::open(pager).unwrap();
                let field_id = 7777;

                let mut metadata = HashMap::new();
                metadata.insert("field_id".to_string(), field_id.to_string());
                let field = Field::new("data", DataType::UInt64, false).with_metadata(metadata);
                let schema = Arc::new(Schema::new(vec![field]));

                let vals: Vec<u64> = (0..NUM_ROWS_SIMPLE).map(|i| i as u64).collect();
                let array = Arc::new(UInt64Array::from(vals));
                let batch = RecordBatch::try_new(schema, vec![array]).unwrap();

                store.append(&batch).unwrap();
                (store, field_id)
            },
            // Measurement: Scan the column and sum the values in each array chunk.
            |(store, fid)| {
                let mut acc: u128 = 0;
                for array_result in store.scan(fid).unwrap() {
                    let array = array_result.unwrap();
                    let u64_array = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                    if let Some(chunk_sum) = compute::sum(u64_array) {
                        acc += chunk_sum as u128;
                    }
                }
                black_box(acc);
            },
            BatchSize::SmallInput,
        );
    });

    // --- Benchmark for i32 column ---
    group.bench_function("sum_i32", |b| {
        b.iter_batched(
            // Setup: Create a store and append one chunk of 1M i32 values.
            || {
                let pager = Arc::new(MemPager::new());
                let store = ColumnStore::open(pager).unwrap();
                let field_id = 8888;

                let mut metadata = HashMap::new();
                metadata.insert("field_id".to_string(), field_id.to_string());
                let field = Field::new("data", DataType::Int32, false).with_metadata(metadata);
                let schema = Arc::new(Schema::new(vec![field]));

                let vals: Vec<i32> = (0..NUM_ROWS_SIMPLE).map(|i| i as i32).collect();
                let array = Arc::new(Int32Array::from(vals));
                let batch = RecordBatch::try_new(schema, vec![array]).unwrap();

                store.append(&batch).unwrap();
                (store, field_id)
            },
            // Measurement: Scan the column and sum the values in each array chunk.
            |(store, fid)| {
                let mut acc: i128 = 0;
                for array_result in store.scan(fid).unwrap() {
                    let array = array_result.unwrap();
                    let i32_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
                    if let Some(chunk_sum) = compute::sum(i32_array) {
                        acc += chunk_sum as i128;
                    }
                }
                black_box(acc);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmarks for fragmented data with deletes and updates.
fn bench_fragmented_deletes_and_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_store_fragmented_1M");
    group.sample_size(10); // This is a slower test, so fewer samples.

    group.bench_function("sum_u64_fragmented_with_deletes", |b| {
        b.iter_batched(
            // --- Setup ---
            || {
                let field_id = 9001;
                let pager = Arc::new(MemPager::new());
                let store = ColumnStore::open(pager).unwrap();

                let mut metadata = HashMap::new();
                metadata.insert("field_id".to_string(), field_id.to_string());
                let field = Field::new("data", DataType::UInt64, false).with_metadata(metadata);
                let schema = Arc::new(Schema::new(vec![field]));

                // 1. Ingest data in many small, fragmented chunks.
                let initial_data: Vec<u64> = (0..NUM_ROWS_FRAGMENTED).collect();
                for i in 0..NUM_CHUNKS_FRAGMENTED {
                    let start = (i * CHUNK_SIZE_FRAGMENTED) as usize;
                    let end = start + CHUNK_SIZE_FRAGMENTED as usize;
                    let array = Arc::new(UInt64Array::from(initial_data[start..end].to_vec()));
                    let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
                    store.append(&batch).unwrap();
                }

                // 2. Delete every 10th row.
                let rows_to_delete: RoaringBitmap = (0..NUM_ROWS_FRAGMENTED)
                    .step_by(10)
                    .map(|i| i as u32)
                    .collect();
                store.delete_rows(field_id, &rows_to_delete).unwrap();

                // 3. Append more data after the deletion.
                let new_data: Vec<u64> =
                    (NUM_ROWS_FRAGMENTED..NUM_ROWS_FRAGMENTED + CHUNK_SIZE_FRAGMENTED).collect();
                let array = Arc::new(UInt64Array::from(new_data.clone()));
                let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
                store.append(&batch).unwrap();

                // 4. Pre-calculate the correct final sum.
                let initial_sum: u128 = initial_data.iter().map(|&x| x as u128).sum();
                let deleted_sum: u128 = rows_to_delete.iter().map(|x| x as u128).sum();
                let new_sum: u128 = new_data.iter().map(|&x| x as u128).sum();
                let expected_final_sum = initial_sum - deleted_sum + new_sum;

                (store, field_id, expected_final_sum)
            },
            // --- Measurement and Verification ---
            |(store, fid, expected_sum)| {
                let mut acc: u128 = 0;
                // The scan will traverse all chunks, apply tombstones, and yield live rows.
                for array_result in store.scan(fid).unwrap() {
                    let array = array_result.unwrap();
                    let u64_array = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                    if let Some(chunk_sum) = compute::sum(u64_array) {
                        acc += chunk_sum as u128;
                    }
                }

                // This is the crucial part: verify correctness inside the benchmark.
                assert_eq!(acc, expected_sum);
                black_box(acc);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_arrow_store_sum,
    bench_fragmented_deletes_and_updates
);
criterion_main!(benches);
