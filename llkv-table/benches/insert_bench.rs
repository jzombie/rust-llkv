//! Insert benchmark: measure throughput of consecutive insert operations.
//!
//! What it measures
//! - Builds an in-memory table (MemPager) and appends 1M rows across multiple
//!   numeric columns using the LlkvTableBuilder API.
//! - Tests different batch sizes (1000, 5000, 10000, 50000) to expose the
//!   overhead of batching vs. raw throughput.
//! - Uses a 20-column schema with mixed integer types (u64, i64, u32, i32,
//!   u16, i16, u8, i8, f64, f32) to simulate realistic workloads.
//!
//! Performance characteristics:
//! - Append-only workloads with sequential row_ids are O(1) per batch due to
//!   fast-path optimization that skips LWW rewrite checks when row_id ranges
//!   don't overlap with existing data.
//! - Performance should scale linearly with total row count regardless of batch size.
//!
//! Run:
//!   cargo bench --bench insert_bench

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use llkv_column_map::store::ColumnStore;
use llkv_storage::pager::MemPager;
use llkv_table::LlkvTableBuilder;

const TOTAL_ROWS: usize = 1_000_000;
const NUM_COLUMNS: usize = 20;

/// Build a schema with 20 numeric columns of mixed types.
fn build_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("col_u64_1", DataType::UInt64, false),
        Field::new("col_u64_2", DataType::UInt64, false),
        Field::new("col_i64_1", DataType::Int64, false),
        Field::new("col_i64_2", DataType::Int64, false),
        Field::new("col_u32_1", DataType::UInt32, false),
        Field::new("col_u32_2", DataType::UInt32, false),
        Field::new("col_i32_1", DataType::Int32, false),
        Field::new("col_i32_2", DataType::Int32, false),
        Field::new("col_u16_1", DataType::UInt16, false),
        Field::new("col_u16_2", DataType::UInt16, false),
        Field::new("col_i16_1", DataType::Int16, false),
        Field::new("col_i16_2", DataType::Int16, false),
        Field::new("col_u8_1", DataType::UInt8, false),
        Field::new("col_u8_2", DataType::UInt8, false),
        Field::new("col_i8_1", DataType::Int8, false),
        Field::new("col_i8_2", DataType::Int8, false),
        Field::new("col_f64_1", DataType::Float64, false),
        Field::new("col_f64_2", DataType::Float64, false),
        Field::new("col_f32_1", DataType::Float32, false),
        Field::new("col_f32_2", DataType::Float32, false),
    ]))
}

/// Generate deterministic numeric data for a range of rows.
fn build_batch(schema: &Arc<Schema>, start: usize, end: usize) -> RecordBatch {
    // Generate deterministic values based on row index
    let col_u64_1: Vec<u64> = (start..end).map(|i| (i as u64) * 7).collect();
    let col_u64_2: Vec<u64> = (start..end).map(|i| (i as u64) * 13).collect();
    let col_i64_1: Vec<i64> = (start..end).map(|i| (i as i64) * 11).collect();
    let col_i64_2: Vec<i64> = (start..end).map(|i| (i as i64) * 17).collect();
    let col_u32_1: Vec<u32> = (start..end)
        .map(|i| ((i * 19) & 0xFFFFFFFF) as u32)
        .collect();
    let col_u32_2: Vec<u32> = (start..end)
        .map(|i| ((i * 23) & 0xFFFFFFFF) as u32)
        .collect();
    let col_i32_1: Vec<i32> = (start..end)
        .map(|i| ((i * 29) & 0x7FFFFFFF) as i32)
        .collect();
    let col_i32_2: Vec<i32> = (start..end)
        .map(|i| ((i * 31) & 0x7FFFFFFF) as i32)
        .collect();
    let col_u16_1: Vec<u16> = (start..end).map(|i| ((i * 37) & 0xFFFF) as u16).collect();
    let col_u16_2: Vec<u16> = (start..end).map(|i| ((i * 41) & 0xFFFF) as u16).collect();
    let col_i16_1: Vec<i16> = (start..end).map(|i| ((i * 43) & 0x7FFF) as i16).collect();
    let col_i16_2: Vec<i16> = (start..end).map(|i| ((i * 47) & 0x7FFF) as i16).collect();
    let col_u8_1: Vec<u8> = (start..end).map(|i| ((i * 53) & 0xFF) as u8).collect();
    let col_u8_2: Vec<u8> = (start..end).map(|i| ((i * 59) & 0xFF) as u8).collect();
    let col_i8_1: Vec<i8> = (start..end).map(|i| ((i * 61) & 0x7F) as i8).collect();
    let col_i8_2: Vec<i8> = (start..end).map(|i| ((i * 67) & 0x7F) as i8).collect();
    let col_f64_1: Vec<f64> = (start..end).map(|i| (i as f64) * 0.71).collect();
    let col_f64_2: Vec<f64> = (start..end).map(|i| (i as f64) * 0.73).collect();
    let col_f32_1: Vec<f32> = (start..end).map(|i| (i as f32) * 0.79).collect();
    let col_f32_2: Vec<f32> = (start..end).map(|i| (i as f32) * 0.83).collect();

    let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(UInt64Array::from(col_u64_1)),
        Arc::new(UInt64Array::from(col_u64_2)),
        Arc::new(Int64Array::from(col_i64_1)),
        Arc::new(Int64Array::from(col_i64_2)),
        Arc::new(UInt32Array::from(col_u32_1)),
        Arc::new(UInt32Array::from(col_u32_2)),
        Arc::new(Int32Array::from(col_i32_1)),
        Arc::new(Int32Array::from(col_i32_2)),
        Arc::new(UInt16Array::from(col_u16_1)),
        Arc::new(UInt16Array::from(col_u16_2)),
        Arc::new(Int16Array::from(col_i16_1)),
        Arc::new(Int16Array::from(col_i16_2)),
        Arc::new(UInt8Array::from(col_u8_1)),
        Arc::new(UInt8Array::from(col_u8_2)),
        Arc::new(Int8Array::from(col_i8_1)),
        Arc::new(Int8Array::from(col_i8_2)),
        Arc::new(Float64Array::from(col_f64_1)),
        Arc::new(Float64Array::from(col_f64_2)),
        Arc::new(Float32Array::from(col_f32_1)),
        Arc::new(Float32Array::from(col_f32_2)),
    ];

    RecordBatch::try_new(schema.clone(), columns).expect("build batch")
}

fn bench_consecutive_inserts(c: &mut Criterion) {
    let schema = build_schema();
    let total_cells = (TOTAL_ROWS as u64) * (NUM_COLUMNS as u64);

    let mut group = c.benchmark_group("insert_1M_rows_20_cols");
    group.sample_size(10);
    group.throughput(Throughput::Elements(total_cells));

    for &batch_size in &[1000usize, 5000, 10000, 50000] {
        group.bench_function(
            BenchmarkId::from_parameter(format!("batch_size={}", batch_size)),
            |b| {
                b.iter(|| {
                    let pager = Arc::new(MemPager::new());
                    let store = Arc::new(ColumnStore::open(pager).expect("open store"));
                    let mut builder =
                        LlkvTableBuilder::new(Arc::clone(&store), 1, Arc::clone(&schema))
                            .expect("builder");

                    let mut offset = 0;
                    while offset < TOTAL_ROWS {
                        let end = (offset + batch_size).min(TOTAL_ROWS);
                        let batch = build_batch(&schema, offset, end);
                        builder.append_batch(&batch).expect("append");
                        offset = end;
                    }

                    let provider = builder.finish().expect("finish");
                    // Touch the provider to prevent optimizer from eliding work
                    black_box(provider.row_count());
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_consecutive_inserts);
criterion_main!(benches);
