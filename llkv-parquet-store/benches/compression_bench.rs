//! Benchmark comparing different compression algorithms for large datasets.
//!
//! Tests writing and reading 1 million rows with 20 columns using different
//! Parquet compression algorithms. Batches are processed in 65K row chunks.

use arrow::array::{Float64Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llkv_parquet_store::{add_mvcc_columns, ParquetStore, WriterConfig};
use llkv_storage::pager::{MemPager, Pager};
use parquet::basic::Compression;
use std::hint::black_box;
use std::sync::Arc;

const TOTAL_ROWS: usize = 1_000_000;
const BATCH_SIZE: usize = 65_000;
const NUM_BATCHES: usize = TOTAL_ROWS / BATCH_SIZE;

/// Generate a 20-column schema with mixed data types.
fn create_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("id", DataType::Int64, false),
        Field::new("value1", DataType::Float64, false),
        Field::new("value2", DataType::Float64, false),
        Field::new("value3", DataType::Float64, false),
        Field::new("value4", DataType::Float64, false),
        Field::new("counter", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("product", DataType::Utf8, false),
        Field::new("customer_id", DataType::Int64, false),
        Field::new("order_id", DataType::Int64, false),
        Field::new("discount", DataType::Float64, false),
        Field::new("tax", DataType::Float64, false),
        Field::new("total", DataType::Float64, false),
        Field::new("notes", DataType::Utf8, false),
    ]))
}

/// Generate a batch of synthetic data with realistic patterns.
fn generate_batch(schema: Arc<Schema>, start_row: u64, num_rows: usize) -> RecordBatch {
    let categories = vec!["Electronics", "Clothing", "Food", "Books", "Toys"];
    let statuses = vec!["Pending", "Shipped", "Delivered", "Cancelled"];
    let regions = vec!["North", "South", "East", "West", "Central"];
    let products = vec!["ProductA", "ProductB", "ProductC", "ProductD", "ProductE"];

    let row_ids: Vec<u64> = (start_row..start_row + num_rows as u64).collect();
    let ids: Vec<i64> = row_ids.iter().map(|&r| r as i64).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(row_ids.clone())),
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| (i as f64) * 1.5),
            )),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| (i as f64) * 2.3),
            )),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| (i as f64) * 0.7),
            )),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| (i as f64) * 1.1),
            )),
            Arc::new(Int64Array::from_iter(
                (0..num_rows).map(|i| (i % 1000) as i64),
            )),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| 100.0 + (i as f64) * 0.5),
            )),
            Arc::new(Int64Array::from_iter(
                (0..num_rows).map(|i| (i % 100) as i64),
            )),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| 10.0 + (i as f64) * 0.1),
            )),
            Arc::new(StringArray::from_iter(
                (0..num_rows).map(|i| Some(categories[i % categories.len()])),
            )),
            Arc::new(StringArray::from_iter(
                (0..num_rows).map(|i| Some(statuses[i % statuses.len()])),
            )),
            Arc::new(StringArray::from_iter(
                (0..num_rows).map(|i| Some(regions[i % regions.len()])),
            )),
            Arc::new(StringArray::from_iter(
                (0..num_rows).map(|i| Some(products[i % products.len()])),
            )),
            Arc::new(Int64Array::from_iter(
                (0..num_rows).map(|i| (i % 10000) as i64),
            )),
            Arc::new(Int64Array::from_iter((0..num_rows).map(|i| i as i64))),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| (i % 20) as f64 * 0.05),
            )),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| (i as f64) * 0.08),
            )),
            Arc::new(Float64Array::from_iter(
                (0..num_rows).map(|i| 100.0 + (i as f64) * 1.23),
            )),
            Arc::new(StringArray::from_iter((0..num_rows).map(|i| {
                if i % 10 == 0 {
                    Some("Special order")
                } else {
                    Some("Regular")
                }
            }))),
        ],
    )
    .unwrap()
}

/// Benchmark writing data with a specific compression algorithm.
fn bench_write(compression: Compression, _compression_name: &str) -> (usize, usize) {
    let pager = Arc::new(MemPager::new());
    let config = WriterConfig::default().with_compression(compression);
    let store = ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap();

    let schema = create_schema();
    let table_id = store.create_table("benchmark", schema.clone()).unwrap();

    let mut total_input_bytes = 0;

    for batch_idx in 0..NUM_BATCHES {
        let start_row = (batch_idx * BATCH_SIZE) as u64;
        let batch = generate_batch(schema.clone(), start_row, BATCH_SIZE);
        let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();

        // Calculate approximate input size
        total_input_bytes += batch_with_mvcc.get_array_memory_size();

        store.append_many(table_id, vec![batch_with_mvcc]).unwrap();
    }

    // Calculate total storage size
    let keys = pager.enumerate_keys().unwrap();
    let mut total_storage_bytes = 0;
    for &key in &keys {
        use llkv_storage::pager::{BatchGet, GetResult};
        if let Some(GetResult::Raw { bytes, .. }) =
            pager.batch_get(&[BatchGet::Raw { key }]).unwrap().first()
        {
            total_storage_bytes += bytes.as_ref().len();
        }
    }

    (total_input_bytes, total_storage_bytes)
}

/// Benchmark reading data with a specific compression algorithm.
fn bench_read(compression: Compression) -> usize {
    let pager = Arc::new(MemPager::new());
    let config = WriterConfig::default().with_compression(compression);
    let store = ParquetStore::open_with_config(pager, config).unwrap();

    let schema = create_schema();
    let table_id = store.create_table("benchmark", schema.clone()).unwrap();

    // Write data first
    for batch_idx in 0..NUM_BATCHES {
        let start_row = (batch_idx * BATCH_SIZE) as u64;
        let batch = generate_batch(schema.clone(), start_row, BATCH_SIZE);
        let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
        store.append_many(table_id, vec![batch_with_mvcc]).unwrap();
    }

    // Read all data back
    let batches = store.read_table_files(table_id, None).unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    total_rows
}

fn compression_write_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_write");
    group.throughput(Throughput::Elements(TOTAL_ROWS as u64));

    let compressions = vec![
        (Compression::UNCOMPRESSED, "uncompressed"),
        (Compression::SNAPPY, "snappy"),
    ];

    for (compression, name) in compressions {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &compression,
            |b, &comp| {
                b.iter(|| {
                    let (input_bytes, storage_bytes) = bench_write(comp, name);
                    black_box((input_bytes, storage_bytes))
                });
            },
        );
    }

    group.finish();
}

fn compression_read_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_read");
    group.throughput(Throughput::Elements(TOTAL_ROWS as u64));

    let compressions = vec![
        (Compression::UNCOMPRESSED, "uncompressed"),
        (Compression::SNAPPY, "snappy"),
    ];

    for (compression, name) in compressions {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &compression,
            |b, &comp| {
                b.iter(|| {
                    let total_rows = bench_read(comp);
                    black_box(total_rows)
                });
            },
        );
    }

    group.finish();
}

fn compression_ratio_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_analysis");
    group.sample_size(10); // Fewer samples for this analysis

    let compressions = vec![
        (Compression::UNCOMPRESSED, "uncompressed"),
        (Compression::SNAPPY, "snappy"),
    ];

    println!("\n=== Compression Ratio Analysis ===");
    println!(
        "{:<15} {:<15} {:<15} {:<10}",
        "Compression", "Input (MB)", "Storage (MB)", "Ratio"
    );
    println!("{}", "-".repeat(60));

    for (compression, name) in &compressions {
        let (input_bytes, storage_bytes) = bench_write(*compression, name);
        let ratio = input_bytes as f64 / storage_bytes as f64;
        let input_mb = input_bytes as f64 / 1_048_576.0;
        let storage_mb = storage_bytes as f64 / 1_048_576.0;

        println!(
            "{:<15} {:<15.2} {:<15.2} {:<10.2}x",
            name, input_mb, storage_mb, ratio
        );
    }
    println!();

    group.finish();
}

criterion_group!(
    benches,
    compression_write_benchmark,
    compression_read_benchmark,
    compression_ratio_analysis
);
criterion_main!(benches);
