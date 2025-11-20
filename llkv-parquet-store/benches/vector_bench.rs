//! Benchmark for vector storage and brute-force similarity search using Parquet.
//!
//! Tests storing 1 million 1024-dimensional float32 vectors and performing
//! cosine similarity search using the simsimd library.

use arrow::array::{FixedSizeListArray, Float32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llkv_parquet_store::{add_mvcc_columns, ParquetStore, WriterConfig};
use llkv_storage::pager::{BatchGet, GetResult, MemPager, Pager};
use llkv_storage::types::PhysicalKey;
use parquet::basic::Compression;
use rand::Rng;
use std::hint::black_box;
use std::sync::Arc;

const VECTOR_DIM: usize = 1024;
const VECTOR_COUNT: usize = 1_000_000;
const VECTOR_BATCH_SIZE: usize = 65_000; // Large batches OK now that Parquet encoding is parallelized

/// Generate a schema for storing 1024-dimensional float32 vectors.
fn create_vector_schema() -> Arc<Schema> {
    // Use FixedSizeList for efficient storage of vectors as a single column
    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                VECTOR_DIM as i32,
            ),
            false,
        ),
    ]))
}

/// Generate a batch of random vectors.
fn generate_vector_batch(schema: Arc<Schema>, start_id: u64, num_vectors: usize) -> RecordBatch {
    let mut rng = rand::rng();

    // Generate vector IDs (row_id)
    let ids: Vec<u64> = (start_id..start_id + num_vectors as u64).collect();
    let id_array = UInt64Array::from(ids);

    // Generate all random float32 values in a single flat array
    let total_values = num_vectors * VECTOR_DIM;
    let values: Vec<f32> = (0..total_values).map(|_| rng.random::<f32>()).collect();
    let values_array = Float32Array::from(values);

    // Create FixedSizeList array wrapping the flat values
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let embedding_array =
        FixedSizeListArray::new(field, VECTOR_DIM as i32, Arc::new(values_array), None);

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(embedding_array)]).unwrap()
}

/// Write vectors to ParquetStore and return (input_bytes, storage_bytes).
fn bench_write_vectors(compression: Compression, _compression_name: &str) -> (usize, usize) {
    let pager = Arc::new(MemPager::new());
    let config = WriterConfig::default()
        .with_compression(compression)
        .with_max_row_group_size(VECTOR_BATCH_SIZE);

    let store = ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap();
    let schema = create_vector_schema();
    let table_id = store.create_table("vectors", schema.clone()).unwrap();

    let mut input_bytes = 0;
    let num_batches = VECTOR_COUNT / VECTOR_BATCH_SIZE;

    let mut batches = Vec::with_capacity(num_batches);
    for batch_idx in 0..num_batches {
        let start_id = (batch_idx * VECTOR_BATCH_SIZE) as u64;
        let batch = generate_vector_batch(schema.clone(), start_id, VECTOR_BATCH_SIZE);
        let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();

        // Calculate input size
        input_bytes += batch_with_mvcc.get_array_memory_size();

        batches.push(batch_with_mvcc);
    }

    // Append all batches at once
    store.append_many(table_id, batches).unwrap();

    // Calculate total storage
    let keys: Vec<PhysicalKey> = pager
        .enumerate_keys()
        .unwrap()
        .into_iter()
        .filter(|&key| key != 0) // Skip catalog key
        .collect();

    let gets: Vec<BatchGet> = keys.iter().map(|&key| BatchGet::Raw { key }).collect();
    let results = pager.batch_get(&gets).unwrap();
    let storage_bytes: usize = results
        .iter()
        .map(|r| match r {
            GetResult::Raw { bytes, .. } => bytes.as_ref().len(),
            GetResult::Missing { .. } => 0,
        })
        .sum();

    (input_bytes, storage_bytes)
}

/// Read all vectors and return total count.
fn bench_read_vectors(compression: Compression) -> usize {
    let pager = Arc::new(MemPager::new());
    let config = WriterConfig::default()
        .with_compression(compression)
        .with_max_row_group_size(VECTOR_BATCH_SIZE);

    let store = ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap();
    let schema = create_vector_schema();
    let table_id = store.create_table("vectors", schema.clone()).unwrap();

    // Write data first
    let num_batches = VECTOR_COUNT / VECTOR_BATCH_SIZE;
    let mut batches = Vec::with_capacity(num_batches);
    for batch_idx in 0..num_batches {
        let start_id = (batch_idx * VECTOR_BATCH_SIZE) as u64;
        let batch = generate_vector_batch(schema.clone(), start_id, VECTOR_BATCH_SIZE);
        let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
        batches.push(batch_with_mvcc);
    }
    store.append_many(table_id, batches).unwrap();

    // Read back and count rows
    let batches = store.read_table_files(table_id, None).unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Perform brute-force cosine similarity search on vectors.
fn bench_vector_similarity_search(compression: Compression) -> (usize, f64) {
    let pager = Arc::new(MemPager::new());
    let config = WriterConfig::default()
        .with_compression(compression)
        .with_max_row_group_size(VECTOR_BATCH_SIZE);

    let store = ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap();
    let schema = create_vector_schema();
    let table_id = store.create_table("vectors", schema.clone()).unwrap();

    // Write vectors
    let num_batches = VECTOR_COUNT / VECTOR_BATCH_SIZE;
    let mut batches = Vec::with_capacity(num_batches);
    for batch_idx in 0..num_batches {
        let start_id = (batch_idx * VECTOR_BATCH_SIZE) as u64;
        let batch = generate_vector_batch(schema.clone(), start_id, VECTOR_BATCH_SIZE);
        let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
        batches.push(batch_with_mvcc);
    }
    store.append_many(table_id, batches).unwrap();

    // Generate query vector
    let mut rng = rand::rng();
    let query_vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.random::<f32>()).collect();

    // Read all vectors and compute cosine similarity
    let batches = store.read_table_files(table_id, None).unwrap();
    let mut best_similarity = f64::NEG_INFINITY;
    let mut vectors_searched = 0;

    for batch in batches {
        let num_rows = batch.num_rows();
        vectors_searched += num_rows;

        // Extract vectors and compute similarity
        let embedding_column = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();

        for row_idx in 0..num_rows {
            // Extract the vector from the FixedSizeList
            let list_value = embedding_column.value(row_idx);
            let vector_array = list_value.as_any().downcast_ref::<Float32Array>().unwrap();

            let vector: Vec<f32> = (0..VECTOR_DIM).map(|i| vector_array.value(i)).collect();

            // Compute cosine similarity using simsimd
            let similarity = simsimd::SpatialSimilarity::cosine(&query_vector, &vector)
                .expect("cosine similarity computation failed");

            if similarity > best_similarity {
                best_similarity = similarity;
            }
        }
    }

    (vectors_searched, best_similarity)
}

fn vector_write_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_write");
    group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
    group.sample_size(10); // Reduce sample size for longer benchmarks

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
                    let (input, storage) = bench_write_vectors(comp, name);
                    black_box((input, storage))
                });
            },
        );
    }

    group.finish();
}

fn vector_read_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_read");
    group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
    group.sample_size(10);

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
                    let total_rows = bench_read_vectors(comp);
                    black_box(total_rows)
                });
            },
        );
    }

    group.finish();
}

fn vector_similarity_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_similarity_search");
    group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
    group.sample_size(10);

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
                    let (count, similarity) = bench_vector_similarity_search(comp);
                    black_box((count, similarity))
                });
            },
        );
    }

    group.finish();
}

fn vector_compression_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_compression_analysis");
    group.sample_size(10);

    let compressions = vec![
        (Compression::UNCOMPRESSED, "uncompressed"),
        (Compression::SNAPPY, "snappy"),
    ];

    println!("\n=== Vector Compression Ratio Analysis ===");
    println!("Vectors: {}, Dimensions: {}", VECTOR_COUNT, VECTOR_DIM);
    println!(
        "{:<15} {:<15} {:<15} {:<10}",
        "Compression", "Input (MB)", "Storage (MB)", "Ratio"
    );
    println!("{}", "-".repeat(60));

    for (compression, name) in &compressions {
        let (input_bytes, storage_bytes) = bench_write_vectors(*compression, name);
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
    vector_write_benchmark,
    vector_read_benchmark,
    vector_similarity_benchmark,
    vector_compression_analysis
);
criterion_main!(benches);
