//! Benchmark for vector storage and brute-force similarity search using Parquet.
//!
//! Tests storing 1 million 1024-dimensional float32 vectors and performing
//! cosine similarity search using the simsimd library.

use arrow::array::{FixedSizeListArray, Float32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llkv_parquet_store::{add_mvcc_columns, ParquetStore, TableId, WriterConfig};
use llkv_result::Result;
use llkv_storage::pager::{BatchGet, GetResult, MemPager, Pager};
use llkv_storage::types::PhysicalKey;
use parquet::basic::Compression;
use rand::Rng;
use simd_r_drive_entry_handle::EntryHandle;
use std::hint::black_box;
use std::sync::Arc;

const VECTOR_DIM: usize = 1024;
const VECTOR_COUNT: usize = 1_000_000;

// Batch size for generating test data - ParquetStore will automatically
// optimize this to ~4MB chunks (TARGET_BATCH_SIZE_BYTES) before writing
const VECTOR_BATCH_SIZE: usize = 65_000;

// ============================================================================
// Parquet-based benchmarks
// ============================================================================

mod parquet_based {
    use super::*;
    use std::collections::HashMap;

    /// Generate a schema for storing 1024-dimensional float32 vectors.
    fn create_vector_schema() -> Arc<Schema> {
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

    /// Generate a schema with external storage for embeddings.
    fn create_external_vector_schema() -> Arc<Schema> {
        let mut metadata = HashMap::new();
        metadata.insert(
            llkv_parquet_store::EXTERNAL_STORAGE_KEY.to_string(),
            llkv_parquet_store::EXTERNAL_STORAGE_VALUE.to_string(),
        );

        Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    VECTOR_DIM as i32,
                ),
                false,
            )
            .with_metadata(metadata),
        ]))
    }

    /// Generate a batch of random vectors.
    fn generate_vector_batch(
        schema: Arc<Schema>,
        start_id: u64,
        num_vectors: usize,
    ) -> RecordBatch {
        let mut rng = rand::rng();

        let ids: Vec<u64> = (start_id..start_id + num_vectors as u64).collect();
        let id_array = UInt64Array::from(ids);

        let total_values = num_vectors * VECTOR_DIM;
        let values: Vec<f32> = (0..total_values).map(|_| rng.random::<f32>()).collect();
        let values_array = Float32Array::from(values);

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

            input_bytes += batch_with_mvcc.get_array_memory_size();
            batches.push(batch_with_mvcc);
        }

        store.append_many(table_id, batches).unwrap();

        // Calculate total storage
        let keys: Vec<PhysicalKey> = pager
            .enumerate_keys()
            .unwrap()
            .into_iter()
            .filter(|&key| key != 0)
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

    /// Read all vectors and compute sum of all values (forces full data access).
    fn bench_read_vectors<P: Pager<Blob = EntryHandle> + Sync>(
        store: &ParquetStore<P>,
        table_id: TableId,
    ) -> f64 {
        let batches = store.scan_parallel(table_id, &[], None, None).unwrap();

        let mut sum = 0.0f64;
        for batch in batches {
            let embedding_column = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            let values = embedding_column
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

            // Access ALL elements using raw slice for better performance
            let slice = values.values();
            for &val in slice.iter() {
                sum += val as f64;
            }
        }
        sum
    }

    /// Setup store with pre-written vectors for read benchmarks.
    fn setup_store_for_read(compression: Compression) -> (Arc<ParquetStore<MemPager>>, TableId) {
        let pager = Arc::new(MemPager::new());
        let config = WriterConfig::default()
            .with_compression(compression)
            .with_max_row_group_size(VECTOR_BATCH_SIZE);

        let store = Arc::new(ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap());
        let schema = create_vector_schema();
        let table_id = store.create_table("vectors", schema.clone()).unwrap();

        let num_batches = VECTOR_COUNT / VECTOR_BATCH_SIZE;
        let mut batches = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let start_id = (batch_idx * VECTOR_BATCH_SIZE) as u64;
            let batch = generate_vector_batch(schema.clone(), start_id, VECTOR_BATCH_SIZE);
            let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
            batches.push(batch_with_mvcc);
        }
        store.append_many(table_id, batches).unwrap();

        (store, table_id)
    }

    /// Perform brute-force cosine similarity search on vectors.
    fn bench_vector_similarity_search<P: Pager<Blob = EntryHandle> + Sync>(
        store: &ParquetStore<P>,
        table_id: TableId,
        query_vector: &[f32],
    ) -> (usize, f64) {
        let batches = store.scan_parallel(table_id, &[], None, None).unwrap();
        let mut best_similarity = f64::NEG_INFINITY;
        let mut vectors_searched = 0;

        for batch in batches {
            let num_rows = batch.num_rows();
            vectors_searched += num_rows;

            let embedding_column = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            // Get flat slice into the underlying Float32Array
            let values = embedding_column
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

            for row_idx in 0..num_rows {
                let start = row_idx * VECTOR_DIM;
                let end = start + VECTOR_DIM;
                let slice = &values.values()[start..end];

                let similarity = simsimd::SpatialSimilarity::cosine(query_vector, slice)
                    .expect("cosine similarity computation failed");

                if similarity > best_similarity {
                    best_similarity = similarity;
                }
            }
        }

        (vectors_searched, best_similarity)
    }

    /// Setup store with pre-written vectors for similarity search.
    fn setup_store_for_similarity(
        compression: Compression,
    ) -> (Arc<ParquetStore<MemPager>>, TableId, Vec<f32>) {
        let pager = Arc::new(MemPager::new());
        let config = WriterConfig::default()
            .with_compression(compression)
            .with_max_row_group_size(VECTOR_BATCH_SIZE);

        let store = Arc::new(ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap());
        let schema = create_vector_schema();
        let table_id = store.create_table("vectors", schema.clone()).unwrap();

        let num_batches = VECTOR_COUNT / VECTOR_BATCH_SIZE;
        let mut batches = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let start_id = (batch_idx * VECTOR_BATCH_SIZE) as u64;
            let batch = generate_vector_batch(schema.clone(), start_id, VECTOR_BATCH_SIZE);
            let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
            batches.push(batch_with_mvcc);
        }
        store.append_many(table_id, batches).unwrap();

        let mut rng = rand::rng();
        let query_vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.random::<f32>()).collect();

        (store, table_id, query_vector)
    }

    // Benchmark entry point for write performance.
    // pub fn run_write_benchmark(c: &mut Criterion) {
    //     let mut group = c.benchmark_group("parquet_vector_write");
    //     group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
    //     group.sample_size(10);

    //     let compressions = vec![
    //         (Compression::UNCOMPRESSED, "uncompressed"),
    //         (Compression::SNAPPY, "snappy"),
    //     ];

    //     for (compression, name) in compressions {
    //         group.bench_with_input(
    //             BenchmarkId::from_parameter(name),
    //             &compression,
    //             |b, &comp| {
    //                 b.iter(|| {
    //                     let (input, storage) = bench_write_vectors(comp, name);
    //                     black_box((input, storage))
    //                 });
    //             },
    //         );
    //     }

    //     group.finish();
    // }

    /// Benchmark entry point for read performance.
    pub fn run_read_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("parquet_vector_read");
        group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
        group.sample_size(10);

        let compressions = vec![
            (Compression::UNCOMPRESSED, "uncompressed"),
            (Compression::SNAPPY, "snappy"),
        ];

        for (compression, name) in compressions {
            let (store, table_id) = setup_store_for_read(compression);
            group.bench_with_input(
                BenchmarkId::from_parameter(name),
                &compression,
                |b, _comp| {
                    b.iter(|| {
                        let sum = bench_read_vectors(&store, table_id);
                        black_box(sum)
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark entry point for similarity search performance.
    pub fn run_similarity_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("parquet_vector_similarity_search");
        group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
        group.sample_size(10);

        let compressions = vec![
            (Compression::UNCOMPRESSED, "uncompressed"),
            (Compression::SNAPPY, "snappy"),
        ];

        for (compression, name) in compressions {
            let (store, table_id, query_vector) = setup_store_for_similarity(compression);
            group.bench_with_input(
                BenchmarkId::from_parameter(name),
                &compression,
                |b, _comp| {
                    b.iter(|| {
                        let (count, similarity) =
                            bench_vector_similarity_search(&store, table_id, &query_vector);
                        black_box((count, similarity))
                    });
                },
            );
        }

        group.finish();
    }

    /// Run compression analysis (not a timed benchmark).
    pub fn run_compression_analysis(c: &mut Criterion) {
        let mut group = c.benchmark_group("parquet_compression_analysis");
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

    /// Setup store with external storage for embeddings.
    fn setup_store_for_read_external(
        compression: Compression,
    ) -> (Arc<ParquetStore<MemPager>>, TableId) {
        let pager = Arc::new(MemPager::new());
        let config = WriterConfig::default()
            .with_compression(compression)
            .with_max_row_group_size(VECTOR_BATCH_SIZE);

        let store = Arc::new(ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap());
        let schema = create_external_vector_schema();
        let table_id = store
            .create_table("vectors_external", schema.clone())
            .unwrap();

        let num_batches = VECTOR_COUNT / VECTOR_BATCH_SIZE;
        let mut batches = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let start_id = (batch_idx * VECTOR_BATCH_SIZE) as u64;
            let batch = generate_vector_batch(schema.clone(), start_id, VECTOR_BATCH_SIZE);
            let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
            batches.push(batch_with_mvcc);
        }
        store.append_many(table_id, batches).unwrap();

        (store, table_id)
    }

    /// Setup store with external storage for similarity search.
    fn setup_store_for_similarity_external(
        compression: Compression,
    ) -> (Arc<ParquetStore<MemPager>>, TableId, Vec<f32>) {
        let pager = Arc::new(MemPager::new());
        let config = WriterConfig::default()
            .with_compression(compression)
            .with_max_row_group_size(VECTOR_BATCH_SIZE);

        let store = Arc::new(ParquetStore::open_with_config(Arc::clone(&pager), config).unwrap());
        let schema = create_external_vector_schema();
        let table_id = store
            .create_table("vectors_external", schema.clone())
            .unwrap();

        let num_batches = VECTOR_COUNT / VECTOR_BATCH_SIZE;
        let mut batches = Vec::with_capacity(num_batches);
        for batch_idx in 0..num_batches {
            let start_id = (batch_idx * VECTOR_BATCH_SIZE) as u64;
            let batch = generate_vector_batch(schema.clone(), start_id, VECTOR_BATCH_SIZE);
            let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
            batches.push(batch_with_mvcc);
        }
        store.append_many(table_id, batches).unwrap();

        let mut rng = rand::rng();
        let query_vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.random::<f32>()).collect();

        (store, table_id, query_vector)
    }

    /// Benchmark read performance with external storage.
    pub fn run_read_benchmark_external(c: &mut Criterion) {
        let mut group = c.benchmark_group("parquet_vector_read_external");
        group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
        group.sample_size(10);

        let compressions = vec![(Compression::UNCOMPRESSED, "uncompressed")];

        for (compression, name) in compressions {
            let (store, table_id) = setup_store_for_read_external(compression);
            group.bench_with_input(
                BenchmarkId::from_parameter(name),
                &compression,
                |b, _comp| {
                    b.iter(|| {
                        let sum = bench_read_vectors(&store, table_id);
                        black_box(sum)
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark similarity search with external storage.
    pub fn run_similarity_benchmark_external(c: &mut Criterion) {
        let mut group = c.benchmark_group("parquet_vector_similarity_external");
        group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
        group.sample_size(10);

        let compressions = vec![(Compression::UNCOMPRESSED, "uncompressed")];

        for (compression, name) in compressions {
            let (store, table_id, query_vector) = setup_store_for_similarity_external(compression);
            group.bench_with_input(
                BenchmarkId::from_parameter(name),
                &compression,
                |b, _comp| {
                    b.iter(|| {
                        let (count, similarity) =
                            bench_vector_similarity_search(&store, table_id, &query_vector);
                        black_box((count, similarity))
                    });
                },
            );
        }

        group.finish();
    }
}

// ============================================================================
// Flat in-memory baseline (no Parquet/Arrow overhead)
// ============================================================================

mod flat_baseline {
    use super::*;

    /// Pure brute-force cosine similarity on flat Vec<f32>.
    fn bench_cosine_bruteforce(data: &[f32], query: &[f32]) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for i in 0..VECTOR_COUNT {
            let start = i * VECTOR_DIM;
            let slice = &data[start..start + VECTOR_DIM];

            let sim =
                simsimd::SpatialSimilarity::cosine(black_box(query), black_box(slice)).unwrap();

            if sim > best {
                best = sim;
            }
        }
        best
    }

    /// Benchmark entry point for flat baseline.
    pub fn run_baseline_benchmark(c: &mut Criterion) {
        let mut rng = rand::rng();

        let mut data = vec![0.0f32; VECTOR_COUNT * VECTOR_DIM];
        for x in &mut data {
            *x = rng.random();
        }

        let query: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.random()).collect();

        let mut group = c.benchmark_group("flat_cosine_bruteforce");
        group.throughput(Throughput::Elements(VECTOR_COUNT as u64));

        group.bench_function("baseline", |b| {
            b.iter(|| {
                let best = bench_cosine_bruteforce(&data, &query);
                black_box(best);
            });
        });

        group.finish();
    }
}

// ============================================================================
// Criterion entry points
// ============================================================================

criterion_group!(
    benches,
    // -- flat_baseline::run_baseline_benchmark,
    // parquet_based::run_write_benchmark,
    // -- parquet_based::run_read_benchmark,
    parquet_based::run_read_benchmark_external,
    // parquet_based::run_similarity_benchmark,
    // parquet_based::run_similarity_benchmark_external,
    // -- parquet_based::run_compression_analysis
);
criterion_main!(benches);
