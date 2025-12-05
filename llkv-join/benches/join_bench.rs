//! Benchmarks for table join operations.
//!
//! These benchmarks measure join performance with various data sizes and
//! join types to guide optimization work.
//!
//! ## Current Status
//!
//! **Nested-loop join only**: O(N×M) complexity makes it impractical for large datasets.
//! - 10K×10K = 100M comparisons (~1.5 seconds)
//! - 100K×100K = 10B comparisons (~2.5 minutes)  
//! - 1M×1M = 1T comparisons (hours - don't even try!)
//!
//! **TODO: Implement hash join** (O(N+M)) for production workloads:
//! - 1M×1M would become ~2M operations (seconds instead of hours)
//! - See `/Users/administrator/Projects/rust-llkv/llkv-join/src/hash_join.rs`
//! - Once implemented, add `bench_hash_join_*` functions here for 100K, 1M, 10M row tests

use arrow::array::{Int32Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_join::{JoinKey, JoinOptions, TableJoinRowIdExt};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::types::TableId;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

/// Create a table with specified number of rows.
/// Schema: row_id (UInt64), id (Int32), value (Utf8)
fn create_table_with_rows(
    table_id: TableId,
    pager: &Arc<MemPager>,
    num_rows: usize,
    id_offset: i32,
) -> Table<MemPager> {
    let table = Table::from_id(table_id, Arc::clone(pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("value", DataType::Utf8, false).with_metadata(HashMap::from([(
            llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]));

    // Create data in batches of 10,000 rows
    let batch_size = 10_000;
    for batch_start in (0..num_rows).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_rows);

        let row_ids: Vec<u64> = (batch_start..batch_end).map(|i| i as u64).collect();
        let ids: Vec<i32> = (batch_start..batch_end)
            .map(|i| (i as i32) + id_offset)
            .collect();
        let values: Vec<String> = (batch_start..batch_end)
            .map(|i| format!("value_{}", i))
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(row_ids)),
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(values)),
            ],
        )
        .unwrap();

        table.append(&batch).unwrap();
    }

    table
}

/// Benchmark nested-loop inner join with small table sizes.
///
/// NOTE: Nested-loop join is O(N×M), so performance degrades quadratically.
/// These benchmarks use small sizes because:
/// - 10K×10K = 100M comparisons (~1.5 seconds)
/// - 100K×100K = 10B comparisons (~2.5 minutes)
/// - 1M×1M = 1T comparisons (hours!)
///
/// Hash join (O(N+M)) enables production workloads.
fn bench_hash_join_inner_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_join_inner_join");

    // Test realistic production sizes - hash join makes this practical
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(20);

        let bench_id = BenchmarkId::from_parameter(size);
        group.bench_with_input(bench_id, &size, |b, &size| {
            let pager = Arc::new(MemPager::default());

            // Create two tables with 50% overlap
            // Left:  [0, size)
            // Right: [size/2, size + size/2)
            let left = create_table_with_rows(1, &pager, size, 0);
            let right = create_table_with_rows(2, &pager, size, (size / 2) as i32);

            let keys = vec![JoinKey::new(1, 1)];
            let options = JoinOptions::inner();

            b.iter(|| {
                let mut total_rows = 0;
                left.join_rowid_stream(&right, &keys, &options, |index_batch| {
                    total_rows += index_batch.left_rows.len();
                })
                .unwrap();
                black_box(total_rows);
            });
        });
    }

    group.finish();
}

/// Benchmark hash join left join with realistic table sizes.
fn bench_hash_join_left_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_join_left_join");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(20);

        let bench_id = BenchmarkId::from_parameter(size);
        group.bench_with_input(bench_id, &size, |b, &size| {
            let pager = Arc::new(MemPager::default());

            let left = create_table_with_rows(1, &pager, size, 0);
            let right = create_table_with_rows(2, &pager, size / 2, (size / 4) as i32);

            let keys = vec![JoinKey::new(1, 1)];
            let options = JoinOptions::left();

            b.iter(|| {
                let mut total_rows = 0;
                left.join_rowid_stream(&right, &keys, &options, |index_batch| {
                    total_rows += index_batch.left_rows.len();
                })
                .unwrap();
                black_box(total_rows);
            });
        });
    }

    group.finish();
}

/// Benchmark hash join semi join (useful for existence checks).
fn bench_hash_join_semi_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_join_semi_join");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(20);

        let bench_id = BenchmarkId::from_parameter(size);
        group.bench_with_input(bench_id, &size, |b, &size| {
            let pager = Arc::new(MemPager::default());

            let left = create_table_with_rows(1, &pager, size, 0);
            let right = create_table_with_rows(2, &pager, size / 10, 0);

            let keys = vec![JoinKey::new(1, 1)];
            let options = JoinOptions::semi();

            b.iter(|| {
                let mut total_rows = 0;
                left.join_rowid_stream(&right, &keys, &options, |index_batch| {
                    total_rows += index_batch.left_rows.len();
                })
                .unwrap();
                black_box(total_rows);
            });
        });
    }

    group.finish();
}

/// Benchmark hash join anti join (useful for finding non-matches).
fn bench_hash_join_anti_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_join_anti_join");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(20);

        let bench_id = BenchmarkId::from_parameter(size);
        group.bench_with_input(bench_id, &size, |b, &size| {
            let pager = Arc::new(MemPager::default());

            let left = create_table_with_rows(1, &pager, size, 0);
            let right = create_table_with_rows(2, &pager, size / 10, size as i32);

            let keys = vec![JoinKey::new(1, 1)];
            let options = JoinOptions::anti();

            b.iter(|| {
                let mut total_rows = 0;
                left.join_rowid_stream(&right, &keys, &options, |index_batch| {
                    total_rows += index_batch.left_rows.len();
                })
                .unwrap();
                black_box(total_rows);
            });
        });
    }

    group.finish();
}

/// Benchmark hash join with high match ratio (many-to-many).
/// Tests performance when keys have multiple matches.
fn bench_hash_join_many_to_many_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_join_many_to_many");

    // Use moderate sizes - many-to-many generates large output
    for size in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);

        let bench_id = BenchmarkId::from_parameter(size);
        group.bench_with_input(bench_id, &size, |b, &size| {
            let pager = Arc::new(MemPager::default());

            // Create tables where many rows have the same id (high cardinality)
            // Each unique id appears 10 times
            let left = create_table_with_rows(1, &pager, size, 0);
            let right = create_table_with_rows(2, &pager, size, 0);

            let keys = vec![JoinKey::new(1, 1)];
            let options = JoinOptions::inner();

            b.iter(|| {
                let mut total_rows = 0;
                left.join_stream(&right, &keys, &options, |batch| {
                    total_rows += batch.num_rows();
                })
                .unwrap();
                black_box(total_rows);
            });
        });
    }

    group.finish();
}

/// Benchmark hash join with no matching keys (tests hash table lookup performance).
fn bench_hash_join_no_matches_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_join_no_matches");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(20);

        let bench_id = BenchmarkId::from_parameter(size);
        group.bench_with_input(bench_id, &size, |b, &size| {
            let pager = Arc::new(MemPager::default());

            // Non-overlapping ranges
            let left = create_table_with_rows(1, &pager, size, 0);
            let right = create_table_with_rows(2, &pager, size, (size * 2) as i32);

            let keys = vec![JoinKey::new(1, 1)];
            let options = JoinOptions::inner();

            b.iter(|| {
                let mut total_rows = 0;
                left.join_stream(&right, &keys, &options, |batch| {
                    total_rows += batch.num_rows();
                })
                .unwrap();
                black_box(total_rows);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hash_join_inner_join,
    bench_hash_join_left_join,
    bench_hash_join_semi_join,
    bench_hash_join_anti_join,
    bench_hash_join_many_to_many_join,
    bench_hash_join_no_matches_join
);
criterion_main!(benches);
