use arrow::compute::{filter, sum};
use arrow_array::{Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use llkv_arrowhead::{ColumnMap, KeySpec, ReadCodec, ReadSpec, batch_to_puts, read_record_batch};
use llkv_column_map::ColumnStore;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, LogicalKeyBytes, ValueMode};
use std::hint::black_box;
use std::sync::Arc;

// NEW: scalar comparisons live in arrow-ord
use arrow_ord::cmp::gt;

// ==================================
//          BENCHMARK SETUP
// ==================================

const NUM_ROWS: usize = 1_000_000;
// const COL_KEY: LogicalFieldId = 0; // TODO: Remove?
const COL_VAL1: LogicalFieldId = 1;
const COL_VAL2: LogicalFieldId = 2;

/// Creates an Arrow RecordBatch with 1 million rows.
/// - column 0 (key): u64 values from 0 to 999,999
/// - column 1 (val1): i64 values from 1,000,000 down to 1
/// - column 2 (val2): u64 values, repeating pattern of (i % 100)
fn create_test_batch(num_rows: usize) -> (RecordBatch, Arc<Schema>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt64, false),
        Field::new("val1", DataType::Int64, false),
        Field::new("val2", DataType::UInt64, false),
    ]));

    let keys: Vec<u64> = (0..num_rows as u64).collect();
    let val1: Vec<i64> = (0..num_rows as i64).rev().map(|i| i + 1).collect();
    let val2: Vec<u64> = (0..num_rows as u64).map(|i| i % 100).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(keys)),
            Arc::new(Int64Array::from(val1)),
            Arc::new(UInt64Array::from(val2)),
        ],
    )
    .unwrap();

    (batch, schema)
}

/// Sets up the store and ingests the test data.
fn setup_store_with_data(num_rows: usize) -> (ColumnStore<'static, MemPager>, RecordBatch) {
    let (batch, _) = create_test_batch(num_rows);

    // Use 'static lifetime for the pager in the benchmark context.
    let pager = Box::leak(Box::new(MemPager::new()));
    let store = ColumnStore::init_empty(pager);

    let key_spec = KeySpec::U64Be { col_idx: 0 };
    let column_map = ColumnMap {
        cols: vec![(1, COL_VAL1), (2, COL_VAL2)],
    };

    let puts = batch_to_puts(&batch, &key_spec, &column_map);
    store.append_many(
        puts,
        AppendOptions {
            mode: ValueMode::Auto,
            segment_max_entries: 65536, // Create a few segments
            ..Default::default()
        },
    );

    (store, batch)
}

// ==================================
//            BENCHMARKS
// ==================================

fn bench_ingest(c: &mut Criterion) {
    let (batch, _) = create_test_batch(NUM_ROWS);
    let key_spec = KeySpec::U64Be { col_idx: 0 };
    let column_map = ColumnMap {
        cols: vec![(1, COL_VAL1), (2, COL_VAL2)],
    };

    // Chain .sample_size() to the benchmark definition
    c.benchmark_group("ingestion")
        .sample_size(10) // Set the number of samples to 10
        .bench_function("1_ingest_1m_rows", |b| {
            b.iter(|| {
                let pager = MemPager::new();
                let store = ColumnStore::init_empty(&pager);
                let puts = batch_to_puts(black_box(&batch), &key_spec, &column_map);
                store.append_many(black_box(puts), Default::default());
            })
        });
}

fn bench_read(c: &mut Criterion) {
    let (store, batch) = setup_store_with_data(NUM_ROWS);
    let key_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let keys: Vec<LogicalKeyBytes> = key_col
        .values()
        .iter()
        .map(|k| k.to_be_bytes().to_vec())
        .collect();

    let read_specs = vec![
        ReadSpec {
            field_id: COL_VAL1,
            field: Field::new("val1", DataType::Int64, true),
            codec: ReadCodec::I64Le,
        },
        ReadSpec {
            field_id: COL_VAL2,
            field: Field::new("val2", DataType::UInt64, true),
            codec: ReadCodec::U64Le,
        },
    ];

    c.bench_function("2_read_1m_rows", |b| {
        b.iter(|| {
            let _batch =
                read_record_batch(black_box(&store), black_box(&keys), black_box(&read_specs))
                    .unwrap();
        })
    });
}

fn bench_olap_query(c: &mut Criterion) {
    let (store, batch) = setup_store_with_data(NUM_ROWS);
    let key_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let keys: Vec<LogicalKeyBytes> = key_col
        .values()
        .iter()
        .map(|k| k.to_be_bytes().to_vec())
        .collect();

    let read_specs = vec![
        ReadSpec {
            field_id: COL_VAL1,
            field: Field::new("val1", DataType::Int64, true),
            codec: ReadCodec::I64Le,
        },
        ReadSpec {
            field_id: COL_VAL2,
            field: Field::new("val2", DataType::UInt64, true),
            codec: ReadCodec::U64Le,
        },
    ];

    c.bench_function("3_olap_query_filter_sum_1m_rows", |b| {
        b.iter(|| {
            // Step 1: materialize to Arrow
            let batch =
                read_record_batch(black_box(&store), black_box(&keys), black_box(&read_specs))
                    .unwrap();

            // Step 2: OLAP on Arrow
            let val1_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let val2_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();

            // Filter: val2 > 50 (scalar compare via arrow-ord)
            let scalar = UInt64Array::new_scalar(50u64);
            let filter_mask = gt(val2_col, &scalar).unwrap();

            // Apply mask and aggregate
            let filtered_val1 = filter(val1_col, &filter_mask).unwrap();
            let sum_val =
                sum(filtered_val1.as_any().downcast_ref::<Int64Array>().unwrap()).unwrap();

            black_box(sum_val);
        })
    });
}

criterion_group!(benches, bench_ingest, bench_read, bench_olap_query);
criterion_main!(benches);
