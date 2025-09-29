#![forbid(unsafe_code)]

use std::sync::Arc;
use std::hint::black_box;

use arrow::array::{Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{Criterion, criterion_group, criterion_main};

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_table::Table;
use llkv_storage::pager::MemPager;

const NUM_ROWS: usize = 1_000_000;
const TABLE_ID: llkv_table::types::TableId = 42;

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let row_id = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![row_id, field]))
}

fn setup_table() -> Table {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(TABLE_ID, Arc::clone(&pager)).expect("table creation");

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("field_id".to_string(), "1001".to_string());
    let data_field = Field::new("data", DataType::Utf8, false).with_metadata(metadata);
    let schema = schema_with_row_id(data_field);

    // Build 1M strings. We create predictable strings so tests are deterministic.
    let mut row_ids = Vec::with_capacity(NUM_ROWS);
    let mut values = Vec::with_capacity(NUM_ROWS);
    for i in 0..NUM_ROWS {
        row_ids.push(i as u64);
        // Make strings like "row-<i>-payload" where some contain the substring "needle".
        if i % 1000 == 0 {
            values.push(format!("row-{}-payload-needle", i));
        } else {
            values.push(format!("row-{}-payload", i));
        }
    }

    let row_id_array = Arc::new(arrow::array::UInt64Array::from(row_ids));
    let value_array = Arc::new(StringArray::from(values));

    let batch = RecordBatch::try_new(schema, vec![row_id_array, value_array]).expect("record batch");
    table.append(&batch).expect("append batch");

    table
}

fn bench_substring_scan(table: &Table) -> usize {
    // Scan the column by reading the projection directly from the store via scan_stream
    let mut found = 0usize;
    let projections = vec![llkv_column_map::store::Projection::from( llkv_column_map::types::LogicalFieldId::for_user(TABLE_ID, 1001))];
    let filter = llkv_expr::Expr::Pred(llkv_expr::Filter {
        field_id: 1001,
        op: llkv_expr::Operator::Range { lower: std::ops::Bound::Unbounded, upper: std::ops::Bound::Unbounded },
    });

    table
        .scan_stream(&projections, &filter, llkv_table::table::ScanStreamOptions::default(), |batch| {
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("projection yields StringArray");

            for i in 0..arr.len() {
                let s = arr.value(i);
                if s.contains("needle") {
                    found += 1;
                }
            }
        })
        .expect("scan_stream succeeds");

    found
}

fn bench_string_substring(c: &mut Criterion) {
    let table = setup_table();

    let mut group = c.benchmark_group("string_substring_search");
    group.sample_size(20);

    group.bench_function("scan_contains_needle", |b| {
        b.iter(|| {
            let n = bench_substring_scan(&table);
            // We expect roughly NUM_ROWS / 1000 matches
            assert_eq!(n, NUM_ROWS / 1000);
            black_box(n);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_string_substring);
criterion_main!(benches);
