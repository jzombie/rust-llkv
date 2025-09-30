#![forbid(unsafe_code)]

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{Criterion, criterion_group, criterion_main};

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{Expr, Filter, Operator};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::ScanStreamOptions;
use llkv_table::types::FieldId;

const NUM_ROWS: usize = 1_000_000;
const TABLE_ID: llkv_table::types::TableId = 42;
const FIELD_ID: FieldId = 1_001;

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let row_id = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![row_id, field]))
}

fn setup_table() -> Table {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(TABLE_ID, Arc::clone(&pager)).expect("table creation");

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("field_id".to_string(), FIELD_ID.to_string());
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

    let batch =
        RecordBatch::try_new(schema, vec![row_id_array, value_array]).expect("record batch");
    table.append(&batch).expect("append batch");

    table
}

fn bench_substring_scan(table: &Table) -> usize {
    let logical_field_id = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);
    let projections = vec![Projection::from(logical_field_id)];
    let filter = Expr::Pred(Filter {
        field_id: FIELD_ID,
        op: Operator::contains("needle", true),
    });

    let mut found = 0usize;
    table
        .scan_stream(
            &projections,
            &filter,
            ScanStreamOptions::default(),
            |batch| {
                found += batch.num_rows();
            },
        )
        .expect("scan_stream contains substring");

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
