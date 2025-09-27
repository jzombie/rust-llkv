#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::compute;
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
use llkv_table::types::{FieldId, TableId};

const NUM_ROWS: usize = 1_000_000;
const TABLE_ID: TableId = 42;
const FIELD_ID: FieldId = 1_001;

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let row_id = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![row_id, field]))
}

fn expected_sum() -> u128 {
    let n = NUM_ROWS as u128;
    n * (n - 1) / 2
}

fn setup_table() -> Table {
    let pager = Arc::new(MemPager::default());
    let table = Table::new(TABLE_ID, Arc::clone(&pager)).expect("table creation");

    let mut metadata = HashMap::new();
    metadata.insert("field_id".to_string(), FIELD_ID.to_string());
    let data_field = Field::new("data", DataType::UInt64, false).with_metadata(metadata);
    let schema = schema_with_row_id(data_field);

    let row_ids: Vec<u64> = (0..NUM_ROWS as u64).collect();
    let values: Vec<u64> = row_ids.clone();

    let row_id_array = Arc::new(UInt64Array::from(row_ids));
    let value_array = Arc::new(UInt64Array::from(values));

    let batch =
        RecordBatch::try_new(schema, vec![row_id_array, value_array]).expect("record batch");
    table.append(&batch).expect("append batch");

    table
}

fn scan_sum(table: &Table, projections: &[Projection], filter: &Expr<'static, FieldId>) -> u128 {
    let mut total: u128 = 0;
    table
        .scan_stream(projections, filter, ScanStreamOptions::default(), |batch| {
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("projection yields UInt64Array");
            if let Some(partial) = compute::sum(arr) {
                total += partial as u128;
            }
        })
        .expect("scan_stream succeeds");

    total
}

fn bench_table_sum(c: &mut Criterion) {
    let table = setup_table();
    let projections = vec![Projection::from(LogicalFieldId::for_user(
        TABLE_ID, FIELD_ID,
    ))];
    let filter: Expr<'static, FieldId> = Expr::Pred(Filter {
        field_id: FIELD_ID,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });
    let expected = expected_sum();

    let mut group = c.benchmark_group("llkv_table_sum_1M");
    group.sample_size(20);

    group.bench_function("scan_stream_sum_u64", |b| {
        b.iter(|| {
            let total = scan_sum(&table, &projections, &filter);
            assert_eq!(total, expected);
            black_box(total);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_table_sum);
criterion_main!(benches);
