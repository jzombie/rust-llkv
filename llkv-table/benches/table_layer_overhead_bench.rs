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
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_expr::{Expr, Filter, Operator};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::ScanStreamOptions;
use llkv_table::types::{FieldId, TableId};
use llkv_types::LogicalFieldId;

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
    let table = Table::from_id(TABLE_ID, Arc::clone(&pager)).expect("table creation");

    let mut metadata = HashMap::new();
    metadata.insert(
        llkv_table::constants::FIELD_ID_META_KEY.to_string(),
        FIELD_ID.to_string(),
    );
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

// Fast path: Direct column store access for simple aggregations
fn bench_direct_columnstore_sum(table: &Table) -> u128 {
    struct SumVisitor {
        total: u128,
    }

    impl SumVisitor {
        fn new() -> Self {
            Self { total: 0 }
        }
    }

    impl PrimitiveVisitor for SumVisitor {
        fn u64_chunk(&mut self, a: &UInt64Array) {
            if let Some(s) = compute::sum(a) {
                self.total += s as u128;
            }
        }
    }

    impl PrimitiveSortedVisitor for SumVisitor {}
    impl PrimitiveWithRowIdsVisitor for SumVisitor {}
    impl PrimitiveSortedWithRowIdsVisitor for SumVisitor {}

    let mut visitor = SumVisitor::new();
    let logical_field_id = LogicalFieldId::for_user(TABLE_ID, FIELD_ID);

    table
        .store()
        .scan(
            logical_field_id,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: false,
                limit: None,
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut visitor,
        )
        .expect("scan succeeds");

    visitor.total
}

// Current table layer implementation
fn bench_table_layer_sum(table: &Table) -> u128 {
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

    let mut total: u128 = 0;
    table
        .scan_stream(
            &projections,
            &filter,
            ScanStreamOptions::default(),
            |batch| {
                let arr = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .expect("projection yields UInt64Array");
                if let Some(partial) = compute::sum(arr) {
                    total += partial as u128;
                }
            },
        )
        .expect("scan_stream succeeds");

    total
}

fn bench_table_performance_comparison(c: &mut Criterion) {
    let table = setup_table();
    let expected = expected_sum();

    let mut group = c.benchmark_group("table_layer_overhead_analysis");
    group.sample_size(100);

    group.bench_function("direct_columnstore_sum", |b| {
        b.iter(|| {
            let total = bench_direct_columnstore_sum(&table);
            assert_eq!(total, expected);
            black_box(total);
        });
    });

    group.bench_function("table_layer_sum", |b| {
        b.iter(|| {
            let total = bench_table_layer_sum(&table);
            assert_eq!(total, expected);
            black_box(total);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_table_performance_comparison);
criterion_main!(benches);
