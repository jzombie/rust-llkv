#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::hint::black_box;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Float64Array, UInt64Array};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use criterion::{Criterion, criterion_group, criterion_main};

use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::Projection;
use llkv_expr::{BinaryOp, Expr, Filter, Operator, ScalarExpr};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::table::{ScanProjection, ScanStreamOptions};
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

fn scan_sum_projection(
    table: &Table,
    projections: &[Projection],
    filter: &Expr<'static, FieldId>,
) -> u128 {
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

fn scan_expr_total(
    table: &Table,
    expr: &ScalarExpr<FieldId>,
    filter: &Expr<'static, FieldId>,
) -> f64 {
    let projections = vec![ScanProjection::computed(expr.clone(), "expr")];
    let mut total: f64 = 0.0;

    table
        .scan_stream_with_exprs(
            &projections,
            filter,
            ScanStreamOptions::default(),
            |batch| {
                let column = batch.column(0);
                let any = column.as_any();
                if let Some(arr) = any.downcast_ref::<Float64Array>() {
                    if let Some(partial) = compute::sum(arr) {
                        total += partial;
                    }
                } else if let Some(arr) = any.downcast_ref::<UInt64Array>() {
                    if let Some(partial) = compute::sum(arr) {
                        total += partial as f64;
                    }
                } else {
                    panic!(
                        "computed projection yielded unexpected array type: {:?}",
                        column.data_type()
                    );
                }
            },
        )
        .expect("scan_stream_with_exprs succeeds");

    total
}

fn approx_eq(a: f64, b: f64) -> bool {
    let tol = b.abs() * 1e-9 + 0.5; // half-unit slack for integer sums represented as f64.
    (a - b).abs() <= tol
}

fn bench_table_math(c: &mut Criterion) {
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

    let expected_u128 = expected_sum();
    let expected_f64 = expected_u128 as f64;
    let expected_add = expected_f64 + NUM_ROWS as f64;
    let expected_sub = expected_f64 - NUM_ROWS as f64;
    let expected_mul = expected_f64 * 2.0;
    let expected_div = expected_f64 / 2.0;

    let expr_add_one = ScalarExpr::binary(
        ScalarExpr::column(FIELD_ID),
        BinaryOp::Add,
        ScalarExpr::literal(1_i64),
    );
    let expr_sub_one = ScalarExpr::binary(
        ScalarExpr::column(FIELD_ID),
        BinaryOp::Subtract,
        ScalarExpr::literal(1_i64),
    );
    let expr_mul_two = ScalarExpr::binary(
        ScalarExpr::column(FIELD_ID),
        BinaryOp::Multiply,
        ScalarExpr::literal(2_i64),
    );
    let expr_div_two = ScalarExpr::binary(
        ScalarExpr::column(FIELD_ID),
        BinaryOp::Divide,
        ScalarExpr::literal(2_i64),
    );

    let mut group = c.benchmark_group("llkv_table_math_1M");
    group.sample_size(20);

    group.bench_function("scan_stream_sum_u64", |b| {
        b.iter(|| {
            let total = scan_sum_projection(&table, &projections, &filter);
            assert_eq!(total, expected_u128);
            black_box(total);
        });
    });

    group.bench_function("scan_stream_expr_add_one", |b| {
        b.iter(|| {
            let total = scan_expr_total(&table, &expr_add_one, &filter);
            assert!(
                approx_eq(total, expected_add),
                "add_one total={total} expected={expected_add}"
            );
            black_box(total);
        });
    });

    group.bench_function("scan_stream_expr_subtract_one", |b| {
        b.iter(|| {
            let total = scan_expr_total(&table, &expr_sub_one, &filter);
            assert!(
                approx_eq(total, expected_sub),
                "subtract_one total={total} expected={expected_sub}"
            );
            black_box(total);
        });
    });

    group.bench_function("scan_stream_expr_multiply_two", |b| {
        b.iter(|| {
            let total = scan_expr_total(&table, &expr_mul_two, &filter);
            assert!(
                approx_eq(total, expected_mul),
                "multiply_two total={total} expected={expected_mul}"
            );
            black_box(total);
        });
    });

    group.bench_function("scan_stream_expr_divide_two", |b| {
        b.iter(|| {
            let total = scan_expr_total(&table, &expr_div_two, &filter);
            assert!(
                approx_eq(total, expected_div),
                "divide_two total={total} expected={expected_div}"
            );
            black_box(total);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_table_math);
criterion_main!(benches);
