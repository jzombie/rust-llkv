use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[test]
fn select_without_from_handles_constant_expression() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let batches = engine
        .sql("SELECT 20 / - - 96 + CAST ( 90 AS INTEGER ) AS col2")
        .expect("execute constant expression select");

    assert_eq!(batches.len(), 1, "expected single record batch");
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "expected single result row");
    assert_eq!(batch.num_columns(), 1, "expected single projection");

    assert_eq!(
        value_as_i64(batch.column(0), 0),
        Some(90),
        "20 / - - 96 + CAST(90 AS INTEGER) should yield 90"
    );
}

#[test]
fn select_without_from_handles_logical_and() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let batches = engine
        .sql("SELECT 1 AND 1 AS all_true, 1 AND 0 AS mixed, NULL AND 1 AS null_branch")
        .expect("execute logical AND select");

    assert_eq!(batches.len(), 1, "expected single record batch");
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "expected single result row");
    assert_eq!(batch.num_columns(), 3, "expected three projections");

    assert_eq!(
        value_as_i64(batch.column(0), 0),
        Some(1),
        "1 AND 1 should yield 1"
    );

    assert_eq!(
        value_as_i64(batch.column(1), 0),
        Some(0),
        "1 AND 0 should yield 0"
    );

    assert!(
        value_as_i64(batch.column(2), 0).is_none(),
        "NULL AND 1 should yield NULL"
    );
}

#[test]
fn select_without_from_handles_logical_or() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let batches = engine
        .sql("SELECT 0 OR 0 AS all_false, 0 OR 1 AS truthy, 0 OR NULL AS false_or_null, NULL OR 1 AS null_or_true")
        .expect("execute logical OR select");

    assert_eq!(batches.len(), 1, "expected single record batch");
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "expected single result row");
    assert_eq!(batch.num_columns(), 4, "expected four projections");

    assert_eq!(
        value_as_i64(batch.column(0), 0),
        Some(0),
        "0 OR 0 should yield 0"
    );

    assert_eq!(
        value_as_i64(batch.column(1), 0),
        Some(1),
        "0 OR 1 should yield 1"
    );

    assert!(
        value_as_i64(batch.column(2), 0).is_none(),
        "0 OR NULL should yield NULL"
    );

    assert_eq!(
        value_as_i64(batch.column(3), 0),
        Some(1),
        "NULL OR 1 should yield 1"
    );
}

fn value_as_i64(column: &ArrayRef, idx: usize) -> Option<i64> {
    if matches!(column.data_type(), DataType::Null) {
        return None;
    }

    if column.is_null(idx) {
        return None;
    }

    if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
        return Some(values.value(idx));
    }

    let casted = cast(column, &DataType::Int64).expect("cast value to Int64");
    let values = casted
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("cast produced Int64Array");
    Some(values.value(idx))
}
