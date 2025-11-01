use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[test]
fn sum_distinct_on_column_and_expression_supported() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE numbers(val INTEGER)")
        .expect("create numbers table");
    engine
        .execute("INSERT INTO numbers VALUES (1), (1), (2)")
        .expect("insert numbers");

    let mut results = engine
        .execute(
            "SELECT SUM(DISTINCT val) AS sum_distinct_val, SUM(DISTINCT + + 52) AS sum_distinct_const FROM numbers",
        )
        .expect("execute sum distinct query");

    assert_eq!(results.len(), 1, "expected single statement result");
    let select_result = results.remove(0);

    let batches = match select_result {
        RuntimeStatementResult::Select { execution, .. } => {
            execution.collect().expect("collect query batches")
        }
        other => panic!("expected select result, got {other:?}"),
    };
    assert_eq!(batches.len(), 1, "expected single record batch");
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "expected single result row");

    let sum_distinct_val = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("downcast sum_distinct_val to Int64Array");
    assert_eq!(sum_distinct_val.value(0), 3, "expected distinct column sum");

    let sum_distinct_const = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("downcast sum_distinct_const to Int64Array");
    assert_eq!(
        sum_distinct_const.value(0),
        52,
        "expected distinct constant sum"
    );
}
