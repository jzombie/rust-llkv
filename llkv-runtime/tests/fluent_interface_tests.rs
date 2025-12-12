use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::DataType;
use llkv_runtime::{
    AggregateExpr, InsertConflictAction, InsertPlan, InsertSource, PlanValue, RuntimeContext,
    RuntimeStatementResult, TransactionKind, row,
};
use llkv_storage::pager::{BoxedPager, MemPager};

#[test]
fn fluent_create_insert_select() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));

    let create_result = context
        .create_table_builder("people")
        .with_column("id", DataType::Int64)
        .with_column("name", DataType::Utf8)
        .finish()
        .expect("create table");
    assert!(matches!(
        create_result,
        RuntimeStatementResult::CreateTable { .. }
    ));

    let table = context.table("people").expect("table handle");
    let insert_result = table
        .insert_rows([
            row().with("id", 1_i64).with("name", "alice"),
            row().with("id", 2_i64).with("name", "bob"),
        ])
        .expect("insert rows");
    assert!(matches!(
        insert_result,
        RuntimeStatementResult::Insert {
            rows_inserted: 2,
            ..
        }
    ));

    let batches = table
        .lazy()
        .expect("lazy scan")
        .select_columns(["id", "name"])
        .collect_batches()
        .expect("collect batches");

    let schema = batches
        .first()
        .map(|b| b.schema())
        .expect("non-empty batches");
    let combined = arrow::compute::concat_batches(&schema, batches.iter()).expect("concat batches");
    let ids = combined
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id column");
    let names = combined
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name column");

    let mut rows = Vec::new();
    for idx in 0..combined.num_rows() {
        rows.push((ids.value(idx), names.value(idx).to_string()));
    }

    assert_eq!(rows, vec![(1, "alice".into()), (2, "bob".into())]);
}

#[test]
fn fluent_transaction_flow() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));
    let session = context.create_session();

    context
        .create_table_builder("numbers")
        .with_column("n", DataType::Int64)
        .finish()
        .expect("create table");

    let table = context.table("numbers").expect("table handle");

    session.begin_transaction().expect("begin tx");

    let insert_plan = InsertPlan {
        table: "numbers".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![vec![PlanValue::Integer(41)]]),
        on_conflict: InsertConflictAction::None,
    };
    let insert_result = session.execute_insert_plan(insert_plan).expect("insert");
    assert!(matches!(
        insert_result,
        RuntimeStatementResult::Insert {
            rows_inserted: 1,
            ..
        }
    ));

    let rows = session.table_batches("numbers").expect("session rows");
    let row_schema = rows.first().map(|b| b.schema()).expect("non-empty batches");
    let combined =
        arrow::compute::concat_batches(&row_schema, rows.iter()).expect("concat batches");
    let values = combined
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("numbers column");
    assert_eq!(values.len(), 1);
    assert_eq!(values.value(0), 41);

    let commit = session.commit_transaction().expect("commit");
    assert!(matches!(
        commit,
        RuntimeStatementResult::Transaction {
            kind: TransactionKind::Commit
        }
    ));

    let persisted_batches = table
        .lazy()
        .expect("lazy scan")
        .select_columns(["n"])
        .collect_batches()
        .expect("rows");
    let persisted_schema = persisted_batches
        .first()
        .map(|b| b.schema())
        .expect("non-empty batches");
    let persisted = arrow::compute::concat_batches(&persisted_schema, persisted_batches.iter())
        .expect("concat batches");
    let persisted_values = persisted
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("numbers column");
    assert_eq!(persisted_values.len(), 1);
    assert_eq!(persisted_values.value(0), 41);

    let aggregate_batches = table
        .lazy()
        .expect("lazy scan")
        .aggregate(vec![AggregateExpr::sum_int64("n", "sum")])
        .collect_batches()
        .expect("aggregate rows");
    let aggregate_schema = aggregate_batches
        .first()
        .map(|b| b.schema())
        .expect("non-empty batches");
    let aggregate = arrow::compute::concat_batches(&aggregate_schema, aggregate_batches.iter())
        .expect("concat batches");
    let sum_values = aggregate
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("sum column");
    assert_eq!(sum_values.len(), 1);
    assert_eq!(sum_values.value(0), 41);
}
