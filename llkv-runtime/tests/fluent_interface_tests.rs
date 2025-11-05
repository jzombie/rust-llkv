use std::sync::Arc;

use arrow::datatypes::DataType;
use llkv_runtime::{
    AggregateExpr, InsertPlan, InsertSource, PlanValue, RuntimeContext, RuntimeStatementResult,
    TransactionKind, row,
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

    let rows = table
        .lazy()
        .expect("lazy scan")
        .select_columns(["id", "name"])
        .collect_rows_vec()
        .expect("collect rows");

    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows,
        vec![
            vec![PlanValue::Integer(1), PlanValue::String("alice".into())],
            vec![PlanValue::Integer(2), PlanValue::String("bob".into())]
        ]
    );
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
    };
    let insert_result = session.execute_insert_plan(insert_plan).expect("insert");
    assert!(matches!(
        insert_result,
        RuntimeStatementResult::Insert {
            rows_inserted: 1,
            ..
        }
    ));

    let rows = session.table_rows("numbers").expect("session rows");
    assert_eq!(rows, vec![vec![PlanValue::Integer(41)]]);

    let commit = session.commit_transaction().expect("commit");
    assert!(matches!(
        commit,
        RuntimeStatementResult::Transaction {
            kind: TransactionKind::Commit
        }
    ));

    let persisted_rows = table
        .lazy()
        .expect("lazy scan")
        .select_columns(["n"])
        .collect_rows_vec()
        .expect("rows");
    assert_eq!(persisted_rows, vec![vec![PlanValue::Integer(41)]]);

    let aggregate_rows = table
        .lazy()
        .expect("lazy scan")
        .aggregate(vec![AggregateExpr::sum_int64("n", "sum")])
        .collect_rows_vec()
        .expect("aggregate rows");
    assert_eq!(aggregate_rows, vec![vec![PlanValue::Integer(41)]]);
}
