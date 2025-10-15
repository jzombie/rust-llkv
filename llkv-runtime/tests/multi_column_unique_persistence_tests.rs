use std::sync::Arc;

use arrow::datatypes::DataType;
use llkv_result::Error;
use llkv_runtime::{
    CreateIndexPlan, IndexColumnPlan, PlanValue, RuntimeContext, RuntimeStatementResult, row,
};
use llkv_storage::pager::MemPager;

#[test]
fn multi_column_unique_survives_restart() {
    let pager = Arc::new(MemPager::default());
    let context = Arc::new(RuntimeContext::new(Arc::clone(&pager)));

    context
        .create_table_builder("people")
        .with_column("first_name", DataType::Utf8)
        .with_column("last_name", DataType::Utf8)
        .with_column("age", DataType::Int64)
        .finish()
        .expect("create table");

    let table = context.table("people").expect("table handle");
    table
        .insert_rows([
            row()
                .with("first_name", "Ada")
                .with("last_name", "Lovelace")
                .with("age", 36_i64),
            row()
                .with("first_name", "Grace")
                .with("last_name", "Hopper")
                .with("age", 44_i64),
        ])
        .expect("initial insert");

    let create_index_plan = CreateIndexPlan::new("people")
        .with_name(Some("uniq_fullname".into()))
        .with_unique(true)
        .with_columns(vec![
            IndexColumnPlan::new("first_name"),
            IndexColumnPlan::new("last_name"),
        ]);

    let index_result = context
        .create_index(create_index_plan)
        .expect("create index");
    assert!(matches!(
        index_result,
        RuntimeStatementResult::CreateIndex { .. }
    ));

    drop(table);
    drop(context);

    let context = Arc::new(RuntimeContext::new(pager));
    let table = context.table("people").expect("table handle");

    let duplicate = table.insert_rows([row()
        .with("first_name", "Ada")
        .with("last_name", "Lovelace")
        .with("age", 40_i64)]);

    match duplicate {
        Err(Error::ConstraintError(message)) => assert!(
            message.contains("first_name") && message.contains("last_name"),
            "unexpected message: {message}"
        ),
        other => panic!("expected constraint error, got {:?}", other),
    }

    let distinct = table
        .insert_rows([row()
            .with("first_name", "Ada")
            .with("last_name", "Byron")
            .with("age", 40_i64)])
        .expect("insert distinct");
    assert!(matches!(
        distinct,
        RuntimeStatementResult::Insert {
            rows_inserted: 1,
            ..
        }
    ));

    let rows = table
        .lazy()
        .expect("lazy scan")
        .select_columns(["first_name", "last_name", "age"])
        .collect_rows_vec()
        .expect("collect rows");
    assert!(rows.contains(&vec![
        PlanValue::String("Ada".into()),
        PlanValue::String("Byron".into()),
        PlanValue::Integer(40),
    ]));
}
