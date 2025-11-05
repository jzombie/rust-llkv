use llkv_plan::PlanValue;
use llkv_runtime::{
    AggregateExpr, CreateTablePlan, InsertConflictAction, InsertPlan, InsertSource,
    PlanColumnSpec, RuntimeContext, RuntimeStatementResult, SelectPlan,
};
use llkv_storage::pager::{BoxedPager, MemPager};
use llkv_table::CatalogDdl;
use std::sync::Arc;

#[test]
fn test_transaction_select() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let ctx = Arc::new(RuntimeContext::new(Arc::clone(&pager)));
    let session = ctx.create_session();

    // Create table
    let create_plan = CreateTablePlan {
        name: "users".into(),
        columns: vec![
            PlanColumnSpec::new("id", arrow::datatypes::DataType::Int64, true),
            PlanColumnSpec::new("name", arrow::datatypes::DataType::Utf8, true),
        ],
        if_not_exists: false,
        or_replace: false,
        source: None,
        namespace: None,
        foreign_keys: Vec::new(),
        multi_column_uniques: Vec::new(),
    };
    CatalogDdl::create_table(&session, create_plan).unwrap();

    // Insert data
    let insert_plan = InsertPlan {
        table: "users".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![
            vec![PlanValue::Integer(1), PlanValue::String("Alice".into())],
            vec![PlanValue::Integer(2), PlanValue::String("Bob".into())],
        ]),
        on_conflict: InsertConflictAction::None,
    };
    session.execute_insert_plan(insert_plan).unwrap();

    // Begin transaction
    session.begin_transaction().unwrap();

    // Insert more data in transaction
    let insert_plan2 = InsertPlan {
        table: "users".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![vec![
            PlanValue::Integer(3),
            PlanValue::String("Charlie".into()),
        ]]),
        on_conflict: InsertConflictAction::None,
    };
    session.execute_insert_plan(insert_plan2).unwrap();

    // Select all rows (should see 3)
    let select_plan = SelectPlan::new("users");
    let result1 = session.execute_select_plan(select_plan).unwrap();

    if let RuntimeStatementResult::Select { execution, .. } = result1 {
        let batches = execution.collect().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
    } else {
        panic!("Expected SELECT result");
    }

    // Rollback
    session.rollback_transaction().unwrap();

    // Select again (should see only 2 - Charlie should be gone)
    let select_plan2 = SelectPlan::new("users");
    let result2 = session.execute_select_plan(select_plan2).unwrap();

    if let RuntimeStatementResult::Select { execution, .. } = result2 {
        let batches = execution.collect().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
    } else {
        panic!("Expected SELECT result");
    }
}

#[test]
fn test_transaction_select_with_aggregates() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let ctx = Arc::new(RuntimeContext::new(Arc::clone(&pager)));
    let session = ctx.create_session();

    // Create table
    let create_plan = CreateTablePlan {
        name: "products".into(),
        columns: vec![
            PlanColumnSpec::new("id", arrow::datatypes::DataType::Int64, true),
            PlanColumnSpec::new("price", arrow::datatypes::DataType::Int64, true),
        ],
        if_not_exists: false,
        or_replace: false,
        source: None,
        namespace: None,
        foreign_keys: Vec::new(),
        multi_column_uniques: Vec::new(),
    };
    CatalogDdl::create_table(&session, create_plan).unwrap();

    // Insert data
    let insert_plan = InsertPlan {
        table: "products".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![
            vec![PlanValue::Integer(1), PlanValue::Integer(100)],
            vec![PlanValue::Integer(2), PlanValue::Integer(200)],
        ]),
        on_conflict: InsertConflictAction::None,
    };
    session.execute_insert_plan(insert_plan).unwrap();

    // Begin transaction
    session.begin_transaction().unwrap();

    // Insert more data in transaction
    let insert_plan2 = InsertPlan {
        table: "products".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![vec![PlanValue::Integer(3), PlanValue::Integer(300)]]),
        on_conflict: InsertConflictAction::None,
    };
    session.execute_insert_plan(insert_plan2).unwrap();

    // SELECT with aggregate within transaction should see all 3 rows
    let select_plan = SelectPlan::new("products")
        .with_aggregates(vec![AggregateExpr::sum_int64("price", "total_price")]);
    let result = session.execute_select_plan(select_plan).unwrap();

    if let RuntimeStatementResult::Select { execution, .. } = result {
        let batches = execution.collect().unwrap();
        assert_eq!(batches.len(), 1);
        let array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        let sum = array.value(0);
        // 100 + 200 + 300 = 600
        assert_eq!(sum, 600, "Should see all rows in aggregate");
    } else {
        panic!("Expected SELECT result");
    }

    // Rollback
    session.rollback_transaction().unwrap();

    // SELECT after rollback should only see first 2 rows
    let select_plan2 = SelectPlan::new("products")
        .with_aggregates(vec![AggregateExpr::sum_int64("price", "total_price")]);
    let result2 = session.execute_select_plan(select_plan2).unwrap();

    if let RuntimeStatementResult::Select { execution, .. } = result2 {
        let batches = execution.collect().unwrap();
        assert_eq!(batches.len(), 1);
        let array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        let sum = array.value(0);
        // 100 + 200 = 300
        assert_eq!(sum, 300, "Should see only original 2 rows after rollback");
    } else {
        panic!("Expected SELECT result");
    }
}
