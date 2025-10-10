use llkv_dsl::{
    AggregateExpr, ColumnSpec, CreateTablePlan, DslContext, DslValue, InsertPlan, InsertSource,
    SelectPlan, SelectProjection,
};
use llkv_storage::pager::MemPager;
use std::sync::Arc;

#[test]
fn test_dsl_transaction_select() {
    let pager = Arc::new(MemPager::default());
    let ctx = Arc::new(DslContext::new(pager));
    let session = ctx.create_session();

    // Create table
    let create_plan = CreateTablePlan {
        name: "users".into(),
        columns: vec![
            ColumnSpec::new("id", arrow::datatypes::DataType::Int64, true),
            ColumnSpec::new("name", arrow::datatypes::DataType::Utf8, true),
        ],
        if_not_exists: false,
        source: None,
    };
    session.create_table_plan(create_plan).unwrap();

    // Insert data
    let insert_plan = InsertPlan {
        table: "users".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![
            vec![DslValue::Integer(1), DslValue::String("Alice".into())],
            vec![DslValue::Integer(2), DslValue::String("Bob".into())],
        ]),
    };
    session.insert(insert_plan).unwrap();

    // Begin transaction
    session.begin_transaction().unwrap();

    // Insert more data in transaction
    let insert_plan2 = InsertPlan {
        table: "users".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![vec![
            DslValue::Integer(3),
            DslValue::String("Charlie".into()),
        ]]),
    };
    session.insert(insert_plan2).unwrap();

    // SELECT within transaction should see all 3 rows
    let select_plan = SelectPlan::new("users").with_projections(vec![SelectProjection::AllColumns]);
    let result = session.select(select_plan).unwrap();

    if let llkv_dsl::StatementResult::Select { execution, .. } = result {
        let batches = execution.collect().unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3, "Should see 3 rows in transaction");
    } else {
        panic!("Expected SELECT result");
    }

    // Rollback
    session.rollback_transaction().unwrap();

    // SELECT after rollback should only see 2 rows
    let select_plan2 =
        SelectPlan::new("users").with_projections(vec![SelectProjection::AllColumns]);
    let result2 = session.select(select_plan2).unwrap();

    if let llkv_dsl::StatementResult::Select { execution, .. } = result2 {
        let batches = execution.collect().unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "Should see only 2 rows after rollback");
    } else {
        panic!("Expected SELECT result");
    }
}

#[test]
fn test_dsl_transaction_select_with_aggregates() {
    let pager = Arc::new(MemPager::default());
    let ctx = Arc::new(DslContext::new(pager));
    let session = ctx.create_session();

    // Create table
    let create_plan = CreateTablePlan {
        name: "products".into(),
        columns: vec![
            ColumnSpec::new("id", arrow::datatypes::DataType::Int64, true),
            ColumnSpec::new("price", arrow::datatypes::DataType::Int64, true),
        ],
        if_not_exists: false,
        source: None,
    };
    session.create_table_plan(create_plan).unwrap();

    // Insert data
    let insert_plan = InsertPlan {
        table: "products".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![
            vec![DslValue::Integer(1), DslValue::Integer(100)],
            vec![DslValue::Integer(2), DslValue::Integer(200)],
        ]),
    };
    session.insert(insert_plan).unwrap();

    // Begin transaction
    session.begin_transaction().unwrap();

    // Insert more data in transaction
    let insert_plan2 = InsertPlan {
        table: "products".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![vec![DslValue::Integer(3), DslValue::Integer(300)]]),
    };
    session.insert(insert_plan2).unwrap();

    // SELECT with aggregate within transaction should see all 3 rows
    let select_plan = SelectPlan::new("products")
        .with_aggregates(vec![AggregateExpr::sum_int64("price", "total_price")]);
    let result = session.select(select_plan).unwrap();

    if let llkv_dsl::StatementResult::Select { execution, .. } = result {
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
    let result2 = session.select(select_plan2).unwrap();

    if let llkv_dsl::StatementResult::Select { execution, .. } = result2 {
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
