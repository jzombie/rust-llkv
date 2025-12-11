use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::DataType;
use llkv_expr::expr::{CompareOp, Expr, ScalarExpr};
use llkv_plan::PlanValue;
use llkv_plan::SelectPlan;
use llkv_plan::plans::{JoinMetadata, JoinPlan, SelectProjection, TableRef};
use llkv_runtime::{
    CreateTablePlan, InsertConflictAction, InsertPlan, InsertSource, RuntimeContext,
    RuntimeStatementResult,
};
use llkv_storage::pager::{BoxedPager, MemPager};
use llkv_table::CatalogDdl;

#[test]
fn executes_inner_join_two_tables() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let ctx = Arc::new(RuntimeContext::new(Arc::clone(&pager)));
    let session = ctx.create_session();

    let users_plan = CreateTablePlan {
        name: "users".into(),
        columns: vec![
            llkv_plan::PlanColumnSpec::new("id", DataType::Int64, true),
            llkv_plan::PlanColumnSpec::new("name", DataType::Utf8, true),
        ],
        if_not_exists: false,
        or_replace: false,
        source: None,
        namespace: None,
        foreign_keys: Vec::new(),
        multi_column_uniques: Vec::new(),
    };
    CatalogDdl::create_table(&session, users_plan).unwrap();

    let orders_plan = CreateTablePlan {
        name: "orders".into(),
        columns: vec![
            llkv_plan::PlanColumnSpec::new("id", DataType::Int64, true),
            llkv_plan::PlanColumnSpec::new("user_id", DataType::Int64, true),
            llkv_plan::PlanColumnSpec::new("amount", DataType::Int64, true),
        ],
        if_not_exists: false,
        or_replace: false,
        source: None,
        namespace: None,
        foreign_keys: Vec::new(),
        multi_column_uniques: Vec::new(),
    };
    CatalogDdl::create_table(&session, orders_plan).unwrap();

    let insert_users = InsertPlan {
        table: "users".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![
            vec![PlanValue::Integer(1), PlanValue::String("alice".into())],
            vec![PlanValue::Integer(2), PlanValue::String("bob".into())],
        ]),
        on_conflict: InsertConflictAction::None,
    };
    session.execute_insert_plan(insert_users).unwrap();

    let insert_orders = InsertPlan {
        table: "orders".into(),
        columns: vec![],
        source: InsertSource::Rows(vec![
            vec![
                PlanValue::Integer(10),
                PlanValue::Integer(1),
                PlanValue::Integer(50),
            ],
            vec![
                PlanValue::Integer(11),
                PlanValue::Integer(2),
                PlanValue::Integer(75),
            ],
            vec![
                PlanValue::Integer(12),
                PlanValue::Integer(3),
                PlanValue::Integer(125),
            ],
        ]),
        on_conflict: InsertConflictAction::None,
    };
    session.execute_insert_plan(insert_orders).unwrap();

    let mut select_plan = SelectPlan::with_tables(vec![
        TableRef::new("", "users"),
        TableRef::new("", "orders"),
    ]);
    select_plan.joins.push(JoinMetadata {
        left_table_index: 0,
        join_type: JoinPlan::Inner,
        strategy: None,
        on_condition: Some(Expr::Compare {
            left: ScalarExpr::column("users.id".to_string()),
            op: CompareOp::Eq,
            right: ScalarExpr::column("orders.user_id".to_string()),
        }),
    });
    select_plan.projections = vec![
        SelectProjection::Column {
            name: "users.id".into(),
            alias: None,
        },
        SelectProjection::Column {
            name: "orders.amount".into(),
            alias: None,
        },
    ];
    let result = session.execute_select_plan(select_plan).unwrap();

    let RuntimeStatementResult::Select { execution, .. } = result else {
        panic!("expected SELECT result");
    };

    let batches = execution.collect().unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2, "only matching user rows should remain");

    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id column should be Int64");
    let amounts = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("amount column should be Int64");

    // Rows should correspond to user_id 1 and 2 (ordered by amount asc)
    assert_eq!(ids.value(0), 1);
    assert_eq!(amounts.value(0), 50);
    assert_eq!(ids.value(1), 2);
    assert_eq!(amounts.value(1), 75);
}
