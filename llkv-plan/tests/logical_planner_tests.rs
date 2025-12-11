use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use llkv_expr::Literal;
use llkv_expr::expr::{CompareOp, Expr, Filter, Operator, ScalarExpr};
use llkv_plan::logical_planner::{LogicalPlan, SingleTableLogicalPlan, projection_name};
use llkv_plan::schema::{PlanColumn, PlanSchema};
use llkv_plan::table_provider::{ExecutionTable, TableProvider};
use llkv_plan::{
    JoinMetadata, JoinPlan, LogicalPlanner, OrderByPlan, OrderSortType, OrderTarget, SelectFilter,
    SelectPlan, SelectProjection, TableRef,
};
use llkv_scan::{ScanProjection, ScanStreamOptions};
use llkv_storage::pager::MemPager;
use llkv_types::{FieldId, LogicalFieldId, QueryContext, TableId};
use rustc_hash::FxHashMap;

type TestPager = MemPager;

#[derive(Debug)]
struct DummyTable {
    name: String,
    table_id: TableId,
    schema: Arc<PlanSchema>,
}

impl DummyTable {
    fn new(name: &str, schema: Arc<PlanSchema>, table_id: TableId) -> Self {
        Self {
            name: name.to_string(),
            table_id,
            schema,
        }
    }
}

impl ExecutionTable<TestPager> for DummyTable {
    fn schema(&self) -> &PlanSchema {
        &self.schema
    }

    fn table_id(&self) -> TableId {
        self.table_id
    }

    fn approximate_row_count(&self) -> Option<usize> {
        None
    }

    fn scan_stream(
        &self,
        _projections: &[ScanProjection],
        _predicate: &Expr<'static, FieldId>,
        _options: ScanStreamOptions<TestPager>,
        _callback: &mut dyn FnMut(RecordBatch),
    ) -> llkv_result::Result<()> {
        Err(llkv_result::Error::Internal(
            "not implemented for tests".to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn into_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }
}

struct DummyProvider {
    table: Arc<DummyTable>,
    name_lower: String,
}

impl DummyProvider {
    fn new(table: Arc<DummyTable>) -> Arc<Self> {
        let name_lower = table.name.to_ascii_lowercase();
        Arc::new(Self { table, name_lower })
    }
}

impl TableProvider<TestPager> for DummyProvider {
    fn get_table(&self, name: &str) -> Option<Arc<dyn ExecutionTable<TestPager>>> {
        if name == self.table.name {
            Some(Arc::new(DummyTable::new(
                &self.table.name,
                self.table.schema.clone(),
                self.table.table_id,
            )))
        } else {
            None
        }
    }
}

struct MultiProvider {
    tables: FxHashMap<String, Arc<DummyTable>>, // lowercased name -> table
}

impl MultiProvider {
    fn new(tables: Vec<Arc<DummyTable>>) -> Arc<Self> {
        let mut map = FxHashMap::default();
        for table in tables {
            map.insert(table.name.to_ascii_lowercase(), table);
        }
        Arc::new(Self { tables: map })
    }
}

impl TableProvider<TestPager> for MultiProvider {
    fn get_table(&self, name: &str) -> Option<Arc<dyn ExecutionTable<TestPager>>> {
        let key = name.to_ascii_lowercase();
        self.tables
            .get(&key)
            .cloned()
            .map(|t| t as Arc<dyn ExecutionTable<TestPager>>)
    }
}

fn make_schema() -> Arc<PlanSchema> {
    let cols = vec![
        PlanColumn {
            name: "id".to_string(),
            data_type: DataType::Int64,
            field_id: 1,
            is_nullable: false,
            is_primary_key: true,
            is_unique: true,
            default_value: None,
            check_expr: None,
        },
        PlanColumn {
            name: "val".to_string(),
            data_type: DataType::Int64,
            field_id: 2,
            is_nullable: true,
            is_primary_key: false,
            is_unique: false,
            default_value: None,
            check_expr: None,
        },
    ];
    Arc::new(PlanSchema::new(cols))
}

fn build_planner() -> (LogicalPlanner<TestPager>, SingleTableLogicalPlan<TestPager>) {
    let schema = make_schema();
    let table = Arc::new(DummyTable::new("t", schema.clone(), 42));
    let provider = DummyProvider::new(table);
    let planner = LogicalPlanner::new(provider);

    let mut select_plan = SelectPlan::new("t");
    select_plan.projections = vec![SelectProjection::Column {
        name: "val".to_string(),
        alias: None,
    }];
    select_plan.order_by = vec![OrderByPlan {
        target: OrderTarget::Column("id".to_string()),
        sort_type: OrderSortType::Native,
        ascending: true,
        nulls_first: false,
    }];

    let ctx = QueryContext::new();
    let logical_plan = planner
        .create_logical_plan(&select_plan, &ctx)
        .expect("logical planning succeeds");
    let single = match logical_plan {
        LogicalPlan::Single(plan) => plan,
        _ => panic!("expected single-table logical plan"),
    };
    (planner, single)
}

fn make_schema_three() -> Arc<PlanSchema> {
    let cols = vec![
        PlanColumn {
            name: "id".to_string(),
            data_type: DataType::Int64,
            field_id: 1,
            is_nullable: false,
            is_primary_key: true,
            is_unique: true,
            default_value: None,
            check_expr: None,
        },
        PlanColumn {
            name: "a_id".to_string(),
            data_type: DataType::Int64,
            field_id: 2,
            is_nullable: false,
            is_primary_key: false,
            is_unique: false,
            default_value: None,
            check_expr: None,
        },
        PlanColumn {
            name: "val".to_string(),
            data_type: DataType::Int64,
            field_id: 3,
            is_nullable: true,
            is_primary_key: false,
            is_unique: false,
            default_value: None,
            check_expr: None,
        },
    ];
    Arc::new(PlanSchema::new(cols))
}

#[test]
fn logical_planner_adds_order_by_columns() {
    let (_planner, logical_plan) = build_planner();

    assert_eq!(
        logical_plan.requested_projections.len(),
        1,
        "requested projections should honor SELECT list"
    );
    assert_eq!(
        logical_plan.scan_projections.len(),
        2,
        "ORDER BY column should be appended for planning"
    );
    assert_eq!(
        logical_plan.extra_columns,
        vec!["id".to_string()],
        "logical planner should track extra ORDER BY columns"
    );
    assert!(
        matches!(
            logical_plan.resolved_order_by.as_slice(),
            [order] if matches!(&order.target, OrderTarget::Column(col) if col == "id")
        ),
        "resolved ORDER BY should target the real column name"
    );
}

#[test]
fn logical_planner_translates_filter_to_field_ids() {
    let schema = make_schema();
    let table = Arc::new(DummyTable::new("t", schema, 77));
    let provider = DummyProvider::new(table);
    let planner = LogicalPlanner::new(provider);

    let mut select_plan = SelectPlan::new("t");
    select_plan.filter = Some(SelectFilter {
        predicate: Expr::Pred(Filter {
            field_id: "val".to_string(),
            op: Operator::GreaterThan(Literal::Int128(5)),
        }),
        subqueries: Vec::new(),
    });

    let logical_plan = planner
        .create_logical_plan(&select_plan, &QueryContext::new())
        .expect("logical planning succeeds");

    let logical_plan = match logical_plan {
        LogicalPlan::Single(plan) => plan,
        _ => panic!("expected single-table plan"),
    };

    let translated = logical_plan.filter.expect("filter is translated");
    match translated {
        Expr::Pred(Filter { field_id, op }) => {
            assert_eq!(field_id, 2, "column name should resolve to field id");
            assert!(matches!(op, Operator::GreaterThan(_)));
        }
        other => panic!("unexpected translated filter: {other:?}"),
    }

    let original = logical_plan
        .original_filter
        .expect("original filter should be preserved");
    match original {
        Expr::Pred(Filter { field_id, .. }) => {
            assert_eq!(field_id, "val");
        }
        other => panic!("unexpected original filter: {other:?}"),
    }
}

#[test]
fn logical_planner_adds_filter_columns_to_scan_projections() {
    let schema = make_schema();
    let table = Arc::new(DummyTable::new("t", schema, 88));
    let provider = DummyProvider::new(table);
    let planner = LogicalPlanner::new(provider);

    let mut select_plan = SelectPlan::new("t");
    select_plan.projections = vec![SelectProjection::Column {
        name: "id".to_string(),
        alias: None,
    }];
    select_plan.filter = Some(SelectFilter {
        predicate: Expr::Pred(Filter {
            field_id: "val".to_string(),
            op: Operator::GreaterThan(Literal::Int128(10)),
        }),
        subqueries: Vec::new(),
    });

    let logical_plan = planner
        .create_logical_plan(&select_plan, &QueryContext::new())
        .expect("logical planning succeeds");
    let logical_plan = match logical_plan {
        LogicalPlan::Single(plan) => plan,
        _ => panic!("expected single-table plan"),
    };

    let projected_names: Vec<String> = logical_plan
        .scan_projections
        .iter()
        .map(|p| projection_name(p, &logical_plan.schema))
        .collect();

    assert!(
        projected_names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("val")),
        "filter column should be included in scan projections"
    );
    assert!(
        projected_names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("id")),
        "selected column should remain in projections"
    );
}

#[test]
fn logical_planner_resolves_multi_table_plan() {
    let table_a = Arc::new(DummyTable::new("a", make_schema(), 10));
    let table_b = Arc::new(DummyTable::new("b", make_schema_three(), 20));
    let provider = MultiProvider::new(vec![table_a.clone(), table_b.clone()]);
    let planner = LogicalPlanner::new(provider);

    let mut select_plan =
        SelectPlan::with_tables(vec![TableRef::new("", "a"), TableRef::new("", "b")]);
    select_plan.projections = vec![
        SelectProjection::Column {
            name: "a.id".to_string(),
            alias: None,
        },
        SelectProjection::Column {
            name: "b.val".to_string(),
            alias: Some("b_val".to_string()),
        },
    ];
    select_plan.joins.push(JoinMetadata {
        left_table_index: 0,
        join_type: JoinPlan::Inner,
        on_condition: Some(Expr::Compare {
            left: ScalarExpr::Column("a.id".to_string()),
            op: CompareOp::Eq,
            right: ScalarExpr::Column("b.a_id".to_string()),
        }),
    });
    select_plan.group_by = vec!["a.id".to_string()];
    select_plan.aggregates = vec![llkv_plan::AggregateExpr::count_column(
        "b.val", "cnt", false,
    )];
    select_plan.order_by = vec![OrderByPlan {
        target: OrderTarget::Column("b.val".to_string()),
        sort_type: OrderSortType::Native,
        ascending: false,
        nulls_first: false,
    }];
    select_plan.filter = Some(SelectFilter {
        predicate: Expr::Pred(Filter {
            field_id: "b.val".to_string(),
            op: Operator::GreaterThan(Literal::Int128(0)),
        }),
        subqueries: Vec::new(),
    });
    select_plan.having = Some(Expr::Compare {
        left: ScalarExpr::Column("a.id".to_string()),
        op: CompareOp::Gt,
        right: ScalarExpr::Literal(Literal::Int128(0)),
    });

    let plan = planner
        .create_logical_plan(&select_plan, &QueryContext::new())
        .expect("logical planning succeeds");

    let multi = match plan {
        LogicalPlan::Multi(plan) => plan,
        _ => panic!("expected multi-table plan"),
    };

    assert_eq!(multi.projections.len(), 2, "projections should be resolved");
    match &multi.projections[0] {
        llkv_plan::logical_planner::ResolvedProjection::Column {
            table_index,
            logical_field_id,
            ..
        } => {
            assert_eq!(*table_index, 0);
            assert_eq!(
                *logical_field_id,
                LogicalFieldId::for_user(table_a.table_id(), 1)
            );
        }
        other => panic!("unexpected projection: {other:?}"),
    }

    let join = multi.joins.first().expect("join metadata exists");
    let on = join.on.as_ref().expect("join ON resolved");
    match on {
        Expr::Compare { left, right, .. } => {
            match left {
                ScalarExpr::Column(fid) => {
                    assert_eq!(fid.table_index, 0);
                    assert_eq!(fid.logical_field_id.table_id(), table_a.table_id());
                }
                other => panic!("unexpected left expr: {other:?}"),
            }
            match right {
                ScalarExpr::Column(fid) => {
                    assert_eq!(fid.table_index, 1);
                    assert_eq!(fid.logical_field_id.table_id(), table_b.table_id());
                }
                other => panic!("unexpected right expr: {other:?}"),
            }
        }
        other => panic!("unexpected ON predicate: {other:?}"),
    }

    let filter = multi.filter.as_ref().expect("filter resolved");
    match filter {
        Expr::Pred(Filter { field_id, .. }) => {
            assert_eq!(field_id.table_index, 1);
            assert_eq!(field_id.logical_field_id.table_id(), table_b.table_id());
        }
        other => panic!("unexpected filter: {other:?}"),
    }

    let having = multi.having.as_ref().expect("having resolved");
    match having {
        Expr::Compare { left, .. } => match left {
            ScalarExpr::Column(fid) => {
                assert_eq!(fid.table_index, 0);
                assert_eq!(fid.logical_field_id.table_id(), table_a.table_id());
            }
            other => panic!("unexpected having expr: {other:?}"),
        },
        other => panic!("unexpected having predicate: {other:?}"),
    }

    let group_key = multi.group_by.first().expect("group by resolved");
    assert_eq!(group_key.table_index, 0);
    assert_eq!(group_key.logical_field_id.table_id(), table_a.table_id());

    let agg = multi.aggregates.first().expect("aggregate resolved");
    let (agg_table, agg_lfid) = agg.input.expect("aggregate input resolved");
    assert_eq!(agg_table, 1, "aggregate should reference second table");
    assert_eq!(agg_lfid.table_id(), table_b.table_id());

    let order = multi.order_by.first().expect("order by resolved");
    let resolved_column = order
        .resolved_column
        .as_ref()
        .expect("order by column resolved");
    assert_eq!(resolved_column.0, 1);
    assert_eq!(resolved_column.1.table_id(), table_b.table_id());
}

#[test]
fn logical_planner_adds_group_by_and_aggregate_columns() {
    let schema = make_schema();
    let table = Arc::new(DummyTable::new("t", schema, 99));
    let provider = DummyProvider::new(table);
    let planner = LogicalPlanner::new(provider);

    let mut select_plan = SelectPlan::new("t");
    select_plan.projections = vec![SelectProjection::Computed {
        expr: llkv_expr::expr::ScalarExpr::column("id".to_string()),
        alias: "gid".to_string(),
    }];
    select_plan.group_by = vec!["id".to_string()];
    select_plan.aggregates = vec![llkv_plan::plans::AggregateExpr::count_column(
        "val", "cnt", false,
    )];

    let logical_plan = planner
        .create_logical_plan(&select_plan, &QueryContext::new())
        .expect("logical planning succeeds");
    let logical_plan = match logical_plan {
        LogicalPlan::Single(plan) => plan,
        _ => panic!("expected single-table plan"),
    };

    let projected_names: Vec<String> = logical_plan
        .scan_projections
        .iter()
        .map(|p| projection_name(p, &logical_plan.schema))
        .collect();

    assert!(
        projected_names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("id")),
        "group-by column should be included in scan projections"
    );
    assert!(
        projected_names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("val")),
        "aggregate input column should be included in scan projections"
    );
}

#[derive(Clone)]
struct MultiTableProvider {
    tables: FxHashMap<String, Arc<DummyTable>>,
}

impl MultiTableProvider {
    fn new(tables: Vec<Arc<DummyTable>>) -> Arc<Self> {
        let mut map = FxHashMap::default();
        for table in tables {
            map.insert(table.name.to_ascii_lowercase(), table);
        }
        Arc::new(Self { tables: map })
    }
}

impl TableProvider<TestPager> for MultiTableProvider {
    fn get_table(&self, name: &str) -> Option<Arc<dyn ExecutionTable<TestPager>>> {
        if let Some(table) = self.tables.get(&name.to_ascii_lowercase()) {
            Some(table.clone())
        } else {
            None
        }
    }
}

fn make_schema_with(id_name: &str, fid: FieldId) -> Arc<PlanSchema> {
    Arc::new(PlanSchema::new(vec![PlanColumn {
        name: id_name.to_string(),
        data_type: DataType::Int64,
        field_id: fid,
        is_nullable: false,
        is_primary_key: false,
        is_unique: false,
        default_value: None,
        check_expr: None,
    }]))
}

#[test]
fn multi_table_resolution_assigns_columns_to_tables() {
    let cust_schema = make_schema_with("id", 1);
    let orders_schema = make_schema_with("customer_id", 1);
    let cust_table = Arc::new(DummyTable::new("customers", cust_schema.clone(), 10));
    let orders_table = Arc::new(DummyTable::new("orders", orders_schema.clone(), 20));

    let provider = MultiTableProvider::new(vec![cust_table.clone(), orders_table.clone()]);
    let planner = LogicalPlanner::new(provider);

    let tables = vec![
        TableRef::with_alias("", "customers", Some("c".to_string())),
        TableRef::with_alias("", "orders", Some("o".to_string())),
    ];

    let joins = vec![JoinMetadata {
        left_table_index: 0,
        join_type: JoinPlan::Inner,
        on_condition: Some(Expr::Compare {
            left: ScalarExpr::column("c.id".to_string()),
            op: CompareOp::Eq,
            right: ScalarExpr::column("o.customer_id".to_string()),
        }),
    }];

    let mut plan = SelectPlan::with_tables(tables.clone()).with_joins(joins);
    plan.projections = vec![
        SelectProjection::Column {
            name: "c.id".to_string(),
            alias: None,
        },
        SelectProjection::Column {
            name: "o.customer_id".to_string(),
            alias: None,
        },
    ];

    let logical_plan = planner
        .create_logical_plan(&plan, &QueryContext::new())
        .expect("multi-table planning succeeds");

    let (resolved, unresolved) = match logical_plan {
        LogicalPlan::Multi(multi) => (multi.resolved_required, multi.unresolved_required),
        _ => panic!("expected multi-table plan"),
    };

    assert!(
        unresolved.is_empty(),
        "expected all columns to resolve, found: {:?}",
        unresolved
    );

    let mut cust_seen = false;
    let mut orders_seen = false;
    for col in resolved {
        if col.table_index == 0 {
            cust_seen = true;
            assert_eq!(
                col.logical_field_id,
                LogicalFieldId::for_user(cust_table.table_id, 1)
            );
        } else if col.table_index == 1 {
            orders_seen = true;
            assert_eq!(
                col.logical_field_id,
                LogicalFieldId::for_user(orders_table.table_id, 1)
            );
        }
    }

    assert!(cust_seen, "customer column should resolve to table 0");
    assert!(orders_seen, "orders column should resolve to table 1");
}
