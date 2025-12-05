use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use llkv_expr::Literal;
use llkv_expr::expr::{Expr, Filter, Operator};
use llkv_plan::logical_planner::{LogicalPlan, SingleTableLogicalPlan};
use llkv_plan::physical::table::{ExecutionTable, TableProvider};
use llkv_plan::schema::{PlanColumn, PlanSchema};
use llkv_plan::{
    LogicalPlanner, OrderByPlan, OrderSortType, OrderTarget, SelectFilter, SelectPlan,
    SelectProjection,
};
use llkv_scan::{ScanProjection, ScanStreamOptions};
use llkv_storage::pager::MemPager;
use llkv_types::{FieldId, TableId};

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
    fn schema(&self) -> Arc<PlanSchema> {
        self.schema.clone()
    }

    fn table_id(&self) -> TableId {
        self.table_id
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
    fn get_table(&self, name: &str) -> llkv_result::Result<Arc<dyn ExecutionTable<TestPager>>> {
        if name.eq_ignore_ascii_case(&self.name_lower) {
            Ok(self.table.clone())
        } else {
            Err(llkv_result::Error::CatalogError(format!(
                "table '{name}' not found"
            )))
        }
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

    let logical_plan = planner
        .create_logical_plan(&select_plan)
        .expect("logical planning succeeds");
    let single = match logical_plan {
        LogicalPlan::Single(plan) => plan,
        _ => panic!("expected single-table logical plan"),
    };
    (planner, single)
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
        .create_logical_plan(&select_plan)
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
