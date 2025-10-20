use std::sync::Arc;

use llkv_plan::{InsertPlan, InsertSource, PlanValue};
use llkv_runtime::{
    RuntimeStatementResult, storage_namespace, storage_namespace::TemporaryNamespace,
};
use llkv_sql::SqlEngine;
use llkv_storage::pager::BoxedPager;
use llkv_storage::pager::MemPager;

#[test]
fn temporary_tables_support_core_dml() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let mut results = engine
        .execute("CREATE TEMPORARY TABLE integers(value INTEGER);")
        .expect("create temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::CreateTable { table_name }
        if table_name.eq_ignore_ascii_case("integers")
    ));

    let mut results = engine
        .execute("INSERT INTO integers VALUES (1), (2);")
        .expect("insert into temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::Insert {
            table_name,
            rows_inserted
        }
        if table_name.eq_ignore_ascii_case("integers") && rows_inserted == 2
    ));

    let registry = engine.session().namespace_registry();
    let namespace_id = registry
        .read()
        .expect("namespace registry lock")
        .namespace_for_table("integers");
    assert_eq!(
        namespace_id,
        llkv_runtime::storage_namespace::TEMPORARY_NAMESPACE_ID
    );

    let mut results = engine
        .execute("CREATE UNIQUE INDEX \"uidx\" ON \"integers\" (\"value\");")
        .expect("create index on temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::CreateIndex { table_name, .. }
        if table_name.eq_ignore_ascii_case("integers")
    ));

    let mut results = engine
        .execute("UPDATE integers SET value = value + 10 WHERE value = 1;")
        .expect("update temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::Update {
            table_name,
            rows_updated
        }
        if table_name.eq_ignore_ascii_case("integers") && rows_updated == 1
    ));

    let mut results = engine
        .execute("DELETE FROM integers WHERE value = 11;")
        .expect("delete from temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::Delete {
            table_name,
            rows_deleted
        }
        if table_name.eq_ignore_ascii_case("integers") && rows_deleted == 1
    ));

    let mut results = engine
        .execute("DROP TABLE integers;")
        .expect("drop temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(results.remove(0), RuntimeStatementResult::NoOp));
}

#[test]
fn temporary_tables_allow_inserts_after_unique_index() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let mut results = engine
        .execute("CREATE TEMPORARY TABLE numbers(i INTEGER, j TEXT);")
        .expect("create temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::CreateTable { table_name }
        if table_name.eq_ignore_ascii_case("numbers")
    ));

    let mut results = engine
        .execute("CREATE UNIQUE INDEX idx_numbers_j ON numbers(j);")
        .expect("create unique index succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::CreateIndex { table_name, .. }
        if table_name.eq_ignore_ascii_case("numbers")
    ));

    let mut results = engine
        .execute("SELECT * FROM numbers;")
        .expect("select from empty temp table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::Select { .. }
    ));

    let temp_namespace = engine
        .session()
        .namespace_registry()
        .read()
        .expect("namespace registry lock")
        .namespace::<TemporaryNamespace<BoxedPager>>(storage_namespace::TEMPORARY_NAMESPACE_ID)
        .expect("temporary namespace present");

    let temp_context = temp_namespace.context();
    let table = temp_context
        .lookup_table("numbers")
        .expect("temporary context lookup succeeds");
    let unique_flag = table
        .schema
        .columns
        .iter()
        .find(|col| col.name.eq_ignore_ascii_case("j"))
        .map(|col| col.unique)
        .unwrap_or_default();
    assert!(
        unique_flag,
        "unique index updates executor schema for column j"
    );

    let insert_plan = InsertPlan {
        table: "numbers".to_string(),
        columns: Vec::new(),
        source: InsertSource::Rows(vec![
            vec![PlanValue::Integer(1), PlanValue::String("a".into())],
            vec![PlanValue::Integer(2), PlanValue::String("b".into())],
        ]),
    };

    let result = engine
        .session()
        .insert(insert_plan)
        .expect("insert now succeeds after catalog synchronization fix");

    assert!(
        matches!(result, RuntimeStatementResult::Insert { table_name, rows_inserted: 2 }
        if table_name.eq_ignore_ascii_case("numbers"))
    );
}
