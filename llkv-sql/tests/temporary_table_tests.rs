use std::sync::Arc;

use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
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
        if table_name.to_ascii_lowercase() == "integers"
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
        if table_name.to_ascii_lowercase() == "integers" && rows_inserted == 2
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
        if table_name.to_ascii_lowercase() == "integers"
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
        if table_name.to_ascii_lowercase() == "integers" && rows_updated == 1
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
        if table_name.to_ascii_lowercase() == "integers" && rows_deleted == 1
    ));

    let mut results = engine
        .execute("DROP TABLE integers;")
        .expect("drop temporary table succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(results.remove(0), RuntimeStatementResult::NoOp));
}
