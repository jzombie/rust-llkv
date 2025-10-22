use arrow::array::{Array, Int64Array};
use llkv_runtime::RuntimeContext;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use std::sync::Arc;

#[test]
fn insert_rollback_smoke() {
    let context = Arc::new(RuntimeContext::new(Arc::new(MemPager::default())));
    let engine = SqlEngine::with_context(Arc::clone(&context), false);
    engine.execute("CREATE TABLE integers(i INTEGER)").unwrap();
    engine.execute("BEGIN TRANSACTION").unwrap();
    engine
        .execute("INSERT INTO integers VALUES (0), (1), (2)")
        .unwrap();
    engine.execute("SELECT COUNT(*) FROM integers").unwrap();
    engine.execute("ROLLBACK").unwrap();
    engine.execute("SELECT COUNT(*) FROM integers").unwrap();
}

#[test]
fn standalone_engine_insert_select() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    engine
        .execute("CREATE TABLE numbers(i INTEGER)")
        .expect("create table");
    engine
        .execute("SELECT COUNT(*) FROM numbers")
        .expect("count empty table");
    let mut sum_result = engine
        .execute("SELECT SUM(i) FROM numbers")
        .expect("sum empty table");
    let select = sum_result.remove(0);
    if let llkv_runtime::RuntimeStatementResult::Select { execution, .. } = select {
        let batches = execution.collect().expect("collect");
        if let Some(batch) = batches.first() {
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64");
            assert!(array.is_null(0), "expected NULL sum for empty table");
        }
    } else {
        panic!("expected select result");
    }
    engine
        .execute("INSERT INTO numbers VALUES (1), (2)")
        .expect("insert rows");
    engine
        .execute("SELECT COUNT(*) FROM numbers")
        .expect("count rows");
}

#[test]
fn commit_rollback_require_active_transaction() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    assert!(engine.execute("COMMIT").is_err());
    assert!(engine.execute("ROLLBACK").is_err());

    engine.execute("BEGIN").expect("begin");
    assert!(engine.execute("BEGIN").is_err());
    engine.execute("ROLLBACK").expect("rollback");
    assert!(engine.execute("COMMIT").is_err());
}

#[test]
fn nested_begin_does_not_clear_transaction() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    engine.execute("BEGIN").expect("initial begin");
    assert!(engine.execute("BEGIN").is_err());
    engine
        .execute("ROLLBACK")
        .expect("rollback after nested begin");
}

#[test]
fn transaction_functionality_script() {
    let pager = Arc::new(MemPager::default());
    let engine = SqlEngine::with_context(Arc::new(RuntimeContext::new(pager)), false);

    engine
        .execute("PRAGMA enable_verification;")
        .expect("pragma");
    assert!(engine.execute("COMMIT;").is_err());
    assert!(engine.execute("ROLLBACK;").is_err());

    engine
        .execute("START TRANSACTION;")
        .expect("start transaction");
    assert!(engine.execute("START TRANSACTION;").is_err());
    engine
        .execute("ROLLBACK;")
        .expect("rollback after start transaction");
}

#[test]
fn basic_transaction_visibility() {
    let pager = Arc::new(MemPager::default());
    let context = Arc::new(RuntimeContext::new(pager));
    let conn1 = SqlEngine::with_context(Arc::clone(&context), false);
    let conn2 = SqlEngine::with_context(Arc::clone(&context), false);

    conn1.execute("BEGIN TRANSACTION").expect("conn1 begin");
    conn2.execute("BEGIN TRANSACTION").expect("conn2 begin");

    conn1
        .execute("CREATE TABLE integers(i INTEGER)")
        .expect("create table");

    let (_, canonical_name) = llkv_table::canonical_table_name("integers").unwrap();
    assert!(
        context.catalog().table_column_specs(&canonical_name).is_err(),
        "base context should not see transactional table"
    );

    conn1
        .execute("SELECT * FROM integers")
        .expect("conn1 sees table");
    assert!(conn2.execute("SELECT * FROM integers").is_err());

    conn1.execute("ROLLBACK").expect("conn1 rollback");
    assert!(conn1.execute("SELECT * FROM integers").is_err());
    assert!(conn2.execute("SELECT * FROM integers").is_err());

    conn1
        .execute("BEGIN TRANSACTION")
        .expect("conn1 begin again");
    conn1
        .execute("CREATE TABLE integers(i INTEGER)")
        .expect("create table second time");

    let (_, canonical_name) = llkv_table::canonical_table_name("integers").unwrap();
    assert!(
        context.catalog().table_column_specs(&canonical_name).is_err(),
        "base context should still not see table before commit"
    );
    conn1.execute("COMMIT").expect("conn1 commit");

    assert!(conn2.execute("SELECT * FROM integers").is_err());
    conn2.execute("ROLLBACK").expect("conn2 rollback");
    conn2
        .execute("SELECT * FROM integers")
        .expect("conn2 sees committed table");
}

#[test]
fn commit_visibility_across_connections() {
    let pager = Arc::new(MemPager::default());
    let shared = Arc::new(RuntimeContext::new(pager));
    let con1 = SqlEngine::with_context(Arc::clone(&shared), false);
    let con2 = SqlEngine::with_context(Arc::clone(&shared), false);

    con1.execute("BEGIN TRANSACTION;").expect("con1 begin");
    con2.execute("BEGIN TRANSACTION;").expect("con2 begin");

    con1.execute("CREATE TABLE integers(i INTEGER);")
        .expect("create table");

    con1.execute("COMMIT;").expect("con1 commit");
    // con2 started before commit, still in old snapshot.
    assert!(con2.execute("SELECT * FROM integers;").is_err());

    con2.execute("ROLLBACK;").expect("con2 rollback");
    con2.execute("SELECT * FROM integers;")
        .expect("con2 sees committed table");
}

#[test]
fn duckdb_basic_transactions_script() {
    let pager = Arc::new(MemPager::default());
    let context = Arc::new(RuntimeContext::new(pager));
    let con1 = SqlEngine::with_context(Arc::clone(&context), false);
    let con2 = SqlEngine::with_context(Arc::clone(&context), false);

    con1.execute("BEGIN TRANSACTION;").unwrap();
    con2.execute("BEGIN TRANSACTION;").unwrap();

    con1.execute("CREATE TABLE integers(i INTEGER);").unwrap();

    // con1 should be able to query.
    con1.execute("SELECT * FROM integers;").unwrap();

    // con2 should not.
    let err = con2.execute("SELECT * FROM integers;").unwrap_err();
    assert!(format!("{err}").contains("Catalog Error"));

    // rollback con1, table disappears for everyone.
    con1.execute("ROLLBACK;").unwrap();
    let err = con1.execute("SELECT * FROM integers;").unwrap_err();
    assert!(format!("{err}").contains("Catalog Error"));
    let err = con2.execute("SELECT * FROM integers;").unwrap_err();
    assert!(format!("{err}").contains("Catalog Error"));

    // recreate and commit on con1.
    con1.execute("BEGIN TRANSACTION;").unwrap();
    con1.execute("CREATE TABLE integers(i INTEGER);").unwrap();
    con1.execute("COMMIT;").unwrap();

    // con2 still in old transaction shouldn't see it.
    let err = con2.execute("SELECT * FROM integers;").unwrap_err();
    assert!(format!("{err}").contains("Catalog Error"));

    // rollback and try again.
    con2.execute("ROLLBACK;").unwrap();
    con2.execute("SELECT * FROM integers;").unwrap();

    // serialization conflict scenario.
    con1.execute("BEGIN TRANSACTION;").unwrap();
    con2.execute("BEGIN TRANSACTION;").unwrap();
    con1.execute("CREATE TABLE integers2(i INTEGER);").unwrap();
    let err = con1
        .execute("CREATE TABLE integers2(i INTEGER);")
        .unwrap_err();
    assert!(format!("{err}").contains("already exists"));
}

#[test]
fn duckdb_transaction_functionality_script() {
    let pager = Arc::new(MemPager::default());
    let context = Arc::new(RuntimeContext::new(pager));
    let engine = SqlEngine::with_context(Arc::clone(&context), false);

    engine
        .execute("PRAGMA enable_verification")
        .expect("pragma should succeed");

    let err = engine.execute("COMMIT").unwrap_err();
    assert!(format!("{err}").contains("no transaction"));

    let err = engine.execute("ROLLBACK").unwrap_err();
    assert!(format!("{err}").contains("no transaction"));

    engine
        .execute("START TRANSACTION")
        .expect("start transaction");

    let err = engine.execute("START TRANSACTION").unwrap_err();
    assert!(format!("{err}").contains("already in progress"));

    engine.execute("ROLLBACK").expect("rollback active tx");
}
