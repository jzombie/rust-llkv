use arrow::array::{Array, Int64Array};
use llkv_dsl::DslContext;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use std::sync::Arc;

#[test]
fn insert_rollback_smoke() {
    let context = Arc::new(DslContext::new(Arc::new(MemPager::default())));
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
    if let llkv_sql::StatementResult::Select { execution, .. } = select {
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
