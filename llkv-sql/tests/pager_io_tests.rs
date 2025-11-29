use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
use llkv_storage::pager::{InstrumentedPager, MemPager};
use std::sync::Arc;

fn collect_select(engine: &SqlEngine, sql: &str) {
    let mut results = engine.execute(sql).expect("execute select statement");
    assert_eq!(results.len(), 1, "expected single statement result");
    match results.pop().unwrap() {
        RuntimeStatementResult::Select { execution, .. } => {
            execution.collect().expect("collect select results");
        }
        _ => panic!("expected SELECT result"),
    }
}

#[test]
fn crud_statements_update_pager_counters() {
    let (pager, stats) = InstrumentedPager::new(MemPager::default());
    let engine = SqlEngine::new(Arc::new(pager));

    engine
        .execute("CREATE TABLE pager_io(id INTEGER PRIMARY KEY, note TEXT);")
        .expect("create table");

    let insert_before = stats.snapshot();
    engine
        .execute(
            "INSERT INTO pager_io VALUES
            (1, 'alpha'),
            (2, 'beta'),
            (3, 'gamma');",
        )
        .expect("insert rows");
    let insert_after = stats.snapshot();
    let insert_delta = insert_after.delta_since(&insert_before);
    assert_eq!(
        insert_delta.physical_allocs, 8,
        "unexpected insert alloc count: {insert_delta:?}"
    );
    assert_eq!(
        insert_delta.alloc_batches, 8,
        "unexpected insert alloc batches: {insert_delta:?}"
    );
    assert_eq!(
        insert_delta.physical_puts, 24,
        "unexpected insert put count: {insert_delta:?}"
    );
    assert_eq!(
        insert_delta.put_batches, 1,
        "unexpected insert put batches: {insert_delta:?}"
    );
    assert_eq!(
        insert_delta.fresh_puts, 8,
        "unexpected fresh puts for insert: {insert_delta:?}"
    );
    assert_eq!(
        insert_delta.overwritten_puts, 16,
        "unexpected overwritten puts for insert: {insert_delta:?}"
    );

    let select_before = stats.snapshot();
    collect_select(&engine, "SELECT COUNT(*) FROM pager_io;");
    let select_after = stats.snapshot();
    let select_delta = select_after.delta_since(&select_before);
    assert_eq!(
        select_delta.physical_gets, 40,
        "unexpected select get count: {select_delta:?}"
    );
    assert_eq!(
        select_delta.get_batches, 26,
        "unexpected select get batches: {select_delta:?}"
    );

    let delete_before = stats.snapshot();
    engine
        .execute("DELETE FROM pager_io;")
        .expect("delete rows");
    let delete_after = stats.snapshot();
    let delete_delta = delete_after.delta_since(&delete_before);
    assert_eq!(
        delete_delta.physical_puts, 6,
        "unexpected delete put count: {delete_delta:?}"
    );
    assert_eq!(
        delete_delta.put_batches, 1,
        "unexpected delete put batches: {delete_delta:?}"
    );
    assert_eq!(
        delete_delta.physical_gets, 54,
        "unexpected delete get count: {delete_delta:?}"
    );
    assert_eq!(
        delete_delta.get_batches, 38,
        "unexpected delete get batches: {delete_delta:?}"
    );

    let drop_before = stats.snapshot();
    engine
        .execute("DROP TABLE pager_io;")
        .expect("drop table cleanup");
    let drop_after = stats.snapshot();
    let drop_delta = drop_after.delta_since(&drop_before);
    assert_eq!(
        drop_delta.physical_frees, 24,
        "unexpected drop free count: {drop_delta:?}"
    );
    assert_eq!(
        drop_delta.free_batches, 4,
        "unexpected drop free batches: {drop_delta:?}"
    );
    assert_eq!(
        drop_delta.physical_puts, 24,
        "unexpected drop put count: {drop_delta:?}"
    );
    assert_eq!(
        drop_delta.put_batches, 7,
        "unexpected drop put batches: {drop_delta:?}"
    );
}
