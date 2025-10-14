use std::sync::Arc;

use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[test]
fn create_schema_simple_noop() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    let mut results = engine
        .execute("CREATE SCHEMA test")
        .expect("create schema succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(results.remove(0), RuntimeStatementResult::NoOp));
}

#[test]
fn schema_qualified_column_with_alias_errors() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    engine.execute("CREATE SCHEMA test").expect("create schema");
    engine
        .execute("CREATE TABLE test.tbl(col INTEGER);")
        .expect("create table");

    let err = engine
        .execute("SELECT test.tbl.col FROM test.tbl t;")
        .expect_err("query should fail when using table alias with schema qualifier");
    assert!(
        format!("{err}")
            .to_ascii_lowercase()
            .contains("binder error"),
        "unexpected error: {err}"
    );
}

#[test]
fn create_table_with_row_type() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    let mut results = engine
        .execute("CREATE TABLE row_struct(payload ROW(a INTEGER, b VARCHAR));")
        .expect("create table with ROW type succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::CreateTable { .. }
    ));
}

#[test]
fn create_table_with_double_precision_row_field() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    let mut results = engine
        .execute("CREATE TABLE row_double(payload ROW(d DOUBLE PRECISION));")
        .expect("create table with DOUBLE PRECISION ROW field");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::CreateTable { .. }
    ));
}
