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

#[test]
fn create_table_with_table_level_primary_key() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Accept table-level PRIMARY KEY constraints (single and composite)
    let mut results = engine
        .execute("CREATE TABLE pk_combo(a INTEGER, b INTEGER, PRIMARY KEY (a, b));")
        .expect("create table with table-level PRIMARY KEY succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(
        results.remove(0),
        RuntimeStatementResult::CreateTable { .. }
    ));

    // Insert distinct composite keys succeeds
    let mut insert1 = engine
        .execute("INSERT INTO pk_combo VALUES (1, 1);")
        .expect("first insert succeeds");
    assert!(matches!(
        insert1.remove(0),
        RuntimeStatementResult::Insert { .. }
    ));

    let mut insert2 = engine
        .execute("INSERT INTO pk_combo VALUES (1, 2);")
        .expect("second insert succeeds");
    assert!(matches!(
        insert2.remove(0),
        RuntimeStatementResult::Insert { .. }
    ));

    // Duplicate composite key should fail
    let err = engine
        .execute("INSERT INTO pk_combo VALUES (1, 1);")
        .expect_err("duplicate primary key should error");
    let err_str = format!("{err}").to_ascii_lowercase();
    assert!(
        err_str.contains("constraint violation"),
        "unexpected error: {err_str}"
    );

    // PRIMARY KEY columns may not accept NULL
    let err = engine
        .execute("INSERT INTO pk_combo VALUES (NULL, 3);")
        .expect_err("primary key NULL should error");
    let err_str = format!("{err}").to_ascii_lowercase();
    assert!(
        err_str.contains("null"),
        "unexpected error: {err_str}"
    );
}
