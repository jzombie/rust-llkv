use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use llkv_plan::plans::PlanValue;
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
    assert!(err_str.contains("null"), "unexpected error: {err_str}");
}

#[test]
fn drop_view_basic_flow() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE base(value INTEGER);")
        .expect("create base table");
    let mut created = engine
        .execute("CREATE VIEW base_view AS SELECT value FROM base;")
        .expect("create view succeeds");
    assert_eq!(created.len(), 1);
    assert!(matches!(created.remove(0), RuntimeStatementResult::NoOp));

    let mut dropped = engine
        .execute("DROP VIEW base_view;")
        .expect("drop view succeeds");
    assert_eq!(dropped.len(), 1);
    assert!(matches!(dropped.remove(0), RuntimeStatementResult::NoOp));

    let mut recreated = engine
        .execute("CREATE VIEW base_view AS SELECT value FROM base;")
        .expect("recreate view succeeds");
    assert_eq!(recreated.len(), 1);
    assert!(matches!(recreated.remove(0), RuntimeStatementResult::NoOp));
}

#[test]
fn drop_view_if_exists_missing_noop() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let mut results = engine
        .execute("DROP VIEW IF EXISTS missing_view;")
        .expect("drop view IF EXISTS succeeds");
    assert_eq!(results.len(), 1);
    assert!(matches!(results.remove(0), RuntimeStatementResult::NoOp));
}

#[test]
fn select_from_view_uses_column_alias() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE base(pk INTEGER, payload TEXT);")
        .expect("create base table");
    engine
        .execute("INSERT INTO base VALUES (1, 'a');")
        .expect("insert row");
    engine
        .execute("CREATE VIEW base_view(pk_alias) AS SELECT pk FROM base;")
        .expect("create view with column alias");

    let mut results = engine
        .execute("SELECT pk_alias FROM base_view;")
        .expect("select from view should succeed");
    assert_eq!(results.len(), 1);
    let RuntimeStatementResult::Select { execution, .. } = results.remove(0) else {
        panic!("expected select result");
    };
    let batches = execution.collect().expect("collect batches");
    let schema = batches
        .first()
        .map(|b| b.schema())
        .expect("non-empty batches");

    let combined = arrow::compute::concat_batches(&schema, batches.iter())
    .expect("concat batches");
    let column = combined
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("int column");
    assert_eq!(combined.num_rows(), 1);
    assert_eq!(column.value(0), 1);
}

#[test]
fn drop_view_errors_when_target_is_table() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE base(value INTEGER);")
        .expect("create base table");

    let err = engine
        .execute("DROP VIEW base;")
        .expect_err("DROP VIEW should fail when target is a table");
    let err_text = format!("{err}");
    assert!(
        err_text.contains("use DROP TABLE to delete table"),
        "unexpected error: {err_text}"
    );

    let err_if_exists = engine
        .execute("DROP VIEW IF EXISTS base;")
        .expect_err("IF EXISTS does not downgrade table vs view mismatch");
    let err_text_if_exists = format!("{err_if_exists}");
    assert!(
        err_text_if_exists.contains("use DROP TABLE to delete table"),
        "unexpected error: {err_text_if_exists}"
    );
}

#[test]
fn information_schema_tables_lists_user_tables() {
    use arrow::array::StringArray;

    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    engine
        .execute("CREATE TABLE first_table(id INTEGER);")
        .expect("create first table");
    engine
        .execute("CREATE TABLE second_table(id INTEGER);")
        .expect("create second table");

    let batches = engine
        .sql("SELECT table_name FROM information_schema.tables ORDER BY table_name;")
        .expect("information_schema query");
    let mut names = Vec::new();
    for batch in batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        for idx in 0..column.len() {
            if !column.is_null(idx) {
                names.push(column.value(idx).to_string());
            }
        }
    }

    assert!(
        names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("first_table")),
        "first_table not found in information_schema.tables: {:?}",
        names
    );
    assert!(
        names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("second_table")),
        "second_table not found in information_schema.tables: {:?}",
        names
    );
}

#[test]
fn information_schema_columns_reports_column_metadata() {
    use arrow::array::{BooleanArray, Int32Array, StringArray};

    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    engine
        .execute(
            "
        CREATE TABLE audit_log (
            entry_id INTEGER PRIMARY KEY,
            payload TEXT NOT NULL
        );",
        )
        .expect("create table");

    let batches = engine
        .sql(
            "SELECT table_name, column_name, ordinal_position, is_nullable
             FROM information_schema.columns
             WHERE table_name = 'audit_log'
             ORDER BY ordinal_position;",
        )
        .expect("information_schema.columns query");
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2);

    let table_names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("table_name column");
    let column_names = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column_name column");
    let ordinal_positions = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("ordinal_position column");
    let nullable_flags = batch
        .column(3)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("is_nullable column");

    for idx in 0..table_names.len() {
        assert!(
            table_names.value(idx).eq_ignore_ascii_case("audit_log"),
            "unexpected table name {}",
            table_names.value(idx)
        );
    }
    assert_eq!(column_names.value(0), "entry_id");
    assert_eq!(column_names.value(1), "payload");
    assert_eq!(ordinal_positions.value(0), 1);
    assert_eq!(ordinal_positions.value(1), 2);
    assert_eq!(nullable_flags.len(), 2);
}
