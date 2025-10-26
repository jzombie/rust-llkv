use indoc::indoc;
use llkv_slt_tester::LlkvSltRunner;

const MULTI_COLUMN_SLT: &str = indoc! { r#"
    statement ok
    CREATE TABLE t(a INT, b INT, c INT);

    statement ok
    INSERT INTO t VALUES (1, 2, 3);

    statement ok
    INSERT INTO t VALUES (4, 5, 6);

    query III
    SELECT a, b, c FROM t ORDER BY a;
    ----
    1
    2
    3
    4
    5
    6

    # DuckDB tests may return multi-column results in a single line
    query III rowsort
    SELECT a, b, c FROM t;
    ----
    1 2 3
    4 5 6
"# };

const MULTI_COLUMN_BAD_SLT: &str = indoc! { r#"
    statement ok
    CREATE TABLE t(a INT, b INT, c INT);

    statement ok
    INSERT INTO t VALUES (1, 2, 3);

    statement ok
    INSERT INTO t VALUES (4, 5, 6);

    query III
    SELECT a, b, c FROM t ORDER BY a;
    ----
    1
    2
    3
    4
    5
    999
"# };

#[test]
fn multi_column_queries_succeed_with_flattening() {
    LlkvSltRunner::in_memory()
        .run_script("multi-column.slt", MULTI_COLUMN_SLT)
        .expect("multi-column SLT script should pass");
}

#[test]
fn multi_column_queries_still_fail_on_real_mismatches() {
    let result =
        LlkvSltRunner::in_memory().run_script("multi-column-bad.slt", MULTI_COLUMN_BAD_SLT);
    assert!(result.is_err(), "expected mismatch should still fail");
}
