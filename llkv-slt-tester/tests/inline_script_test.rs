use indoc::indoc;
use llkv_slt_tester::LlkvSltRunner;

const SIMPLE_SLT_GOOD: &str = indoc! { r#"
    statement ok
    CREATE TABLE t(a INT);

    statement ok
    INSERT INTO t VALUES (1);

    query I
    SELECT a FROM t ORDER BY 1;
    ----
    1
"# };

// Slightly incorrect expected output (empty) to force a failure.
const SIMPLE_SLT_BAD: &str = indoc! { r#"
    statement ok
    CREATE TABLE t(a INT);

    statement ok
    INSERT INTO t VALUES (1);

    query I
    SELECT a FROM t ORDER BY 1;
    ----
    9
"# };

#[test]
fn run_script_from_string() {
    LlkvSltRunner::in_memory()
        .run_script("inline-test.slt", SIMPLE_SLT_GOOD)
        .expect("inline SLT should pass");
}

#[test]
fn run_script_from_reader() {
    let cursor = std::io::Cursor::new(SIMPLE_SLT_GOOD);
    LlkvSltRunner::in_memory()
        .run_reader("cursor-test.slt", cursor)
        .expect("reader SLT should pass");
}

#[test]
fn run_script_bad_fails() {
    let res = LlkvSltRunner::in_memory().run_script("bad-test.slt", SIMPLE_SLT_BAD);
    assert!(res.is_err(), "expected the bad SLT to fail but it succeeded");
}
