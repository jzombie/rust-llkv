use indoc::indoc;
use llkv_slt_tester::LlkvSltRunner;

#[test]
fn sum_of_negated_column_matches_reference() {
    let script = indoc! { r#"
        statement ok
        CREATE TABLE t(v INT);

        statement ok
        INSERT INTO t VALUES (1),(2);

        query I
        SELECT sum(-v) FROM t;
        ----
        -3
    "# };

    let runner = LlkvSltRunner::in_memory();
    let result = runner.run_script("sum-neg.slt", script);
    assert!(result.is_ok(), "{:?}", result.err());
}
