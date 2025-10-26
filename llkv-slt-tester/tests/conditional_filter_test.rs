/// Integration tests for conditional directive filtering (onlyif/skipif).
///
/// These tests verify that the SLT runner correctly filters test blocks based on
/// database engine compatibility directives.
use indoc::indoc;
use llkv_slt_tester::LlkvSltRunner;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper to create a temporary .slt file with the given content
fn create_temp_slt(content: &str) -> NamedTempFile {
    let mut temp_file = NamedTempFile::with_suffix(".slt").expect("Failed to create temp file");
    temp_file
        .write_all(content.as_bytes())
        .expect("Failed to write");
    temp_file.flush().expect("Failed to flush");
    temp_file
}

#[test]
fn test_onlyif_sqlite_included() {
    // Test that "onlyif sqlite" blocks ARE included (sqlite is in our compat list)
    let content = indoc! {"
        statement ok
        CREATE TABLE test1 (id INTEGER)

        onlyif sqlite
        statement ok
        INSERT INTO test1 VALUES (1)

        statement ok
        INSERT INTO test1 VALUES (2)

        query I rowsort
        SELECT * FROM test1
        ----
        1
        2
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - both inserts should run
    assert!(
        result.is_ok(),
        "Expected success for onlyif sqlite block: {:?}",
        result.err()
    );
}

#[test]
fn test_onlyif_duckdb_included() {
    // Test that "onlyif duckdb" blocks ARE included (duckdb is in our compat list)
    let content = indoc! {"
        statement ok
        CREATE TABLE test2 (id INTEGER)

        onlyif duckdb
        statement ok
        INSERT INTO test2 VALUES (10)

        query I
        SELECT * FROM test2
        ----
        10
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - duckdb-specific insert should run
    assert!(
        result.is_ok(),
        "Expected success for onlyif duckdb block: {:?}",
        result.err()
    );
}

#[test]
fn test_onlyif_mysql_excluded() {
    // Test that "onlyif mysql" blocks are EXCLUDED (mysql not in our compat list)
    let content = indoc! {"
        statement ok
        CREATE TABLE test3 (id INTEGER)

        statement ok
        INSERT INTO test3 VALUES (1)

        onlyif mysql
        statement ok
        INSERT INTO test3 VALUES (999)

        query I
        SELECT * FROM test3
        ----
        1
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - mysql-only insert should be skipped, only value 1 present
    assert!(
        result.is_ok(),
        "Expected success with mysql block excluded: {:?}",
        result.err()
    );
}

#[test]
fn test_skipif_sqlite_excluded() {
    // Test that "skipif sqlite" blocks are EXCLUDED
    let content = indoc! {"
        statement ok
        CREATE TABLE test4 (id INTEGER)

        statement ok
        INSERT INTO test4 VALUES (1)

        skipif sqlite
        statement ok
        INSERT INTO test4 VALUES (999)

        query I
        SELECT * FROM test4
        ----
        1
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - skipif sqlite means skip for us, only value 1 present
    assert!(
        result.is_ok(),
        "Expected success with skipif sqlite block excluded: {:?}",
        result.err()
    );
}

#[test]
fn test_skipif_duckdb_excluded() {
    // Test that "skipif duckdb" blocks are EXCLUDED
    let content = indoc! {"
        statement ok
        CREATE TABLE test5 (id INTEGER)

        statement ok
        INSERT INTO test5 VALUES (1)

        skipif duckdb
        statement ok
        INSERT INTO test5 VALUES (999)

        query I
        SELECT * FROM test5
        ----
        1
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - skipif duckdb means skip for us, only value 1 present
    assert!(
        result.is_ok(),
        "Expected success with skipif duckdb block excluded: {:?}",
        result.err()
    );
}

#[test]
fn test_skipif_mysql_included() {
    // Test that "skipif mysql" blocks are INCLUDED (we're not mysql)
    let content = indoc! {"
        statement ok
        CREATE TABLE test6 (id INTEGER)

        skipif mysql
        statement ok
        INSERT INTO test6 VALUES (42)

        query I
        SELECT * FROM test6
        ----
        42
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - skipif mysql means include for us
    assert!(
        result.is_ok(),
        "Expected success with skipif mysql block included: {:?}",
        result.err()
    );
}

#[test]
fn test_multiple_conditionals() {
    // Test multiple conditional directives in sequence
    let content = indoc! {"
        statement ok
        CREATE TABLE test7 (id INTEGER)

        statement ok
        INSERT INTO test7 VALUES (1)

        onlyif mysql
        statement ok
        INSERT INTO test7 VALUES (100)

        skipif postgresql
        statement ok
        INSERT INTO test7 VALUES (2)

        onlyif sqlite
        statement ok
        INSERT INTO test7 VALUES (3)

        skipif sqlite
        statement ok
        INSERT INTO test7 VALUES (200)

        query I rowsort
        SELECT * FROM test7
        ----
        1
        2
        3
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed with values 1, 2, 3 (skipping mysql-only and skipif sqlite)
    assert!(
        result.is_ok(),
        "Expected success with multiple conditionals: {:?}",
        result.err()
    );
}

#[test]
fn test_onlyif_postgresql_excluded() {
    // Test that "onlyif postgresql" blocks are EXCLUDED
    let content = indoc! {"
        statement ok
        CREATE TABLE test8 (id INTEGER)

        onlyif postgresql
        statement ok
        INSERT INTO test8 VALUES (999)

        statement ok
        INSERT INTO test8 VALUES (1)

        query I
        SELECT * FROM test8
        ----
        1
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - postgresql-only block should be skipped
    assert!(
        result.is_ok(),
        "Expected success with postgresql block excluded: {:?}",
        result.err()
    );
}

#[test]
fn test_no_conditionals() {
    // Test that files without conditional directives work normally
    let content = indoc! {"
        statement ok
        CREATE TABLE test9 (id INTEGER)

        statement ok
        INSERT INTO test9 VALUES (1)

        statement ok
        INSERT INTO test9 VALUES (2)

        query I rowsort
        SELECT * FROM test9
        ----
        1
        2
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - no conditionals means run everything
    assert!(
        result.is_ok(),
        "Expected success with no conditionals: {:?}",
        result.err()
    );
}

#[test]
fn test_conditional_with_query_block() {
    // Test conditionals with query blocks (not just statements)
    let content = indoc! {"
        statement ok
        CREATE TABLE test10 (id INTEGER)

        statement ok
        INSERT INTO test10 VALUES (1)

        onlyif mysql
        query I
        SELECT * FROM test10 WHERE id = 999
        ----
        999

        query I
        SELECT * FROM test10
        ----
        1
    "};

    let temp_file = create_temp_slt(content);
    let runner = LlkvSltRunner::in_memory();

    let result = runner.run_file(temp_file.path());

    // Should succeed - mysql-only query should be skipped
    assert!(
        result.is_ok(),
        "Expected success with conditional query block: {:?}",
        result.err()
    );
}
