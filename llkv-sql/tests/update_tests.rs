use arrow::util::pretty::pretty_format_batches;
use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_update_via_datafusion() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager).expect("sql processor");
    let rt = Runtime::new().expect("runtime");

    rt.block_on(async {
        engine
            .execute("CREATE TABLE t1(id INT, val TEXT);")
            .await
            .expect("create table");

        engine
            .execute("INSERT INTO t1(id, val) VALUES (1, 'a'), (2, 'b');")
            .await
            .expect("insert");

        let pre_update = engine
            .execute("SELECT id, val FROM t1 ORDER BY id")
            .await
            .expect("select pre");

        let batches = match pre_update.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+----+-----+",
            "| id | val |",
            "+----+-----+",
            "| 1  | a   |",
            "| 2  | b   |",
            "+----+-----+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        // This currently uses the manual path.
        // If we want to test DataFusion path, we might need to bypass the manual check
        // or just observe that it works currently.
        engine
            .execute("UPDATE t1 SET val = 'c' WHERE id = 1;")
            .await
            .expect("update");

        let results = engine
            .execute("SELECT id, val FROM t1 ORDER BY id")
            .await
            .expect("select");

        let batches = match results.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();

        let expected = vec![
            "+----+-----+",
            "| id | val |",
            "+----+-----+",
            "| 1  | c   |",
            "| 2  | b   |",
            "+----+-----+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        // Test Range Update
        engine
            .execute("UPDATE t1 SET val = 'd' WHERE id > 1;")
            .await
            .expect("update range");

        // TODO: Fix so that `SELECT *` doesn't show meta `row_id` by default
        let results_range = engine
            .execute("SELECT id, val FROM t1 ORDER BY id")
            .await
            .expect("select range");

        let batches_range = match results_range.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted_range = pretty_format_batches(&batches_range).unwrap().to_string();

        let expected_range = vec![
            "+----+-----+",
            "| id | val |",
            "+----+-----+",
            "| 1  | c   |",
            "| 2  | d   |",
            "+----+-----+",
        ];
        assert_eq!(formatted_range.trim(), expected_range.join("\n").trim());
    });
}

#[test]
fn test_update_example() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager).expect("sql processor");
    let rt = Runtime::new().expect("runtime");

    rt.block_on(async {
        engine
            .execute("CREATE TABLE t1(x INTEGER, y VARCHAR(8))")
            .await
            .expect("create table");
        engine
            .execute("INSERT INTO t1(x, y) VALUES(1,'true'), (0,'false'), (NULL,'NULL')")
            .await
            .expect("insert");

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY x NULLS LAST")
            .await
            .expect("select after insert");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+---+-------+",
            "| x | y     |",
            "+---+-------+",
            "| 0 | false |",
            "| 1 | true  |",
            "|   | NULL  |",
            "+---+-------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        engine.execute("UPDATE t1 SET x=3").await.expect("update");
        // Check update result if possible, but usually it returns row count or empty

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY y")
            .await
            .expect("select after update");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+---+-------+",
            "| x | y     |",
            "+---+-------+",
            "| 3 | NULL  |",
            "| 3 | false |",
            "| 3 | true  |",
            "+---+-------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        let result = engine
            .execute("SELECT count(*) FROM t1 WHERE x=3")
            .await
            .expect("count");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+----------+",
            "| count(*) |",
            "+----------+",
            "| 3        |",
            "+----------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());
    });
}

#[test]
fn test_update2() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager).expect("sql processor");
    let rt = Runtime::new().expect("runtime");

    rt.block_on(async {
        engine
            .execute("CREATE TABLE t1(x INTEGER, y VARCHAR(8))")
            .await
            .expect("create table");
        engine
            .execute("INSERT INTO t1(x, y) VALUES(1,'true'), (0,'false'), (NULL,'NULL')")
            .await
            .expect("insert");

        // Update just one row first
        engine
            .execute("UPDATE t1 SET x=99 WHERE x=1")
            .await
            .expect("update single");

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY y")
            .await
            .expect("select 1");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+----+-------+",
            "| x  | y     |",
            "+----+-------+",
            "|    | NULL  |",
            "| 0  | false |",
            "| 99 | true  |",
            "+----+-------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        // Now update all
        engine
            .execute("UPDATE t1 SET x=3")
            .await
            .expect("update all");

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY y")
            .await
            .expect("select 2");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+---+-------+",
            "| x | y     |",
            "+---+-------+",
            "| 3 | NULL  |",
            "| 3 | false |",
            "| 3 | true  |",
            "+---+-------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        let result = engine
            .execute("SELECT count(*) FROM t1 WHERE x=3")
            .await
            .expect("count");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+----------+",
            "| count(*) |",
            "+----------+",
            "| 3        |",
            "+----------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());
    });
}

#[test]
fn test_update3() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager).expect("sql processor");
    let rt = Runtime::new().expect("runtime");

    rt.block_on(async {
        engine
            .execute("CREATE TABLE t1(x INTEGER, y VARCHAR(8))")
            .await
            .expect("create table");
        engine
            .execute("INSERT INTO t1(x, y) VALUES(1,'true'), (0,'false'), (NULL,'NULL')")
            .await
            .expect("insert");

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY x NULLS LAST")
            .await
            .expect("select before");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+---+-------+",
            "| x | y     |",
            "+---+-------+",
            "| 0 | false |",
            "| 1 | true  |",
            "|   | NULL  |",
            "+---+-------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        engine
            .execute("UPDATE t1 SET x=3 WHERE x IS NULL")
            .await
            .expect("update");

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY x")
            .await
            .expect("select after");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+---+-------+",
            "| x | y     |",
            "+---+-------+",
            "| 0 | false |",
            "| 1 | true  |",
            "| 3 | NULL  |",
            "+---+-------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        let result = engine
            .execute("SELECT count(*) FROM t1 WHERE x=3")
            .await
            .expect("count");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+----------+",
            "| count(*) |",
            "+----------+",
            "| 1        |",
            "+----------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());
    });
}
