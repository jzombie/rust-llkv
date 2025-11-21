use arrow::util::pretty::pretty_format_batches;
use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_null_update() {
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
            .execute("SELECT x, y FROM t1")
            .await
            .expect("select after insert");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let _formatted = pretty_format_batches(&batches).unwrap().to_string();
        // Order is not guaranteed without ORDER BY, but let's assume insertion order or sort it
        // Actually, let's use ORDER BY x NULLS LAST to be deterministic
        // But we didn't use ORDER BY in the query above.
        // Let's just check that we got 3 rows.
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 3);

        // Do the exact sequence from the SLT test
        engine
            .execute("UPDATE t1 SET x=1 WHERE x>0")
            .await
            .expect("update 1");

        engine
            .execute("UPDATE t1 SET x=2 WHERE x>0")
            .await
            .expect("update 2");

        engine
            .execute("UPDATE t1 SET y='true' WHERE x>0")
            .await
            .expect("update 3");

        engine
            .execute("UPDATE t1 SET y='unknown' WHERE x>0")
            .await
            .expect("update 4");

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY x NULLS LAST")
            .await
            .expect("select after updates");

        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+---+---------+",
            "| x | y       |",
            "+---+---------+",
            "| 0 | false   |",
            "| 2 | unknown |",
            "|   | NULL    |",
            "+---+---------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        engine
            .execute("UPDATE t1 SET x=99")
            .await
            .expect("update 5");

        let result = engine
            .execute("SELECT x, y FROM t1 ORDER BY y")
            .await
            .expect("select *");
        let batches = match result.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => batches,
            _ => panic!("expected query results"),
        };
        let formatted = pretty_format_batches(&batches).unwrap().to_string();
        let expected = vec![
            "+----+---------+",
            "| x  | y       |",
            "+----+---------+",
            "| 99 | NULL    |",
            "| 99 | false   |",
            "| 99 | unknown |",
            "+----+---------+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        let result = engine
            .execute("SELECT COUNT(*) FROM t1 WHERE x=99")
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
