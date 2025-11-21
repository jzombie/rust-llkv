use arrow::util::pretty::pretty_format_batches;
use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_delete_via_datafusion() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager).expect("sql engine");
    let rt = Runtime::new().expect("runtime");

    rt.block_on(async {
        engine
            .execute("CREATE TABLE t1(id INT, val TEXT);")
            .await
            .expect("create table");

        engine
            .execute("INSERT INTO t1(id, val) VALUES (1, 'a'), (2, 'b'), (3, 'c')")
            .await
            .expect("insert");

        // Delete single row
        engine
            .execute("DELETE FROM t1 WHERE id = 2;")
            .await
            .expect("delete");

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
            "| 1  | a   |",
            "| 3  | c   |",
            "+----+-----+",
        ];
        assert_eq!(formatted.trim(), expected.join("\n").trim());

        // Delete range
        engine
            .execute("DELETE FROM t1 WHERE id > 1;")
            .await
            .expect("delete range");

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
            "| 1  | a   |",
            "+----+-----+",
        ];
        assert_eq!(formatted_range.trim(), expected_range.join("\n").trim());
    });
}
