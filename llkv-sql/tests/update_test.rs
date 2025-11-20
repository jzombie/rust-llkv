use arrow::util::pretty::pretty_format_batches;
use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_update_via_datafusion() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager).expect("sql engine");
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
            .execute("SELECT id, val FROM t1")
            .await
            .expect("select pre");
        println!("Pre-update results: {:?}", pre_update);

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
