use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
use llkv_table::{ColumnMapTableProvider, LlkvTableProvider};
use std::sync::Arc;
use tokio::runtime::Runtime;

#[test]
fn test_backend_selection() {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager).expect("sql engine");
    let rt = Runtime::new().expect("runtime");

    rt.block_on(async {
        // 1. Create table with default backend (Parquet)
        engine
            .execute("CREATE TABLE t_parquet(id INT, val TEXT);")
            .await
            .expect("create table parquet");

        // 2. Create table with explicit backend (ColumnStore) via WITH options
        engine
            .execute("CREATE TABLE t_columnstore(id INT) WITH (backend = 'columnstore');")
            .await
            .expect("create table columnstore");

        // 3. Verify backends by checking catalog metadata
        let catalog = engine.catalog();

        let t_parquet = catalog
            .get_table("t_parquet")
            .expect("t_parquet exists")
            .expect("t_parquet is some");
        assert!(
            t_parquet
                .as_any()
                .downcast_ref::<LlkvTableProvider<BoxedPager>>()
                .is_some(),
            "t_parquet should be LlkvTableProvider (Parquet)"
        );

        let t_columnstore = catalog
            .get_table("t_columnstore")
            .expect("t_columnstore exists")
            .expect("t_columnstore is some");
        assert!(
            t_columnstore
                .as_any()
                .downcast_ref::<ColumnMapTableProvider<BoxedPager>>()
                .is_some(),
            "t_columnstore should be ColumnMapTableProvider"
        );

        // 4. Verify operations work
        engine
            .execute("INSERT INTO t_parquet(id, val) VALUES (1, 'a');")
            .await
            .expect("insert parquet");

        engine
            .execute("INSERT INTO t_columnstore(id) VALUES (1);")
            .await
            .expect("insert columnstore");

        // 5. Verify persistence
        // We can check if we can read back.
        let res = engine
            .execute("SELECT * FROM t_parquet")
            .await
            .expect("select parquet");
        match res.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => assert_eq!(batches[0].num_rows(), 1),
            _ => panic!("expected query"),
        }

        let res = engine
            .execute("SELECT * FROM t_columnstore")
            .await
            .expect("select columnstore");
        match res.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => assert_eq!(batches[0].num_rows(), 1),
            _ => panic!("expected query"),
        }
    });
}
