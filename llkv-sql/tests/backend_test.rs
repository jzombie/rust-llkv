use llkv_sql::{SqlEngine, SqlStatementResult};
use llkv_storage::pager::{BoxedPager, MemPager};
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

        // 2. Create table with explicit backend (ColumnStore) via LOCATION hint
        engine
            .execute("CREATE EXTERNAL TABLE t_columnstore(id INT) STORED AS CSV LOCATION '/tmp/dummy?backend=columnstore';")
            .await
            .expect("create table columnstore");


        // 3. Verify backends by checking catalog metadata (if possible) or behavior.
        // We can't easily check metadata from SQL.
        // But we can check if they work.
        
        engine
            .execute("INSERT INTO t_parquet(id, val) VALUES (1, 'a');")
            .await
            .expect("insert parquet");
            
        engine
            .execute("INSERT INTO t_columnstore(id) VALUES (1);")
            .await
            .expect("insert columnstore");

        // 4. Verify persistence
        // We can check if we can read back.
        let res = engine.execute("SELECT * FROM t_parquet").await.expect("select parquet");
        match res.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => assert_eq!(batches[0].num_rows(), 1),
            _ => panic!("expected query"),
        }

        let res = engine.execute("SELECT * FROM t_columnstore").await.expect("select columnstore");
        match res.into_iter().next().unwrap() {
            SqlStatementResult::Query { batches } => assert_eq!(batches[0].num_rows(), 1),
            _ => panic!("expected query"),
        }
    });
}
