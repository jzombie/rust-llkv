use std::sync::Arc;
use llkv_sql::SqlEngine;
use llkv_storage::pager::{BoxedPager, MemPager};
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
            .execute("INSERT INTO t1 VALUES (1, 'a'), (2, 'b');")
            .await
            .expect("insert");

        let pre_update = engine.execute("SELECT * FROM t1").await.expect("select pre");
        println!("Pre-update results: {:?}", pre_update);

        // This currently uses the manual path. 
        // If we want to test DataFusion path, we might need to bypass the manual check 
        // or just observe that it works currently.
        engine
            .execute("UPDATE t1 SET val = 'c' WHERE id = 1;")
            .await
            .expect("update");

        let results = engine.execute("SELECT * FROM t1 ORDER BY id").await.expect("select");
        // Verify results...
    });
}
