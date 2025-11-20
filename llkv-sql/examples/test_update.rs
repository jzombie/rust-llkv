use llkv_sql::SqlEngine;
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager_arc = Arc::new(MemPager::default());
    let pager = Arc::new(BoxedPager::from_arc(pager_arc));
    let engine = SqlEngine::new(pager)?;

    engine
        .execute("CREATE TABLE t1(x INTEGER, y VARCHAR(8))")
        .await?;
    engine.execute("INSERT INTO t1 VALUES(1,'true')").await?;
    engine.execute("INSERT INTO t1 VALUES(0,'false')").await?;
    engine.execute("INSERT INTO t1 VALUES(NULL,'NULL')").await?;

    println!("After INSERT:");
    let result = engine.execute("SELECT * FROM t1").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }

    println!("\nExecuting: UPDATE t1 SET x=3");
    let update_result = engine.execute("UPDATE t1 SET x=3").await?;
    println!("Update result: {:?}", update_result);

    println!("\nAfter UPDATE:");
    let result = engine.execute("SELECT * FROM t1").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }

    println!("\nExecuting: SELECT count(*) FROM t1 WHERE x=3");
    let result = engine.execute("SELECT count(*) FROM t1 WHERE x=3").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }

    Ok(())
}
