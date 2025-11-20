use llkv_sql::SqlEngine;
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
    let engine = SqlEngine::new(pager)?;
    
    engine.execute("CREATE TABLE t1(x INTEGER, y VARCHAR(8))").await?;
    engine.execute("INSERT INTO t1 VALUES(1,'true')").await?;
    engine.execute("INSERT INTO t1 VALUES(0,'false')").await?;
    engine.execute("INSERT INTO t1 VALUES(NULL,'NULL')").await?;
    
    // Update just one row first
    println!("Test 1: UPDATE single row WHERE x=1");
    engine.execute("UPDATE t1 SET x=99 WHERE x=1").await?;
    
    let result = engine.execute("SELECT * FROM t1 ORDER BY y").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }
    
    // Now update all
    println!("\nTest 2: UPDATE all rows");
    engine.execute("UPDATE t1 SET x=3").await?;
    
    let result = engine.execute("SELECT * FROM t1 ORDER BY y").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }
    
    let result = engine.execute("SELECT count(*) FROM t1 WHERE x=3").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("\ncount(*) WHERE x=3: {:?}", batch);
            }
        }
    }
    
    Ok(())
}
