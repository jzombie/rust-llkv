use llkv_sql::SqlEngine;
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
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

    // Do the exact sequence from the SLT test
    println!("\n1. UPDATE t1 SET x=1 WHERE x>0");
    engine.execute("UPDATE t1 SET x=1 WHERE x>0").await?;

    println!("\n2. UPDATE t1 SET x=2 WHERE x>0");
    engine.execute("UPDATE t1 SET x=2 WHERE x>0").await?;

    println!("\n3. UPDATE t1 SET y='true' WHERE x>0");
    engine.execute("UPDATE t1 SET y='true' WHERE x>0").await?;

    println!("\n4. UPDATE t1 SET y='unknown' WHERE x>0");
    engine
        .execute("UPDATE t1 SET y='unknown' WHERE x>0")
        .await?;

    println!("\nAfter WHERE updates:");
    let result = engine.execute("SELECT * FROM t1").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }

    println!("\n5. UPDATE t1 SET x=99 (no WHERE)");
    engine.execute("UPDATE t1 SET x=99").await?;

    println!("\nAfter UPDATE - SELECT *:");
    let result = engine.execute("SELECT * FROM t1").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }
    
    println!("\nAfter UPDATE - SELECT x:");
    let result = engine.execute("SELECT x FROM t1").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }
    
    println!("\nAfter UPDATE - COUNT(*):");
    let result = engine.execute("SELECT COUNT(*) FROM t1 WHERE x=99").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("{:?}", batch);
            }
        }
    }

    Ok(())
}
