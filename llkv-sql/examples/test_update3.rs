use llkv_column_map::store::GatherNullPolicy;
use llkv_column_map::types::LogicalFieldId;
use llkv_sql::SqlEngine;
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager_arc = Arc::new(MemPager::default());
    let pager = Arc::new(BoxedPager::from_arc(Arc::clone(&pager_arc)));
    let engine = SqlEngine::new(pager)?;

    engine
        .execute("CREATE TABLE t1(x INTEGER, y VARCHAR(8))")
        .await?;
    engine.execute("INSERT INTO t1 VALUES(1,'true')").await?;
    engine.execute("INSERT INTO t1 VALUES(0,'false')").await?;
    engine.execute("INSERT INTO t1 VALUES(NULL,'NULL')").await?;

    println!("After INSERT - SQL:");
    let result = engine.execute("SELECT * FROM t1 ORDER BY y").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("  {:?}", batch);
            }
        }
    }

    println!("\nExecuting: UPDATE t1 SET x=3");
    engine.execute("UPDATE t1 SET x=3").await?;

    // DIRECT column store check - removed as store is no longer accessible directly
    // println!("\nDirect ColumnStore check after UPDATE:");
    // let catalog = engine.catalog();
    // let store = catalog.store();
    // let field_id = LogicalFieldId::for_user_table_0(1); // First field of first table
    // let result = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    // println!("  gather_rows([1,2,3]): {:?}", result);

    println!("\nAfter UPDATE - SQL:");
    let result = engine.execute("SELECT * FROM t1 ORDER BY y").await?;
    for stmt_result in &result {
        if let llkv_sql::SqlStatementResult::Query { batches } = stmt_result {
            for batch in batches {
                println!("  {:?}", batch);
            }
        }
    }

    Ok(())
}
