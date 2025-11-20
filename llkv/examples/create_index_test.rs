//! Test CREATE INDEX functionality through llkv-sql

use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let pager = Arc::new(MemPager::default());
    let engine = SqlEngine::new(Arc::new(llkv_storage::pager::BoxedPager::from_arc(pager)))
        .expect("failed to create SQL engine");

    // Create a table
    engine
        .execute("CREATE TABLE t1 (x INTEGER, y TEXT)")
        .await
        .expect("failed to create table");

    // Insert some data
    engine
        .execute("INSERT INTO t1 VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        .await
        .expect("failed to insert data");

    // Create an index on column x
    engine
        .execute("CREATE INDEX t1i1 ON t1(x)")
        .await
        .expect("failed to create index");

    println!("✓ CREATE INDEX succeeded");

    // Try to create the same index again (should fail)
    match engine.execute("CREATE INDEX t1i1 ON t1(x)").await {
        Err(e) => println!("✓ Duplicate index correctly rejected: {}", e),
        Ok(_) => panic!("Should have failed to create duplicate index"),
    }

    // Create index with IF NOT EXISTS (should succeed silently)
    engine
        .execute("CREATE INDEX IF NOT EXISTS t1i1 ON t1(x)")
        .await
        .expect("failed to create index with IF NOT EXISTS");

    println!("✓ CREATE INDEX IF NOT EXISTS succeeded");

    // Create another index on column y
    engine
        .execute("CREATE INDEX t1i2 ON t1(y)")
        .await
        .expect("failed to create second index");

    println!("✓ Created second index");

    // Query the table to verify indexes don't break queries
    let results = engine
        .execute("SELECT * FROM t1 WHERE x = 2")
        .await
        .expect("failed to query table");

    println!("✓ Query after index creation succeeded: {:?}", results);

    println!("\n✅ All CREATE INDEX tests passed!");
}
