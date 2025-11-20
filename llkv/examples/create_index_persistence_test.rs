//! Test that CREATE INDEX persists indexes to the pager and they survive restarts

use llkv_sql::SqlEngine;
use llkv_storage::pager::SimdRDrivePager;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let db_path = std::env::temp_dir().join(format!("llkv_index_test_{}.db", std::process::id()));

    println!("📁 Test database: {:?}", db_path);

    // Phase 1: Create database, table, and index
    {
        println!("\n=== Phase 1: Creating database, table, and index ===");
        let pager = Arc::new(SimdRDrivePager::open(&db_path).expect("failed to create pager"));
        let engine = SqlEngine::new(Arc::new(llkv_storage::pager::BoxedPager::from_arc(pager)))
            .expect("failed to create SQL engine");

        // Create a table
        engine
            .execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
            .await
            .expect("failed to create table");
        println!("✓ Created table 'users'");

        // Insert some data
        engine
            .execute(
                "INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35)",
            )
            .await
            .expect("failed to insert data");
        println!("✓ Inserted 3 rows");

        // Create indexes
        engine
            .execute("CREATE INDEX idx_id ON users(id)")
            .await
            .expect("failed to create index on id");
        println!("✓ Created index 'idx_id' on column 'id'");

        engine
            .execute("CREATE INDEX idx_name ON users(name)")
            .await
            .expect("failed to create index on name");
        println!("✓ Created index 'idx_name' on column 'name'");

        engine
            .execute("CREATE INDEX idx_age ON users(age)")
            .await
            .expect("failed to create index on age");
        println!("✓ Created index 'idx_age' on column 'age'");

        // Query to verify data is accessible
        let results = engine
            .execute("SELECT * FROM users WHERE id = 2")
            .await
            .expect("failed to query");
        println!("✓ Query returned: {:?}", results);

        // Drop engine to close the database
        drop(engine);
        println!("✓ Closed database");
    }

    // Phase 2: Reopen database and verify indexes still exist
    {
        println!("\n=== Phase 2: Reopening database ===");
        let pager = Arc::new(SimdRDrivePager::open(&db_path).expect("failed to reopen pager"));
        let engine = SqlEngine::new(Arc::new(llkv_storage::pager::BoxedPager::from_arc(pager)))
            .expect("failed to create SQL engine");

        println!("✓ Reopened database");

        // Query the table - indexes should still exist
        let results = engine
            .execute("SELECT * FROM users WHERE name = 'Bob'")
            .await
            .expect("failed to query after restart");
        println!("✓ Query after restart returned: {:?}", results);

        // Try to create the same index again - should fail if index persisted
        match engine.execute("CREATE INDEX idx_id ON users(id)").await {
            Err(e) => {
                println!(
                    "✓ Index 'idx_id' persisted! (duplicate creation failed: {})",
                    e
                );
            }
            Ok(_) => {
                panic!("❌ Index did NOT persist - was able to create duplicate index!");
            }
        }

        match engine.execute("CREATE INDEX idx_name ON users(name)").await {
            Err(e) => {
                println!(
                    "✓ Index 'idx_name' persisted! (duplicate creation failed: {})",
                    e
                );
            }
            Ok(_) => {
                panic!("❌ Index did NOT persist - was able to create duplicate index!");
            }
        }

        match engine.execute("CREATE INDEX idx_age ON users(age)").await {
            Err(e) => {
                println!(
                    "✓ Index 'idx_age' persisted! (duplicate creation failed: {})",
                    e
                );
            }
            Ok(_) => {
                panic!("❌ Index did NOT persist - was able to create duplicate index!");
            }
        }

        // Use IF NOT EXISTS - should succeed without error
        engine
            .execute("CREATE INDEX IF NOT EXISTS idx_id ON users(id)")
            .await
            .expect("failed with IF NOT EXISTS");
        println!("✓ IF NOT EXISTS works correctly on persisted index");

        drop(engine);
    }

    // Phase 3: Reopen again and insert more data
    {
        println!("\n=== Phase 3: Adding more data to indexed table ===");
        let pager = Arc::new(SimdRDrivePager::open(&db_path).expect("failed to reopen pager"));
        let engine = SqlEngine::new(Arc::new(llkv_storage::pager::BoxedPager::from_arc(pager)))
            .expect("failed to create SQL engine");

        // Insert more data
        engine
            .execute("INSERT INTO users VALUES (4, 'Diana', 28), (5, 'Eve', 32)")
            .await
            .expect("failed to insert more data");
        println!("✓ Inserted 2 more rows");

        // Query all data
        let results = engine
            .execute("SELECT * FROM users ORDER BY id")
            .await
            .expect("failed to query all");
        println!("✓ Final query returned {} batches", results.len());

        drop(engine);
    }

    // Clean up test database
    let _ = std::fs::remove_file(&db_path);

    println!("\n✅ All persistence tests passed!");
    println!("✅ Indexes successfully persist to pager across database restarts!");
}
