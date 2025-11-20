//! Integration tests for CREATE INDEX statement

use llkv_sql::SqlEngine;
use llkv_storage::pager::{BoxedPager, MemPager, SimdRDrivePager};
use std::sync::Arc;

#[tokio::test]
async fn test_create_index_basic() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

    // Create a table
    engine
        .execute("CREATE TABLE t1 (x INTEGER, y TEXT)")
        .await
        .expect("failed to create table");

    // Create an index
    engine
        .execute("CREATE INDEX idx1 ON t1(x)")
        .await
        .expect("failed to create index");

    // Try to create the same index again - should fail
    let result = engine.execute("CREATE INDEX idx1 ON t1(x)").await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("index already exists")
    );
}

#[tokio::test]
async fn test_create_index_if_not_exists() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

    // Create a table
    engine
        .execute("CREATE TABLE t1 (x INTEGER)")
        .await
        .expect("failed to create table");

    // Create an index
    engine
        .execute("CREATE INDEX idx1 ON t1(x)")
        .await
        .expect("failed to create index");

    // Create the same index with IF NOT EXISTS - should succeed
    engine
        .execute("CREATE INDEX IF NOT EXISTS idx1 ON t1(x)")
        .await
        .expect("IF NOT EXISTS should succeed");
}

#[tokio::test]
async fn test_create_multiple_indexes() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

    // Create a table
    engine
        .execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
        .await
        .expect("failed to create table");

    // Create multiple indexes
    engine
        .execute("CREATE INDEX idx_id ON users(id)")
        .await
        .expect("failed to create index on id");

    engine
        .execute("CREATE INDEX idx_name ON users(name)")
        .await
        .expect("failed to create index on name");

    engine
        .execute("CREATE INDEX idx_age ON users(age)")
        .await
        .expect("failed to create index on age");

    // All indexes should be created successfully
}

#[tokio::test]
async fn test_create_index_nonexistent_table() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

    // Try to create an index on a non-existent table
    let result = engine.execute("CREATE INDEX idx1 ON t1(x)").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_create_index_nonexistent_column() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

    // Create a table
    engine
        .execute("CREATE TABLE t1 (x INTEGER, y TEXT)")
        .await
        .expect("failed to create table");

    // Try to create an index on a non-existent column
    let result = engine.execute("CREATE INDEX idx1 ON t1(z)").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[tokio::test]
async fn test_create_index_persists() {
    let db_path = std::env::temp_dir().join(format!("llkv_test_{}.db", std::process::id()));

    // Phase 1: Create database, table, and index
    {
        let pager = Arc::new(SimdRDrivePager::open(&db_path).expect("failed to create pager"));
        let engine = SqlEngine::new(Arc::new(BoxedPager::from_arc(pager)))
            .expect("failed to create SQL engine");

        engine
            .execute("CREATE TABLE t1 (x INTEGER, y TEXT)")
            .await
            .expect("failed to create table");

        engine
            .execute("CREATE INDEX idx1 ON t1(x)")
            .await
            .expect("failed to create index");

        // Engine drops here, closing the database
    }

    // Phase 2: Reopen database and verify index persisted
    {
        let pager = Arc::new(SimdRDrivePager::open(&db_path).expect("failed to reopen pager"));
        let engine = SqlEngine::new(Arc::new(BoxedPager::from_arc(pager)))
            .expect("failed to create SQL engine");

        // Try to create the same index again - should fail if index persisted
        let result = engine.execute("CREATE INDEX idx1 ON t1(x)").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("index already exists")
        );
    }

    // Clean up
    let _ = std::fs::remove_file(&db_path);
}

#[tokio::test]
async fn test_create_index_with_data() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

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

    // Create an index on existing data
    engine
        .execute("CREATE INDEX idx1 ON t1(x)")
        .await
        .expect("failed to create index");

    // Query should still work
    let results = engine
        .execute("SELECT * FROM t1 WHERE x = 2")
        .await
        .expect("failed to query");

    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_multiple_indexes_persist() {
    let db_path = std::env::temp_dir().join(format!("llkv_test_multi_{}.db", std::process::id()));

    // Phase 1: Create multiple indexes
    {
        let pager = Arc::new(SimdRDrivePager::open(&db_path).expect("failed to create pager"));
        let engine = SqlEngine::new(Arc::new(BoxedPager::from_arc(pager)))
            .expect("failed to create SQL engine");

        engine
            .execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
            .await
            .expect("failed to create table");

        engine
            .execute("CREATE INDEX idx_id ON users(id)")
            .await
            .expect("failed to create index on id");

        engine
            .execute("CREATE INDEX idx_name ON users(name)")
            .await
            .expect("failed to create index on name");

        engine
            .execute("CREATE INDEX idx_age ON users(age)")
            .await
            .expect("failed to create index on age");
    }

    // Phase 2: Reopen and verify all indexes persisted
    {
        let pager = Arc::new(SimdRDrivePager::open(&db_path).expect("failed to reopen pager"));
        let engine = SqlEngine::new(Arc::new(BoxedPager::from_arc(pager)))
            .expect("failed to create SQL engine");

        // All indexes should fail to recreate because they already exist
        let result = engine.execute("CREATE INDEX idx_id ON users(id)").await;
        assert!(result.is_err());

        let result = engine.execute("CREATE INDEX idx_name ON users(name)").await;
        assert!(result.is_err());

        let result = engine.execute("CREATE INDEX idx_age ON users(age)").await;
        assert!(result.is_err());
    }

    // Clean up
    let _ = std::fs::remove_file(&db_path);
}
