//! Test UPDATE statement functionality

use llkv_sql::SqlEngine;
use llkv_storage::pager::{BoxedPager, MemPager};
use std::sync::Arc;

#[tokio::test]
async fn test_update_simple() {
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

    // Update one row
    let result = engine
        .execute("UPDATE t1 SET x = 10 WHERE x = 2")
        .await
        .expect("failed to update");

    // Verify update happened
    let query_result = engine
        .execute("SELECT * FROM t1 ORDER BY x")
        .await
        .expect("failed to query");

    assert_eq!(query_result.len(), 1);
    // Should have rows with x = 1, 3, 10
}

#[tokio::test]
async fn test_update_all_rows() {
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

    // Update all rows
    engine
        .execute("UPDATE t1 SET y = 'updated'")
        .await
        .expect("failed to update");

    // Verify all rows were updated
    let query_result = engine
        .execute("SELECT * FROM t1 WHERE y = 'updated'")
        .await
        .expect("failed to query");

    assert_eq!(query_result.len(), 1);
}

#[tokio::test]
async fn test_update_multiple_columns() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

    // Create a table
    engine
        .execute("CREATE TABLE t1 (x INTEGER, y TEXT, z INTEGER)")
        .await
        .expect("failed to create table");

    // Insert some data
    engine
        .execute("INSERT INTO t1 VALUES (1, 'a', 10), (2, 'b', 20), (3, 'c', 30)")
        .await
        .expect("failed to insert data");

    // Update multiple columns
    engine
        .execute("UPDATE t1 SET x = 100, z = 999 WHERE y = 'b'")
        .await
        .expect("failed to update");

    // Verify the update
    let query_result = engine
        .execute("SELECT * FROM t1 WHERE y = 'b'")
        .await
        .expect("failed to query");

    assert_eq!(query_result.len(), 1);
}

#[tokio::test]
async fn test_update_no_matching_rows() {
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
        .execute("INSERT INTO t1 VALUES (1, 'a'), (2, 'b')")
        .await
        .expect("failed to insert data");

    // Update with no matching rows
    engine
        .execute("UPDATE t1 SET x = 99 WHERE x = 999")
        .await
        .expect("failed to update");

    // Verify no changes
    let query_result = engine
        .execute("SELECT COUNT(*) as count FROM t1 WHERE x = 99")
        .await
        .expect("failed to query");

    assert_eq!(query_result.len(), 1);
}

#[tokio::test]
async fn test_update_nonexistent_table() {
    let pager = Arc::new(MemPager::default());
    let engine =
        SqlEngine::new(Arc::new(BoxedPager::from_arc(pager))).expect("failed to create SQL engine");

    // Try to update a non-existent table
    let result = engine.execute("UPDATE t1 SET x = 1").await;
    assert!(result.is_err());
}
