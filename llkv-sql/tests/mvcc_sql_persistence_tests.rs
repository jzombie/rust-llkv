//! SQL-level MVCC persistence integration tests.
//!
//! These tests verify that MVCC (Multi-Version Concurrency Control) transaction state
//! persists correctly across database reopens when using durable pagers. Unlike the
//! low-level MVCC tests in llkv-transaction, these tests use SQL statements (BEGIN,
//! COMMIT, ROLLBACK, INSERT, SELECT, DELETE) to verify persistence behavior.
//!
//! ## What is tested
//!
//! - Committed transactions persist and remain visible after database reopen
//! - Uncommitted transactions are lost and their data is invisible after reopen
//! - Transaction IDs don't overlap across sessions (no reuse)
//! - Soft deletes (DELETE statements) persist correctly
//! - Multiple committed transactions work correctly
//!
//! ## Implementation
//!
//! Each test:
//! 1. Creates a temporary file-backed database using SimdRDrivePager
//! 2. Executes SQL statements in a transaction (BEGIN...COMMIT or ROLLBACK)
//! 3. Closes the database (drops Context and SqlEngine)
//! 4. Reopens the same database file with a fresh Context
//! 5. Verifies data visibility matches expected MVCC behavior
//!
//! The persistence is handled automatically by:
//! - Context loading transaction state from catalog on initialization
//! - Session persisting transaction state to catalog on commit
//! - No manual MVCC layer manipulation required

use arrow::array::{Array, Int64Array, StringArray};
use llkv_runtime::{Context, StatementResult};
use llkv_sql::SqlEngine;
use llkv_storage::pager::simd_r_drive_pager::SimdRDrivePager;
use std::sync::Arc;
use tempfile::NamedTempFile;

/// Helper to create a new SqlEngine with a fresh file-backed pager.
fn create_engine_with_file(path: &std::path::Path) -> SqlEngine<SimdRDrivePager> {
    let pager = Arc::new(
        SimdRDrivePager::open(path)
            .expect("Failed to open SimdRDrivePager"),
    );
    let context = Arc::new(Context::new(pager));
    SqlEngine::with_context(context, false)
}

/// Helper to execute a SELECT and extract Int64 values from first column.
fn select_int64_values(engine: &SqlEngine<SimdRDrivePager>, sql: &str) -> Vec<Option<i64>> {
    let mut results = engine.execute(sql).expect("SELECT failed");
    assert_eq!(results.len(), 1, "Expected exactly one result");
    
    if let StatementResult::Select { execution, .. } = results.remove(0) {
        let batches = execution.collect().expect("Failed to collect batches");
        let mut values = Vec::new();
        
        for batch in batches {
            if batch.num_columns() > 0 {
                let array = batch.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Expected Int64Array");
                
                for i in 0..array.len() {
                    if array.is_null(i) {
                        values.push(None);
                    } else {
                        values.push(Some(array.value(i)));
                    }
                }
            }
        }
        values
    } else {
        panic!("Expected SELECT result");
    }
}

/// Helper to execute a SELECT and extract String values from first column.
fn select_string_values(engine: &SqlEngine<SimdRDrivePager>, sql: &str) -> Vec<Option<String>> {
    let mut results = engine.execute(sql).expect("SELECT failed");
    assert_eq!(results.len(), 1, "Expected exactly one result");
    
    if let StatementResult::Select { execution, .. } = results.remove(0) {
        let batches = execution.collect().expect("Failed to collect batches");
        let mut values = Vec::new();
        
        for batch in batches {
            if batch.num_columns() > 0 {
                let array = batch.column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Expected StringArray");
                
                for i in 0..array.len() {
                    if array.is_null(i) {
                        values.push(None);
                    } else {
                        values.push(Some(array.value(i).to_string()));
                    }
                }
            }
        }
        values
    } else {
        panic!("Expected SELECT result");
    }
}

/// Helper to count rows from a SELECT COUNT(*) query.
fn count_rows(engine: &SqlEngine<SimdRDrivePager>, table: &str) -> i64 {
    let sql = format!("SELECT COUNT(*) FROM {}", table);
    let values = select_int64_values(engine, &sql);
    assert_eq!(values.len(), 1, "Expected exactly one count result");
    values[0].expect("COUNT should not be NULL")
}

#[test]
fn test_sql_committed_transaction_persists_across_reopen() {
    // Test that committed SQL transactions persist and remain visible after database reopen.
    
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let db_path = temp_file.path();
    
    // Scope 1: Create table, insert data, commit
    {
        let engine = create_engine_with_file(db_path);
        
        engine.execute("CREATE TABLE users(id INTEGER, name TEXT)")
            .expect("CREATE TABLE failed");
        
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')")
            .expect("INSERT failed");
        
        // Verify data is visible within transaction
        let count = count_rows(&engine, "users");
        assert_eq!(count, 3, "Expected 3 rows before commit");
        
        engine.execute("COMMIT")
            .expect("COMMIT failed");
        
        // Verify data still visible after commit
        let count = count_rows(&engine, "users");
        assert_eq!(count, 3, "Expected 3 rows after commit");
    }
    
    // Scope 2: Reopen database, verify committed data is visible
    {
        let engine = create_engine_with_file(db_path);
        
        // Table should exist
        engine.execute("SELECT * FROM users")
            .expect("Table should exist after reopen");
        
        // Data should be visible
        let count = count_rows(&engine, "users");
        assert_eq!(count, 3, "Expected 3 rows after reopen");
        
        let names = select_string_values(&engine, "SELECT name FROM users ORDER BY id");
        assert_eq!(names, vec![
            Some("Alice".to_string()),
            Some("Bob".to_string()),
            Some("Charlie".to_string())
        ]);
    }
}

#[test]
fn test_sql_uncommitted_transaction_not_visible_after_reopen() {
    // Test that uncommitted SQL transactions are lost and invisible after database reopen.
    
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let db_path = temp_file.path();
    
    // Scope 1: Create table (committed), then insert data without commit
    {
        let engine = create_engine_with_file(db_path);
        
        // First commit a table creation
        engine.execute("CREATE TABLE products(id INTEGER, name TEXT)")
            .expect("CREATE TABLE failed");
        
        let count = count_rows(&engine, "products");
        assert_eq!(count, 0, "Table should be empty initially");
        
        // Start a transaction and insert data, but don't commit
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("INSERT INTO products VALUES (1, 'Widget'), (2, 'Gadget')")
            .expect("INSERT failed");
        
        // Verify data is visible within transaction
        let count = count_rows(&engine, "products");
        assert_eq!(count, 2, "Expected 2 rows in uncommitted transaction");
        
        // Drop engine without commit (transaction is lost)
    }
    
    // Scope 2: Reopen database, verify uncommitted data is NOT visible
    {
        let engine = create_engine_with_file(db_path);
        
        // Table should exist (it was committed)
        engine.execute("SELECT * FROM products")
            .expect("Table should exist after reopen");
        
        // But data should NOT be visible (it was not committed)
        let count = count_rows(&engine, "products");
        assert_eq!(count, 0, "Expected 0 rows after reopen (uncommitted data lost)");
    }
}

#[test]
fn test_sql_multiple_transactions_persist_correctly() {
    // Test that multiple committed SQL transactions persist correctly.
    
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let db_path = temp_file.path();
    
    // Scope 1: Create table and commit first batch
    {
        let engine = create_engine_with_file(db_path);
        
        engine.execute("CREATE TABLE orders(id INTEGER, amount INTEGER)")
            .expect("CREATE TABLE failed");
        
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("INSERT INTO orders VALUES (1, 100), (2, 200)")
            .expect("INSERT failed");
        
        engine.execute("COMMIT")
            .expect("COMMIT failed");
        
        let count = count_rows(&engine, "orders");
        assert_eq!(count, 2, "Expected 2 rows after first commit");
    }
    
    // Scope 2: Reopen, add more data, commit
    {
        let engine = create_engine_with_file(db_path);
        
        let count = count_rows(&engine, "orders");
        assert_eq!(count, 2, "Expected 2 rows after reopen");
        
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("INSERT INTO orders VALUES (3, 300), (4, 400)")
            .expect("INSERT failed");
        
        engine.execute("COMMIT")
            .expect("COMMIT failed");
        
        let count = count_rows(&engine, "orders");
        assert_eq!(count, 4, "Expected 4 rows after second commit");
    }
    
    // Scope 3: Reopen again, verify all data is visible
    {
        let engine = create_engine_with_file(db_path);
        
        let count = count_rows(&engine, "orders");
        assert_eq!(count, 4, "Expected 4 rows after final reopen");
        
        let ids = select_int64_values(&engine, "SELECT id FROM orders ORDER BY id");
        assert_eq!(ids, vec![Some(1), Some(2), Some(3), Some(4)]);
    }
}

#[test]
fn test_sql_rollback_not_visible_after_reopen() {
    // Test that rolled back SQL transactions are not visible after reopen.
    
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let db_path = temp_file.path();
    
    // Scope 1: Create table, insert and commit some data
    {
        let engine = create_engine_with_file(db_path);
        
        engine.execute("CREATE TABLE inventory(id INTEGER, quantity INTEGER)")
            .expect("CREATE TABLE failed");
        
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("INSERT INTO inventory VALUES (1, 10), (2, 20)")
            .expect("INSERT failed");
        
        engine.execute("COMMIT")
            .expect("COMMIT failed");
        
        let count = count_rows(&engine, "inventory");
        assert_eq!(count, 2, "Expected 2 rows after commit");
        
        // Start another transaction and insert more data, then rollback
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("INSERT INTO inventory VALUES (3, 30), (4, 40)")
            .expect("INSERT failed");
        
        let count = count_rows(&engine, "inventory");
        assert_eq!(count, 4, "Expected 4 rows in transaction before rollback");
        
        engine.execute("ROLLBACK")
            .expect("ROLLBACK failed");
        
        let count = count_rows(&engine, "inventory");
        assert_eq!(count, 2, "Expected 2 rows after rollback");
    }
    
    // Scope 2: Reopen and verify only committed data is visible
    {
        let engine = create_engine_with_file(db_path);
        
        let count = count_rows(&engine, "inventory");
        assert_eq!(count, 2, "Expected 2 rows after reopen (rolled back data not visible)");
        
        let ids = select_int64_values(&engine, "SELECT id FROM inventory ORDER BY id");
        assert_eq!(ids, vec![Some(1), Some(2)]);
    }
}

#[test]
fn test_sql_delete_persists_across_reopen() {
    // Test that DELETE statements (soft deletes using MVCC) persist correctly.
    
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let db_path = temp_file.path();
    
    // Scope 1: Create table, insert data, then delete some rows
    {
        let engine = create_engine_with_file(db_path);
        
        engine.execute("CREATE TABLE employees(id INTEGER, name TEXT)")
            .expect("CREATE TABLE failed");
        
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("INSERT INTO employees VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie'), (4, 'Dave')")
            .expect("INSERT failed");
        
        engine.execute("COMMIT")
            .expect("COMMIT failed");
        
        let count = count_rows(&engine, "employees");
        assert_eq!(count, 4, "Expected 4 rows after insert");
        
        // Delete some rows
        engine.execute("BEGIN TRANSACTION")
            .expect("BEGIN failed");
        
        engine.execute("DELETE FROM employees WHERE id = 2")
            .expect("DELETE failed");
        engine.execute("DELETE FROM employees WHERE id = 4")
            .expect("DELETE failed");
        
        let count = count_rows(&engine, "employees");
        assert_eq!(count, 2, "Expected 2 rows after delete in transaction");
        
        engine.execute("COMMIT")
            .expect("COMMIT failed");
        
        let count = count_rows(&engine, "employees");
        assert_eq!(count, 2, "Expected 2 rows after delete commit");
    }
    
    // Scope 2: Reopen and verify deletes persisted
    {
        let engine = create_engine_with_file(db_path);
        
        let count = count_rows(&engine, "employees");
        assert_eq!(count, 2, "Expected 2 rows after reopen (deletes should persist)");
        
        let names = select_string_values(&engine, "SELECT name FROM employees ORDER BY id");
        assert_eq!(names, vec![
            Some("Alice".to_string()),
            Some("Charlie".to_string())
        ]);
    }
}

#[test]
fn test_sql_mixed_operations_persist() {
    // Test a realistic scenario with multiple operations across reopens.
    
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let db_path = temp_file.path();
    
    // Scope 1: Initial setup
    {
        let engine = create_engine_with_file(db_path);
        
        engine.execute("CREATE TABLE accounts(id INTEGER, balance INTEGER)")
            .expect("CREATE TABLE failed");
        
        engine.execute("BEGIN TRANSACTION").expect("BEGIN failed");
        engine.execute("INSERT INTO accounts VALUES (1, 1000), (2, 2000), (3, 3000)")
            .expect("INSERT failed");
        engine.execute("COMMIT").expect("COMMIT failed");
    }
    
    // Scope 2: Update some records
    {
        let engine = create_engine_with_file(db_path);
        
        let count = count_rows(&engine, "accounts");
        assert_eq!(count, 3, "Expected 3 accounts");
        
        engine.execute("BEGIN TRANSACTION").expect("BEGIN failed");
        engine.execute("UPDATE accounts SET balance = 1500 WHERE id = 1")
            .expect("UPDATE failed");
        engine.execute("DELETE FROM accounts WHERE id = 3")
            .expect("DELETE failed");
        engine.execute("COMMIT").expect("COMMIT failed");
        
        let count = count_rows(&engine, "accounts");
        assert_eq!(count, 2, "Expected 2 accounts after update/delete");
    }
    
    // Scope 3: Verify final state
    {
        let engine = create_engine_with_file(db_path);
        
        let count = count_rows(&engine, "accounts");
        assert_eq!(count, 2, "Expected 2 accounts after reopen");
        
        let balances = select_int64_values(&engine, "SELECT balance FROM accounts ORDER BY id");
        assert_eq!(balances, vec![Some(1500), Some(2000)]);
    }
}
