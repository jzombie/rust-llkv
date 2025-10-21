use llkv_sql::SqlEngine;
use llkv_storage::pager::SimdRDrivePager;
use std::sync::Arc;

#[test]
fn test_custom_type_persistence_across_restarts() {
    let test_dir = tempfile::tempdir().expect("create temp dir");
    let db_path = test_dir.path().join("test.db");

    // Session 1: Create a custom type
    {
        let pager = SimdRDrivePager::open(&db_path).expect("open pager");
        let engine = SqlEngine::new(Arc::new(pager));

        // Create a custom type
        let results = engine
            .execute("CREATE TYPE user_id AS INTEGER;")
            .expect("create type");
        assert_eq!(results.len(), 1);

        // Create a table using the custom type
        let results = engine
            .execute("CREATE TABLE users (id user_id, name TEXT);")
            .expect("create table with custom type");
        assert_eq!(results.len(), 1);

        // Insert data
        let results = engine
            .execute("INSERT INTO users VALUES (1, 'Alice');")
            .expect("insert data");
        assert_eq!(results.len(), 1);

        // Verify data
        let results = engine.execute("SELECT * FROM users;").expect("select data");
        assert_eq!(results.len(), 1);
    }
    // Drop engine to close the database

    // Session 2: Reopen database and verify type persists
    {
        let pager = SimdRDrivePager::open(&db_path).expect("reopen pager");
        let engine = SqlEngine::new(Arc::new(pager));

        // The type should still exist, so we can use it to insert more data
        let results = engine
            .execute("INSERT INTO users VALUES (2, 'Bob');")
            .expect("insert using persisted type");
        assert_eq!(results.len(), 1);

        // Verify both rows exist
        let results = engine
            .execute("SELECT * FROM users ORDER BY id;")
            .expect("select all data");
        assert_eq!(results.len(), 1);

        // Drop the table and type
        engine.execute("DROP TABLE users;").expect("drop table");
        engine.execute("DROP TYPE user_id;").expect("drop type");
    }
    // Drop engine to close the database

    // Session 3: Reopen and verify type no longer exists
    {
        let pager = SimdRDrivePager::open(&db_path).expect("reopen pager");
        let engine = SqlEngine::new(Arc::new(pager));

        // Try to create a table with the dropped type - should fail
        let result = engine.execute("CREATE TABLE test (id user_id);");
        assert!(
            result.is_err(),
            "Should fail because user_id type was dropped"
        );
    }

    test_dir.close().expect("cleanup temp dir");
}
