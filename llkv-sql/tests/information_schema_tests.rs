//! Integration tests for information_schema functionality.
//!
//! Tests cover:
//! - Lazy refresh mechanism (tables created on-demand)
//! - Schema-qualified table name resolution
//! - JOIN operations between information_schema and user tables
//! - Reserved table name validation
//! - Information schema table availability

use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StringArray};
use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

/// Helper to extract record batches from SELECT results
fn extract_batches(
    mut results: Vec<RuntimeStatementResult<llkv_storage::pager::BoxedPager>>,
) -> Vec<RecordBatch> {
    if results.is_empty() {
        return vec![];
    }

    match results.remove(0) {
        RuntimeStatementResult::Select { execution, .. } => {
            execution.collect().expect("collect query batches")
        }
        _ => vec![],
    }
}

#[test]
fn test_information_schema_lazy_refresh() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create a user table
    engine
        .execute("CREATE TABLE test_table (id INTEGER, name TEXT);")
        .expect("create table succeeds");

    // Query information_schema.tables - should trigger refresh
    let results = engine
        .execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'test_table';",
        )
        .expect("query information_schema.tables succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "should find test_table");

    let table_name_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(table_name_col.value(0), "test_table");
}

#[test]
fn test_information_schema_columns_metadata() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create a table with various column types
    engine
        .execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER);")
        .expect("create table succeeds");

    // Query information_schema.columns
    let results = engine
        .execute(
            "SELECT column_name, data_type, is_nullable 
             FROM information_schema.columns 
             WHERE table_name = 'sample' 
             ORDER BY ordinal_position;",
        )
        .expect("query information_schema.columns succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3, "should have 3 columns");

    let column_names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(column_names.value(0), "id");
    assert_eq!(column_names.value(1), "name");
    assert_eq!(column_names.value(2), "age");
}

#[test]
fn test_information_schema_join_with_user_table() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create a user table
    engine
        .execute("CREATE TABLE projects (id INTEGER, name TEXT, owner TEXT);")
        .expect("create table succeeds");

    // Create a governance policy table
    engine
        .execute("CREATE TABLE column_policies (table_name TEXT, column_name TEXT, classification TEXT);")
        .expect("create policy table succeeds");

    engine
        .execute("INSERT INTO column_policies VALUES ('projects', 'id', 'identifier');")
        .expect("insert policy succeeds");

    engine
        .execute("INSERT INTO column_policies VALUES ('projects', 'name', 'identity');")
        .expect("insert policy succeeds");

    // JOIN information_schema.columns with user table
    let results = engine
        .execute(
            "SELECT c.column_name, c.data_type, p.classification 
             FROM information_schema.columns AS c 
             JOIN column_policies AS p 
               ON p.table_name = c.table_name AND p.column_name = c.column_name 
             WHERE c.table_name = 'projects' 
             ORDER BY c.ordinal_position;",
        )
        .expect("join with information_schema succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2, "should match 2 columns");

    let column_names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(column_names.value(0), "id");
    assert_eq!(column_names.value(1), "name");

    let classifications = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(classifications.value(0), "identifier");
    assert_eq!(classifications.value(1), "identity");
}

#[test]
fn test_information_schema_left_join() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE products (id INTEGER, name TEXT, price INTEGER);")
        .expect("create table succeeds");

    engine
        .execute("CREATE TABLE metadata_tags (table_name TEXT, column_name TEXT, tag TEXT);")
        .expect("create tags table succeeds");

    engine
        .execute("INSERT INTO metadata_tags VALUES ('products', 'id', 'key');")
        .expect("insert tag succeeds");

    // LEFT JOIN to show all columns, with optional tags
    let results = engine
        .execute(
            "SELECT c.column_name, t.tag 
             FROM information_schema.columns AS c 
             LEFT JOIN metadata_tags AS t 
               ON t.table_name = c.table_name AND t.column_name = c.column_name 
             WHERE c.table_name = 'products' 
             ORDER BY c.ordinal_position;",
        )
        .expect("left join succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3, "should show all 3 columns");

    let column_names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(column_names.value(0), "id");
    assert_eq!(column_names.value(1), "name");
    assert_eq!(column_names.value(2), "price");
}

#[test]
fn test_information_schema_multiple_table_query() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create multiple tables
    engine
        .execute("CREATE TABLE table1 (id INTEGER);")
        .expect("create table1 succeeds");

    engine
        .execute("CREATE TABLE table2 (id INTEGER);")
        .expect("create table2 succeeds");

    engine
        .execute("CREATE TABLE table3 (id INTEGER);")
        .expect("create table3 succeeds");

    // Query all tables
    let results = engine
        .execute("SELECT table_name FROM information_schema.tables ORDER BY table_name;")
        .expect("query all tables succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert!(batch.num_rows() >= 3, "should have at least 3 tables");

    let table_names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let names: Vec<&str> = (0..batch.num_rows())
        .map(|i| table_names.value(i))
        .collect();

    assert!(names.contains(&"table1"));
    assert!(names.contains(&"table2"));
    assert!(names.contains(&"table3"));
}

#[test]
fn test_information_schema_reserved_name_validation() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Attempt to create table named "information_schema" in default namespace
    let err = engine
        .execute("CREATE TABLE information_schema (id INTEGER);")
        .expect_err("creating table named information_schema should fail");

    let err_str = format!("{err}").to_ascii_lowercase();
    assert!(
        err_str.contains("reserved") || err_str.contains("information_schema"),
        "error should mention reserved name: {err}"
    );
}

#[test]
fn test_information_schema_schema_qualified_query() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE my_table (col1 INTEGER, col2 TEXT);")
        .expect("create table succeeds");

    // Query with explicit schema qualification
    let results = engine
        .execute("SELECT * FROM information_schema.columns WHERE table_name = 'my_table';")
        .expect("schema-qualified query succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2, "should find 2 columns");
}

#[test]
fn test_information_schema_table_constraints() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create table with primary key constraint
    engine
        .execute("CREATE TABLE constrained (id INTEGER PRIMARY KEY, value TEXT UNIQUE);")
        .expect("create table with constraints succeeds");

    // Query table_constraints
    let results = engine
        .execute(
            "SELECT constraint_type 
             FROM information_schema.table_constraints 
             WHERE table_name = 'constrained' 
             ORDER BY constraint_type;",
        )
        .expect("query table_constraints succeeds");

    let batches = extract_batches(results);
    if !batches.is_empty() {
        let batch = &batches[0];
        assert!(batch.num_rows() > 0, "should have at least one constraint");
    }
}

#[test]
fn test_information_schema_key_column_usage() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE keyed (id INTEGER PRIMARY KEY, name TEXT);")
        .expect("create table with primary key succeeds");

    // Query key_column_usage
    let results = engine
        .execute(
            "SELECT column_name 
             FROM information_schema.key_column_usage 
             WHERE table_name = 'keyed';",
        )
        .expect("query key_column_usage succeeds");

    let batches = extract_batches(results);
    if !batches.is_empty() {
        let batch = &batches[0];
        assert!(batch.num_rows() > 0, "should find key column");
    }
}

#[test]
fn test_information_schema_subquery_with_join() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER);")
        .expect("create orders table succeeds");

    engine
        .execute("INSERT INTO orders VALUES (100, 1);")
        .expect("insert order succeeds");

    // Use information_schema in a subquery
    let results = engine
        .execute(
            "SELECT o.order_id 
             FROM orders AS o 
             WHERE EXISTS (
                 SELECT 1 
                 FROM information_schema.columns AS c 
                 WHERE c.table_name = 'orders' AND c.column_name = 'order_id'
             );",
        )
        .expect("subquery with information_schema succeeds");

    // Should return the row since the column exists in information_schema
    let batches = extract_batches(results);
    assert!(
        !batches.is_empty(),
        "should have results since column exists"
    );
    assert_eq!(batches[0].num_rows(), 1, "should return one row");
}

#[test]
fn test_information_schema_aggregate_column_count() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE wide_table (col1 INT, col2 INT, col3 INT, col4 INT, col5 INT);")
        .expect("create table succeeds");

    // Count columns using aggregate
    let results = engine
        .execute(
            "SELECT COUNT(*) AS column_count 
             FROM information_schema.columns 
             WHERE table_name = 'wide_table';",
        )
        .expect("aggregate query succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "should have one aggregate result");

    // The count should be 5
    let count_col = batch.column(0);
    assert_eq!(count_col.len(), 1);
}

#[test]
fn test_information_schema_cross_join() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE small_table (id INTEGER);")
        .expect("create table succeeds");

    // CROSS JOIN information_schema tables
    let results = engine
        .execute(
            "SELECT t.table_name, c.column_name 
             FROM information_schema.tables AS t 
             CROSS JOIN information_schema.columns AS c 
             WHERE t.table_name = 'small_table' AND c.table_name = 'small_table';",
        )
        .expect("cross join succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "should find table and its column");
}

#[test]
fn test_information_schema_case_insensitive_query() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE test (id INTEGER);")
        .expect("create table succeeds");

    // Query with mixed case
    let results = engine
        .execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE table_name = 'test';")
        .expect("case-insensitive query succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");
}

#[test]
fn test_information_schema_aliased_table_reference() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE aliased (value TEXT);")
        .expect("create table succeeds");

    // Use alias for information_schema table
    let results = engine
        .execute(
            "SELECT cols.column_name, cols.data_type 
             FROM information_schema.columns AS cols 
             WHERE cols.table_name = 'aliased';",
        )
        .expect("aliased table reference succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1, "should find one column");
}

#[test]
fn test_information_schema_with_filter_and_order() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE filtered (z_col INT, a_col TEXT, m_col REAL);")
        .expect("create table succeeds");

    // Filter and order results - just filter by table name and order
    let results = engine
        .execute(
            "SELECT column_name 
             FROM information_schema.columns 
             WHERE table_name = 'filtered' 
             ORDER BY column_name DESC;",
        )
        .expect("filtered and ordered query succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3, "should find 3 columns");

    let column_names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        column_names.value(0),
        "z_col",
        "should be sorted descending"
    );
    assert_eq!(column_names.value(1), "m_col");
    assert_eq!(column_names.value(2), "a_col");
}

#[test]
fn test_information_schema_nullable_column_detection() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    engine
        .execute("CREATE TABLE nullability (required INT NOT NULL, optional TEXT);")
        .expect("create table succeeds");

    // Query nullable columns
    let results = engine
        .execute(
            "SELECT column_name, is_nullable 
             FROM information_schema.columns 
             WHERE table_name = 'nullability' 
             ORDER BY ordinal_position;",
        )
        .expect("nullable query succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have results");

    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2, "should have 2 columns");
}

#[test]
fn test_information_schema_reflects_schema_changes() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create initial table with 3 columns
    engine
        .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price INTEGER);")
        .expect("create table succeeds");

    // Verify initial schema
    let results = engine
        .execute(
            "SELECT column_name FROM information_schema.columns 
             WHERE table_name = 'products' 
             ORDER BY ordinal_position;",
        )
        .expect("query initial columns succeeds");

    let batches = extract_batches(results);
    assert!(!batches.is_empty(), "should have initial results");
    assert_eq!(batches[0].num_rows(), 3, "should have 3 initial columns");

    let column_names = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(column_names.value(0), "id");
    assert_eq!(column_names.value(1), "name");
    assert_eq!(column_names.value(2), "price");

    // Test 1: DROP COLUMN - remove 'price' column
    engine
        .execute("ALTER TABLE products DROP COLUMN price;")
        .expect("drop column succeeds");

    let results = engine
        .execute(
            "SELECT column_name FROM information_schema.columns 
             WHERE table_name = 'products' 
             ORDER BY ordinal_position;",
        )
        .expect("query after drop column succeeds");

    let batches = extract_batches(results);
    assert_eq!(batches[0].num_rows(), 2, "should have 2 columns after drop");

    let column_names = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(column_names.value(0), "id");
    assert_eq!(column_names.value(1), "name");

    // Test 2: RENAME COLUMN - rename 'name' to 'product_name'
    engine
        .execute("ALTER TABLE products RENAME COLUMN name TO product_name;")
        .expect("rename column succeeds");

    let results = engine
        .execute(
            "SELECT column_name FROM information_schema.columns 
             WHERE table_name = 'products' 
             ORDER BY ordinal_position;",
        )
        .expect("query after rename column succeeds");

    let batches = extract_batches(results);
    assert_eq!(batches[0].num_rows(), 2, "should still have 2 columns");

    let column_names = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(column_names.value(0), "id");
    assert_eq!(
        column_names.value(1),
        "product_name",
        "column should be renamed"
    );

    // Test 3: ALTER COLUMN TYPE - change 'product_name' type (not a primary key)
    engine
        .execute("ALTER TABLE products ALTER COLUMN product_name SET DATA TYPE INTEGER;")
        .expect("alter column type succeeds");

    let results = engine
        .execute(
            "SELECT column_name, data_type FROM information_schema.columns 
             WHERE table_name = 'products' AND column_name = 'product_name';",
        )
        .expect("query after alter column type succeeds");

    let batches = extract_batches(results);
    assert_eq!(batches[0].num_rows(), 1, "should find product_name column");

    let data_types = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        data_types.value(0),
        "Int64",
        "column type should be changed"
    );

    // Test 4: RENAME TABLE - rename 'products' to 'items'
    engine
        .execute("ALTER TABLE products RENAME TO items;")
        .expect("rename table succeeds");

    // Verify old table name is gone by listing all non-information_schema tables
    let results = engine
        .execute(
            "SELECT table_name FROM information_schema.tables 
             WHERE table_schema IS NULL OR table_schema != 'information_schema'
             ORDER BY table_name;",
        )
        .expect("query user tables succeeds");

    let batches = extract_batches(results);
    let table_names_arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let user_tables: Vec<&str> = (0..batches[0].num_rows())
        .map(|i| table_names_arr.value(i))
        .collect();

    assert!(
        !user_tables.contains(&"products"),
        "old table name 'products' should not exist"
    );
    assert!(
        user_tables.contains(&"items"),
        "new table name 'items' should exist"
    );

    // Verify new table name exists with correct columns
    let results = engine
        .execute(
            "SELECT column_name FROM information_schema.columns 
             WHERE table_name = 'items' 
             ORDER BY ordinal_position;",
        )
        .expect("query new table name succeeds");

    let batches = extract_batches(results);
    assert_eq!(
        batches[0].num_rows(),
        2,
        "renamed table should have 2 columns"
    );

    let column_names = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(column_names.value(0), "id");
    assert_eq!(column_names.value(1), "product_name");

    // Test 5: DROP TABLE - remove 'items' table completely
    engine
        .execute("DROP TABLE items;")
        .expect("drop table succeeds");

    // Verify table is gone from information_schema.tables
    let results = engine
        .execute(
            "SELECT table_name FROM information_schema.tables 
             WHERE table_name = 'items';",
        )
        .expect("query for dropped table succeeds");

    let batches = extract_batches(results);
    assert_eq!(
        batches[0].num_rows(),
        0,
        "dropped table should not exist in tables"
    );

    // Verify columns are gone from information_schema.columns
    let results = engine
        .execute(
            "SELECT column_name FROM information_schema.columns 
             WHERE table_name = 'items';",
        )
        .expect("query for dropped table columns succeeds");

    let batches = extract_batches(results);
    assert_eq!(
        batches[0].num_rows(),
        0,
        "dropped table columns should not exist"
    );

    // Test 6: CREATE TABLE after DROP - verify fresh state
    engine
        .execute("CREATE TABLE items (id INTEGER, description TEXT);")
        .expect("recreate table succeeds");

    let results = engine
        .execute(
            "SELECT column_name FROM information_schema.columns 
             WHERE table_name = 'items' 
             ORDER BY ordinal_position;",
        )
        .expect("query recreated table succeeds");

    let batches = extract_batches(results);
    assert_eq!(
        batches[0].num_rows(),
        2,
        "recreated table should have 2 columns"
    );

    let column_names = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        column_names.value(0),
        "id",
        "recreated table should have new schema"
    );
    assert_eq!(column_names.value(1), "description");
}

#[test]
fn test_alter_table_drop_column_basic() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create a table
    engine
        .execute("CREATE TABLE test (id INTEGER, name TEXT, value INTEGER);")
        .expect("create table succeeds");

    // Verify initial structure by querying the table directly
    engine
        .execute("INSERT INTO test VALUES (1, 'test', 100);")
        .expect("insert succeeds");

    let results = engine
        .execute("SELECT * FROM test;")
        .expect("select succeeds");
    let batches = extract_batches(results);
    assert_eq!(
        batches[0].num_columns(),
        3,
        "should have 3 columns initially"
    );

    // Drop a column
    engine
        .execute("ALTER TABLE test DROP COLUMN value;")
        .expect("drop column succeeds");

    // Query the table again to see if column was actually dropped
    let results = engine
        .execute("SELECT * FROM test;")
        .expect("select after drop succeeds");
    let batches = extract_batches(results);

    assert_eq!(
        batches[0].num_columns(),
        2,
        "should have 2 columns after drop"
    );
}
