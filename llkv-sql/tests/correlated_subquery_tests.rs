use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;
use std::sync::Arc;

#[test]
fn test_correlated_scalar_subquery_simple() {
    let pager = Arc::new(MemPager::default());
    let engine = SqlEngine::new(pager);

    // Create outer table
    engine
        .execute("CREATE TABLE outer_table (id INTEGER, value INTEGER)")
        .unwrap();
    engine
        .execute("INSERT INTO outer_table VALUES (1, 10), (2, 20)")
        .unwrap();

    // Create inner table
    engine
        .execute("CREATE TABLE inner_table (outer_id INTEGER, amount INTEGER)")
        .unwrap();
    engine
        .execute("INSERT INTO inner_table VALUES (1, 5), (1, 3), (2, 8)")
        .unwrap();

    // Correlated scalar subquery
    let sql = "SELECT id, value, (SELECT MIN(amount) FROM inner_table WHERE outer_id = id) as min_amount FROM outer_table";
    eprintln!("\n=== Executing SQL: {} ===\n", sql);
    let result = engine.sql(sql);

    match result {
        Ok(batches) => {
            println!("Success! Batches: {}", batches.len());
            for batch in &batches {
                println!("Batch: {:?}", batch);
            }
        }
        Err(e) => {
            println!("Error: {:?}", e);
            panic!("Query failed: {:?}", e);
        }
    }
}

#[test]
fn test_correlated_scalar_subquery_cross_product() {
    let pager = Arc::new(MemPager::default());
    let engine = SqlEngine::new(pager);

    // Simplified version of TPC-H Q2 structure
    // Outer query: cross product of part, supplier, partsupp
    // Subquery: cross product of partsupp, supplier (NO part)
    // Correlated column: p_partkey from outer query's part table

    engine
        .execute("CREATE TABLE part (p_partkey INTEGER, p_name TEXT, p_size INTEGER)")
        .unwrap();
    engine
        .execute("INSERT INTO part VALUES (1, 'Widget', 10), (2, 'Gadget', 20)")
        .unwrap();

    engine
        .execute("CREATE TABLE supplier (s_suppkey INTEGER, s_name TEXT)")
        .unwrap();
    engine
        .execute("INSERT INTO supplier VALUES (1, 'Supplier A'), (2, 'Supplier B')")
        .unwrap();

    engine
        .execute(
            "CREATE TABLE partsupp (ps_partkey INTEGER, ps_suppkey INTEGER, ps_supplycost INTEGER)",
        )
        .unwrap();
    engine
        .execute("INSERT INTO partsupp VALUES (1, 1, 100), (1, 2, 90), (2, 1, 150), (2, 2, 140)")
        .unwrap();

    // Query structure similar to Q2:
    // Outer: SELECT ... FROM part, supplier, partsupp WHERE ...
    // Subquery: SELECT MIN(ps_supplycost) FROM partsupp, supplier WHERE p_partkey = ps_partkey (p_partkey is correlated)
    let sql = r#"
        SELECT p_partkey, p_name, s_name, ps_supplycost
        FROM part, supplier, partsupp
        WHERE p_partkey = ps_partkey
          AND s_suppkey = ps_suppkey
          AND ps_supplycost = (
              SELECT MIN(ps_supplycost)
              FROM partsupp, supplier
              WHERE p_partkey = ps_partkey
                AND s_suppkey = ps_suppkey
          )
    "#;

    eprintln!("\n=== Executing cross product correlated subquery ===\n");
    let result = engine.sql(sql);

    match result {
        Ok(batches) => {
            println!("Success! Batches: {}", batches.len());
            for batch in &batches {
                println!("Batch rows: {}", batch.num_rows());
                println!("Schema: {:?}", batch.schema());
            }
            // Should return rows where ps_supplycost matches the minimum for that part
            assert!(!batches.is_empty());
            assert!(batches[0].num_rows() > 0);
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
            panic!("Cross product correlated subquery failed: {:?}", e);
        }
    }
}
