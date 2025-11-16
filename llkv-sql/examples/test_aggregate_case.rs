use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

fn main() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    println!("=== Testing Aggregates with information_schema ===\n");

    // Setup
    engine.execute("CREATE TABLE test1 (id INT PRIMARY KEY, name TEXT);").unwrap();
    engine.execute("CREATE TABLE test2 (id INT PRIMARY KEY, value INT, category TEXT);").unwrap();
    engine.execute("INSERT INTO test2 VALUES (1, 100, 'A'), (2, 200, 'B'), (3, 150, 'A');").unwrap();

    // Test 0: GROUP BY on regular table (baseline)
    println!("Test 0: GROUP BY on regular user table (baseline)");
    match engine.sql("SELECT category, COUNT(*) as cnt, SUM(value) as total FROM test2 GROUP BY category ORDER BY category;") {
        Ok(batches) if !batches.is_empty() => {
            println!("{}\n", pretty_format_batches(&batches).unwrap());
        }
        Ok(_) => println!("Empty result\n"),
        Err(e) => eprintln!("Error: {:?}\n", e),
    }

    // Test 1: Simple COUNT without GROUP BY
    println!("Test 1: Simple COUNT on information_schema");
    match engine.sql("SELECT COUNT(*) as total FROM information_schema.columns WHERE table_name = 'test1';") {
        Ok(batches) if !batches.is_empty() => {
            println!("{}\n", pretty_format_batches(&batches).unwrap());
        }
        Ok(_) => println!("Empty result\n"),
        Err(e) => eprintln!("Error: {:?}\n", e),
    }

    // Test 1.5: Simple SELECT (no aggregate) to see if data is visible
    println!("Test 1.5: Simple SELECT to verify data visibility");
    match engine.sql("SELECT table_name, column_name FROM information_schema.columns WHERE table_name = 'test1';") {
        Ok(batches) => {
            println!("Got {} batches", batches.len());
            if !batches.is_empty() {
                println!("{}\n", pretty_format_batches(&batches).unwrap());
            } else {
                println!("Empty result\n");
            }
        }
        Err(e) => eprintln!("Error: {:?}\n", e),
    }

    // Test 2: COUNT with GROUP BY
    println!("Test 2: COUNT with GROUP BY on information_schema");
    match engine.sql("SELECT table_name, COUNT(*) as col_count FROM information_schema.columns WHERE table_name IN ('test1', 'test2') GROUP BY table_name ORDER BY table_name;") {
        Ok(batches) => {
            println!("Got {} batches", batches.len());
            if !batches.is_empty() {
                for (i, batch) in batches.iter().enumerate() {
                    println!("Batch {}: {} rows, {} columns", i, batch.num_rows(), batch.num_columns());
                }
                println!("{}\n", pretty_format_batches(&batches).unwrap());
            } else {
                println!("Empty result (0 batches)\n");
            }
        }
        Err(e) => eprintln!("Error: {:?}\n", e),
    }

    // Test 3: SUM with CAST
    println!("Test 3: SUM with CAST of boolean");
    match engine.sql("SELECT SUM(CAST(is_primary_key AS INTEGER)) as pk_count FROM information_schema.columns WHERE table_name IN ('test1', 'test2');") {
        Ok(batches) if !batches.is_empty() => {
            println!("{}\n", pretty_format_batches(&batches).unwrap());
        }
        Ok(_) => println!("Empty result\n"),
        Err(e) => eprintln!("Error: {:?}\n", e),
    }

    // Test 4: CASE expression
    println!("Test 4: CASE expression");
    match engine.sql("SELECT column_name, CASE WHEN is_nullable THEN 'yes' ELSE 'no' END as nullable_text FROM information_schema.columns WHERE table_name = 'test1';") {
        Ok(batches) if !batches.is_empty() => {
            println!("{}\n", pretty_format_batches(&batches).unwrap());
        }
        Ok(_) => println!("Empty result\n"),
        Err(e) => eprintln!("Error: {:?}\n", e),
    }
}
