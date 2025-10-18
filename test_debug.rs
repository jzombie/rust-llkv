use std::sync::Arc;
use llkv_runtime::RuntimeStatementResult;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("llkv_runtime=trace,llkv_table=trace")
        .init();

    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    // Create table with composite primary key
    engine
        .execute("CREATE TABLE pk_combo(a INTEGER, b INTEGER, PRIMARY KEY (a, b));")
        .expect("create table");

    println!("\n=== INSERTING (1, 1) ===");
    engine
        .execute("INSERT INTO pk_combo VALUES (1, 1);")
        .expect("first insert");

    println!("\n=== INSERTING (1, 2) ===");
    match engine.execute("INSERT INTO pk_combo VALUES (1, 2);") {
        Ok(_) => println!("Second insert succeeded"),
        Err(e) => println!("Second insert failed: {}", e),
    }
}
