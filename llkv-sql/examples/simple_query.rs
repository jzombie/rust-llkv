use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[allow(clippy::print_stdout, clippy::print_stderr)]
fn main() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let setup = r#"
        CREATE TABLE users (id INT PRIMARY KEY, name TEXT);
        INSERT INTO users (id, name) VALUES (1, 'Ada');
        INSERT INTO users (id, name) VALUES (2, 'Sam');
        
        -- Comments begin with `--` and are ignored by the SQL engine.
        -- BEGIN TRANSACTION;
        -- INSERT INTO users (id, name) VALUES (3, 'Bob');
        -- ROLLBACK;
        -- COMMIT;

        INSERT INTO users (id, name) VALUES (3, 'Alice');
    "#;

    // Execute setup statements
    match engine.execute(setup) {
        Ok(results) => {
            println!("Setup executed: {} statements", results.len());
        }
        Err(e) => {
            eprintln!("Setup failed: {:?}", e);
            std::process::exit(1);
        }
    }

    // Run a select and pretty-print the resulting RecordBatches
    match engine.sql("SELECT * FROM users ORDER BY id DESC;") {
        Ok(batches) => {
            if batches.is_empty() {
                println!("No batches returned");
                return;
            }

            match pretty_format_batches(&batches) {
                Ok(s) => println!("Query result:\n{}", s),
                Err(e) => println!("Query executed but failed to format batches: {:?}", e),
            }
        }
        Err(e) => {
            eprintln!("Query failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
