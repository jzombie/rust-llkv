use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[allow(clippy::print_stdout, clippy::print_stderr)]
fn main() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let setup = r#"
            CREATE TABLE people (
                id INT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                age INT,
                city TEXT
            );

            -- Insert several rows including NULLs and duplicates for sorting edge cases
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (1, 'Ada', 'Lovelace', 36, 'London');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (2, 'Alan', 'Turing', 41, 'Cambridge');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (3, 'Grace', 'Hopper', 85, 'New York');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (4, 'Edsger', 'Dijkstra', 72, NULL);
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (5, 'Barbara', 'Liskov', NULL, 'Boston');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (6, 'Donald', 'Knuth', 83, 'Stanford');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (7, NULL, 'Unknown', 28, 'Unknown');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (8, 'Ada', 'Byron', 36, 'London');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (9, 'Alan', 'Turing', 41, 'Princeton');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (10, 'Zoe', 'Zeta', 29, 'Zurich');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (11, 'Ada', 'Lovelace', 36, 'London');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (12, 'Bob', 'Alice', 36, 'Cambridge');
            INSERT INTO people (id, first_name, last_name, age, city) VALUES (13, 'Charlie', 'Brown', 36, 'Cambridge');
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
    match engine.sql("SELECT * FROM people ORDER BY last_name ASC, first_name DESC, age ASC;") {
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
