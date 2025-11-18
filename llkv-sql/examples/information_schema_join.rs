use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[allow(clippy::print_stdout, clippy::print_stderr)]
fn main() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let setup = r#"
        CREATE TABLE projects (
            id INT PRIMARY KEY,
            name TEXT,
            owner TEXT,
            budget INT,
            status TEXT
        );

        INSERT INTO projects (id, name, owner, budget, status) VALUES
            (1, 'Vector Indexer', 'Search', 175000, 'active'),
            (2, 'Stream Alerts', 'Observability', 92000, 'paused'),
            (3, 'Metrics Rewrite', NULL, 210000, 'active');

        CREATE TABLE column_policies (
            table_name TEXT,
            column_name TEXT,
            classification TEXT,
            must_capture BOOLEAN,
            PRIMARY KEY (table_name, column_name)
        );

        INSERT INTO column_policies (table_name, column_name, classification, must_capture) VALUES
            ('projects', 'id', 'identifier', true),
            ('projects', 'name', 'identity', true),
            ('projects', 'owner', 'steward', false),
            ('projects', 'budget', 'finance', true),
            ('projects', 'status', 'lifecycle', false);
    "#;

    if let Err(error) = engine.execute(setup) {
        eprintln!("Setup failed: {:?}", error);
        std::process::exit(1);
    }

    let catalog = engine.runtime_context().table_catalog();
    println!("Tables: {:?}", catalog.table_names());

    // Combine metadata from information_schema with local policy data to plan governance work.
    let query = r#"
        SELECT
            c.column_name,
            c.data_type,
            CASE WHEN c.is_nullable THEN 'optional' ELSE 'required' END AS null_policy,
            p.classification,
            CASE
                WHEN p.must_capture THEN 'capture + document'
                WHEN c.is_nullable THEN 'optional reference'
                ELSE 'monitor usage'
            END AS governance_action
        FROM "information_schema.columns" AS c
        JOIN column_policies AS p
          ON p.table_name = c.table_name
         AND p.column_name = c.column_name
        WHERE c.table_name = 'projects'
        ORDER BY c.ordinal_position;
    "#;

    match engine.sql(query) {
        Ok(batches) => {
            if batches.is_empty() {
                println!("Query produced no data");
                return;
            }

            match pretty_format_batches(&batches) {
                Ok(pretty) => println!("Information schema join result:\n{}", pretty),
                Err(error) => eprintln!("Query succeeded but formatting failed: {:?}", error),
            }
        }
        Err(error) => {
            eprintln!("Query failed: {:?}", error);
            std::process::exit(1);
        }
    }
}
