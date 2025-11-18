//! Data audit report using information_schema.
//!
//! Demonstrates combining information_schema metadata with user-defined audit rules
//! to generate a comprehensive data governance report. Shows nullable columns,
//! primary keys, and data type distribution across tables.

use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use llkv_sql::SqlEngine;
use llkv_storage::pager::MemPager;

#[allow(clippy::print_stdout, clippy::print_stderr)]
fn main() {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));

    let setup = r#"
        -- Create sample application tables
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            name TEXT,
            created_at INTEGER,
            credit_score INTEGER
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total_amount INTEGER NOT NULL,
            status TEXT,
            shipped_date INTEGER
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price INTEGER NOT NULL,
            stock_level INTEGER
        );

        -- Insert sample data
        INSERT INTO customers VALUES (1, 'alice@example.com', 'Alice Smith', 1700000000, 750);
        INSERT INTO customers VALUES (2, 'bob@example.com', 'Bob Jones', 1700000100, 680);
        INSERT INTO orders VALUES (101, 1, 15000, 'shipped', 1700000200);
        INSERT INTO orders VALUES (102, 2, 28000, 'pending', NULL);
        INSERT INTO products VALUES (1, 'Laptop', 'Electronics', 120000, 50);
        INSERT INTO products VALUES (2, 'Desk Chair', 'Furniture', 35000, 125);

        -- Define audit rules for each table/column
        CREATE TABLE audit_rules (
            table_name TEXT,
            column_name TEXT,
            sensitivity TEXT,
            requires_encryption INTEGER,
            PRIMARY KEY (table_name, column_name)
        );

        INSERT INTO audit_rules VALUES ('customers', 'customer_id', 'low', 0);
        INSERT INTO audit_rules VALUES ('customers', 'email', 'high', 1);
        INSERT INTO audit_rules VALUES ('customers', 'name', 'medium', 0);
        INSERT INTO audit_rules VALUES ('customers', 'credit_score', 'high', 1);
        INSERT INTO audit_rules VALUES ('orders', 'order_id', 'low', 0);
        INSERT INTO audit_rules VALUES ('orders', 'customer_id', 'medium', 0);
        INSERT INTO audit_rules VALUES ('orders', 'total_amount', 'high', 1);
        INSERT INTO audit_rules VALUES ('products', 'product_id', 'low', 0);
        INSERT INTO audit_rules VALUES ('products', 'price', 'medium', 1);
    "#;

    if let Err(error) = engine.execute(setup) {
        eprintln!("Setup failed: {:?}", error);
        std::process::exit(1);
    }

    println!("=== DATA GOVERNANCE AUDIT REPORT ===\n");

    // Report 1: Column-level audit with nullable and encryption requirements
    println!("📋 Column Audit (Nullable + Encryption Requirements):");
    let audit_query = r#"
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable AS nullable,
            c.is_primary_key AS is_pk,
            a.sensitivity,
            a.requires_encryption AS needs_encrypt
        FROM information_schema.columns AS c
        LEFT JOIN audit_rules AS a
          ON a.table_name = c.table_name
         AND a.column_name = c.column_name
        WHERE c.table_name IN ('customers', 'orders', 'products')
        ORDER BY c.table_name, c.ordinal_position;
    "#;

    match engine.sql(audit_query) {
        Ok(batches) if !batches.is_empty() => {
            if let Ok(pretty) = pretty_format_batches(&batches) {
                println!("{}\n", pretty);
            }
        }
        Ok(_) => println!("No audit data found\n"),
        Err(e) => eprintln!("Audit query failed: {:?}\n", e),
    }

    // Report 2: Table-level summary with column counts and key status
    println!("📊 Table Summary (Column Counts & Primary Keys):");
    let summary_query = r#"
        SELECT
            table_name,
            COUNT(*) AS total_columns,
            SUM(CAST(is_primary_key AS INTEGER)) AS pk_columns,
            SUM(CAST(is_nullable AS INTEGER)) AS nullable_columns
        FROM information_schema.columns
        WHERE table_name IN ('customers', 'orders', 'products')
        GROUP BY table_name
        ORDER BY table_name;
    "#;

    match engine.sql(summary_query) {
        Ok(batches) if !batches.is_empty() => {
            if let Ok(pretty) = pretty_format_batches(&batches) {
                println!("{}\n", pretty);
            }
        }
        Ok(_) => println!("No summary data found\n"),
        Err(e) => eprintln!("Summary query failed: {:?}\n", e),
    }

    // Report 3: High-sensitivity columns requiring encryption
    println!("🔒 High-Sensitivity Encrypted Columns:");
    let sensitive_query = r#"
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            a.sensitivity
        FROM information_schema.columns AS c
        JOIN audit_rules AS a
          ON a.table_name = c.table_name
         AND a.column_name = c.column_name
        WHERE a.requires_encryption = 1
        ORDER BY c.table_name, c.column_name;
    "#;

    match engine.sql(sensitive_query) {
        Ok(batches) if !batches.is_empty() => {
            if let Ok(pretty) = pretty_format_batches(&batches) {
                println!("{}\n", pretty);
            }
        }
        Ok(_) => println!("No sensitive columns found\n"),
        Err(e) => eprintln!("Sensitive columns query failed: {:?}\n", e),
    }

    // Report 4: Data type distribution across all tables
    println!("📈 Data Type Distribution:");
    let type_distribution_query = r#"
        SELECT
            data_type,
            COUNT(*) AS column_count
        FROM information_schema.columns
        WHERE table_name IN ('customers', 'orders', 'products', 'audit_rules')
        GROUP BY data_type
        ORDER BY column_count DESC, data_type;
    "#;

    match engine.sql(type_distribution_query) {
        Ok(batches) if !batches.is_empty() => {
            if let Ok(pretty) = pretty_format_batches(&batches) {
                println!("{}", pretty);
            }
        }
        Ok(_) => println!("No type distribution data found"),
        Err(e) => eprintln!("Type distribution query failed: {:?}", e),
    }
}
