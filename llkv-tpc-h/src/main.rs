use std::process;
use std::sync::Arc;

use arrow::util::pretty::print_batches;
use llkv::{SqlEngine, storage::MemPager};
use llkv_tpc_h::{TpchError, install_default_schema};

fn main() {
    if let Err(err) = run() {
        eprintln!("tpch bootstrap failed: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), TpchError> {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    let schema = install_default_schema(&engine)?;

    println!(
        "Installed {} TPC-H tables into schema '{}':",
        schema.tables.len(),
        schema.schema_name
    );
    for table in &schema.tables {
        println!(
            "  - {:<10} {:>10} base rows ({})",
            table.name, table.base_rows, table.description
        );
    }

    print_query(
        &engine,
        "information_schema.tables",
        "SELECT table_schema, table_name, table_type \
         FROM information_schema.tables \
         ORDER BY table_schema, table_name;",
    )?;

    let columns_sql = format!(
        "SELECT table_schema, table_name, column_name, ordinal_position, data_type \
         FROM information_schema.columns \
         WHERE table_schema = '{}' \
         ORDER BY table_name, ordinal_position;",
        schema.schema_name
    );
    print_query(&engine, "information_schema.columns", &columns_sql)?;
    Ok(())
}

fn print_query(engine: &SqlEngine, label: &str, sql: &str) -> Result<(), TpchError> {
    println!("\n{label}");
    let batches = engine.sql(sql)?;
    if batches.is_empty() {
        println!("  (no rows)");
        return Ok(());
    }
    print_batches(&batches).map_err(|err| TpchError::Sql(err.into()))
}
