use std::process;
use std::sync::Arc;

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
    for table in schema.tables {
        println!(
            "  - {:<10} {:>10} base rows ({})",
            table.name, table.base_rows, table.description
        );
    }
    Ok(())
}
