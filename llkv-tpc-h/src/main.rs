use std::env;
use std::process;
use std::sync::Arc;

use arrow::util::pretty::print_batches;
use llkv::{SqlEngine, storage::MemPager};
use llkv_tpc_h::{LoadSummary, TpchError, install_default_schema, load_tpch_data};

const DEFAULT_SCALE_FACTOR: f64 = 0.01;
const DEFAULT_BATCH_SIZE: usize = 500;

fn main() {
    if let Err(err) = run_from_args() {
        eprintln!("tpch bootstrap failed: {err}");
        process::exit(1);
    }
}

fn run_from_args() -> Result<(), TpchError> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("load") => {
            let scale_arg = args
                .next()
                .unwrap_or_else(|| DEFAULT_SCALE_FACTOR.to_string());
            let scale_factor = scale_arg
                .parse::<f64>()
                .map_err(|_| TpchError::Parse(format!("invalid scale factor '{scale_arg}'")))?;
            run_load(scale_factor)
        }
        _ => run_install(),
    }
}

fn run_install() -> Result<(), TpchError> {
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

fn run_load(scale_factor: f64) -> Result<(), TpchError> {
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    let schema = install_default_schema(&engine)?;

    println!(
        "Installing TPC-H schema '{}' and loading data at scale factor {:.3}",
        schema.schema_name, scale_factor
    );

    let summary = load_tpch_data(
        &engine,
        &schema.schema_name,
        scale_factor,
        DEFAULT_BATCH_SIZE,
    )?;
    print_load_summary(&summary);

    let query_specs = vec![
        (
            "TPCD.LINEITEM row count",
            format!(
                "SELECT COUNT(*) AS line_count FROM {}.LINEITEM;",
                schema.schema_name
            ),
        ),
        (
            "Customers by market segment",
            format!(
                "SELECT c_mktsegment, COUNT(*) AS customers \
                 FROM {}.CUSTOMER \
                 GROUP BY c_mktsegment \
                 ORDER BY c_mktsegment;",
                schema.schema_name
            ),
        ),
        (
            "Orders by status",
            format!(
                "SELECT o_orderstatus, COUNT(*) AS orders \
                 FROM {}.ORDERS \
                 GROUP BY o_orderstatus \
                 ORDER BY o_orderstatus;",
                schema.schema_name
            ),
        ),
    ];

    for (label, sql) in query_specs {
        print_query(&engine, label, &sql)?;
    }

    Ok(())
}

fn print_load_summary(summary: &LoadSummary) {
    println!(
        "\nLoaded {} rows across {} tables:",
        summary.total_rows(),
        summary.tables.len()
    );
    for entry in &summary.tables {
        println!("  - {:<9} {:>10} rows", entry.table, entry.rows);
    }
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
