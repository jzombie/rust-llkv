// TODO: If running in development mode, include warning about unoptimized performance.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process;
use std::sync::Arc;
use std::time::Instant;

use arrow::util::pretty::print_batches;
use clap::{Args, Parser, Subcommand};
use llkv::{SqlEngine, storage::MemPager};
use llkv_tpch::qualification::{QualificationOptions, QualificationStatus};
use llkv_tpch::queries::{QueryOptions, StatementKind, render_tpch_query};
use llkv_tpch::{
    LoadSummary, SchemaPaths, TableLoadEvent, TpchError, TpchToolkit, install_default_schema,
    load_tpch_data,
};

const DEFAULT_SCALE_FACTOR: f64 = 0.01;
const DEFAULT_BATCH_SIZE: usize = 65_536;

fn main() {
    // Initialize tracing subscriber to respect RUST_LOG environment variable
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    if let Err(err) = run() {
        tracing::debug!("tpch bootstrap failed: {err}");
        process::exit(1);
    }
}

#[derive(Parser)]
#[command(
    name = "llkv-tpch",
    about = "TPC-H bootstrap and query runner for LLKV"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Install the canonical TPC-H schema and print catalog metadata.
    Install,
    /// Install the schema, load data, and optionally execute benchmark queries.
    Load(LoadArgs),
    /// Install the schema, load data, and run specific TPC-H queries.
    Query(QueryArgs),
    /// Install, load, and validate query answers against the canonical reference output.
    Qualify(QualifyArgs),
}

#[derive(Args, Clone)]
struct LoadArgs {
    /// Scale factor to load (defaults to 0.01, i.e. 10MB).
    #[arg(value_name = "SCALE", default_value_t = DEFAULT_SCALE_FACTOR)]
    scale: f64,
    /// One or more TPC-H query numbers (1-22) to execute after loading.
    #[arg(
        long = "query",
        short = 'q',
        value_delimiter = ',',
        value_parser = parse_query_number,
        value_name = "Q"
    )]
    queries: Vec<u8>,
    /// TPC-H stream number used when rendering parameterized templates.
    #[arg(long, default_value_t = 1)]
    stream: u32,
    /// Override a positional parameter, e.g. --param 3=VALUE.
    #[arg(long = "param", value_name = "INDEX=VALUE", value_parser = parse_param)]
    params: Vec<(usize, String)>,
}

#[derive(Args, Clone)]
struct QueryArgs {
    /// One or more TPC-H query numbers (1-22) to execute.
    #[arg(
        value_name = "Q",
        value_delimiter = ',',
        value_parser = parse_query_number,
        num_args = 1..
    )]
    queries: Vec<u8>,
    /// Scale factor to load before running the queries.
    #[arg(long, default_value_t = DEFAULT_SCALE_FACTOR)]
    scale: f64,
    /// TPC-H stream number used when rendering parameterized templates.
    #[arg(long, default_value_t = 1)]
    stream: u32,
    /// Override a positional parameter, e.g. --param 3=VALUE.
    #[arg(long = "param", value_name = "INDEX=VALUE", value_parser = parse_param)]
    params: Vec<(usize, String)>,
}

#[derive(Args, Clone)]
struct QualifyArgs {
    /// Scale factor to load before running qualification.
    #[arg(long, default_value_t = 1.0)]
    scale: f64,
    /// TPC-H stream number used when rendering parameterized templates.
    #[arg(long, default_value_t = 1)]
    stream: u32,
    /// One or more TPC-H query numbers (1-22) to validate.
    #[arg(
        long = "query",
        short = 'q',
        value_delimiter = ',',
        value_parser = parse_query_number,
        value_name = "Q"
    )]
    queries: Vec<u8>,
    /// Override the qualification dataset directory (defaults to the bundled ref_data).
    #[arg(long = "dataset", value_name = "PATH")]
    dataset_dir: Option<PathBuf>,
    /// Select the bundled qualification dataset scale (ref_data/<scale>).
    #[arg(long = "ref-scale", value_name = "NAME", default_value = "1")]
    ref_scale: String,
}

fn run() -> Result<(), TpchError> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Install) | None => run_install(),
        Some(Command::Load(args)) => run_load_command(args),
        Some(Command::Query(args)) => run_query_command(args),
        Some(Command::Qualify(args)) => run_qualify_command(args),
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
    print_query(&engine, "information_schema.columns", &columns_sql)
}

fn run_load_command(args: LoadArgs) -> Result<(), TpchError> {
    let execution = build_query_execution(&args.queries, args.stream, &args.params);
    run_load_internal(args.scale, execution)
}

fn run_query_command(args: QueryArgs) -> Result<(), TpchError> {
    let execution = build_query_execution(&args.queries, args.stream, &args.params)
        .ok_or_else(|| TpchError::Parse("no TPC-H queries provided".to_string()))?;
    run_load_internal(args.scale, Some(execution))
}

fn run_load_internal(
    scale_factor: f64,
    execution: Option<QueryExecution>,
) -> Result<(), TpchError> {
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

    if let Some(execution) = execution {
        run_tpch_queries(&engine, &schema.schema_name, execution)?;
    }

    Ok(())
}

fn run_qualify_command(args: QualifyArgs) -> Result<(), TpchError> {
    let toolkit = TpchToolkit::with_default_paths()?;
    let engine = SqlEngine::new(Arc::new(MemPager::default()));
    let schema = toolkit.install(&engine)?;

    println!(
        "Installing TPC-H schema '{}' and loading data at scale factor {:.3} for qualification",
        schema.schema_name, args.scale
    );

    let summary = toolkit.load_data_with_progress(
        &engine,
        &schema.schema_name,
        args.scale,
        DEFAULT_BATCH_SIZE,
        |event| match event {
            TableLoadEvent::Begin {
                table,
                estimated_rows,
            } => {
                if let Some(rows) = estimated_rows {
                    println!("  -> Loading {table} (~{rows} rows)...");
                } else {
                    println!("  -> Loading {table}...");
                }
            }
            TableLoadEvent::Progress { table, rows } => {
                println!("     {table}: {rows} rows loaded...");
            }
            TableLoadEvent::Complete {
                table,
                rows,
                elapsed,
            } => {
                println!(
                    "     finished {table} ({rows} rows) in {:.2}s",
                    elapsed.as_secs_f64()
                );
            }
        },
    )?;
    print_load_summary(&summary);

    let mut options = if let Some(dir) = &args.dataset_dir {
        QualificationOptions::new(dir.clone())
    } else {
        QualificationOptions::from_scale(toolkit.schema_paths(), &args.ref_scale)
    };
    options = options.with_stream_number(args.stream);
    if !args.queries.is_empty() {
        options = options.with_queries(args.queries.clone());
    }

    let reports = toolkit.run_qualification(&engine, &schema.schema_name, &options)?;

    println!(
        "\nQualification results for stream {} (dataset: {}):",
        args.stream,
        options.dataset_dir().display()
    );

    let mut passed = 0usize;
    for report in &reports {
        let status = if report.status == QualificationStatus::Pass {
            passed += 1;
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "  Q{:02}: {} (expected {} rows, actual {})",
            report.query, status, report.expected_row_count, report.actual_row_count
        );
        if report.status == QualificationStatus::Fail {
            if !report.missing_rows.is_empty() {
                println!("    missing rows:");
                for row in &report.missing_rows {
                    println!("      {}", row.join(" | "));
                }
            }
            if !report.extra_rows.is_empty() {
                println!("    extra rows:");
                for row in &report.extra_rows {
                    println!("      {}", row.join(" | "));
                }
            }
        }
    }

    println!("\nSummary: {passed}/{} queries passed", reports.len());

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

    // DEBUG: Print the actual SQL being executed
    tracing::debug!("\n=== DEBUG: ACTUAL SQL ===");
    tracing::debug!("{}", sql);
    tracing::debug!("=== END DEBUG ===\n");

    let batches = engine.sql(sql)?;
    if batches.is_empty() {
        println!("  (no rows)");
        return Ok(());
    }
    // Debug: print schema to see actual data types
    if !batches.is_empty() {
        tracing::debug!("DEBUG Schema:");
        for field in batches[0].schema().fields() {
            tracing::debug!("  {} -> {:?}", field.name(), field.data_type());
        }
    }
    print_batches(&batches).map_err(|err| TpchError::Sql(err.into()))
}

struct QueryExecution {
    numbers: Vec<u8>,
    options: QueryOptions,
}

fn build_query_execution(
    queries: &[u8],
    stream: u32,
    params: &[(usize, String)],
) -> Option<QueryExecution> {
    if queries.is_empty() {
        return None;
    }

    let mut seen = BTreeSet::new();
    let mut numbers = Vec::new();
    for query in queries {
        if seen.insert(*query) {
            numbers.push(*query);
        }
    }

    let mut overrides = BTreeMap::new();
    for (idx, value) in params {
        overrides.insert(*idx, value.clone());
    }

    Some(QueryExecution {
        numbers,
        options: QueryOptions {
            stream_number: stream,
            parameter_overrides: overrides,
        },
    })
}

fn run_tpch_queries(
    engine: &SqlEngine,
    schema: &str,
    execution: QueryExecution,
) -> Result<(), TpchError> {
    let QueryExecution { numbers, options } = execution;
    let paths = SchemaPaths::discover();

    println!(
        "\nRunning {} TPC-H quer{} against schema '{}' (stream {}):",
        numbers.len(),
        if numbers.len() == 1 { "y" } else { "ies" },
        schema,
        options.stream_number
    );

    for number in numbers {
        let rendered = render_tpch_query(&paths, number, schema, &options)?;
        let heading = rendered
            .title
            .as_deref()
            .map(|title| format!(" - {title}"))
            .unwrap_or_default();
        println!("\nTPC-H Q{:02}{}", rendered.number, heading);
        println!("Template: {}", rendered.path.display());

        let start = Instant::now();
        for statement in rendered.statements {
            match statement.kind {
                StatementKind::Command => {
                    engine.execute(&statement.sql).map_err(TpchError::Sql)?;
                }
                StatementKind::Query => {
                    // DEBUG: Print the SQL for Query 1
                    if rendered.number == 1 {
                        tracing::debug!("\n=== DEBUG: QUERY 1 SQL ===");
                        tracing::debug!("{}", statement.sql);
                        tracing::debug!("=== END DEBUG ===\n");
                    }

                    let batches = engine.sql(&statement.sql).map_err(TpchError::Sql)?;
                    if batches.is_empty() {
                        println!("  (no rows)");
                    } else {
                        // Debug: print schema to see actual data types
                        tracing::debug!("DEBUG Schema for Q{:02}:", rendered.number);
                        for field in batches[0].schema().fields() {
                            tracing::debug!("  {} -> {:?}", field.name(), field.data_type());
                        }
                        print_batches(&batches).map_err(|err| TpchError::Sql(err.into()))?;
                    }
                }
            }
        }
        println!("Completed Q{:02} in {:?}", rendered.number, start.elapsed());
    }

    Ok(())
}

fn parse_query_number(raw: &str) -> Result<u8, String> {
    let value = raw
        .parse::<u8>()
        .map_err(|err| format!("invalid query number '{raw}': {err}"))?;
    if (1..=22).contains(&value) {
        Ok(value)
    } else {
        Err(format!(
            "query number must be between 1 and 22, got {value}"
        ))
    }
}

fn parse_param(raw: &str) -> Result<(usize, String), String> {
    let (index, value) = raw
        .split_once('=')
        .ok_or_else(|| "expected INDEX=VALUE".to_string())?;
    let index = index
        .parse::<usize>()
        .map_err(|err| format!("invalid parameter index '{index}': {err}"))?;
    if index == 0 {
        return Err("parameter indices start at 1".to_string());
    }
    Ok((index, value.to_string()))
}
