// TODO: If running in development mode, include warning about unoptimized performance.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::PathBuf;
use std::process;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Int32Array, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::print_batches;
use clap::{Args, Parser, Subcommand};
use llkv::{
    Error as LlkvError, SqlEngine,
    storage::{InstrumentedPager, IoStats, IoStatsSnapshot, MemPager, PagerDiagnostics},
};
use llkv_table::diagnostics::{TablePagerIngestionDiagnostics, TablePagerIngestionSample};
use llkv_tpch::qualification::{QualificationOptions, QualificationStatus, verify_qualification_assets};
use llkv_tpch::queries::{QueryOptions, StatementKind, render_tpch_query};
use llkv_tpch::{
    LoadSummary, SchemaPaths, TableLoadEvent, TpchError, TpchToolkit, install_default_schema,
    resolve_loader_batch_size,
};

const DEFAULT_SCALE_FACTOR: f64 = 0.01;

fn parse_batch_size(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid batch size '{value}': {err}"))?;
    if parsed == 0 {
        return Err("batch size must be greater than zero".into());
    }
    Ok(parsed)
}

fn diagnostics_requested(cli_flag: bool) -> bool {
    if cli_flag {
        return true;
    }
    match env::var("LLKV_TPCH_PAGER_DIAGNOSTICS") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                true
            } else {
                !matches!(normalized.as_str(), "0" | "false" | "off")
            }
        }
        Err(env::VarError::NotPresent) => false,
        Err(env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                target: "tpch-loader",
                "LLKV_TPCH_PAGER_DIAGNOSTICS is not valid UTF-8; enabling diagnostics"
            );
            true
        }
    }
}

fn build_engine_with_diagnostics(enable: bool) -> (SqlEngine, Option<Arc<IoStats>>) {
    if enable {
        let (pager, stats) = InstrumentedPager::new(MemPager::default());
        (SqlEngine::new(Arc::new(pager)), Some(stats))
    } else {
        (SqlEngine::new(Arc::new(MemPager::default())), None)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes == 0 {
        return "0.00 MiB".to_string();
    }
    let mib = bytes as f64 / (1024.0 * 1024.0);
    format!("{mib:.2} MiB")
}

fn render_table_block(entry: &TablePagerIngestionSample) {
    let fresh_bytes = format_bytes(entry.delta.fresh_put_bytes);
    let overwrite_bytes = format_bytes(entry.delta.overwritten_put_bytes);
    let unknown_bytes = format_bytes(entry.delta.unknown_put_bytes);
    let overwrite_pct = entry.overwrite_pct();

    println!("\n    [pager] {}", entry.table);
    println!(
        "        Rows {:>10} | Duration {:>6.2}s",
        entry.rows,
        entry.elapsed.as_secs_f64()
    );
    println!(
        "        Fresh {:>5} puts ({fresh_bytes}) | Overwrite {:>5} puts ({overwrite_bytes}, {overwrite_pct:>5.1}%)",
        entry.delta.fresh_puts, entry.delta.overwritten_puts,
    );
    if entry.delta.unknown_puts > 0 {
        println!(
            "        Unknown {:>3} puts ({unknown_bytes})",
            entry.delta.unknown_puts
        );
    }
    let puts_per_batch = entry.puts_per_batch();
    let gets_per_batch = entry.gets_per_batch();

    println!(
        "        Put ops {:>6} ({:>3} batches, {:>5.1} ops/batch) | Get ops {:>6} ({:>3} batches, {:>5.1} ops/batch)",
        entry.delta.physical_puts,
        entry.delta.put_batches,
        puts_per_batch,
        entry.delta.physical_gets,
        entry.delta.get_batches,
        gets_per_batch,
    );
    println!(
        "        Alloc {:>6} keys ({:>3} batches) | Free {:>6} keys ({:>3} batches)",
        entry.delta.physical_allocs,
        entry.delta.alloc_batches,
        entry.delta.physical_frees,
        entry.delta.free_batches,
    );
}

fn print_table_summary(entries: &[TablePagerIngestionSample]) {
    if entries.is_empty() {
        return;
    }
    println!(
        "\nPager diagnostics summary:\n  {:<10} {:>10} {:>10} {:>14} {:>16} {:>12}",
        "Table", "Rows", "Seconds", "Fresh MiB", "Overwrite MiB", "Overwrite%"
    );
    for entry in entries {
        let fresh = entry.fresh_mib();
        let overwrite = entry.overwrite_mib();
        let pct = entry.overwrite_pct();
        println!(
            "  {:<10} {:>10} {:>10.2} {:>14.2} {:>16.2} {:>11.1}%",
            entry.table,
            entry.rows,
            entry.elapsed.as_secs_f64(),
            fresh,
            overwrite,
            pct,
        );
    }
}

fn print_pager_totals(totals: &IoStatsSnapshot) {
    let fresh = format_bytes(totals.fresh_put_bytes);
    let overwrite = format_bytes(totals.overwritten_put_bytes);
    let pct = totals.overwrite_pct();
    println!("\nPager totals:");
    println!(
        "  Fresh puts {:>8} ({fresh}) | Overwrites {:>8} ({overwrite}, {pct:>5.1}%)",
        totals.fresh_puts, totals.overwritten_puts
    );
    if totals.unknown_puts > 0 {
        let unknown = format_bytes(totals.unknown_put_bytes);
        println!("  Unknown puts {:>6} ({unknown})", totals.unknown_puts);
    }
    let total_puts_per_batch = totals.puts_per_batch();
    let total_gets_per_batch = totals.gets_per_batch();

    println!(
        "  Put ops {:>8} ({:>4} batches, {:>5.1} ops/batch) | Get ops {:>8} ({:>4} batches, {:>5.1} ops/batch)",
        totals.physical_puts,
        totals.put_batches,
        total_puts_per_batch,
        totals.physical_gets,
        totals.get_batches,
        total_gets_per_batch,
    );
    println!(
        "  Alloc keys {:>8} ({:>4} batches) | Free keys {:>8} ({:>4} batches)",
        totals.physical_allocs, totals.alloc_batches, totals.physical_frees, totals.free_batches,
    );
}

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
    /// Override the loader batch size (rows per INSERT statement).
    #[arg(long = "batch-size", value_name = "ROWS", value_parser = parse_batch_size)]
    batch_size: Option<usize>,
    /// Emit pager overwrite diagnostics during ingest (or set LLKV_TPCH_PAGER_DIAGNOSTICS=1).
    #[arg(long = "pager-diagnostics")]
    pager_diagnostics: bool,
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
    /// Override the loader batch size (rows per INSERT statement).
    #[arg(long = "batch-size", value_name = "ROWS", value_parser = parse_batch_size)]
    batch_size: Option<usize>,
    /// Emit pager overwrite diagnostics during ingest (or set LLKV_TPCH_PAGER_DIAGNOSTICS=1).
    #[arg(long = "pager-diagnostics")]
    pager_diagnostics: bool,
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
    /// Override the loader batch size (rows per INSERT statement).
    #[arg(long = "batch-size", value_name = "ROWS", value_parser = parse_batch_size)]
    batch_size: Option<usize>,
    /// Emit pager overwrite diagnostics during ingest (or set LLKV_TPCH_PAGER_DIAGNOSTICS=1).
    #[arg(long = "pager-diagnostics")]
    pager_diagnostics: bool,
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
    print_query(&engine, "information_schema.columns", &columns_sql)?;

    print_schema_constraints(&engine, &schema.schema_name)
}

fn run_load_command(args: LoadArgs) -> Result<(), TpchError> {
    let execution = build_query_execution(&args.queries, args.stream, &args.params);
    let diagnostics = diagnostics_requested(args.pager_diagnostics);
    run_load_internal(args.scale, execution, args.batch_size, diagnostics)
}

fn run_query_command(args: QueryArgs) -> Result<(), TpchError> {
    let execution = build_query_execution(&args.queries, args.stream, &args.params)
        .ok_or_else(|| TpchError::Parse("no TPC-H queries provided".to_string()))?;
    let diagnostics = diagnostics_requested(args.pager_diagnostics);
    run_load_internal(args.scale, Some(execution), args.batch_size, diagnostics)
}

fn run_load_internal(
    scale_factor: f64,
    execution: Option<QueryExecution>,
    batch_override: Option<usize>,
    diagnostics_enabled: bool,
) -> Result<(), TpchError> {
    let (engine, stats_handle) = build_engine_with_diagnostics(diagnostics_enabled);
    let toolkit = TpchToolkit::with_default_paths()?;
    let schema = toolkit.install(&engine)?;

    println!(
        "Installing TPC-H schema '{}' and loading data at scale factor {:.3}",
        schema.schema_name, scale_factor
    );

    let batch_size = resolve_loader_batch_size(&engine, batch_override);
    tracing::info!(
        target: "tpch-loader",
        batch_size,
        override_rows = ?batch_override,
        "resolved loader batch size"
    );
    println!("  -> Using loader batch size of {batch_size} rows");

    let diagnostics = stats_handle.as_ref().map(|stats| {
        let pager = Arc::new(PagerDiagnostics::new(Arc::clone(stats)));
        Arc::new(TablePagerIngestionDiagnostics::new(pager))
    });
    let diagnostics_hook = diagnostics.clone();
    let summary = toolkit.load_data_with_progress(
        &engine,
        &schema.schema_name,
        scale_factor,
        batch_size,
        |event| {
            if let Some(diag) = diagnostics_hook.as_ref() {
                match event {
                    TableLoadEvent::Begin { table, .. } => {
                        diag.begin_table(table);
                    }
                    TableLoadEvent::Complete {
                        table,
                        rows,
                        elapsed,
                    } => {
                        let entry = diag.finish_table(table, rows, elapsed);
                        render_table_block(&entry);
                    }
                    _ => {}
                }
            }
        },
    )?;
    print_load_summary(&summary);

    if let Some(diag) = diagnostics.as_ref() {
        print_table_summary(&diag.completed_tables());
        print_pager_totals(&diag.totals());
    }

    verify_schema_overview(&engine, &schema.schema_name)?;
    maybe_run_query_validation(&toolkit, &engine, &schema.schema_name, scale_factor)?;

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
    let diagnostics_flag = diagnostics_requested(args.pager_diagnostics);
    let (engine, stats_handle) = build_engine_with_diagnostics(diagnostics_flag);
    let diagnostics = stats_handle.as_ref().map(|stats| {
        let pager = Arc::new(PagerDiagnostics::new(Arc::clone(stats)));
        Arc::new(TablePagerIngestionDiagnostics::new(pager))
    });
    let schema = toolkit.install(&engine)?;

    println!(
        "Installing TPC-H schema '{}' and loading data at scale factor {:.3} for qualification",
        schema.schema_name, args.scale
    );

    let batch_size = resolve_loader_batch_size(&engine, args.batch_size);
    tracing::info!(
        target: "tpch-loader",
        batch_size,
        override_rows = ?args.batch_size,
        "resolved loader batch size for qualification"
    );
    println!("  -> Using loader batch size of {batch_size} rows");

    let diagnostics_hook = diagnostics.clone();
    let summary = toolkit.load_data_with_progress(
        &engine,
        &schema.schema_name,
        args.scale,
        batch_size,
        |event| {
            if let Some(diag) = diagnostics_hook.as_ref() {
                match event {
                    TableLoadEvent::Begin { table, .. } => diag.begin_table(table),
                    TableLoadEvent::Complete {
                        table,
                        rows,
                        elapsed,
                    } => {
                        let entry = diag.finish_table(table, rows, elapsed);
                        render_table_block(&entry);
                    }
                    _ => {}
                }
            }
            match event {
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
                TableLoadEvent::Progress {
                    table,
                    rows,
                    elapsed,
                    since_last,
                } => {
                    println!(
                        "     {table}: {rows} rows loaded... (+{:.2}s, total {:.2}s)",
                        since_last.as_secs_f64(),
                        elapsed.as_secs_f64()
                    );
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
            }
        },
    )?;
    print_load_summary(&summary);
    if let Some(diag) = diagnostics.as_ref() {
        print_table_summary(&diag.completed_tables());
        print_pager_totals(&diag.totals());
    }

    let mut options = if let Some(dir) = &args.dataset_dir {
        QualificationOptions::new(dir.clone())
    } else {
        QualificationOptions::from_scale(toolkit.schema_paths(), &args.ref_scale)
    };
    options = options.with_stream_number(args.stream);
    if !args.queries.is_empty() {
        options = options.with_queries(args.queries.clone());
    }

    options = ensure_dataset_available(&toolkit, options)?;

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

fn ensure_dataset_available(
    toolkit: &TpchToolkit,
    options: QualificationOptions,
) -> Result<QualificationOptions, TpchError> {
    match verify_qualification_assets(&options) {
        Ok(()) => Ok(options),
        Err(primary_err) => {
            let fallback_dir = toolkit.schema_paths().check_answers_dir();
            if fallback_dir != options.dataset_dir() {
                let fallback = options.clone().with_dataset(fallback_dir);
                if verify_qualification_assets(&fallback).is_ok() {
                    println!(
                        "Qualification dataset '{}' missing canonical results; using fallback '{}'",
                        options.dataset_dir().display(),
                        fallback.dataset_dir().display()
                    );
                    return Ok(fallback);
                }
            }
            Err(primary_err)
        }
    }
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

fn verify_schema_overview(engine: &SqlEngine, schema: &str) -> Result<(), TpchError> {
    let safe_schema = escape_identifier(schema);
    println!("\nSchema verification for '{schema}':");

    let tables_sql = format!(
        "SELECT table_name, table_type FROM information_schema.tables \
         WHERE table_schema = '{safe_schema}' ORDER BY table_name;"
    );
    print_query(engine, "Tables present", &tables_sql)?;
    print_schema_constraints(engine, schema)
}

fn print_schema_constraints(engine: &SqlEngine, schema: &str) -> Result<(), TpchError> {
    let constraints = load_constraints(engine, schema)?;
    let key_usage = load_key_column_usage(engine, schema)?;
    let referential = load_referential_constraints(engine, schema)?;
    let mut column_usage_schemas = BTreeSet::new();
    column_usage_schemas.insert(schema.to_string());
    for target in referential.values() {
        column_usage_schemas.insert(target.unique_schema.clone());
    }
    let constraint_columns = load_constraint_column_usage(engine, &column_usage_schemas)?;

    let pk_batch = build_constraint_overview_batch(&constraints, &key_usage)?;
    print_result_batch("Primary/Unique constraints", pk_batch)?;

    let fk_batch = build_foreign_key_overview_batch(
        &constraints,
        &key_usage,
        &constraint_columns,
        &referential,
    )?;
    print_result_batch("Foreign key relationships", fk_batch)
}

fn maybe_run_query_validation(
    toolkit: &TpchToolkit,
    engine: &SqlEngine,
    schema: &str,
    scale_factor: f64,
) -> Result<(), TpchError> {
    println!("\nValidating TPC-H query answers (focus: Q1)");
    let Some(scale_label) = qualification_scale_label(scale_factor) else {
        println!(
            "  - Skipping query validation: no reference dataset for scale factor {:.3}",
            scale_factor
        );
        return Ok(());
    };

    let dataset_dir = toolkit.schema_paths().ref_data_dir(&scale_label);
    if !dataset_dir.is_dir() {
        println!(
            "  - Skipping query validation: reference dataset missing at {}",
            dataset_dir.display()
        );
        return Ok(());
    }

    let mut options = QualificationOptions::from_scale(toolkit.schema_paths(), &scale_label);
    options = options.with_stream_number(1).with_queries(vec![1]);
    let reports = toolkit.run_qualification(engine, schema, &options)?;
    if let Some(report) = reports.first() {
        if report.passed() {
            println!(
                "  - Q{:02} PASS (expected {} rows, actual {})",
                report.query, report.expected_row_count, report.actual_row_count
            );
        } else {
            println!(
                "  - Q{:02} FAIL (expected {} rows, actual {})",
                report.query, report.expected_row_count, report.actual_row_count
            );
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

    Ok(())
}

fn qualification_scale_label(scale_factor: f64) -> Option<String> {
    if scale_factor <= 0.0 {
        return None;
    }
    let rounded = scale_factor.round();
    if (scale_factor - rounded).abs() < 1e-9 {
        Some(format!("{:.0}", rounded))
    } else {
        None
    }
}

fn escape_identifier(value: &str) -> String {
    value.replace('\'', "''")
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

fn print_result_batch(label: &str, batch: Option<RecordBatch>) -> Result<(), TpchError> {
    println!("\n{label}");
    if let Some(batch) = batch {
        let batches = vec![batch];
        print_batches(&batches).map_err(|err| TpchError::Sql(err.into()))
    } else {
        println!("  (no rows)");
        Ok(())
    }
}

fn build_constraint_overview_batch(
    constraints: &BTreeMap<ConstraintKey, ConstraintMeta>,
    key_usage: &BTreeMap<ConstraintKey, Vec<KeyColumn>>,
) -> Result<Option<RecordBatch>, TpchError> {
    let mut entries: Vec<&ConstraintMeta> = constraints
        .values()
        .filter(|meta| matches!(meta.constraint_type.as_str(), "PRIMARY KEY" | "UNIQUE"))
        .collect();
    entries.sort_by(|a, b| {
        a.table_name
            .cmp(&b.table_name)
            .then_with(|| a.constraint_name.cmp(&b.constraint_name))
    });

    let mut rows = Vec::new();
    for meta in entries {
        let key = ConstraintKey::new(&meta.schema, &meta.constraint_name);
        if let Some(columns) = key_usage.get(&key) {
            for column in columns {
                rows.push(vec![
                    meta.table_name.clone(),
                    meta.constraint_name.clone(),
                    meta.constraint_type.clone(),
                    column.column_name.clone(),
                ]);
            }
        }
    }

    build_string_batch(
        &[
            "table_name",
            "constraint_name",
            "constraint_type",
            "column_name",
        ],
        rows,
    )
}

fn build_foreign_key_overview_batch(
    constraints: &BTreeMap<ConstraintKey, ConstraintMeta>,
    key_usage: &BTreeMap<ConstraintKey, Vec<KeyColumn>>,
    referenced_columns: &BTreeMap<ConstraintKey, ConstraintColumns>,
    referential: &BTreeMap<ConstraintKey, UniqueConstraintRef>,
) -> Result<Option<RecordBatch>, TpchError> {
    let mut entries: Vec<&ConstraintMeta> = constraints
        .values()
        .filter(|meta| meta.constraint_type.eq_ignore_ascii_case("FOREIGN KEY"))
        .collect();
    entries.sort_by(|a, b| {
        a.table_name
            .cmp(&b.table_name)
            .then_with(|| a.constraint_name.cmp(&b.constraint_name))
    });

    let mut rows = Vec::new();
    for meta in entries {
        let key = ConstraintKey::new(&meta.schema, &meta.constraint_name);
        let Some(columns) = key_usage.get(&key) else {
            continue;
        };
        let Some(target) = referential.get(&key) else {
            continue;
        };
        let reference_key = ConstraintKey::new(&target.unique_schema, &target.unique_name);
        let referenced_entry = referenced_columns.get(&reference_key);
        let referenced_table = referenced_entry
            .map(|entry| entry.table_name.clone())
            .unwrap_or_else(|| "<unknown>".to_string());

        for (idx, column) in columns.iter().enumerate() {
            let referenced_column = referenced_entry
                .and_then(|entry| entry.columns.get(idx))
                .cloned()
                .unwrap_or_else(|| "<unknown>".to_string());
            rows.push(vec![
                meta.table_name.clone(),
                meta.constraint_name.clone(),
                column.column_name.clone(),
                referenced_table.clone(),
                referenced_column,
            ]);
        }
    }

    build_string_batch(
        &[
            "table_name",
            "constraint_name",
            "column_name",
            "referenced_table",
            "referenced_column",
        ],
        rows,
    )
}

fn build_string_batch(
    headers: &[&str],
    rows: Vec<Vec<String>>,
) -> Result<Option<RecordBatch>, TpchError> {
    if rows.is_empty() {
        return Ok(None);
    }

    let mut builders: Vec<StringBuilder> = headers.iter().map(|_| StringBuilder::new()).collect();

    for row in rows {
        for (idx, value) in row.into_iter().enumerate() {
            builders[idx].append_value(&value);
        }
    }

    let fields: Vec<Field> = headers
        .iter()
        .map(|name| Field::new(*name, DataType::Utf8, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let mut columns = Vec::with_capacity(builders.len());
    for mut builder in builders {
        let array = builder.finish();
        columns.push(Arc::new(array) as ArrayRef);
    }

    let batch = RecordBatch::try_new(schema, columns).map_err(|err| TpchError::Sql(err.into()))?;
    Ok(Some(batch))
}

fn load_constraints(
    engine: &SqlEngine,
    schema: &str,
) -> Result<BTreeMap<ConstraintKey, ConstraintMeta>, TpchError> {
    let safe_schema = escape_identifier(schema);
    let sql = format!(
        "SELECT constraint_schema, constraint_name, table_name, constraint_type \
         FROM information_schema.table_constraints \
         WHERE table_schema = '{safe_schema}';"
    );
    let batches = engine.sql(&sql)?;
    let mut constraints = BTreeMap::new();

    for batch in batches {
        let schema_col = downcast_string_column(&batch, "constraint_schema")?;
        let name_col = downcast_string_column(&batch, "constraint_name")?;
        let table_col = downcast_string_column(&batch, "table_name")?;
        let type_col = downcast_string_column(&batch, "constraint_type")?;

        for row in 0..batch.num_rows() {
            let schema_value = string_value(schema_col, row);
            let name_value = name_col.value(row);
            let key = ConstraintKey::new(&schema_value, name_value);
            constraints.insert(
                key,
                ConstraintMeta {
                    schema: schema_value,
                    table_name: table_col.value(row).to_string(),
                    constraint_name: name_value.to_string(),
                    constraint_type: type_col.value(row).to_string(),
                },
            );
        }
    }

    Ok(constraints)
}

fn load_key_column_usage(
    engine: &SqlEngine,
    schema: &str,
) -> Result<BTreeMap<ConstraintKey, Vec<KeyColumn>>, TpchError> {
    let safe_schema = escape_identifier(schema);
    let sql = format!(
        "SELECT constraint_schema, constraint_name, column_name, ordinal_position \
         FROM information_schema.key_column_usage \
         WHERE constraint_schema = '{safe_schema}' \
         ORDER BY constraint_schema, constraint_name, ordinal_position;"
    );
    let batches = engine.sql(&sql)?;
    let mut usage = BTreeMap::new();

    for batch in batches {
        let schema_col = downcast_string_column(&batch, "constraint_schema")?;
        let name_col = downcast_string_column(&batch, "constraint_name")?;
        let column_col = downcast_string_column(&batch, "column_name")?;
        let ordinal_col = downcast_int32_column(&batch, "ordinal_position")?;

        for row in 0..batch.num_rows() {
            let schema_value = string_value(schema_col, row);
            let name_value = name_col.value(row);
            let key = ConstraintKey::new(&schema_value, name_value);
            let ordinal = ordinal_col.value(row);
            usage.entry(key).or_insert_with(Vec::new).push(KeyColumn {
                ordinal,
                column_name: column_col.value(row).to_string(),
            });
        }
    }

    for columns in usage.values_mut() {
        columns.sort_by_key(|column| column.ordinal);
    }

    Ok(usage)
}

fn load_referential_constraints(
    engine: &SqlEngine,
    schema: &str,
) -> Result<BTreeMap<ConstraintKey, UniqueConstraintRef>, TpchError> {
    let safe_schema = escape_identifier(schema);
    let sql = format!(
        "SELECT constraint_schema, constraint_name, unique_constraint_schema, unique_constraint_name \
         FROM information_schema.referential_constraints \
         WHERE constraint_schema = '{safe_schema}';"
    );
    let batches = engine.sql(&sql)?;
    let mut mapping = BTreeMap::new();

    for batch in batches {
        let schema_col = downcast_string_column(&batch, "constraint_schema")?;
        let name_col = downcast_string_column(&batch, "constraint_name")?;
        let unique_schema_col = downcast_string_column(&batch, "unique_constraint_schema")?;
        let unique_name_col = downcast_string_column(&batch, "unique_constraint_name")?;

        for row in 0..batch.num_rows() {
            let schema_value = string_value(schema_col, row);
            let name_value = name_col.value(row);
            let key = ConstraintKey::new(&schema_value, name_value);
            mapping.insert(
                key,
                UniqueConstraintRef {
                    unique_schema: string_value(unique_schema_col, row),
                    unique_name: unique_name_col.value(row).to_string(),
                },
            );
        }
    }

    Ok(mapping)
}

fn load_constraint_column_usage(
    engine: &SqlEngine,
    schemas: &BTreeSet<String>,
) -> Result<BTreeMap<ConstraintKey, ConstraintColumns>, TpchError> {
    if schemas.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut columns = BTreeMap::new();
    for schema_name in schemas {
        let safe_schema = escape_identifier(schema_name);
        let sql = format!(
            "SELECT constraint_schema, constraint_name, table_schema, table_name, column_name \
             FROM information_schema.constraint_column_usage \
             WHERE constraint_schema = '{safe_schema}' \
             ORDER BY constraint_schema, constraint_name;"
        );
        let batches = engine.sql(&sql)?;

        for batch in batches {
            let schema_col = downcast_string_column(&batch, "constraint_schema")?;
            let name_col = downcast_string_column(&batch, "constraint_name")?;
            let table_schema_col = downcast_string_column(&batch, "table_schema")?;
            let table_name_col = downcast_string_column(&batch, "table_name")?;
            let column_col = downcast_string_column(&batch, "column_name")?;

            for row in 0..batch.num_rows() {
                let schema_value = string_value(schema_col, row);
                let name_value = name_col.value(row);
                let key = ConstraintKey::new(&schema_value, name_value);
                let entry = columns.entry(key).or_insert_with(|| ConstraintColumns {
                    schema: table_schema_col.value(row).to_string(),
                    table_name: table_name_col.value(row).to_string(),
                    columns: Vec::new(),
                });
                entry.columns.push(column_col.value(row).to_string());
            }
        }
    }

    Ok(columns)
}

fn downcast_string_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a StringArray, TpchError> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| internal_error(format!("missing column '{name}' in result")))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| internal_error(format!("column '{name}' is not Utf8")))
}

fn downcast_int32_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Int32Array, TpchError> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| internal_error(format!("missing column '{name}' in result")))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| internal_error(format!("column '{name}' is not Int32")))
}

fn string_value(column: &StringArray, row: usize) -> String {
    if column.is_null(row) {
        String::new()
    } else {
        column.value(row).to_string()
    }
}

fn internal_error(message: impl Into<String>) -> TpchError {
    TpchError::Sql(LlkvError::Internal(message.into()))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ConstraintKey {
    schema: String,
    name: String,
}

impl ConstraintKey {
    fn new(schema: &str, name: &str) -> Self {
        Self {
            schema: schema.to_ascii_lowercase(),
            name: name.to_ascii_lowercase(),
        }
    }
}

#[derive(Debug, Clone)]
struct ConstraintMeta {
    schema: String,
    table_name: String,
    constraint_name: String,
    constraint_type: String,
}

#[derive(Debug, Clone)]
struct KeyColumn {
    ordinal: i32,
    column_name: String,
}

#[derive(Debug, Clone)]
struct ConstraintColumns {
    #[allow(dead_code)]
    schema: String,
    table_name: String,
    columns: Vec<String>,
}

#[derive(Debug, Clone)]
struct UniqueConstraintRef {
    unique_schema: String,
    unique_name: String,
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
