use std::env;
use std::io::Write;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, UInt64Builder};
use arrow::csv::WriterBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::print_batches;
use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME, rowid_fid};
use llkv_csv::csv_ingest::append_csv_into_table;
use llkv_csv::{CsvReadOptions, CsvReader};
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::expr::{Expr, Filter, Operator};
use llkv_table::table::{ScanProjection, ScanStreamOptions};
use tempfile::NamedTempFile;

fn main() -> LlkvResult<()> {
    // Usage: ingest_and_stream <path-to-csv> [--no-print] [--null-token TOKEN] [--sample N] [--batch-size N]
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ingest_and_stream <path-to-csv> [--no-print]");
        std::process::exit(2);
    }

    // First non-flag argument is the CSV path
    let mut path_arg: Option<String> = None;
    let mut no_print = false;
    let mut sample: Option<usize> = None;
    let mut batch_size: Option<usize> = None;
    let mut null_token: Option<String> = None;
    let mut iter = args.iter().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--no-print" => {
                no_print = true;
            }
            "--sample" => {
                if let Some(v) = iter.next()
                    && let Ok(n) = v.parse::<usize>()
                {
                    sample = Some(n);
                }
            }
            "--batch-size" => {
                if let Some(v) = iter.next()
                    && let Ok(n) = v.parse::<usize>()
                {
                    batch_size = Some(n);
                }
            }
            "--null-token" => {
                if let Some(v) = iter.next() {
                    null_token = Some(v.to_string());
                }
            }
            other => {
                if path_arg.is_none() {
                    path_arg = Some(other.to_string());
                }
            }
        }
    }

    let path = match path_arg {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Usage: ingest_and_stream <path-to-csv> [--no-print]");
            std::process::exit(2);
        }
    };

    run(path, no_print, sample, batch_size, null_token)
}

fn run(
    path: PathBuf,
    no_print: bool,
    sample: Option<usize>,
    batch_size: Option<usize>,
    null_token: Option<String>,
) -> LlkvResult<()> {
    let pager = Arc::new(MemPager::default());
    // Use a simple table identifier for the in-memory table.
    let table = Table::new(1, Arc::clone(&pager))?;

    let mut options = CsvReadOptions::default();
    if let Some(s) = sample {
        options.max_read_records = Some(s);
    }
    if let Some(b) = batch_size {
        options.batch_size = Some(b);
    }
    if let Some(t) = null_token {
        options.null_token = Some(t);
    }
    // Materialize CSV with injected row ids and time the operation.
    let mat_start = Instant::now();
    let (augmented_csv, _) = materialize_csv_with_row_ids(path.as_path(), &options)?;
    let mat_elapsed = mat_start.elapsed();
    println!("CSV materialization completed in {:.3?}", mat_elapsed);

    // Ingest the augmented CSV into the table and time ingestion.
    let ingest_start = Instant::now();
    append_csv_into_table(&table, augmented_csv.path(), &options)?;
    let ingest_elapsed = ingest_start.elapsed();
    println!("CSV ingest completed in {:.3?}", ingest_elapsed);

    let mut logical_fields = table.store().user_field_ids_for_table(table.table_id());
    if logical_fields.is_empty() {
        println!("The CSV file did not yield any user columns.");
        return Ok(());
    }

    logical_fields.sort_by_key(|lfid| lfid.field_id());

    // Build a canonical Arrow schema for the table and use it for debug output
    // and projection construction.
    let schema = table.schema()?;
    // Pretty-print the table schema as a RecordBatch for clearer output.
    println!("Table schema:");
    let schema_batch = table.schema_recordbatch()?;
    // print_batches takes &[RecordBatch]
    print_batches(&[schema_batch])?;
    println!();

    // Build projections from the schema (skip the row_id field)
    let mut projections: Vec<ScanProjection> = Vec::new();
    for field in schema.fields().iter() {
        if field.name() == ROW_ID_COLUMN_NAME {
            continue;
        }
        let fid = field
            .metadata()
            .get("field_id")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or_else(|| {
                // fallback: try to extract numeric suffix from generated col_<id>
                field
                    .name()
                    .strip_prefix("col_")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0)
            });

        let lfid = llkv_column_map::types::LogicalFieldId::for_user(table.table_id(), fid);
        let alias = field.name().to_string();
        projections.push(ScanProjection::from(Projection::with_alias(lfid, alias)));
    }
    // Debug: the row-id shadow column is created per user field (presence index),
    // not as a single table-wide field. Check the first user field's shadow.
    if let Some(first_lfid) = logical_fields.first() {
        let rid_lfid = rowid_fid(*first_lfid);
        match table.store().data_type(rid_lfid) {
            Ok(dt) => {
                let total = table.store().total_rows_for_field(rid_lfid).ok();
                println!(
                    "Row-id shadow column for first field: logical = {:?}, type = {:?}, total_rows = {:?}",
                    rid_lfid, dt, total
                );
            }
            Err(_) => println!("Row-id shadow column for first field: not present"),
        }
    } else {
        println!("No user fields to inspect for row-id shadow column");
    }
    println!();

    // `projections` was constructed above from the schema; use it directly.

    println!("Loaded columns:");
    for (proj_idx, proj) in projections.iter().enumerate() {
        match proj {
            ScanProjection::Column(proj) => {
                println!(
                    "  [{}] {} (field_id = {})",
                    proj_idx,
                    proj.alias.as_deref().unwrap_or("<unnamed>"),
                    proj.logical_field_id.field_id()
                );
            }
            ScanProjection::Computed { alias, .. } => {
                println!("  [{}] <computed> alias = {}", proj_idx, alias);
            }
        }
    }
    println!();

    let first_field = match projections.first() {
        Some(ScanProjection::Column(p)) => p.logical_field_id.field_id(),
        Some(ScanProjection::Computed { .. }) => {
            // If the first projection is computed, fall back to scanning the
            // first user field id we found earlier.
            logical_fields[0].field_id()
        }
        None => {
            println!("No projections available");
            return Ok(());
        }
    };
    let filter_expr = Expr::Pred(Filter {
        field_id: first_field,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    let mut batch_index = 0usize;
    let mut total_rows = 0usize;
    let mut print_result: LlkvResult<()> = Ok(());

    // Metrics aggregation
    use std::time::Duration;
    let mut total_materialization = Duration::from_secs(0);
    let mut total_print_time = Duration::from_secs(0);
    let mut min_materialization: Option<Duration> = None;
    let mut max_materialization: Option<Duration> = None;
    let mut min_print: Option<Duration> = None;
    let mut max_print: Option<Duration> = None;

    let scan_start = Instant::now();
    // Track the end-time of the previous callback (start equals scan_start)
    let mut prev_callback_end = scan_start;

    table.scan_stream(
        &projections,
        &filter_expr,
        ScanStreamOptions::default(),
        |batch| {
            if print_result.is_err() {
                return;
            }

            // Arrival time = when the batch materialization finished and the
            // callback was invoked. The time since prev_callback_end is the
            // materialization duration for this batch (excludes printing).
            let arrival = Instant::now();
            let mat_dur = arrival.duration_since(prev_callback_end);
            total_materialization += mat_dur;
            min_materialization = Some(min_materialization.map_or(mat_dur, |m| m.min(mat_dur)));
            max_materialization = Some(max_materialization.map_or(mat_dur, |m| m.max(mat_dur)));

            batch_index += 1;
            total_rows += batch.num_rows();

            println!("--- Batch {} ({} rows) ---", batch_index, batch.num_rows());

            if !no_print {
                // Printing/pretty formatting time only
                let print_start = Instant::now();
                let batches = vec![batch];
                if let Err(err) = print_batches(&batches) {
                    print_result = Err(err.into());
                }
                let print_elapsed = print_start.elapsed();
                total_print_time += print_elapsed;
                min_print = Some(min_print.map_or(print_elapsed, |p| p.min(print_elapsed)));
                max_print = Some(max_print.map_or(print_elapsed, |p| p.max(print_elapsed)));
            } else {
                // When printing is disabled, we don't call print_batches and
                // leave printing metrics as zero.
            }

            // Mark end of callback (used to compute next batch's materialization)
            prev_callback_end = Instant::now();
        },
    )?;
    let scan_elapsed = scan_start.elapsed();
    println!("Scan completed in {:.3?}", scan_elapsed);

    print_result?;

    if batch_index == 0 {
        println!("No rows were emitted from the table scan.");
    } else {
        println!(
            "Stream completed: {} batches, {} total rows.",
            batch_index, total_rows
        );

        // Also print total projected columns and total cells (rows x columns)
        let projected_columns = projections.len();
        let total_cells: usize = projected_columns.saturating_mul(total_rows);
        println!("Projected columns: {}", projected_columns);
        println!("Total cells (rows x cols): {}", total_cells);

        // Final metrics report
        let avg_mat = if batch_index > 0 {
            total_materialization / (batch_index as u32)
        } else {
            Duration::from_secs(0)
        };
        let avg_print = if batch_index > 0 {
            total_print_time / (batch_index as u32)
        } else {
            Duration::from_secs(0)
        };

        println!("\n=== Metrics Summary ===");
        println!("CSV materialization: {:.3?}", mat_elapsed);
        println!("CSV ingest: {:.3?}", ingest_elapsed);
        println!("Batches: {}, total rows: {}", batch_index, total_rows);
        println!(
            "Materialization (excluding printing): total = {:.3?}, avg = {:.3?}, min = {:.3?}, max = {:.3?}",
            total_materialization,
            avg_mat,
            min_materialization.unwrap_or_else(|| Duration::from_secs(0)),
            max_materialization.unwrap_or_else(|| Duration::from_secs(0))
        );
        println!(
            "Printing time: total = {:.3?}, avg = {:.3?}, min = {:.3?}, max = {:.3?}",
            total_print_time,
            avg_print,
            min_print.unwrap_or_else(|| Duration::from_secs(0)),
            max_print.unwrap_or_else(|| Duration::from_secs(0))
        );
    }

    Ok(())
}

fn materialize_csv_with_row_ids(
    path: &Path,
    options: &CsvReadOptions,
) -> LlkvResult<(NamedTempFile, Vec<String>)> {
    let reader = CsvReader::with_options(options.clone());
    let mut session = reader
        .open(path)
        .map_err(|err| LlkvError::Internal(format!("failed to read CSV: {err}")))?;
    let schema = session.schema();
    // Capture the original column names (excluding row_id) so we can use
    // them as friendly aliases when printing later on.
    let original_column_names: Vec<String> = schema
        .fields()
        .iter()
        .filter(|f| f.name() != ROW_ID_COLUMN_NAME)
        .map(|f| f.name().to_string())
        .collect();

    let reader_schema = session.reader().schema();
    let mut fields: Vec<Field> = Vec::with_capacity(reader_schema.fields().len() + 1);
    fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
    fields.extend(
        reader_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone()),
    );
    let augmented_schema = Arc::new(Schema::new(fields));

    let mut tmp = NamedTempFile::new()?;
    let mut builder = WriterBuilder::new();
    builder = builder.with_delimiter(options.delimiter);
    builder = builder.with_header(true);
    let mut writer = builder.build(tmp.as_file_mut());

    let mut next_row_id: u64 = 0;

    for batch in session {
        let batch = batch?;
        let len = batch.num_rows();
        if len == 0 {
            continue;
        }

        let mut row_id_builder = UInt64Builder::with_capacity(len);
        for _ in 0..len {
            row_id_builder.append_value(next_row_id);
            next_row_id += 1;
        }
        let row_ids: ArrayRef = Arc::new(row_id_builder.finish());

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() + 1);
        columns.push(row_ids);
        columns.extend(batch.columns().iter().cloned());

        let augmented_batch = RecordBatch::try_new(Arc::clone(&augmented_schema), columns)?;

        // TODO(#55): Remove manual row_id injection once CSV ingest can synthesize it automatically.
        writer.write(&augmented_batch)?;
    }

    let inner = writer.into_inner();
    inner.flush()?;

    Ok((tmp, original_column_names))
}
