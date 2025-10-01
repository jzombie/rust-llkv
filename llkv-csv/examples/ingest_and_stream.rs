use std::env;
use std::io::Write;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Builder};
use arrow::csv::WriterBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::print_batches;
use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME, rowid_fid};
use llkv_csv::csv_ingest::append_csv_into_table;
use llkv_csv::{CsvReadOptions, open_csv_reader};
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::MemPager;
use llkv_table::Table;
use llkv_table::expr::{Expr, Filter, Operator};
use llkv_table::table::{ScanProjection, ScanStreamOptions};
use tempfile::NamedTempFile;

fn main() -> LlkvResult<()> {
    let path = match env::args().nth(1) {
        Some(arg) => PathBuf::from(arg),
        None => {
            eprintln!("Usage: ingest_and_stream <path-to-csv>");
            std::process::exit(2);
        }
    };

    run(path)
}

fn run(path: PathBuf) -> LlkvResult<()> {
    let pager = Arc::new(MemPager::default());
    // Use a simple table identifier for the in-memory table.
    let table = Table::new(1, Arc::clone(&pager))?;

    let options = CsvReadOptions::default();
    let (augmented_csv, _) = materialize_csv_with_row_ids(path.as_path(), &options)?;
    append_csv_into_table(&table, augmented_csv.path(), &options)?;

    let mut logical_fields = table.store().user_field_ids_for_table(table.table_id());
    if logical_fields.is_empty() {
        println!("The CSV file did not yield any user columns.");
        return Ok(());
    }

    logical_fields.sort_by_key(|lfid| lfid.field_id());

    // Build a canonical Arrow schema for the table and use it for debug output
    // and projection construction.
    let schema = table.schema()?;

    // Debug: show the table schema fields (name, field_id, type)
    println!("Table schema:");
    for (idx, field) in schema.fields().iter().enumerate() {
        if field.name() == ROW_ID_COLUMN_NAME {
            println!("  [{}] {} (row id)", idx, field.name());
            continue;
        }
        let fid = field
            .metadata()
            .get("field_id")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        println!(
            "  [{}] {} (field_id = {}, type = {:?})",
            idx,
            field.name(),
            fid,
            field.data_type()
        );
    }
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

    table.scan_stream(
        &projections,
        &filter_expr,
        ScanStreamOptions::default(),
        |batch| {
            if print_result.is_err() {
                return;
            }

            batch_index += 1;
            total_rows += batch.num_rows();

            println!("--- Batch {} ({} rows) ---", batch_index, batch.num_rows());
            let batches = vec![batch];
            if let Err(err) = print_batches(&batches) {
                print_result = Err(err.into());
            }
            println!();
        },
    )?;

    print_result?;

    if batch_index == 0 {
        println!("No rows were emitted from the table scan.");
    } else {
        println!(
            "Stream completed: {} batches, {} total rows.",
            batch_index, total_rows
        );
    }

    Ok(())
}

fn materialize_csv_with_row_ids(
    path: &Path,
    options: &CsvReadOptions,
) -> LlkvResult<(NamedTempFile, Vec<String>)> {
    let (schema, mut reader) = open_csv_reader(path, options)
        .map_err(|err| LlkvError::Internal(format!("failed to read CSV: {err}")))?;
    // Capture the original column names (excluding row_id) so we can use
    // them as friendly aliases when printing later on.
    let original_column_names: Vec<String> = schema
        .fields()
        .iter()
        .filter(|f| f.name() != ROW_ID_COLUMN_NAME)
        .map(|f| f.name().to_string())
        .collect();

    let mut fields: Vec<Field> = Vec::with_capacity(schema.fields().len() + 1);
    fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
    fields.extend(schema.fields().iter().map(|field| field.as_ref().clone()));
    let augmented_schema = Arc::new(Schema::new(fields));

    let mut tmp = NamedTempFile::new()?;
    let mut builder = WriterBuilder::new();
    builder = builder.with_delimiter(options.delimiter);
    builder = builder.with_header(true);
    let mut writer = builder.build(tmp.as_file_mut());

    let mut next_row_id: u64 = 0;

    while let Some(batch) = reader.next() {
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
