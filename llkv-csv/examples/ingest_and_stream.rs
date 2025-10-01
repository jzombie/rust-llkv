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
use llkv_column_map::store::{Projection, ROW_ID_COLUMN_NAME};
use llkv_csv::csv_ingest::append_csv_into_table;
use llkv_csv::{CsvReadOptions, open_csv_reader};
use llkv_result::{Error as LlkvError, Result as LlkvResult};
use llkv_storage::pager::MemPager;
use llkv_table::expr::{Expr, Filter, Operator};
use llkv_table::table::{ScanProjection, ScanStreamOptions};
use llkv_table::Table;
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
    let augmented_csv = materialize_csv_with_row_ids(path.as_path(), &options)?;
    append_csv_into_table(&table, augmented_csv.path(), &options)?;

    let mut logical_fields = table
        .store()
        .user_field_ids_for_table(table.table_id());
    if logical_fields.is_empty() {
        println!("The CSV file did not yield any user columns.");
        return Ok(());
    }

    logical_fields.sort_by_key(|lfid| lfid.field_id());

    let field_ids: Vec<_> = logical_fields.iter().map(|lfid| lfid.field_id()).collect();
    let column_meta = table.get_cols_meta(&field_ids);

    let projections: Vec<ScanProjection> = logical_fields
        .iter()
        .enumerate()
        .map(|(idx, lfid)| {
            let alias = column_meta
                .get(idx)
                .and_then(|meta_opt| meta_opt.as_ref().and_then(|meta| meta.name.clone()))
                .unwrap_or_else(|| format!("col_{}", lfid.field_id()));
            ScanProjection::from(Projection::with_alias(*lfid, alias))
        })
        .collect();

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

    let first_field = field_ids[0];
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

fn materialize_csv_with_row_ids(path: &Path, options: &CsvReadOptions) -> LlkvResult<NamedTempFile> {
    let (schema, mut reader) = open_csv_reader(path, options)
        .map_err(|err| LlkvError::Internal(format!("failed to read CSV: {err}")))?;

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

    Ok(tmp)
}
