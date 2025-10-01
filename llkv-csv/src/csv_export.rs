use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;

use arrow::csv::WriterBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use llkv_table::expr::{Expr, Filter, Operator};
use llkv_table::table::ScanStreamOptions;
use llkv_table::{Table, types::FieldId};

/// Configuration for writing CSV files.
#[derive(Debug, Clone)]
pub struct CsvWriteOptions {
    /// Write a header row with column names when true.
    pub include_header: bool,
    /// Delimiter to use between fields.
    pub delimiter: u8,
    /// Preserve rows that are entirely null when true.
    pub include_nulls: bool,
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        Self {
            include_header: true,
            delimiter: b',',
            include_nulls: true,
        }
    }
}

/// Column specification for CSV export.
#[derive(Debug, Clone)]
pub struct CsvExportColumn {
    pub field_id: FieldId,
    pub alias: Option<String>,
}

impl CsvExportColumn {
    pub fn new(field_id: FieldId) -> Self {
        Self {
            field_id,
            alias: None,
        }
    }

    pub fn with_alias<S: Into<String>>(field_id: FieldId, alias: S) -> Self {
        Self {
            field_id,
            alias: Some(alias.into()),
        }
    }
}

pub fn export_csv_from_table<P, C>(
    table: &Table<P>,
    csv_path: C,
    columns: &[CsvExportColumn],
    options: &CsvWriteOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    C: AsRef<Path>,
{
    if columns.is_empty() {
        return Err(Error::InvalidArgumentError(
            "at least one column must be provided for CSV export".into(),
        ));
    }

    let filter_expr = Expr::Pred(Filter {
        field_id: columns[0].field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    });

    export_csv_from_table_with_filter(table, csv_path, columns, &filter_expr, options)
}

pub fn export_csv_from_table_with_filter<P, C>(
    table: &Table<P>,
    csv_path: C,
    columns: &[CsvExportColumn],
    filter_expr: &Expr<'_, FieldId>,
    options: &CsvWriteOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    C: AsRef<Path>,
{
    let file = File::create(csv_path.as_ref()).map_err(|err| {
        Error::Internal(format!(
            "failed to create CSV file '{}': {err}",
            csv_path.as_ref().display()
        ))
    })?;
    let writer = BufWriter::new(file);
    export_csv_to_writer_with_filter(table, writer, columns, filter_expr, options)
}

pub fn export_csv_to_writer_with_filter<P, W>(
    table: &Table<P>,
    writer: W,
    columns: &[CsvExportColumn],
    filter_expr: &Expr<'_, FieldId>,
    options: &CsvWriteOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    W: Write,
{
    if columns.is_empty() {
        return Err(Error::InvalidArgumentError(
            "at least one column must be provided for CSV export".into(),
        ));
    }

    let field_ids: Vec<FieldId> = columns.iter().map(|c| c.field_id).collect();
    let column_meta = table.get_cols_meta(&field_ids);

    let mut column_names: Vec<String> = Vec::with_capacity(columns.len());
    let mut schema_fields: Vec<Field> = Vec::with_capacity(columns.len());

    for (idx, column) in columns.iter().enumerate() {
        let lfid = LogicalFieldId::for_user(table.table_id(), column.field_id);
        let dtype: DataType = table.store().data_type(lfid)?;

        let resolved_name = column
            .alias
            .clone()
            .or_else(|| {
                column_meta
                    .get(idx)
                    .and_then(|meta| meta.as_ref().and_then(|m| m.name.clone()))
            })
            .unwrap_or_else(|| format!("col_{}", column.field_id));

        schema_fields.push(Field::new(&resolved_name, dtype, true));
        column_names.push(resolved_name);
    }

    let schema = Arc::new(Schema::new(schema_fields));
    let mut projections: Vec<Projection> = Vec::with_capacity(columns.len());
    for (name, column) in column_names.iter().zip(columns.iter()) {
        let lfid = LogicalFieldId::for_user(table.table_id(), column.field_id);
        projections.push(Projection::with_alias(lfid, name.clone()));
    }

    let mut builder = WriterBuilder::new();
    builder = builder.with_delimiter(options.delimiter);
    builder = builder.with_header(options.include_header);
    let mut csv_writer = builder.build(writer);

    let mut write_error: Option<Error> = None;
    let scan_options = ScanStreamOptions {
        include_nulls: options.include_nulls,
    };

    table.scan_stream(&projections, filter_expr, scan_options, |batch| {
        if write_error.is_some() {
            return;
        }
        let arrays = batch.columns().to_vec();
        match RecordBatch::try_new(Arc::clone(&schema), arrays) {
            Ok(out_batch) => {
                if let Err(err) = csv_writer.write(&out_batch) {
                    write_error =
                        Some(Error::Internal(format!("failed to write CSV batch: {err}")));
                }
            }
            Err(err) => {
                write_error = Some(Error::Internal(format!(
                    "failed to materialize CSV batch: {err}"
                )));
            }
        }
    })?;

    if let Some(err) = write_error {
        return Err(err);
    }

    let mut inner_writer = csv_writer.into_inner();
    inner_writer
        .flush()
        .map_err(|err| Error::Internal(format!("failed to flush CSV writer: {err}")))?;

    Ok(())
}
