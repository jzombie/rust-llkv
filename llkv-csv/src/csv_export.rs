use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Bound;
use std::path::Path;

use arrow::csv::WriterBuilder;
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use llkv_table::expr::{Expr, Filter, Operator};
use llkv_table::table::{ScanProjection, ScanStreamOptions};
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
    let projections = build_column_projections(table, columns)?;
    let file = File::create(csv_path.as_ref()).map_err(|err| {
        Error::Internal(format!(
            "failed to create CSV file '{}': {err}",
            csv_path.as_ref().display()
        ))
    })?;
    let writer = BufWriter::new(file);
    export_csv_to_writer_with_projections(
        table,
        writer,
        projections,
        filter_expr,
        options,
    )
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
    let projections = build_column_projections(table, columns)?;
    export_csv_to_writer_with_projections(table, writer, projections, filter_expr, options)
}

pub fn export_csv_from_table_with_projections<P, C, I, SP>(
    table: &Table<P>,
    csv_path: C,
    projections: I,
    filter_expr: &Expr<'_, FieldId>,
    options: &CsvWriteOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    C: AsRef<Path>,
    I: IntoIterator<Item = SP>,
    SP: Into<ScanProjection>,
{
    let file = File::create(csv_path.as_ref()).map_err(|err| {
        Error::Internal(format!(
            "failed to create CSV file '{}': {err}",
            csv_path.as_ref().display()
        ))
    })?;
    let writer = BufWriter::new(file);
    export_csv_to_writer_with_projections(table, writer, projections, filter_expr, options)
}

pub fn export_csv_to_writer_with_projections<P, W, I, SP>(
    table: &Table<P>,
    writer: W,
    projections: I,
    filter_expr: &Expr<'_, FieldId>,
    options: &CsvWriteOptions,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    W: Write,
    I: IntoIterator<Item = SP>,
    SP: Into<ScanProjection>,
{
    let mut projections: Vec<ScanProjection> =
        projections.into_iter().map(|p| p.into()).collect();

    if projections.is_empty() {
        return Err(Error::InvalidArgumentError(
            "at least one projection must be provided for CSV export".into(),
        ));
    }

    ensure_column_aliases(table, &mut projections)?;

    let mut builder = WriterBuilder::new();
    builder = builder.with_delimiter(options.delimiter);
    builder = builder.with_header(options.include_header);
    let mut csv_writer = builder.build(writer);

    let mut write_error: Option<Error> = None;
    let scan_options = ScanStreamOptions {
        include_nulls: options.include_nulls,
    };

    table.scan_stream_with_exprs(&projections, filter_expr, scan_options, |batch| {
        if write_error.is_some() {
            return;
        }
        if let Err(err) = csv_writer.write(&batch) {
            write_error = Some(Error::Internal(format!(
                "failed to write CSV batch: {err}"
            )));
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

fn build_column_projections<P>(
    table: &Table<P>,
    columns: &[CsvExportColumn],
) -> LlkvResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if columns.is_empty() {
        return Err(Error::InvalidArgumentError(
            "at least one column must be provided for CSV export".into(),
        ));
    }

    let field_ids: Vec<FieldId> = columns.iter().map(|c| c.field_id).collect();
    let column_meta = table.get_cols_meta(&field_ids);

    let mut projections: Vec<ScanProjection> = Vec::with_capacity(columns.len());
    for (idx, column) in columns.iter().enumerate() {
        let lfid = LogicalFieldId::for_user(table.table_id(), column.field_id);
    table.store().data_type(lfid)?;

        let resolved_name = column
            .alias
            .clone()
            .or_else(|| {
                column_meta
                    .get(idx)
                    .and_then(|meta| meta.as_ref().and_then(|m| m.name.clone()))
            })
            .unwrap_or_else(|| format!("col_{}", column.field_id));

        let projection = Projection::with_alias(lfid, resolved_name);
        projections.push(ScanProjection::from(projection));
    }

    Ok(projections)
}

fn ensure_column_aliases<P>(table: &Table<P>, projections: &mut [ScanProjection]) -> LlkvResult<()> 
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut missing_field_ids: Vec<FieldId> = Vec::new();
    for projection in projections.iter() {
        if let ScanProjection::Column(col_proj) = projection {
            if col_proj.logical_field_id.table_id() != table.table_id() {
                return Err(Error::InvalidArgumentError(format!(
                    "projection targets table {} but export is on table {}",
                    col_proj.logical_field_id.table_id(),
                    table.table_id(),
                )));
            }

            if col_proj.alias.is_none() {
                missing_field_ids.push(col_proj.logical_field_id.field_id());
            }
        }
    }

    if missing_field_ids.is_empty() {
        return Ok(());
    }

    let metas = table.get_cols_meta(&missing_field_ids);
    let mut alias_map: HashMap<FieldId, String> = HashMap::new();
    for (field_id, meta) in missing_field_ids.iter().zip(metas.into_iter()) {
        let name = meta
            .and_then(|m| m.name)
            .unwrap_or_else(|| format!("col_{}", field_id));
        alias_map.insert(*field_id, name);
    }

    for projection in projections.iter_mut() {
        if let ScanProjection::Column(col_proj) = projection {
            if col_proj.alias.is_none() {
                let field_id = col_proj.logical_field_id.field_id();
                if let Some(name) = alias_map.get(&field_id) {
                    col_proj.alias = Some(name.clone());
                } else {
                    col_proj.alias = Some(format!("col_{}", field_id));
                }
            }
        }
    }

    Ok(())
}
