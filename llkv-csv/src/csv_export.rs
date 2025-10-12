use std::io::Write;
use std::path::Path;

use llkv_result::Result as LlkvResult;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use llkv_table::expr::Expr;
use llkv_table::table::ScanProjection;
use llkv_table::{Table, types::FieldId};

pub use crate::writer::{CsvExportColumn, CsvWriteOptions, CsvWriter};

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
    tracing::trace!("[CSV_EXPORT] export_csv_from_table called with {} columns", columns.len());
    let writer = CsvWriter::with_options(table, options.clone());
    tracing::trace!("[CSV_EXPORT] About to call write_columns_to_path");
    let result = writer.write_columns_to_path(csv_path, columns);
    tracing::trace!("[CSV_EXPORT] write_columns_to_path returned: {:?}", result);
    result
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
    let writer = CsvWriter::with_options(table, options.clone());
    writer.write_columns_to_path_with_filter(csv_path, columns, filter_expr)
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
    let writer_state = CsvWriter::with_options(table, options.clone());
    writer_state.write_columns_to_writer(writer, columns, filter_expr)
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
    let writer = CsvWriter::with_options(table, options.clone());
    writer.write_projections_to_path(csv_path, projections, filter_expr)
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
    let writer_state = CsvWriter::with_options(table, options.clone());
    writer_state.write_projections_to_writer(writer, projections, filter_expr)
}
