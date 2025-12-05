//! CSV export writer built on top of table scan projections.
//!
//! [`CsvWriter`] adapts table scans into Arrow CSV writers, handling alias resolution and
//! optional filtering while streaming batches directly to disk or an arbitrary `Write` sink.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Bound;
use std::path::Path;

use arrow::csv::WriterBuilder;
use llkv_column_map::store::Projection;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_types::LogicalFieldId;
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

/// Builder-style helper for writing CSV data from a table.
#[derive(Clone)]
pub struct CsvWriter<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: &'a Table<P>,
    options: CsvWriteOptions,
}

impl<'a, P> CsvWriter<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(table: &'a Table<P>) -> Self {
        Self {
            table,
            options: CsvWriteOptions::default(),
        }
    }

    pub fn with_options(table: &'a Table<P>, options: CsvWriteOptions) -> Self {
        Self { table, options }
    }

    pub fn options(&self) -> &CsvWriteOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut CsvWriteOptions {
        &mut self.options
    }

    pub fn into_options(self) -> CsvWriteOptions {
        self.options
    }

    pub fn write_columns_to_path<C>(
        &self,
        csv_path: C,
        columns: &[CsvExportColumn],
    ) -> LlkvResult<()>
    where
        C: AsRef<Path>,
    {
        tracing::trace!(
            "[CSV_EXPORT] write_columns_to_path called with {} columns",
            columns.len()
        );
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

        tracing::trace!("[CSV_EXPORT] Calling write_columns_to_path_with_filter");
        let result = self.write_columns_to_path_with_filter(csv_path, columns, &filter_expr);
        tracing::trace!(
            "[CSV_EXPORT] write_columns_to_path_with_filter returned: {:?}",
            result
        );
        result
    }

    pub fn write_columns_to_path_with_filter<C>(
        &self,
        csv_path: C,
        columns: &[CsvExportColumn],
        filter_expr: &Expr<'_, FieldId>,
    ) -> LlkvResult<()>
    where
        C: AsRef<Path>,
    {
        tracing::trace!("[CSV_EXPORT] write_columns_to_path_with_filter called");
        tracing::trace!("[CSV_EXPORT] About to call build_column_projections");
        let projections = build_column_projections(self.table, columns)?;
        tracing::trace!("[CSV_EXPORT] build_column_projections returned successfully");
        self.write_projections_to_path(csv_path, projections, filter_expr)
    }

    pub fn write_columns_to_writer<W>(
        &self,
        writer: W,
        columns: &[CsvExportColumn],
        filter_expr: &Expr<'_, FieldId>,
    ) -> LlkvResult<()>
    where
        W: Write,
    {
        let projections = build_column_projections(self.table, columns)?;
        self.write_projections_to_writer(writer, projections, filter_expr)
    }

    pub fn write_projections_to_path<C, I, SP>(
        &self,
        csv_path: C,
        projections: I,
        filter_expr: &Expr<'_, FieldId>,
    ) -> LlkvResult<()>
    where
        C: AsRef<Path>,
        I: IntoIterator<Item = SP>,
        SP: Into<ScanProjection>,
    {
        tracing::trace!("[CSV_EXPORT] write_projections_to_path called");
        let file = File::create(csv_path.as_ref()).map_err(|err| {
            Error::Internal(format!(
                "failed to create CSV file '{}': {err}",
                csv_path.as_ref().display()
            ))
        })?;
        let writer = BufWriter::new(file);
        tracing::trace!("[CSV_EXPORT] About to call write_projections_to_writer");
        let result = self.write_projections_to_writer(writer, projections, filter_expr);
        tracing::trace!(
            "[CSV_EXPORT] write_projections_to_writer returned: {:?}",
            result
        );
        result
    }

    pub fn write_projections_to_writer<W, I, SP>(
        &self,
        writer: W,
        projections: I,
        filter_expr: &Expr<'_, FieldId>,
    ) -> LlkvResult<()>
    where
        W: Write,
        I: IntoIterator<Item = SP>,
        SP: Into<ScanProjection>,
    {
        tracing::trace!("[CSV_EXPORT] write_projections_to_writer called");
        let mut projections: Vec<ScanProjection> =
            projections.into_iter().map(|p| p.into()).collect();

        tracing::trace!("[CSV_EXPORT] Got {} projections", projections.len());
        if projections.is_empty() {
            return Err(Error::InvalidArgumentError(
                "at least one projection must be provided for CSV export".into(),
            ));
        }

        tracing::trace!("[CSV_EXPORT] About to call ensure_column_aliases");
        let result = ensure_column_aliases(self.table, &mut projections);
        tracing::trace!("[CSV_EXPORT] ensure_column_aliases returned: {:?}", result);
        result?;

        let mut builder = WriterBuilder::new();
        builder = builder.with_delimiter(self.options.delimiter);
        builder = builder.with_header(self.options.include_header);
        let mut csv_writer = builder.build(writer);

        let mut write_error: Option<Error> = None;
        let scan_options = ScanStreamOptions {
            include_nulls: self.options.include_nulls,
            order: None,
            row_id_filter: None,
            include_row_ids: false,
            ranges: None,
            driving_column: None,
        };

        tracing::trace!("[CSV_EXPORT] About to call scan_stream_with_exprs");
        tracing::trace!("[CSV_EXPORT] filter_expr: {:?}", filter_expr);
        let scan_result =
            self.table
                .scan_stream_with_exprs(&projections, filter_expr, scan_options, |batch| {
                    if write_error.is_some() {
                        return;
                    }
                    if let Err(err) = csv_writer.write(&batch) {
                        write_error =
                            Some(Error::Internal(format!("failed to write CSV batch: {err}")));
                    }
                });
        tracing::trace!(
            "[CSV_EXPORT] scan_stream_with_exprs returned: {:?}",
            scan_result
        );
        scan_result?;

        if let Some(err) = write_error {
            return Err(err);
        }

        let mut inner_writer = csv_writer.into_inner();
        inner_writer
            .flush()
            .map_err(|err| Error::Internal(format!("failed to flush CSV writer: {err}")))?;

        Ok(())
    }
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
        tracing::trace!(
            "[CSV_EXPORT] Checking column field_id={}, lfid={:?}, table_id={}",
            column.field_id,
            lfid,
            table.table_id()
        );
        match table.store().data_type(lfid) {
            Ok(dt) => tracing::trace!("[CSV_EXPORT] Found data_type: {:?}", dt),
            Err(e) => {
                tracing::trace!("[CSV_EXPORT] data_type lookup failed: {:?}", e);
                return Err(e);
            }
        }

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
        if let ScanProjection::Column(col_proj) = projection
            && col_proj.alias.is_none()
        {
            let field_id = col_proj.logical_field_id.field_id();
            if let Some(name) = alias_map.get(&field_id) {
                col_proj.alias = Some(name.clone());
            } else {
                col_proj.alias = Some(format!("col_{}", field_id));
            }
        }
    }

    Ok(())
}
