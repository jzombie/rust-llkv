//! System catalog for storing table and column metadata.
//!
//! The system catalog uses table 0 (reserved) to store metadata about all tables
//! and columns in the database. This metadata includes:
//!
//! - **Table metadata** ([`TableMeta`]): Table ID, name, creation time, flags
//! - **Column metadata** ([`ColMeta`]): Column ID, name, flags, default values
//! - **Multi-column unique metadata** ([`TableMultiColumnUniqueMeta`]): Unique index definitions per table
//!
//! # Storage Format
//!
//! The catalog stores metadata as serialized [`bitcode`] blobs in special catalog
//! columns within table 0. See [`CATALOG_TABLE_ID`] and related constants in the
//! [`reserved`](crate::reserved) module.
//!
//! # Usage
//!
//! The [`SysCatalog`] provides methods to:
//! - Insert/update table metadata ([`put_table_meta`](SysCatalog::put_table_meta))
//! - Query table metadata ([`get_table_meta`](SysCatalog::get_table_meta))
//! - Manage column metadata similarly
//!
//! This metadata is used by higher-level components to validate schemas, assign
//! field IDs, and enforce table constraints.

use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use arrow::array::{Array, BinaryArray, BinaryBuilder, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bitcode::{Decode, Encode};

use crate::constraints::{
    ConstraintId, ConstraintRecord, decode_constraint_row_id, encode_constraint_row_id,
};
use crate::types::{FieldId, RowId, TableId};
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
};

use llkv_column_map::types::LogicalFieldId;
use llkv_column_map::{
    ColumnStore,
    store::{GatherNullPolicy, ROW_ID_COLUMN_NAME, rowid_fid},
    types::Namespace,
};
use llkv_result::{self, Result as LlkvResult};
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

// Import all reserved constants and validation functions
use crate::reserved::*;

// ----- Namespacing helpers -----

// TODO: Dedupe with llkv_column_map::types::lfid()
#[inline]
fn lfid(table_id: TableId, col_id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(table_id)
        .with_field_id(col_id)
}

// TODO: Migrate to llkv_column_map::types::rid_table()
#[inline]
fn rid_table(table_id: TableId) -> u64 {
    let fid = LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(table_id)
        .with_field_id(0);
    fid.into()
}

// TODO: Migrate to llkv_column_map::types::rid_col()
#[inline]
fn rid_col(table_id: TableId, col_id: u32) -> u64 {
    lfid(table_id, col_id).into()
}

const CONSTRAINT_SCAN_CHUNK: usize = 256;

#[inline]
fn constraint_meta_lfid() -> LogicalFieldId {
    lfid(CATALOG_TABLE_ID, CATALOG_FIELD_CONSTRAINT_META_ID)
}

#[inline]
fn constraint_row_lfid() -> LogicalFieldId {
    rowid_fid(constraint_meta_lfid())
}

fn decode_constraint_record(bytes: &[u8]) -> LlkvResult<ConstraintRecord> {
    bitcode::decode(bytes).map_err(|err| {
        llkv_result::Error::Internal(format!("failed to decode constraint metadata: {err}"))
    })
}

struct ConstraintRowCollector<'a, P, F>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(Vec<ConstraintRecord>),
{
    store: &'a ColumnStore<P>,
    lfid: LogicalFieldId,
    table_id: TableId,
    on_batch: &'a mut F,
    buffer: Vec<RowId>,
    error: Option<llkv_result::Error>,
}

impl<'a, P, F> ConstraintRowCollector<'a, P, F>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(Vec<ConstraintRecord>),
{
    fn flush_buffer(&mut self) -> LlkvResult<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let row_ids = mem::take(&mut self.buffer);
        let batch =
            self.store
                .gather_rows(&[self.lfid], &row_ids, GatherNullPolicy::IncludeNulls)?;

        if batch.num_columns() == 0 {
            return Ok(());
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal(
                    "constraint metadata column stored unexpected type".into(),
                )
            })?;

        let mut records = Vec::with_capacity(row_ids.len());
        for (idx, row_id) in row_ids.into_iter().enumerate() {
            if array.is_null(idx) {
                continue;
            }

            let record = decode_constraint_record(array.value(idx))?;
            let (table_from_id, constraint_id) = decode_constraint_row_id(row_id);
            if table_from_id != self.table_id {
                continue;
            }
            if record.constraint_id != constraint_id {
                return Err(llkv_result::Error::Internal(
                    "constraint metadata id mismatch".into(),
                ));
            }
            records.push(record);
        }

        if !records.is_empty() {
            (self.on_batch)(records);
        }

        Ok(())
    }

    fn finish(&mut self) -> LlkvResult<()> {
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        self.flush_buffer()
    }
}

impl<'a, P, F> PrimitiveVisitor for ConstraintRowCollector<'a, P, F>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(Vec<ConstraintRecord>),
{
    fn u64_chunk(&mut self, values: &UInt64Array) {
        if self.error.is_some() {
            return;
        }

        for idx in 0..values.len() {
            let row_id = values.value(idx);
            let (table_id, _) = decode_constraint_row_id(row_id);
            if table_id != self.table_id {
                continue;
            }
            self.buffer.push(row_id);
            if self.buffer.len() >= CONSTRAINT_SCAN_CHUNK {
                if let Err(err) = self.flush_buffer() {
                    self.error = Some(err);
                    return;
                }
            }
        }
    }
}

impl<'a, P, F> PrimitiveWithRowIdsVisitor for ConstraintRowCollector<'a, P, F>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(Vec<ConstraintRecord>),
{
}

impl<'a, P, F> PrimitiveSortedVisitor for ConstraintRowCollector<'a, P, F>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(Vec<ConstraintRecord>),
{
}

impl<'a, P, F> PrimitiveSortedWithRowIdsVisitor for ConstraintRowCollector<'a, P, F>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(Vec<ConstraintRecord>),
{
}

// ----- Public catalog types -----

/// Metadata about a table.
///
/// Stored in the system catalog (table 0) and serialized using [`bitcode`].
#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct TableMeta {
    /// Unique identifier for this table.
    pub table_id: TableId,
    /// Optional human-readable name for the table.
    pub name: Option<String>,
    /// When the table was created (microseconds since epoch).
    pub created_at_micros: u64,
    /// Bitflags for table properties (e.g., temporary, system).
    pub flags: u32,
    /// Schema version or modification counter.
    pub epoch: u64,
}

/// Metadata about a column.
///
/// Stored in the system catalog (table 0) and serialized using [`bitcode`].
#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct ColMeta {
    /// Unique identifier for this column within its table.
    pub col_id: u32,
    /// Optional human-readable name for the column.
    pub name: Option<String>,
    /// Bitflags for column properties (e.g., nullable, indexed).
    pub flags: u32,
    /// Optional serialized default value for the column.
    pub default: Option<Vec<u8>>,
}

/// Metadata about a schema.
///
/// Stored in the system catalog (table 0) and serialized using [`bitcode`].
#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct SchemaMeta {
    /// Human-readable schema name (case-preserved).
    pub name: String,
    /// When the schema was created (microseconds since epoch).
    pub created_at_micros: u64,
    /// Bitflags for schema properties (reserved for future use).
    pub flags: u32,
}

/// Metadata describing a single multi-column UNIQUE index.
#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct MultiColumnUniqueEntryMeta {
    /// Optional human-readable index name.
    pub index_name: Option<String>,
    /// Field IDs participating in this UNIQUE constraint.
    pub column_ids: Vec<u32>,
}

/// Metadata describing all multi-column UNIQUE indexes for a table.
#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct TableMultiColumnUniqueMeta {
    /// Table identifier these UNIQUE entries belong to.
    pub table_id: TableId,
    /// Definitions of each persisted multi-column UNIQUE.
    pub uniques: Vec<MultiColumnUniqueEntryMeta>,
}

// ----- SysCatalog -----

/// Interface to the system catalog (table 0).
///
/// The system catalog stores metadata about all tables and columns in the database.
/// It uses special reserved columns within table 0 to persist [`TableMeta`] and
/// [`ColMeta`] structures.
///
/// # Lifetime
///
/// `SysCatalog` borrows a reference to the [`ColumnStore`] and does not own it.
/// This allows multiple catalog instances to coexist with the same storage.
pub struct SysCatalog<'a, P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: &'a ColumnStore<P>,
}

impl<'a, P> SysCatalog<'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn write_null_entries(
        &self,
        meta_field: LogicalFieldId,
        row_ids: &[RowId],
    ) -> LlkvResult<()> {
        if row_ids.is_empty() {
            return Ok(());
        }

        let lfid_val: u64 = meta_field.into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("meta", DataType::Binary, true).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_array = Arc::new(UInt64Array::from(row_ids.to_vec()));
        let mut builder = BinaryBuilder::new();
        for _ in row_ids {
            builder.append_null();
        }
        let meta_array = Arc::new(builder.finish());

        let batch = RecordBatch::try_new(schema, vec![row_array, meta_array])?;
        self.store.append(&batch)?;
        Ok(())
    }

    /// Create a new system catalog interface using the provided column store.
    pub fn new(store: &'a ColumnStore<P>) -> Self {
        Self { store }
    }

    /// Insert or update table metadata.
    ///
    /// This persists the table's metadata to the system catalog. If metadata for
    /// this table ID already exists, it is overwritten (last-write-wins).
    pub fn put_table_meta(&self, meta: &TableMeta) {
        let lfid_val: u64 = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_TABLE_META_ID).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("meta", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_id = Arc::new(UInt64Array::from(vec![rid_table(meta.table_id)]));
        let meta_encoded = bitcode::encode(meta);
        let meta_bytes = Arc::new(BinaryArray::from(vec![meta_encoded.as_slice()]));

        let batch = RecordBatch::try_new(schema, vec![row_id, meta_bytes]).unwrap();
        self.store.append(&batch).unwrap();
    }

    /// Retrieve table metadata by table ID.
    ///
    /// Returns `None` if no metadata exists for the given table ID.
    pub fn get_table_meta(&self, table_id: TableId) -> Option<TableMeta> {
        let row_id = rid_table(table_id);
        let catalog_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_TABLE_META_ID);
        let batch = self
            .store
            .gather_rows(&[catalog_field], &[row_id], GatherNullPolicy::IncludeNulls)
            .ok()?;

        if batch.num_rows() == 0 || batch.num_columns() == 0 {
            return None;
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("table meta column must be BinaryArray");

        if array.is_null(0) {
            return None;
        }

        bitcode::decode(array.value(0)).ok()
    }

    /// Upsert a single column’s metadata.
    pub fn put_col_meta(&self, table_id: TableId, meta: &ColMeta) {
        let lfid_val: u64 = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_COL_META_ID).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("meta", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let rid_value = rid_col(table_id, meta.col_id);
        let row_id = Arc::new(UInt64Array::from(vec![rid_value]));
        let meta_encoded = bitcode::encode(meta);
        let meta_bytes = Arc::new(BinaryArray::from(vec![meta_encoded.as_slice()]));

        let batch = RecordBatch::try_new(schema, vec![row_id, meta_bytes]).unwrap();
        self.store.append(&batch).unwrap();
    }

    /// Batch fetch specific column metas by col_id using a shared keyset.
    pub fn get_cols_meta(&self, table_id: TableId, col_ids: &[u32]) -> Vec<Option<ColMeta>> {
        if col_ids.is_empty() {
            return Vec::new();
        }

        let row_ids: Vec<RowId> = col_ids.iter().map(|&cid| rid_col(table_id, cid)).collect();
        let catalog_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_COL_META_ID);

        let batch =
            match self
                .store
                .gather_rows(&[catalog_field], &row_ids, GatherNullPolicy::IncludeNulls)
            {
                Ok(batch) => batch,
                Err(_) => return vec![None; col_ids.len()],
            };

        let meta_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("catalog meta column should be Binary");

        col_ids
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                if meta_col.is_null(idx) {
                    None
                } else {
                    bitcode::decode(meta_col.value(idx)).ok()
                }
            })
            .collect()
    }

    /// Delete metadata rows for the specified column identifiers.
    pub fn delete_col_meta(&self, table_id: TableId, col_ids: &[FieldId]) -> LlkvResult<()> {
        if col_ids.is_empty() {
            return Ok(());
        }

        let meta_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_COL_META_ID);
        let row_ids: Vec<RowId> = col_ids
            .iter()
            .map(|&col_id| rid_col(table_id, col_id))
            .collect();
        self.write_null_entries(meta_field, &row_ids)
    }

    /// Remove the persisted table metadata record, if present.
    pub fn delete_table_meta(&self, table_id: TableId) -> LlkvResult<()> {
        let meta_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_TABLE_META_ID);
        let row_id = rid_table(table_id);
        self.write_null_entries(meta_field, &[row_id])
    }

    /// Delete constraint records for the provided identifiers.
    pub fn delete_constraint_records(
        &self,
        table_id: TableId,
        constraint_ids: &[ConstraintId],
    ) -> LlkvResult<()> {
        if constraint_ids.is_empty() {
            return Ok(());
        }

        let meta_field = constraint_meta_lfid();
        let row_ids: Vec<RowId> = constraint_ids
            .iter()
            .map(|&constraint_id| encode_constraint_row_id(table_id, constraint_id))
            .collect();
        self.write_null_entries(meta_field, &row_ids)
    }

    /// Delete the multi-column UNIQUE metadata record for a table, if any.
    pub fn delete_multi_column_uniques(&self, table_id: TableId) -> LlkvResult<()> {
        let meta_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_MULTI_COLUMN_UNIQUE_META_ID);
        let row_id = rid_table(table_id);
        self.write_null_entries(meta_field, &[row_id])
    }

    /// Persist the complete set of multi-column UNIQUE definitions for a table.
    pub fn put_multi_column_uniques(
        &self,
        table_id: TableId,
        uniques: &[MultiColumnUniqueEntryMeta],
    ) -> LlkvResult<()> {
        let lfid_val: u64 =
            lfid(CATALOG_TABLE_ID, CATALOG_FIELD_MULTI_COLUMN_UNIQUE_META_ID).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("meta", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_id = Arc::new(UInt64Array::from(vec![rid_table(table_id)]));
        let meta = TableMultiColumnUniqueMeta {
            table_id,
            uniques: uniques.to_vec(),
        };
        let encoded = bitcode::encode(&meta);
        let meta_bytes = Arc::new(BinaryArray::from(vec![encoded.as_slice()]));

        let batch = RecordBatch::try_new(schema, vec![row_id, meta_bytes])?;
        self.store.append(&batch)?;
        Ok(())
    }

    /// Retrieve all persisted multi-column UNIQUE definitions for a table.
    pub fn get_multi_column_uniques(
        &self,
        table_id: TableId,
    ) -> LlkvResult<Vec<MultiColumnUniqueEntryMeta>> {
        let lfid = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_MULTI_COLUMN_UNIQUE_META_ID);
        let row_id = rid_table(table_id);
        let batch = match self
            .store
            .gather_rows(&[lfid], &[row_id], GatherNullPolicy::IncludeNulls)
        {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 || batch.num_rows() == 0 {
            return Ok(Vec::new());
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal(
                    "catalog multi-column unique column stored unexpected type".into(),
                )
            })?;

        if array.is_null(0) {
            return Ok(Vec::new());
        }

        let meta: TableMultiColumnUniqueMeta = bitcode::decode(array.value(0)).map_err(|err| {
            llkv_result::Error::Internal(format!(
                "failed to decode multi-column unique metadata: {err}"
            ))
        })?;

        Ok(meta.uniques)
    }

    /// Retrieve all persisted multi-column UNIQUE definitions across tables.
    pub fn all_multi_column_unique_metas(&self) -> LlkvResult<Vec<TableMultiColumnUniqueMeta>> {
        let meta_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_MULTI_COLUMN_UNIQUE_META_ID);
        let row_field = rowid_fid(meta_field);

        struct RowIdCollector {
            row_ids: Vec<RowId>,
        }

        impl PrimitiveVisitor for RowIdCollector {
            fn u64_chunk(&mut self, values: &UInt64Array) {
                for i in 0..values.len() {
                    self.row_ids.push(values.value(i));
                }
            }
        }
        impl PrimitiveWithRowIdsVisitor for RowIdCollector {}
        impl PrimitiveSortedVisitor for RowIdCollector {}
        impl PrimitiveSortedWithRowIdsVisitor for RowIdCollector {}

        let mut collector = RowIdCollector {
            row_ids: Vec::new(),
        };
        match ScanBuilder::new(self.store, row_field)
            .options(ScanOptions::default())
            .run(&mut collector)
        {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => return Ok(Vec::new()),
            Err(err) => return Err(err),
        }

        if collector.row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let batch = match self.store.gather_rows(
            &[meta_field],
            &collector.row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 {
            return Ok(Vec::new());
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal(
                    "catalog multi-column unique column stored unexpected type".into(),
                )
            })?;

        let mut metas = Vec::with_capacity(batch.num_rows());
        for idx in 0..batch.num_rows() {
            if array.is_null(idx) {
                continue;
            }
            let meta: TableMultiColumnUniqueMeta =
                bitcode::decode(array.value(idx)).map_err(|err| {
                    llkv_result::Error::Internal(format!(
                        "failed to decode multi-column unique metadata: {err}"
                    ))
                })?;
            metas.push(meta);
        }

        Ok(metas)
    }

    // TODO: Use batch APIs for better performance.
    /// Persist or update multiple constraint records for a table in a single batch.
    pub fn put_constraint_records(
        &self,
        table_id: TableId,
        records: &[ConstraintRecord],
    ) -> LlkvResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let lfid_val: u64 = constraint_meta_lfid().into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("constraint", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_ids: Vec<RowId> = records
            .iter()
            .map(|record| encode_constraint_row_id(table_id, record.constraint_id))
            .collect();

        let row_ids_array = Arc::new(UInt64Array::from(row_ids));
        let payload_array = Arc::new(BinaryArray::from_iter_values(
            records.iter().map(|record| bitcode::encode(record)),
        ));

        let batch = RecordBatch::try_new(schema, vec![row_ids_array, payload_array])?;
        self.store.append(&batch)?;
        Ok(())
    }

    /// Fetch multiple constraint records for a table in a single batch.
    pub fn get_constraint_records(
        &self,
        table_id: TableId,
        constraint_ids: &[ConstraintId],
    ) -> LlkvResult<Vec<Option<ConstraintRecord>>> {
        if constraint_ids.is_empty() {
            return Ok(Vec::new());
        }

        let lfid = constraint_meta_lfid();
        let row_ids: Vec<RowId> = constraint_ids
            .iter()
            .map(|&constraint_id| encode_constraint_row_id(table_id, constraint_id))
            .collect();

        let batch = match self
            .store
            .gather_rows(&[lfid], &row_ids, GatherNullPolicy::IncludeNulls)
        {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => {
                return Ok(vec![None; constraint_ids.len()]);
            }
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 || batch.num_rows() == 0 {
            return Ok(vec![None; constraint_ids.len()]);
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal(
                    "constraint metadata column stored unexpected type".into(),
                )
            })?;

        let mut results = Vec::with_capacity(constraint_ids.len());
        for (idx, &constraint_id) in constraint_ids.iter().enumerate() {
            if array.is_null(idx) {
                results.push(None);
                continue;
            }
            let record = decode_constraint_record(array.value(idx))?;
            if record.constraint_id != constraint_id {
                return Err(llkv_result::Error::Internal(
                    "constraint metadata id mismatch".into(),
                ));
            }
            results.push(Some(record));
        }

        Ok(results)
    }

    /// Stream constraint records for a table in batches.
    pub fn scan_constraint_records_for_table<F>(
        &self,
        table_id: TableId,
        mut on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(Vec<ConstraintRecord>),
    {
        let row_field = constraint_row_lfid();
        let mut visitor = ConstraintRowCollector {
            store: self.store,
            lfid: constraint_meta_lfid(),
            table_id,
            on_batch: &mut on_batch,
            buffer: Vec::with_capacity(CONSTRAINT_SCAN_CHUNK),
            error: None,
        };

        match ScanBuilder::new(self.store, row_field)
            .options(ScanOptions::default())
            .run(&mut visitor)
        {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => return Ok(()),
            Err(err) => return Err(err),
        }

        visitor.finish()
    }

    /// Load all constraint records for a table into a vector.
    pub fn constraint_records_for_table(
        &self,
        table_id: TableId,
    ) -> LlkvResult<Vec<ConstraintRecord>> {
        let mut all = Vec::new();
        self.scan_constraint_records_for_table(table_id, |mut chunk| {
            all.append(&mut chunk);
        })?;
        Ok(all)
    }

    pub fn put_next_table_id(&self, next_id: TableId) -> LlkvResult<()> {
        let lfid_val: u64 = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_NEXT_TABLE_ID).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("next_table_id", DataType::UInt64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_id = Arc::new(UInt64Array::from(vec![CATALOG_NEXT_TABLE_ROW_ID]));
        let value_array = Arc::new(UInt64Array::from(vec![next_id as u64]));
        let batch = RecordBatch::try_new(schema, vec![row_id, value_array])?;
        self.store.append(&batch)?;
        Ok(())
    }

    pub fn get_next_table_id(&self) -> LlkvResult<Option<TableId>> {
        let lfid = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_NEXT_TABLE_ID);
        let batch = match self.store.gather_rows(
            &[lfid],
            &[CATALOG_NEXT_TABLE_ROW_ID],
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 || batch.num_rows() == 0 {
            return Ok(None);
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                llkv_result::Error::Internal(
                    "catalog next_table_id column stored unexpected type".into(),
                )
            })?;
        if array.is_empty() || array.is_null(0) {
            return Ok(None);
        }

        let value = array.value(0);
        if value > TableId::MAX as u64 {
            return Err(llkv_result::Error::InvalidArgumentError(
                "persisted next_table_id exceeds TableId range".into(),
            ));
        }

        Ok(Some(value as TableId))
    }

    pub fn max_table_id(&self) -> LlkvResult<Option<TableId>> {
        let meta_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_TABLE_META_ID);
        let row_field = rowid_fid(meta_field);

        let mut collector = MaxRowIdCollector { max: None };
        match ScanBuilder::new(self.store, row_field)
            .options(ScanOptions::default())
            .run(&mut collector)
        {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        }

        let max_value = match collector.max {
            Some(value) => value,
            None => return Ok(None),
        };

        let logical: LogicalFieldId = max_value.into();
        Ok(Some(logical.table_id()))
    }

    /// Scan all table metadata entries from the catalog.
    /// Returns a vector of (table_id, TableMeta) pairs for all persisted tables.
    ///
    /// This method first scans for all row IDs in the table metadata column,
    /// then uses gather_rows to retrieve the actual metadata.
    pub fn all_table_metas(&self) -> LlkvResult<Vec<(TableId, TableMeta)>> {
        let meta_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_TABLE_META_ID);
        let row_field = rowid_fid(meta_field);

        // Collect all row IDs that have table metadata
        struct RowIdCollector {
            row_ids: Vec<RowId>,
        }

        impl PrimitiveVisitor for RowIdCollector {
            fn u64_chunk(&mut self, values: &UInt64Array) {
                for i in 0..values.len() {
                    self.row_ids.push(values.value(i));
                }
            }
        }
        impl PrimitiveWithRowIdsVisitor for RowIdCollector {}
        impl PrimitiveSortedVisitor for RowIdCollector {}
        impl PrimitiveSortedWithRowIdsVisitor for RowIdCollector {}

        let mut collector = RowIdCollector {
            row_ids: Vec::new(),
        };
        match ScanBuilder::new(self.store, row_field)
            .options(ScanOptions::default())
            .run(&mut collector)
        {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => return Ok(Vec::new()),
            Err(err) => return Err(err),
        }

        if collector.row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Gather all table metadata using the collected row IDs
        let batch = self.store.gather_rows(
            &[meta_field],
            &collector.row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let meta_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal("catalog table_meta column should be Binary".into())
            })?;

        let mut result = Vec::new();
        for (idx, &row_id) in collector.row_ids.iter().enumerate() {
            if !meta_col.is_null(idx) {
                let bytes = meta_col.value(idx);
                if let Ok(meta) = bitcode::decode::<TableMeta>(bytes) {
                    let logical: LogicalFieldId = row_id.into();
                    let table_id = logical.table_id();
                    result.push((table_id, meta));
                }
            }
        }

        Ok(result)
    }

    /// Persist the next transaction id to the catalog.
    pub fn put_next_txn_id(&self, next_txn_id: u64) -> LlkvResult<()> {
        let lfid_val: u64 = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_NEXT_TXN_ID).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("next_txn_id", DataType::UInt64, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_id = Arc::new(UInt64Array::from(vec![CATALOG_NEXT_TXN_ROW_ID]));
        let value_array = Arc::new(UInt64Array::from(vec![next_txn_id]));
        let batch = RecordBatch::try_new(schema, vec![row_id, value_array])?;
        self.store.append(&batch)?;
        Ok(())
    }

    /// Load the next transaction id from the catalog.
    pub fn get_next_txn_id(&self) -> LlkvResult<Option<u64>> {
        let lfid = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_NEXT_TXN_ID);
        let batch = match self.store.gather_rows(
            &[lfid],
            &[CATALOG_NEXT_TXN_ROW_ID],
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 || batch.num_rows() == 0 {
            return Ok(None);
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                llkv_result::Error::Internal(
                    "catalog next_txn_id column stored unexpected type".into(),
                )
            })?;
        if array.is_empty() || array.is_null(0) {
            return Ok(None);
        }

        let value = array.value(0);
        Ok(Some(value))
    }

    /// Persist the last committed transaction id to the catalog.
    pub fn put_last_committed_txn_id(&self, last_committed: u64) -> LlkvResult<()> {
        let lfid_val: u64 = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_LAST_COMMITTED_TXN_ID).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("last_committed_txn_id", DataType::UInt64, false).with_metadata(
                HashMap::from([(
                    crate::constants::FIELD_ID_META_KEY.to_string(),
                    lfid_val.to_string(),
                )]),
            ),
        ]));

        let row_id = Arc::new(UInt64Array::from(vec![CATALOG_LAST_COMMITTED_TXN_ROW_ID]));
        let value_array = Arc::new(UInt64Array::from(vec![last_committed]));
        let batch = RecordBatch::try_new(schema, vec![row_id, value_array])?;
        self.store.append(&batch)?;
        Ok(())
    }

    /// Load the last committed transaction id from the catalog.
    pub fn get_last_committed_txn_id(&self) -> LlkvResult<Option<u64>> {
        let lfid = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_LAST_COMMITTED_TXN_ID);
        let batch = match self.store.gather_rows(
            &[lfid],
            &[CATALOG_LAST_COMMITTED_TXN_ROW_ID],
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 || batch.num_rows() == 0 {
            return Ok(None);
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                llkv_result::Error::Internal(
                    "catalog last_committed_txn_id column stored unexpected type".into(),
                )
            })?;
        if array.is_empty() || array.is_null(0) {
            return Ok(None);
        }

        let value = array.value(0);
        Ok(Some(value))
    }

    /// Persist the catalog state to the system catalog.
    ///
    /// Stores the complete catalog state (all tables and fields) as a binary blob
    /// using bitcode serialization.
    pub fn put_catalog_state(&self, state: &crate::catalog::TableCatalogState) -> LlkvResult<()> {
        let lfid_val: u64 = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_CATALOG_STATE).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("catalog_state", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_id = Arc::new(UInt64Array::from(vec![CATALOG_STATE_ROW_ID]));
        let encoded = bitcode::encode(state);
        let state_bytes = Arc::new(BinaryArray::from(vec![encoded.as_slice()]));

        let batch = RecordBatch::try_new(schema, vec![row_id, state_bytes])?;
        self.store.append(&batch)?;
        Ok(())
    }

    /// Load the catalog state from the system catalog.
    ///
    /// Retrieves the complete catalog state including all table and field mappings.
    pub fn get_catalog_state(&self) -> LlkvResult<Option<crate::catalog::TableCatalogState>> {
        let lfid = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_CATALOG_STATE);
        let batch = match self.store.gather_rows(
            &[lfid],
            &[CATALOG_STATE_ROW_ID],
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 || batch.num_rows() == 0 {
            return Ok(None);
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal("catalog state column stored unexpected type".into())
            })?;
        if array.is_empty() || array.is_null(0) {
            return Ok(None);
        }

        let bytes = array.value(0);
        let state = bitcode::decode(bytes).map_err(|e| {
            llkv_result::Error::Internal(format!("Failed to decode catalog state: {}", e))
        })?;
        Ok(Some(state))
    }

    /// Persist schema metadata to the catalog.
    ///
    /// Stores schema metadata at a row ID derived from the schema name hash.
    /// This allows efficient lookup and prevents collisions.
    pub fn put_schema_meta(&self, meta: &SchemaMeta) -> LlkvResult<()> {
        let lfid_val: u64 = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_SCHEMA_META_ID).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new("meta", DataType::Binary, false).with_metadata(HashMap::from([(
                crate::constants::FIELD_ID_META_KEY.to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        // Use hash of canonical (lowercase) schema name as row ID
        let canonical = meta.name.to_ascii_lowercase();
        let row_id_val = schema_name_to_row_id(&canonical);
        let row_id = Arc::new(UInt64Array::from(vec![row_id_val]));
        let meta_encoded = bitcode::encode(meta);
        let meta_bytes = Arc::new(BinaryArray::from(vec![meta_encoded.as_slice()]));

        let batch = RecordBatch::try_new(schema, vec![row_id, meta_bytes])?;
        self.store.append(&batch)?;
        Ok(())
    }

    /// Retrieve schema metadata by name.
    ///
    /// Returns `None` if the schema does not exist.
    pub fn get_schema_meta(&self, schema_name: &str) -> LlkvResult<Option<SchemaMeta>> {
        let canonical = schema_name.to_ascii_lowercase();
        let row_id = schema_name_to_row_id(&canonical);
        let lfid = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_SCHEMA_META_ID);

        let batch = match self
            .store
            .gather_rows(&[lfid], &[row_id], GatherNullPolicy::IncludeNulls)
        {
            Ok(batch) => batch,
            Err(llkv_result::Error::NotFound) => return Ok(None),
            Err(err) => return Err(err),
        };

        if batch.num_columns() == 0 || batch.num_rows() == 0 {
            return Ok(None);
        }

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal("catalog schema_meta column should be Binary".into())
            })?;

        if array.is_empty() || array.is_null(0) {
            return Ok(None);
        }

        let bytes = array.value(0);
        let meta = bitcode::decode(bytes).map_err(|e| {
            llkv_result::Error::Internal(format!("Failed to decode schema metadata: {}", e))
        })?;
        Ok(Some(meta))
    }

    /// Scan all schema metadata entries from the catalog.
    ///
    /// Returns a vector of all persisted schemas.
    pub fn all_schema_metas(&self) -> LlkvResult<Vec<SchemaMeta>> {
        let meta_field = lfid(CATALOG_TABLE_ID, CATALOG_FIELD_SCHEMA_META_ID);
        let row_field = rowid_fid(meta_field);

        // Collect all row IDs that have schema metadata
        struct RowIdCollector {
            row_ids: Vec<RowId>,
        }

        impl PrimitiveVisitor for RowIdCollector {
            fn u64_chunk(&mut self, values: &UInt64Array) {
                for i in 0..values.len() {
                    self.row_ids.push(values.value(i));
                }
            }
        }
        impl PrimitiveWithRowIdsVisitor for RowIdCollector {}
        impl PrimitiveSortedVisitor for RowIdCollector {}
        impl PrimitiveSortedWithRowIdsVisitor for RowIdCollector {}

        let mut collector = RowIdCollector {
            row_ids: Vec::new(),
        };
        match ScanBuilder::new(self.store, row_field)
            .options(ScanOptions::default())
            .run(&mut collector)
        {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => return Ok(Vec::new()),
            Err(err) => return Err(err),
        }

        if collector.row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Gather all schema metadata using the collected row IDs
        let batch = self.store.gather_rows(
            &[meta_field],
            &collector.row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let meta_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                llkv_result::Error::Internal("catalog schema_meta column should be Binary".into())
            })?;

        let mut result = Vec::new();
        for idx in 0..collector.row_ids.len() {
            if !meta_col.is_null(idx) {
                let bytes = meta_col.value(idx);
                if let Ok(meta) = bitcode::decode::<SchemaMeta>(bytes) {
                    result.push(meta);
                }
            }
        }

        Ok(result)
    }
}

/// Generate a row ID for schema metadata based on schema name.
///
/// Uses a simple hash to map schema names to row IDs. This is deterministic
/// and allows direct lookup without scanning.
fn schema_name_to_row_id(canonical_name: &str) -> RowId {
    // Use a simple 64-bit FNV-1a hash for deterministic IDs across platforms and releases
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x1000_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in canonical_name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Use high bit to avoid collision with reserved catalog row IDs (0-3) and table metadata rows
    hash | (1u64 << 63)
}

struct MaxRowIdCollector {
    max: Option<RowId>,
}

impl PrimitiveVisitor for MaxRowIdCollector {
    fn u64_chunk(&mut self, values: &UInt64Array) {
        for i in 0..values.len() {
            let value = values.value(i);
            self.max = match self.max {
                Some(curr) if curr >= value => Some(curr),
                _ => Some(value),
            };
        }
    }
}

impl PrimitiveWithRowIdsVisitor for MaxRowIdCollector {}
impl PrimitiveSortedVisitor for MaxRowIdCollector {}
impl PrimitiveSortedWithRowIdsVisitor for MaxRowIdCollector {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{
        ConstraintKind, ConstraintState, PrimaryKeyConstraint, UniqueConstraint,
    };
    use llkv_column_map::ColumnStore;
    use std::sync::Arc;

    #[test]
    fn constraint_records_roundtrip() {
        let pager = Arc::new(MemPager::default());
        let store = ColumnStore::open(Arc::clone(&pager)).unwrap();
        let catalog = SysCatalog::new(&store);

        let table_id: TableId = 42;
        let record1 = ConstraintRecord {
            constraint_id: 1,
            kind: ConstraintKind::PrimaryKey(PrimaryKeyConstraint {
                field_ids: vec![1, 2],
            }),
            state: ConstraintState::Active,
            revision: 1,
            last_modified_micros: 100,
        };
        let record2 = ConstraintRecord {
            constraint_id: 2,
            kind: ConstraintKind::Unique(UniqueConstraint { field_ids: vec![3] }),
            state: ConstraintState::Active,
            revision: 2,
            last_modified_micros: 200,
        };
        catalog
            .put_constraint_records(table_id, &[record1.clone(), record2.clone()])
            .unwrap();

        let other_table_record = ConstraintRecord {
            constraint_id: 1,
            kind: ConstraintKind::Unique(UniqueConstraint { field_ids: vec![5] }),
            state: ConstraintState::Active,
            revision: 1,
            last_modified_micros: 150,
        };
        catalog
            .put_constraint_records(7, &[other_table_record])
            .unwrap();

        let mut fetched = catalog.constraint_records_for_table(table_id).unwrap();
        fetched.sort_by_key(|record| record.constraint_id);

        assert_eq!(fetched.len(), 2);
        assert_eq!(fetched[0], record1);
        assert_eq!(fetched[1], record2);

        let single = catalog
            .get_constraint_records(table_id, &[record1.constraint_id])
            .unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].as_ref(), Some(&record1));

        let missing = catalog.get_constraint_records(table_id, &[999]).unwrap();
        assert_eq!(missing.len(), 1);
        assert!(missing[0].is_none());
    }
}
