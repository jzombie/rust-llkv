//! System catalog for storing table and column metadata.
//!
//! The system catalog uses table 0 (reserved) to store metadata about all tables
//! and columns in the database. This metadata includes:
//!
//! - **Table metadata** ([`TableMeta`]): Table ID, name, creation time, flags
//! - **Column metadata** ([`ColMeta`]): Column ID, name, flags, default values
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
use std::sync::Arc;

use arrow::array::{Array, BinaryArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bitcode::{Decode, Encode};

use crate::types::TableId;
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
        struct MetaVisitor {
            target_rid: u64,
            meta: Option<TableMeta>,
        }
        impl PrimitiveVisitor for MetaVisitor {}
        impl PrimitiveWithRowIdsVisitor for MetaVisitor {
            fn u64_chunk_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array) {
                for i in 0..r.len() {
                    if r.value(i) == self.target_rid {
                        // This logic assumes the 'meta' column is u64. It needs to be Binary.
                        // This scan implementation is a placeholder and needs to be updated
                        // to correctly handle BinaryArray for the metadata.
                        // For now, this lets the code compile.
                        let _bytes = v.value(i);
                        // self.meta = Some(bitcode::decode(&bytes.to_be_bytes()).unwrap());
                        break;
                    }
                }
            }
        }
        impl PrimitiveSortedVisitor for MetaVisitor {}
        impl PrimitiveSortedWithRowIdsVisitor for MetaVisitor {}

        let mut visitor = MetaVisitor {
            target_rid: rid_table(table_id),
            meta: None,
        };
        // Note: The scan needs `with_row_ids` to be true for `u64_chunk_with_rids` to be called.
        // A full implementation would require passing ScanOptions.
        let scan_opts = llkv_column_map::store::scan::ScanOptions {
            with_row_ids: true,
            ..Default::default()
        };

        let _ = self.store.scan(
            lfid(CATALOG_TABLE_ID, CATALOG_FIELD_TABLE_META_ID),
            scan_opts,
            &mut visitor,
        );
        visitor.meta
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

        let row_ids: Vec<u64> = col_ids.iter().map(|&cid| rid_col(table_id, cid)).collect();
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
            row_ids: Vec<u64>,
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

        let batch = match self.store.gather_rows(
            &[lfid],
            &[row_id],
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
            row_ids: Vec<u64>,
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
fn schema_name_to_row_id(canonical_name: &str) -> u64 {
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
    max: Option<u64>,
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
