//! System catalog stored inside ColumnStore (table id 0).

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

#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct TableMeta {
    pub table_id: TableId,
    pub name: Option<String>,
    pub created_at_micros: u64,
    pub flags: u32,
    pub epoch: u64,
}

#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct ColMeta {
    pub col_id: u32,
    pub name: Option<String>,
    pub flags: u32,
    pub default: Option<Vec<u8>>,
}

// ----- SysCatalog -----

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
    pub fn new(store: &'a ColumnStore<P>) -> Self {
        Self { store }
    }

    /// Upsert table metadata.
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

    /// Fetch table metadata by table_id.
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
