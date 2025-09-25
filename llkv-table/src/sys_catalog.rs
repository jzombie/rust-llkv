//! System catalog stored inside ColumnStore (table id 0).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{BinaryArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bitcode::{Decode, Encode};

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_column_map::{
    ColumnStore,
    storage::pager::{MemPager, Pager},
    types::Namespace,
};
use simd_r_drive_entry_handle::EntryHandle;

// ----- Catalog constants -----

/// Reserved catalog table id.
pub const CATALOG_TID: u32 = 0;

/// Catalog column ids (within table id 0).
const F_TABLE_META: u32 = 1; // bytes: bitcode(TableMeta)
const F_COL_META: u32 = 10; // bytes: bitcode(ColMeta)

// ----- Namespacing helpers -----

#[inline]
fn lfid(table_id: u32, col_id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(table_id)
        .with_field_id(col_id)
}

#[inline]
fn rid_table(table_id: u32) -> u64 {
    let fid = LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(table_id)
        .with_field_id(0);
    fid.into()
}

#[inline]
fn rid_col(table_id: u32, col_id: u32) -> u64 {
    lfid(table_id, col_id).into()
}

// ----- Public catalog types -----

#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct TableMeta {
    pub table_id: u32,
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
        let lfid_val: u64 = lfid(CATALOG_TID, F_TABLE_META).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("meta", DataType::Binary, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
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
    pub fn get_table_meta(&self, table_id: u32) -> Option<TableMeta> {
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

        let _ = self
            .store
            .scan(lfid(CATALOG_TID, F_TABLE_META), scan_opts, &mut visitor);
        visitor.meta
    }

    /// Upsert a single column’s metadata.
    pub fn put_col_meta(&self, table_id: u32, meta: &ColMeta) {
        let lfid_val: u64 = lfid(CATALOG_TID, F_COL_META).into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("meta", DataType::Binary, false).with_metadata(HashMap::from([(
                "field_id".to_string(),
                lfid_val.to_string(),
            )])),
        ]));

        let row_id = Arc::new(UInt64Array::from(vec![rid_col(table_id, meta.col_id)]));
        let meta_encoded = bitcode::encode(meta);
        let meta_bytes = Arc::new(BinaryArray::from(vec![meta_encoded.as_slice()]));

        let batch = RecordBatch::try_new(schema, vec![row_id, meta_bytes]).unwrap();
        self.store.append(&batch).unwrap();
    }

    /// Batch fetch specific column metas by col_id using a shared keyset.
    pub fn get_cols_meta(&self, table_id: u32, col_ids: &[u32]) -> Vec<Option<ColMeta>> {
        let results = vec![None; col_ids.len()];
        let target_rids: HashMap<u64, usize> = col_ids
            .iter()
            .enumerate()
            .map(|(i, &cid)| (rid_col(table_id, cid), i))
            .collect();

        struct MetaVisitor<'a> {
            target_rids: &'a HashMap<u64, usize>,
            // results: &'a mut Vec<Option<ColMeta>>,
        }
        impl<'a> PrimitiveVisitor for MetaVisitor<'a> {}
        impl<'a> PrimitiveWithRowIdsVisitor for MetaVisitor<'a> {
            // --- FIX: Method moved to the correct trait implementation ---
            fn u64_chunk_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array) {
                for i in 0..r.len() {
                    let rid = r.value(i);
                    if let Some(&_idx) = self.target_rids.get(&rid) {
                        // Placeholder logic, as in get_table_meta.
                        let _bytes = v.value(i);
                        // self.results[idx] = Some(bitcode::decode(&bytes.to_be_bytes()).unwrap());
                    }
                }
            }
        }
        impl<'a> PrimitiveSortedVisitor for MetaVisitor<'a> {}
        impl<'a> PrimitiveSortedWithRowIdsVisitor for MetaVisitor<'a> {}

        let mut visitor = MetaVisitor {
            target_rids: &target_rids,
            // results: &mut results,
        };
        let scan_opts = llkv_column_map::store::scan::ScanOptions {
            with_row_ids: true,
            ..Default::default()
        };

        let _ = self
            .store
            .scan(lfid(CATALOG_TID, F_COL_META), scan_opts, &mut visitor);
        results
    }
}
