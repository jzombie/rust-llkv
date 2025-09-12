//! System catalog stored inside ColumnStore (table id 0).
//!
//! - LogicalFieldId = (table_id: u32 << 32) | (col_id: u32)
//! - Catalog rows:
//!     * Table meta   @ row_id = (table_id << 32) | 0
//!     * Column meta  @ row_id = (table_id << 32) | col_id
//! - Presence column (col_id = 0) marks existence for fast key scans.
//! - Encodes metadata with `bitcode` (no serde/bincode).
//!
//! API highlights:
//! - put_table_meta / get_table_meta / list_tables
//! - put_col_meta / get_cols_meta / list_cols_meta
//! - schema(table_id) -> Arrow-ish Schema
//!
//! Notes:
//! - All reads use `get_many_projected` for shared keysets when possible.
//! - Listing columns uses a key-range scan over the catalog’s column-meta field.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Bound;
use std::sync::Arc;

use bitcode::{Decode, Encode};

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ScanError, ValueScanOpts};
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};
use llkv_column_map::views::ValueSlice;

// ----- Namespacing helpers -----

#[inline]
pub fn lfid(table_id: u32, col_id: u32) -> LogicalFieldId {
    ((table_id as u64) << 32) | (col_id as u64)
}

#[inline]
fn rid_table(table_id: u32) -> u64 {
    // Choose (tid << 32) | 0 so table rows and column rows share sort locality.
    (table_id as u64) << 32
}

#[inline]
fn rid_col(table_id: u32, col_id: u32) -> u64 {
    ((table_id as u64) << 32) | (col_id as u64)
}

/// Encode row id as big-endian bytes so lexicographic order == numeric.
#[inline]
fn row_key_bytes(row_id: u64) -> Vec<u8> {
    row_id.to_be_bytes().to_vec()
}

/// Extract owned bytes from a ValueSlice<Arc<[u8]>>.
#[inline]
fn slice_to_owned(v: &ValueSlice<Arc<[u8]>>) -> Vec<u8> {
    let a = v.start() as usize;
    let b = v.end() as usize;
    v.data().as_ref()[a..b].to_vec()
}

/// Compute the tight upper bound for a prefix range ([prefix, upper)) in byte space.
#[inline]
#[allow(dead_code)] // TODO: Clean up or wire up
fn next_prefix_upper(prefix: &[u8]) -> Option<Vec<u8>> {
    if prefix.is_empty() {
        return None;
    }
    let mut up = prefix.to_vec();
    for i in (0..up.len()).rev() {
        if up[i] != 0xFF {
            up[i] = up[i].saturating_add(1);
            up.truncate(i + 1);
            return Some(up);
        }
    }
    None
}

// ----- Catalog constants -----

/// Reserved catalog table id.
pub const CATALOG_TID: u32 = 0;

/// Presence column id used across *all* tables (including catalog).
pub const PRESENCE_COL_ID: u32 = 0;

/// Catalog column ids (within table id 0).
const F_TABLE_META: u32 = 1; // bytes: bitcode(TableMeta)
const F_TABLE_NAME: u32 = 2; // bytes: UTF-8 (optional index)
const F_COL_META: u32 = 10; // bytes: bitcode(ColMeta)
const F_COL_NAME: u32 = 11; // bytes: UTF-8 (optional index)

/// ----- Public catalog types -----

#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct TableMeta {
    pub table_id: u32,
    pub name: Option<String>,
    pub created_at_micros: u64,
    pub flags: u32,
    pub epoch: u64,
}

// TODO: Replace with `llkv-types`
#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub enum ColType {
    Bytes,
    Utf8,
    BeU64,
    BeI64,
    F64,
    Json,
}

#[derive(Encode, Decode, Clone, Debug, PartialEq, Eq)]
pub struct ColMeta {
    pub col_id: u32,
    pub name: Option<String>,
    pub ty: ColType,
    pub flags: u32,
    pub default: Option<Vec<u8>>,
}

/// ----- Arrow-ish schema surfaces -----

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataType {
    Binary,
    Utf8,
    UInt64,
    Int64,
    Float64,
    Json,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool, // column-store is sparse => default true
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    pub table_id: u32,
    pub table_name: Option<String>,
    pub fields: Vec<Field>,
    pub metadata: HashMap<String, String>,
}

#[inline]
fn map_coltype_to_dtype(t: &ColType) -> DataType {
    match t {
        ColType::Bytes => DataType::Binary,
        ColType::Utf8 => DataType::Utf8,
        ColType::BeU64 => DataType::UInt64,
        ColType::BeI64 => DataType::Int64,
        ColType::F64 => DataType::Float64,
        ColType::Json => DataType::Json,
    }
}

// ----- SysCatalog -----

// TODO: Use generic pager, not MemPager!
pub struct SysCatalog<'a> {
    store: &'a ColumnStore<llkv_column_map::storage::pager::MemPager>,
}

impl<'a> SysCatalog<'a> {
    pub fn new(store: &'a ColumnStore<llkv_column_map::storage::pager::MemPager>) -> Self {
        Self { store }
    }

    #[inline]
    fn append_opts(&self) -> AppendOptions {
        AppendOptions {
            mode: ValueMode::Auto,
            // These are modest defaults; you can tune if you want:
            segment_max_entries: 128_000,
            segment_max_bytes: 16 << 20,
            last_write_wins_in_batch: true,
            value_order: None,
        }
    }

    // ----- Table meta -----

    /// Upsert table metadata. Also writes presence + optional name index.
    pub fn put_table_meta(&self, meta: &TableMeta) {
        let k = row_key_bytes(rid_table(meta.table_id));
        let v = bitcode::encode(meta);

        let mut puts = vec![
            Put {
                field_id: lfid(CATALOG_TID, F_TABLE_META),
                items: vec![(Cow::Owned(k.clone()), Cow::Owned(v))],
            },
            Put {
                field_id: lfid(CATALOG_TID, PRESENCE_COL_ID),
                items: vec![(Cow::Owned(k.clone()), Cow::Owned(Vec::new()))],
            },
        ];

        if let Some(name) = &meta.name {
            puts.push(Put {
                field_id: lfid(CATALOG_TID, F_TABLE_NAME),
                items: vec![(Cow::Owned(k), Cow::Owned(name.as_bytes().to_vec()))],
            });
        }

        self.store.append_many(puts, self.append_opts());
    }

    /// Fetch table metadata by table_id.
    pub fn get_table_meta(&self, table_id: u32) -> Option<TableMeta> {
        let key = row_key_bytes(rid_table(table_id));
        let fids = [lfid(CATALOG_TID, F_TABLE_META)];
        let keys = [&key[..]];
        let res = self.store.get_many_projected(&fids, &keys);
        let cell = &res[0][0];
        cell.as_ref().map(|s| {
            let bytes = slice_to_owned(s);
            bitcode::decode::<TableMeta>(&bytes).expect("decode TableMeta")
        })
    }

    /// List all tables (by scanning table-meta keys).
    pub fn list_tables(&self) -> Vec<TableMeta> {
        let mut out = Vec::new();
        let it = match self.store.scan_values_lww(
            lfid(CATALOG_TID, F_TABLE_META),
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 16,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(ScanError::NoActiveSegments) => return out,
            Err(e) => panic!("catalog scan init (tables) error: {:?}", e),
        };

        for item in it {
            let meta = bitcode::decode::<TableMeta>(&slice_to_owned(&item.value))
                .expect("decode TableMeta");
            out.push(meta);
        }
        out
    }

    // ----- Column meta -----

    /// Upsert a single column’s metadata (and presence + optional name).
    pub fn put_col_meta(&self, table_id: u32, meta: &ColMeta) {
        let rid = rid_col(table_id, meta.col_id);
        let k = row_key_bytes(rid);
        let v_meta = bitcode::encode(meta);

        let mut puts = vec![
            Put {
                field_id: lfid(CATALOG_TID, F_COL_META),
                items: vec![(Cow::Owned(k.clone()), Cow::Owned(v_meta))],
            },
            Put {
                field_id: lfid(CATALOG_TID, PRESENCE_COL_ID),
                items: vec![(Cow::Owned(k.clone()), Cow::Owned(Vec::new()))],
            },
        ];
        if let Some(name) = &meta.name {
            puts.push(Put {
                field_id: lfid(CATALOG_TID, F_COL_NAME),
                items: vec![(Cow::Owned(k), Cow::Owned(name.as_bytes().to_vec()))],
            });
        }
        self.store.append_many(puts, self.append_opts());
    }

    /// Batch fetch specific column metas by col_id using a shared keyset.
    pub fn get_cols_meta(&self, table_id: u32, col_ids: &[u32]) -> Vec<Option<ColMeta>> {
        if col_ids.is_empty() {
            return Vec::new();
        }
        let keys_owned: Vec<Vec<u8>> = col_ids
            .iter()
            .map(|&cid| row_key_bytes(rid_col(table_id, cid)))
            .collect();
        let keys: Vec<&[u8]> = keys_owned.iter().map(|k| k.as_slice()).collect();
        let fids = [lfid(CATALOG_TID, F_COL_META)];
        let res = self.store.get_many_projected(&fids, &keys);
        res[0]
            .iter()
            .map(|cell| {
                cell.as_ref().map(|s| {
                    let b = slice_to_owned(s);
                    bitcode::decode::<ColMeta>(&b).expect("decode ColMeta")
                })
            })
            .collect()
    }

    /// Scan all column metas for a table id; `on_col` is called in key order.
    pub fn list_cols_meta<F>(&self, table_id: u32, mut on_col: F)
    where
        F: FnMut(u32, ColMeta),
    {
        // Keys for this table’s columns occupy the range:
        // [ (tid<<32)|0 , ( (tid+1)<<32 ) | 0 )
        let lo_k = row_key_bytes(rid_col(table_id, 0));
        let hi_k = row_key_bytes(rid_col(table_id.saturating_add(1), 0));
        let it = match self.store.scan_values_lww(
            lfid(CATALOG_TID, F_COL_META),
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Forward,
                lo: Bound::Included(&lo_k[..]),
                hi: Bound::Excluded(&hi_k[..]),
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 16,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(ScanError::NoActiveSegments) => return,
            Err(e) => panic!("catalog scan init (cols) error: {:?}", e),
        };

        for item in it {
            // Decode back the column id from the rid (low 32 bits).
            let mut a = [0u8; 8];
            a.copy_from_slice(&item.key[..8]);
            let rid = u64::from_be_bytes(a);
            let col_id = (rid & 0xFFFF_FFFF) as u32;

            let meta =
                bitcode::decode::<ColMeta>(&slice_to_owned(&item.value)).expect("decode ColMeta");
            on_col(col_id, meta);
        }
    }

    // ----- Schema -----

    /// Build an Arrow-ish schema for a table by reading its table meta and all column metas.
    ///
    /// - `nullable` is set to true (sparse columns).
    /// - Field metadata contains `"col_id"` and `"flags"` as strings.
    pub fn schema(&self, table_id: u32) -> Option<Schema> {
        let table = self.get_table_meta(table_id)?;
        let mut fields = Vec::new();

        self.list_cols_meta(table_id, |col_id, cm| {
            let name = cm.name.clone().unwrap_or_else(|| format!("col_{}", col_id));
            let mut md = HashMap::new();
            md.insert("col_id".to_string(), col_id.to_string());
            if cm.flags != 0 {
                md.insert("flags".to_string(), format!("0x{:08x}", cm.flags));
            }
            if cm.default.is_some() {
                md.insert("has_default".to_string(), "true".to_string());
            }

            fields.push(Field {
                name,
                data_type: map_coltype_to_dtype(&cm.ty),
                nullable: true,
                metadata: md,
            });
        });

        let mut md = HashMap::new();
        if table.flags != 0 {
            md.insert("flags".to_string(), format!("0x{:08x}", table.flags));
        }
        md.insert("epoch".to_string(), table.epoch.to_string());
        md.insert(
            "created_at_micros".to_string(),
            table.created_at_micros.to_string(),
        );

        Some(Schema {
            table_id: table.table_id,
            table_name: table.name.clone(),
            fields,
            metadata: md,
        })
    }
}
