//! ColumnMapTable: a column-store-backed table with row-based ingest.
//!
//! Goals:
//! - No B+Tree, no rayon.
//! - Row-based inserts -> column_map append_many.
//! - Scans delegate to column_map read_scan utilities.
//!
//! NOTE: This lives alongside your current Table so nothing breaks until
//! you decide to switch. No indexes here yet by design.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Bound;

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};
use llkv_column_map::views::ValueSlice;

use crate::types::{FieldId, RowId, RowPatch};

/// Encode row id as big-endian bytes so lexicographic order == numeric.
#[inline]
fn row_key_bytes(row_id: RowId) -> Vec<u8> {
    row_id.to_be_bytes().to_vec()
}

/// Parse big-endian row id from the first 8 bytes of a key.
#[inline]
fn parse_row_id(key: &[u8]) -> RowId {
    let mut a = [0u8; 8];
    a.copy_from_slice(&key[..8]);
    u64::from_be_bytes(a)
}

/// Extract owned bytes from a ValueSlice<Arc<[u8]>>.
#[inline]
fn value_slice_bytes(v: &ValueSlice<std::sync::Arc<[u8]>>) -> Vec<u8> {
    let a = v.start() as usize;
    let b = v.end() as usize;
    v.data().as_ref()[a..b].to_vec()
}

/// Thin configuration for ingest segmentation and LWW policy.
#[derive(Clone, Debug)]
pub struct CmTableConfig {
    pub segment_max_entries: usize,
    pub segment_max_bytes: usize,
    pub last_write_wins_in_batch: bool,
}

impl Default for CmTableConfig {
    fn default() -> Self {
        Self {
            segment_max_entries: 100_000,
            segment_max_bytes: 32 << 20,
            last_write_wins_in_batch: true,
        }
    }
}

/// ColumnMap-backed table.
pub struct ColumnMapTable {
    /// Leaked pager to satisfy ColumnStore's lifetime without gymnastics.
    pager: &'static MemPager,
    store: ColumnStore<'static, MemPager>,
    cfg: CmTableConfig,
}

impl ColumnMapTable {
    /// Create a new in-memory table.
    pub fn new(cfg: CmTableConfig) -> Self {
        let pager = Box::leak(Box::new(MemPager::default()));
        let store = ColumnStore::init_empty(pager);
        Self { pager, store, cfg }
    }

    /// Optional: declare a column; ColumnStore creates on first append.
    pub fn add_column(&mut self, _fid: FieldId) {}

    /// Row-based ingest. Each row contains (field_id -> value).
    /// We map to per-column `Put` batches and call append_many.
    pub fn insert_many(&self, rows: &[RowPatch]) {
        if rows.is_empty() {
            return;
        }

        // FieldId -> Vec<(key, val)>
        let mut by_col: HashMap<LogicalFieldId, Vec<(Cow<[u8]>, Cow<[u8]>)>> = HashMap::new();

        for (row_id, cols) in rows.iter() {
            let key = row_key_bytes(*row_id);
            for (fid, (_index_key, col_in)) in cols {
                // Ignoring _index_key for now (no secondary indexes).
                by_col
                    .entry(*fid as LogicalFieldId)
                    .or_default()
                    .push((Cow::Owned(key.clone()), col_in.clone()));
            }
        }

        let puts: Vec<Put> = by_col
            .into_iter()
            .map(|(field_id, items)| Put { field_id, items })
            .collect();

        let opts = AppendOptions {
            mode: ValueMode::Auto,
            segment_max_entries: self.cfg.segment_max_entries,
            segment_max_bytes: self.cfg.segment_max_bytes,
            last_write_wins_in_batch: self.cfg.last_write_wins_in_batch,
        };

        self.store.append_many(puts, opts);
    }

    /// Scan rows by a row-id range, projecting a set of columns.
    ///
    /// Semantics: rows appear if they have a value in the driver column.
    /// Choose `driver_fid` with wide coverage (e.g., PK or common column).
    ///
    /// For each row, projected columns are fetched via `get_many`.
    pub fn scan_by_row_id_range<F>(
        &self,
        driver_fid: FieldId,
        lo: Bound<RowId>,
        hi: Bound<RowId>,
        project: &[FieldId],
        page: usize,
        mut on_row: F,
    ) where
        F: FnMut(RowId, Vec<Option<Vec<u8>>>),
    {
        if page == 0 {
            return;
        }

        let mut after: Option<Vec<u8>> = match lo {
            Bound::Unbounded => None,
            Bound::Included(x) => Some(row_key_bytes(x)),
            Bound::Excluded(x) => Some(row_key_bytes(x)),
        };

        loop {
            // Backing storage for &[u8] bounds.
            let mut lo_buf: Option<Vec<u8>> = None;
            let mut hi_buf: Option<Vec<u8>> = None;

            let lo_bound: Bound<&[u8]> = match (&after, lo) {
                (None, Bound::Unbounded) => Bound::Unbounded,
                (None, Bound::Included(x)) => {
                    lo_buf = Some(row_key_bytes(x));
                    Bound::Included(lo_buf.as_ref().unwrap().as_slice())
                }
                (None, Bound::Excluded(x)) => {
                    lo_buf = Some(row_key_bytes(x));
                    Bound::Excluded(lo_buf.as_ref().unwrap().as_slice())
                }
                (Some(k), _) => {
                    lo_buf = Some(k.clone());
                    Bound::Excluded(lo_buf.as_ref().unwrap().as_slice())
                }
            };

            let hi_bound: Bound<&[u8]> = match hi {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(x) => {
                    hi_buf = Some(row_key_bytes(x));
                    Bound::Included(hi_buf.as_ref().unwrap().as_slice())
                }
                Bound::Excluded(x) => {
                    hi_buf = Some(row_key_bytes(x));
                    Bound::Excluded(hi_buf.as_ref().unwrap().as_slice())
                }
            };

            let it = match self.store.scan_values_lww(
                driver_fid as LogicalFieldId,
                ValueScanOpts {
                    order_by: OrderBy::Key,
                    dir: Direction::Forward,
                    lo: lo_bound,
                    hi: hi_bound,
                    prefix: None,
                    bucket_prefix_len: 2,
                    // 16 minimizes head-tie fallbacks inside column-map
                    head_tag_len: 16,
                    frame_predicate: None,
                },
            ) {
                Ok(it) => it,
                Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => {
                    break;
                }
                Err(e) => panic!("scan init error: {:?}", e),
            };

            let mut keys: Vec<Vec<u8>> = Vec::with_capacity(page);
            let mut last_key: Option<Vec<u8>> = None;

            for item in it.take(page) {
                last_key = Some(item.key.clone());
                keys.push(item.key);
            }

            if keys.is_empty() {
                break;
            }

            // Batch point-lookups across projected columns.
            let queries: Vec<(LogicalFieldId, Vec<Vec<u8>>)> = project
                .iter()
                .map(|fid| (*fid as LogicalFieldId, keys.clone()))
                .collect();

            let results = self.store.get_many(queries);

            // Emit rows in the same order as `keys`.
            for (i, key) in keys.iter().enumerate() {
                let rid = parse_row_id(key);
                let mut row: Vec<Option<Vec<u8>>> = Vec::with_capacity(project.len());
                for col in 0..project.len() {
                    let cell = &results[col][i];
                    let owned = cell.as_ref().map(|s| value_slice_bytes(s));
                    row.push(owned);
                }
                on_row(rid, row);
            }

            after = last_key;
        }
    }

    /// Stream (row_id, value) pairs for one column ordered by VALUE.
    /// Efficient due to segment-local order + heads in column-map.
    pub fn stream_column_values<F>(&self, fid: FieldId, dir: Direction, page: usize, mut on_item: F)
    where
        F: FnMut(RowId, Vec<u8>),
    {
        if page == 0 {
            return;
        }

        let mut cursor_v: Option<Vec<u8>> = None;

        loop {
            let mut lo_buf: Option<Vec<u8>> = None;
            let mut hi_buf: Option<Vec<u8>> = None;

            let (lo_bound, hi_bound): (Bound<&[u8]>, Bound<&[u8]>) = match dir {
                Direction::Forward => {
                    let lo = match cursor_v.as_deref() {
                        None => Bound::Unbounded,
                        Some(v) => {
                            lo_buf = Some(v.to_vec());
                            Bound::Excluded(lo_buf.as_ref().unwrap().as_slice())
                        }
                    };
                    (lo, Bound::Unbounded)
                }
                Direction::Reverse => {
                    let hi = match cursor_v.as_deref() {
                        None => Bound::Unbounded,
                        Some(v) => {
                            hi_buf = Some(v.to_vec());
                            Bound::Excluded(hi_buf.as_ref().unwrap().as_slice())
                        }
                    };
                    (Bound::Unbounded, hi)
                }
            };

            let it = match self.store.scan_values_lww(
                fid as LogicalFieldId,
                ValueScanOpts {
                    order_by: OrderBy::Value,
                    dir,
                    lo: lo_bound,
                    hi: hi_bound,
                    prefix: None,
                    bucket_prefix_len: 2,
                    // 16 minimizes tag-ties across segments
                    head_tag_len: 16,
                    frame_predicate: None,
                },
            ) {
                Ok(it) => it,
                Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => {
                    // Normal end once bound excludes all segments.
                    break;
                }
                Err(e) => panic!("scan init error (value-ordered): {:?}", e),
            };

            let mut took = 0usize;

            for item in it.take(page) {
                let rid = parse_row_id(item.key.as_slice());
                let a = item.value.start() as usize;
                let b = item.value.end() as usize;
                let v = item.value.data().as_ref()[a..b].to_vec();

                cursor_v = Some(v.clone());
                on_item(rid, v);
                took += 1;
            }

            if took == 0 {
                break;
            }
        }
    }

    /// Access to the underlying store for power users.
    pub fn store(&self) -> &ColumnStore<'_, MemPager> {
        &self.store
    }
}
