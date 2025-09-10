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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ColumnInput;
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::ops::Bound;

    /// Build a single row patch from simple (field, value) pairs.
    #[inline]
    fn mk_row(
        rid: RowId,
        kv: &[(FieldId, ColumnInput<'static>)],
    ) -> (RowId, HashMap<FieldId, (u64, ColumnInput<'static>)>) {
        let mut m = HashMap::new();
        for (fid, v) in kv.iter() {
            m.insert(*fid, (0u64, v.clone()));
        }
        (rid, m)
    }

    /// String helper: own bytes to avoid 'static lifetime requirements.
    #[inline]
    fn s(x: &str) -> ColumnInput<'static> {
        Cow::Owned(x.as_bytes().to_vec())
    }

    #[inline]
    fn be_u64(x: u64) -> Vec<u8> {
        x.to_be_bytes().to_vec()
    }

    #[inline]
    fn from_be_u64(b: &[u8]) -> u64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[..8]);
        u64::from_be_bytes(a)
    }

    /// u64 helper: store as big-endian bytes (lex order == numeric).
    #[inline]
    fn be64_bytes(x: u64) -> ColumnInput<'static> {
        Cow::Owned(x.to_be_bytes().to_vec())
    }

    fn mk_rowpatch_u64(
        rid: RowId,
        pairs: &[(FieldId, u64)],
    ) -> (RowId, HashMap<FieldId, (u64, ColumnInput<'static>)>) {
        let mut m = HashMap::new();
        for &(fid, n) in pairs {
            m.insert(fid, (0u64, Cow::Owned(be_u64(n))));
        }
        (rid, m)
    }

    #[test]
    fn insert_many_strings_readable() {
        let table = ColumnMapTable::new(CmTableConfig::default());

        // Columns (FieldId == logical column id)
        const COL_NAME: FieldId = 11;
        const COL_CITY: FieldId = 12;
        const COL_ROLE: FieldId = 13;

        // Three rows, each maps row id -> column values (strings).
        // No synthetic columns; the driver will be COL_NAME.
        let rows: Vec<RowPatch> = vec![
            mk_row(
                1,
                &[
                    (COL_NAME, s("alice")),
                    (COL_CITY, s("austin")),
                    (COL_ROLE, s("dev")),
                ],
            ),
            mk_row(
                2,
                &[
                    (COL_NAME, s("bob")),
                    (COL_CITY, s("boston")),
                    (COL_ROLE, s("pm")),
                ],
            ),
            mk_row(
                3,
                &[
                    (COL_NAME, s("carol")),
                    (COL_CITY, s("chicago")),
                    (COL_ROLE, s("qa")),
                ],
            ),
        ];

        table.insert_many(&rows);

        // Scan row ids 1..=3. The scan uses COL_NAME as the driver
        // (because it exists on all three rows).
        let mut got: Vec<(RowId, Vec<Option<Vec<u8>>>)> = Vec::new();
        table.scan_by_row_id_range(
            COL_NAME,
            Bound::Excluded(0),
            Bound::Included(3),
            &[COL_NAME, COL_CITY, COL_ROLE],
            2,
            |rid, cols| got.push((rid, cols)),
        );

        assert_eq!(got.len(), 3);

        // Check values round-trip cleanly (own strings to avoid lifetimes).
        let to_s = |o: &Option<Vec<u8>>| -> String {
            String::from_utf8(o.as_ref().unwrap().clone()).unwrap()
        };

        assert_eq!(got[0].0, 1);
        assert_eq!(to_s(&got[0].1[0]), "alice");
        assert_eq!(to_s(&got[0].1[1]), "austin");
        assert_eq!(to_s(&got[0].1[2]), "dev");

        assert_eq!(got[1].0, 2);
        assert_eq!(to_s(&got[1].1[0]), "bob");
        assert_eq!(to_s(&got[1].1[1]), "boston");
        assert_eq!(to_s(&got[1].1[2]), "pm");

        assert_eq!(got[2].0, 3);
        assert_eq!(to_s(&got[2].1[0]), "carol");
        assert_eq!(to_s(&got[2].1[1]), "chicago");
        assert_eq!(to_s(&got[2].1[2]), "qa");
    }

    #[test]
    fn insert_many_u64_readable() {
        let table = ColumnMapTable::new(CmTableConfig::default());

        // Columns (FieldId == logical column id)
        const COL_A: FieldId = 21; // driver
        const COL_B: FieldId = 22; // numeric payload
        const COL_C: FieldId = 23; // numeric payload (sparse)

        // Five rows, u64 payloads stored as big-endian bytes.
        // COL_C is only present on odd row ids to show sparsity.
        let rows: Vec<RowPatch> = (1..=5)
            .map(|rid| {
                let mut cols = vec![
                    (COL_A, be64_bytes(10 + rid)),
                    (COL_B, be64_bytes(100 + rid)),
                ];
                if rid % 2 == 1 {
                    cols.push((COL_C, be64_bytes(1000 + rid)));
                }
                mk_row(rid, &cols)
            })
            .collect();

        table.insert_many(&rows);

        // Scan all rows using COL_A as the driver.
        let mut got: Vec<(RowId, Vec<Option<Vec<u8>>>)> = Vec::new();
        table.scan_by_row_id_range(
            COL_A,
            Bound::Excluded(0),
            Bound::Included(5),
            &[COL_A, COL_B, COL_C],
            3,
            |rid, cols| got.push((rid, cols)),
        );

        assert_eq!(got.len(), 5);
        for i in 0..5 {
            let (rid, cols) = &got[i];
            let expect = (i as u64) + 1;
            assert_eq!(*rid, expect);

            let a = from_be_u64(cols[0].as_ref().unwrap());
            let b = from_be_u64(cols[1].as_ref().unwrap());
            assert_eq!(a, 10 + expect);
            assert_eq!(b, 100 + expect);

            if expect % 2 == 1 {
                let c = from_be_u64(cols[2].as_ref().unwrap());
                assert_eq!(c, 1000 + expect);
            } else {
                assert!(cols[2].is_none());
            }
        }
    }

    #[test]
    fn scan_unmatched_range_returns_no_rows_u64() {
        let cfg = CmTableConfig::default();
        let table = ColumnMapTable::new(cfg);

        let fid_row: FieldId = 10;
        let fid_d1: FieldId = 11;
        let fid_d2: FieldId = 12;

        // Seed rows 1..=3 so driver exists.
        let rows: Vec<RowPatch> = (1..=3)
            .map(|rid| {
                mk_rowpatch_u64(rid, &[(fid_row, rid), (fid_d1, rid + 7), (fid_d2, rid + 9)])
            })
            .collect();
        table.insert_many(&rows);

        // Ask for a range above any inserted row ids.
        let mut called = false;
        table.scan_by_row_id_range(
            fid_row,
            Bound::Excluded(10),
            Bound::Included(20),
            &[fid_row, fid_d1, fid_d2],
            8,
            |_rid, _cols| {
                called = true;
            },
        );
        assert!(!called, "no rows should be returned");
    }

    #[test]
    fn stream_value_order_multi_cols() {
        let cfg = CmTableConfig::default();
        let table = ColumnMapTable::new(cfg);

        let fid_row: FieldId = 20; // driver
        let fid_v1: FieldId = 21; // rid * 10
        let fid_v2: FieldId = 22; // 500 - rid (reverse order baseline)
        let fid_sparse: FieldId = 23; // multiples of 3, 3000 + rid

        let rows: Vec<RowPatch> = (1..=12)
            .map(|rid| {
                let mut pairs = vec![(fid_row, rid), (fid_v1, rid * 10), (fid_v2, 500 - rid)];
                if rid % 3 == 0 {
                    pairs.push((fid_sparse, 3000 + rid));
                }
                mk_rowpatch_u64(rid, &pairs)
            })
            .collect();
        table.insert_many(&rows);

        // Stream v1 forward: strictly increasing.
        let mut v1_f: Vec<(RowId, u64)> = Vec::new();
        table.stream_column_values(fid_v1, Direction::Forward, 4, |rid, v| {
            v1_f.push((rid, from_be_u64(&v)));
        });
        assert_eq!(v1_f.len(), 12);
        for w in v1_f.windows(2) {
            assert!(w[0].1 < w[1].1, "v1 must increase forward");
        }

        // Stream v2 reverse: strictly decreasing (since base is 500 - rid).
        let mut v2_r: Vec<(RowId, u64)> = Vec::new();
        table.stream_column_values(fid_v2, Direction::Reverse, 5, |rid, v| {
            v2_r.push((rid, from_be_u64(&v)));
        });
        assert_eq!(v2_r.len(), 12);
        for w in v2_r.windows(2) {
            assert!(w[0].1 > w[1].1, "v2 must decrease reverse");
        }

        // Stream sparse forward: only rid % 3 == 0.
        let mut sp_f: Vec<(RowId, u64)> = Vec::new();
        table.stream_column_values(fid_sparse, Direction::Forward, 3, |rid, v| {
            sp_f.push((rid, from_be_u64(&v)));
        });
        // rows 3,6,9,12
        assert_eq!(sp_f.len(), 4);
        assert_eq!(
            sp_f.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec![3, 6, 9, 12]
        );
        for (i, (_rid, val)) in sp_f.iter().enumerate() {
            assert_eq!(*val, 3000 + ((i as u64 + 1) * 3));
        }
    }

    #[test]
    fn pagination_chunks_are_consistent_u64_multi_proj() {
        let cfg = CmTableConfig::default();
        let table = ColumnMapTable::new(cfg);

        let fid_row: FieldId = 30; // driver
        let fid_d1: FieldId = 31;
        let fid_d2: FieldId = 32;

        // 1..=50 rows, three projected columns.
        let rows: Vec<RowPatch> = (1..=50)
            .map(|rid| {
                mk_rowpatch_u64(
                    rid,
                    &[(fid_row, rid), (fid_d1, 1000 + rid), (fid_d2, 2000 + rid)],
                )
            })
            .collect();
        table.insert_many(&rows);

        // Collect with page=7 and ensure we see all rows once with projections.
        let mut got: Vec<(RowId, Vec<Option<Vec<u8>>>)> = Vec::new();
        table.scan_by_row_id_range(
            fid_row,
            Bound::Excluded(0),
            Bound::Included(50),
            &[fid_row, fid_d1, fid_d2],
            7,
            |rid, cols| got.push((rid, cols)),
        );

        assert_eq!(got.len(), 50);
        for (i, (rid, cols)) in got.into_iter().enumerate() {
            let expect = (i as u64) + 1;
            assert_eq!(rid, expect);
            assert_eq!(from_be_u64(cols[0].as_ref().unwrap()), expect);
            assert_eq!(from_be_u64(cols[1].as_ref().unwrap()), 1000 + expect);
            assert_eq!(from_be_u64(cols[2].as_ref().unwrap()), 2000 + expect);
        }
    }
}
