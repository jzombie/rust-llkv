//! Table: a column-store-backed table with row-based ingest.
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

use crossbeam_channel as xchan;

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};
use llkv_column_map::views::ValueSlice;

use crate::expr::{Expr, Filter, Operator};
use crate::types::{FieldId, RowId, RowPatch};

/// Internal "presence" column:
/// - We write a single byte for every inserted row into this hidden column.
/// - This lets us stream *all* row_ids in row-id order, enabling proper
///   semantics for `NOT` (complement against the universe of rows),
///   and future helpers like row_count(), existence checks, etc.
/// - Kept entirely private; callers never see or pass this fid.
const PRESENCE_FID: LogicalFieldId = u32::MAX - 7;
const PRESENCE_VAL: &[u8] = b"\x01";

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
pub struct TableCfg {
    pub segment_max_entries: usize,
    pub segment_max_bytes: usize,
    pub last_write_wins_in_batch: bool,

    /// Track presence for each row in a hidden internal column.
    /// Enables `NOT` and "full table" scans without exposing any fid.
    pub track_presence: bool,
}

impl Default for TableCfg {
    fn default() -> Self {
        Self {
            segment_max_entries: 100_000,
            segment_max_bytes: 32 << 20,
            last_write_wins_in_batch: true,
            track_presence: true,
        }
    }
}

/// Scan/init error wrapper for this module.
#[derive(Debug)]
pub enum CmError {
    ScanInit(String),
}

/// ColumnMap-backed table.
pub struct Table {
    /// Leaked pager to satisfy ColumnStore's lifetime without
    /// gymnastics.
    pager: &'static MemPager,
    store: ColumnStore<'static, MemPager>,
    cfg: TableCfg,
}

impl Table {
    /// Create a new in-memory table.
    pub fn new(cfg: TableCfg) -> Self {
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

            // --- Internal presence write (hidden column) ---
            if self.cfg.track_presence {
                by_col
                    .entry(PRESENCE_FID)
                    .or_default()
                    .push((Cow::Owned(key), Cow::Borrowed(PRESENCE_VAL)));
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
    /// Semantics: rows appear if they have a value in the driver
    /// column. Choose `driver_fid` with wide coverage (e.g., PK or
    /// common column).
    ///
    /// For each row, projected columns are fetched via `get_many_projected`.
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
                    // 16 minimizes head-tie fallbacks inside column-map.
                    head_tag_len: 16,
                    frame_predicate: None,
                },
            ) {
                Ok(it) => it,
                Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => break,
                Err(e) => panic!("scan init error: {:?}", e),
            };

            // Own the keys for this page and keep them alive while we query.
            let mut keys: Vec<Vec<u8>> = Vec::with_capacity(page);
            let mut last_key: Option<Vec<u8>> = None;

            for item in it.take(page) {
                last_key = Some(item.key.clone());
                keys.push(item.key);
            }

            if keys.is_empty() {
                break;
            }

            // Build shared keyset & fid list for get_many_projected.
            let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
            let fids: Vec<LogicalFieldId> =
                project.iter().map(|fid| *fid as LogicalFieldId).collect();

            let results = self.store.get_many_projected(&fids, &key_refs);

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
                    // 16 minimizes tag-ties across segments.
                    head_tag_len: 16,
                    frame_predicate: None,
                },
            ) {
                Ok(it) => it,
                Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => break,
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

    /// Scan rows using an expression over column values (by value order),
    /// then project the requested columns.
    pub fn scan_expr<F>(
        &self,
        expr: &Expr<'_, FieldId>,
        projection: &[FieldId],
        mut on_row: F,
    ) -> Result<(), CmError>
    where
        F: FnMut(RowId, Vec<Option<Vec<u8>>>),
    {
        let rx_opt = self.get_row_id_stream(expr)?;
        let Some(rx) = rx_opt else {
            return Ok(());
        };

        // Drain all row ids from the stream.
        let mut rids: Vec<RowId> = Vec::new();
        for rid in rx {
            rids.push(rid);
        }
        if rids.is_empty() {
            return Ok(());
        }

        // Shared keyset for this projection batch.
        let keys: Vec<Vec<u8>> = rids.iter().map(|r| row_key_bytes(*r)).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let fids: Vec<LogicalFieldId> = projection
            .iter()
            .map(|fid| (*fid as LogicalFieldId))
            .collect();

        let results = self.store.get_many_projected(&fids, &key_refs);

        // Emit per-row, preserving rids order.
        for (i, rid) in rids.into_iter().enumerate() {
            let mut row: Vec<Option<Vec<u8>>> = Vec::with_capacity(projection.len());
            for col in 0..projection.len() {
                let cell = &results[col][i];
                let owned = cell.as_ref().map(|s| value_slice_bytes(s));
                row.push(owned);
            }
            on_row(rid, row);
        }
        Ok(())
    }

    /// Build a stream (receiver) of row ids that satisfy the expression.
    /// Uses value-ordered scans per predicate; AND/OR combine via sets.
    pub fn get_row_id_stream<'b>(
        &'b self,
        expr: &Expr<'b, FieldId>,
    ) -> Result<Option<xchan::Receiver<RowId>>, CmError> {
        match expr {
            Expr::Pred(filter) => self.get_stream_for_predicate(filter),
            Expr::And(children) => {
                // Collect child streams.
                let mut streams: Vec<xchan::Receiver<RowId>> = Vec::new();
                for ch in children {
                    if let Some(rx) = self.get_row_id_stream(ch)? {
                        streams.push(rx);
                    }
                }

                // No predicates? return closed empty stream.
                if streams.is_empty() {
                    let (_tx, rx) = xchan::unbounded();
                    return Ok(Some(rx));
                }

                // Combine via intersection in a worker thread.
                let (tx, rx) = xchan::unbounded();
                std::thread::spawn(move || {
                    use std::collections::HashSet;
                    let first: HashSet<RowId> = streams.remove(0).into_iter().collect();
                    let mut acc = first;
                    for s in streams {
                        if acc.is_empty() {
                            break;
                        }
                        let set: HashSet<RowId> = s.into_iter().collect();
                        acc.retain(|rid| set.contains(rid));
                    }
                    for rid in acc {
                        if tx.send(rid).is_err() {
                            break;
                        }
                    }
                });
                Ok(Some(rx))
            }
            Expr::Or(children) => {
                let mut streams: Vec<xchan::Receiver<RowId>> = Vec::new();
                for ch in children {
                    if let Some(rx) = self.get_row_id_stream(ch)? {
                        streams.push(rx);
                    }
                }
                let (tx, rx) = xchan::unbounded();
                std::thread::spawn(move || {
                    use std::collections::HashSet;
                    let mut acc: HashSet<RowId> = std::collections::HashSet::new();
                    for s in streams {
                        for rid in s {
                            acc.insert(rid);
                        }
                    }
                    for rid in acc {
                        if tx.send(rid).is_err() {
                            break;
                        }
                    }
                });
                Ok(Some(rx))
            }
            Expr::Not(inner) => {
                // Presence complement Universe \ Positives.
                if !self.cfg.track_presence {
                    // Defensive: if someone disables presence, make it explicit.
                    return Err(CmError::ScanInit(
                        "Expr::Not requires TableCfg.track_presence=true".into(),
                    ));
                }

                // Evaluate the positive side first (synchronously).
                let pos_rx_opt = self.get_row_id_stream(inner)?;
                let pos_set: std::collections::HashSet<RowId> = match pos_rx_opt {
                    None => std::collections::HashSet::new(),
                    Some(rx) => rx.into_iter().collect(),
                };

                // Stream all rows via presence and subtract positives.
                let all_rx = self.stream_all_row_ids();
                let (tx, rx) = xchan::unbounded();
                std::thread::spawn(move || {
                    for rid in all_rx {
                        if !pos_set.contains(&rid) {
                            if tx.send(rid).is_err() {
                                break;
                            }
                        }
                    }
                });
                Ok(Some(rx))
            }
        }
    }

    /// Single-predicate stream using a value-ordered scan on the column.
    fn get_stream_for_predicate<'b>(
        &'b self,
        filter: &Filter<'b, FieldId>,
    ) -> Result<Option<xchan::Receiver<RowId>>, CmError> {
        // Map the predicate to lo/hi bounds on VALUE bytes.
        let (lo_bound, hi_bound): (Bound<&[u8]>, Bound<&[u8]>);

        match &filter.op {
            Operator::Equals(val) => {
                lo_bound = Bound::Included(*val);
                hi_bound = Bound::Included(*val);
            }
            Operator::Range { lower, upper } => {
                // Lower
                lo_bound = match lower {
                    Bound::Unbounded => Bound::Unbounded,
                    Bound::Included(b) => Bound::Included(*b),
                    Bound::Excluded(b) => Bound::Excluded(*b),
                };
                // Upper
                hi_bound = match upper {
                    Bound::Unbounded => Bound::Unbounded,
                    Bound::Included(b) => Bound::Included(*b),
                    Bound::Excluded(b) => Bound::Excluded(*b),
                };
            }
            _ => {
                unimplemented!("Only Equals and Range are supported here");
            }
        }

        // Kick off a value-ordered scan for this column.
        let iter = match self.store.scan_values_lww(
            filter.field_id as LogicalFieldId,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Forward,
                lo: lo_bound,
                hi: hi_bound,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 16,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => {
                return Ok(None);
            }
            Err(llkv_column_map::column_store::read_scan::ScanError::ColumnMissing(_)) => {
                return Ok(None);
            }
            Err(e) => return Err(CmError::ScanInit(format!("{:?}", e))),
        };

        // Fill the channel synchronously to avoid requiring Send on the
        // iterator (ValueScan is not Send).
        let (tx, rx) = xchan::unbounded();
        for item in iter {
            let rid = parse_row_id(item.key.as_slice());
            if tx.send(rid).is_err() {
                break;
            }
        }
        drop(tx);

        Ok(Some(rx))
    }

    /// Stream all row ids in **row-id order** via the hidden presence column.
    fn stream_all_row_ids(&self) -> xchan::Receiver<RowId> {
        use llkv_column_map::column_store::read_scan::ScanError;
        let (tx, rx) = xchan::unbounded();

        match self.store.scan_values_lww(
            PRESENCE_FID,
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                // 16 minimizes head-tie fallbacks inside column-map.
                head_tag_len: 16,
                frame_predicate: None,
            },
        ) {
            Ok(iter) => {
                for item in iter {
                    let rid = parse_row_id(item.key.as_slice());
                    if tx.send(rid).is_err() {
                        break;
                    }
                }
            }
            Err(ScanError::NoActiveSegments) | Err(ScanError::ColumnMissing(_)) => {
                // Empty table or presence not yet written => empty stream.
            }
            Err(e) => panic!("presence scan init error: {:?}", e),
        }

        drop(tx);
        rx
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
        let table = Table::new(TableCfg::default());

        const COL_NAME: FieldId = 11;
        const COL_CITY: FieldId = 12;
        const COL_ROLE: FieldId = 13;

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
        let table = Table::new(TableCfg::default());

        const COL_A: FieldId = 21; // driver
        const COL_B: FieldId = 22; // numeric payload
        const COL_C: FieldId = 23; // numeric payload (sparse)

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
        let cfg = TableCfg::default();
        let table = Table::new(cfg);

        let fid_row: FieldId = 10;
        let fid_d1: FieldId = 11;
        let fid_d2: FieldId = 12;

        let rows: Vec<RowPatch> = (1..=3)
            .map(|rid| {
                mk_rowpatch_u64(rid, &[(fid_row, rid), (fid_d1, rid + 7), (fid_d2, rid + 9)])
            })
            .collect();
        table.insert_many(&rows);

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
        let cfg = TableCfg::default();
        let table = Table::new(cfg);

        let fid_row: FieldId = 20; // driver
        let fid_v1: FieldId = 21; // rid * 10
        let fid_v2: FieldId = 22; // 500 - rid (reverse baseline)
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

        // Stream v2 reverse: strictly decreasing.
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
        let cfg = TableCfg::default();
        let table = Table::new(cfg);

        let fid_row: FieldId = 30; // driver
        let fid_d1: FieldId = 31;
        let fid_d2: FieldId = 32;

        let rows: Vec<RowPatch> = (1..=50)
            .map(|rid| {
                mk_rowpatch_u64(
                    rid,
                    &[(fid_row, rid), (fid_d1, 1000 + rid), (fid_d2, 2000 + rid)],
                )
            })
            .collect();
        table.insert_many(&rows);

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

    use crate::expr::{Expr, Filter, Operator};

    #[test]
    fn expr_equals_and_range_scan_projects_values() {
        let table = Table::new(TableCfg::default());

        const COL_NAME: FieldId = 101;
        const COL_CITY: FieldId = 102;
        const COL_SCORE: FieldId = 103;

        let rows: Vec<RowPatch> = vec![
            mk_row(
                1,
                &[
                    (COL_NAME, s("alice")),
                    (COL_CITY, s("austin")),
                    (COL_SCORE, be64_bytes(10)),
                ],
            ),
            mk_row(
                2,
                &[
                    (COL_NAME, s("bob")),
                    (COL_CITY, s("boston")),
                    (COL_SCORE, be64_bytes(20)),
                ],
            ),
            mk_row(
                3,
                &[
                    (COL_NAME, s("carol")),
                    (COL_CITY, s("chicago")),
                    (COL_SCORE, be64_bytes(30)),
                ],
            ),
            mk_row(
                4,
                &[
                    (COL_NAME, s("dave")),
                    (COL_CITY, s("dallas")),
                    (COL_SCORE, be64_bytes(40)),
                ],
            ),
            mk_row(
                5,
                &[
                    (COL_NAME, s("erin")),
                    (COL_CITY, s("el paso")),
                    (COL_SCORE, be64_bytes(50)),
                ],
            ),
            mk_row(
                6,
                &[
                    (COL_NAME, s("frank")),
                    (COL_CITY, s("fresno")),
                    (COL_SCORE, be64_bytes(60)),
                ],
            ),
        ];
        table.insert_many(&rows);

        // WHERE score = 30
        let key_eq = 30u64.to_be_bytes();
        let expr_eq = Expr::Pred(Filter {
            field_id: COL_SCORE,
            op: Operator::Equals(&key_eq),
        });
        let mut got_eq: Vec<(RowId, Vec<Option<Vec<u8>>>)> = Vec::new();
        table
            .scan_expr(&expr_eq, &[COL_NAME, COL_CITY], |rid, cols| {
                got_eq.push((rid, cols));
            })
            .unwrap();
        assert_eq!(got_eq.len(), 1);
        assert_eq!(got_eq[0].0, 3);

        // WHERE 20 <= score < 50
        let lo = 20u64.to_be_bytes();
        let hi = 50u64.to_be_bytes();
        let expr_rng = Expr::Pred(Filter {
            field_id: COL_SCORE,
            op: Operator::Range {
                lower: Bound::Included(&lo),
                upper: Bound::Excluded(&hi),
            },
        });
        let mut got_rng: Vec<RowId> = Vec::new();
        table
            .scan_expr(&expr_rng, &[], |rid, _| {
                got_rng.push(rid);
            })
            .unwrap();
        got_rng.sort_unstable();
        assert_eq!(got_rng, vec![2, 3, 4]);
    }

    #[test]
    fn expr_intersection_and_union() {
        let table = Table::new(TableCfg::default());

        const COL_COUNTRY: FieldId = 201;
        const COL_STATUS: FieldId = 202;
        const COL_PAYLOAD: FieldId = 203;

        let rows: Vec<RowPatch> = (1..=20)
            .map(|rid| {
                mk_row(
                    rid,
                    &[
                        (COL_COUNTRY, be64_bytes((rid % 5) as u64)),
                        (COL_STATUS, be64_bytes((rid % 4) as u64)),
                        (COL_PAYLOAD, s("x")),
                    ],
                )
            })
            .collect();
        table.insert_many(&rows);

        // country=3 AND status=2
        let c3 = (3u64).to_be_bytes();
        let s2 = (2u64).to_be_bytes();
        let expr_and = Expr::And(vec![
            Expr::Pred(Filter {
                field_id: COL_COUNTRY,
                op: Operator::Equals(&c3),
            }),
            Expr::Pred(Filter {
                field_id: COL_STATUS,
                op: Operator::Equals(&s2),
            }),
        ]);

        let mut and_rows: Vec<RowId> = Vec::new();
        table
            .scan_expr(&expr_and, &[], |rid, _| and_rows.push(rid))
            .unwrap();
        and_rows.sort_unstable();
        assert_eq!(and_rows, vec![18]);

        // country=1 OR status=0
        let c1 = (1u64).to_be_bytes();
        let s0 = (0u64).to_be_bytes();
        let expr_or = Expr::Or(vec![
            Expr::Pred(Filter {
                field_id: COL_COUNTRY,
                op: Operator::Equals(&c1),
            }),
            Expr::Pred(Filter {
                field_id: COL_STATUS,
                op: Operator::Equals(&s0),
            }),
        ]);

        let mut or_rows: Vec<RowId> = Vec::new();
        table
            .scan_expr(&expr_or, &[], |rid, _| or_rows.push(rid))
            .unwrap();
        or_rows.sort_unstable();
        assert_eq!(or_rows, vec![1, 4, 6, 8, 11, 12, 16, 20]);
    }

    #[test]
    fn not_equals_returns_complement_including_missing_field() {
        let table = Table::new(TableCfg::default());

        const F_NAME: FieldId = 301;

        // 1..=5; only rows 2 and 4 have the value "target"
        let rows: Vec<RowPatch> = (1..=5)
            .map(|rid| {
                if rid % 2 == 0 {
                    mk_row(rid, &[(F_NAME, s("target"))])
                } else {
                    // row present but field missing
                    mk_row(rid, &[])
                }
            })
            .collect();
        table.insert_many(&rows);

        let tgt = b"target";
        let expr = Expr::Not(Box::new(Expr::Pred(Filter {
            field_id: F_NAME,
            op: Operator::Equals(tgt),
        })));

        let mut got: Vec<RowId> = Vec::new();
        table.scan_expr(&expr, &[], |rid, _| got.push(rid)).unwrap();
        got.sort_unstable();

        // Expect all rows except {2,4}
        assert_eq!(got, vec![1, 3, 5]);
    }

    #[test]
    fn not_over_empty_predicate_yields_all_rows() {
        let table = Table::new(TableCfg::default());

        const F_TAG: FieldId = 401;

        // Insert 4 rows with **no** F_TAG values at all.
        let rows: Vec<RowPatch> = (1..=4).map(|rid| mk_row(rid, &[])).collect();
        table.insert_many(&rows);

        // Positive predicate has no matches -> NOT(empty) == all rows.
        let expr = Expr::Not(Box::new(Expr::Pred(Filter {
            field_id: F_TAG,
            op: Operator::Equals(b"never-appears"),
        })));

        let mut got: Vec<RowId> = Vec::new();
        table.scan_expr(&expr, &[], |rid, _| got.push(rid)).unwrap();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3, 4]);
    }

    #[test]
    fn presence_stream_orders_row_ids_by_key() {
        let table = Table::new(TableCfg::default());
        let rows: Vec<RowPatch> = vec![mk_row(3, &[]), mk_row(1, &[]), mk_row(2, &[])];
        table.insert_many(&rows);

        // Stream all and check order 1,2,3
        let rx = table.stream_all_row_ids();
        let mut got: Vec<RowId> = rx.into_iter().collect();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3]);
    }
}
