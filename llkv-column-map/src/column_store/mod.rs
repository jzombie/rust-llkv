// High-level append/query API on top of pager + index modules.
use crate::column_index::{Bootstrap, ColumnIndex, IndexSegment, Manifest};
use crate::constants::BOOTSTRAP_PKEY;
use crate::layout::{IndexLayoutInfo, KeyLayout, ValueLayout};
use crate::storage::{
    StorageKind, StorageNode,
    pager::{BatchGet, BatchPut, GetResult, Pager},
};
use crate::types::{ByteOffset, ByteWidth, LogicalFieldId, PhysicalKey, TypedKind, TypedValue};
use rustc_hash::FxHashMap;
use std::fmt::Write;
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicUsize, Ordering},
};
pub mod introspect;
pub mod metrics;
// Legacy read/write paths removed in favor of columnar u64 prototype.
// pub mod read;
// pub mod read_scan;
// pub mod write;
pub mod columnar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction { Asc, Desc }

#[derive(Clone, Copy, Debug)]
pub enum Bounds {
    None,
    /// Per-chunk row bounds: [start_row, end_row) relative to each chunk.
    Rows { start_row: Option<usize>, end_row: Option<usize> },
    /// Value bounds (inclusive). Prunes whole chunks via min/max; no in-chunk selection yet.
    Values { min: Option<u64>, max: Option<u64> },
}

#[derive(Clone, Copy, Debug)]
pub struct ScanOptions {
    pub direction: Direction,
    pub bounds: Bounds,
    /// Desired vector length. If 0, use per-chunk header vector_len.
    pub vector_len: usize,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self { direction: Direction::Asc, bounds: Bounds::None, vector_len: 16_384 }
    }
}

pub struct ColumnStore<P: Pager> {
    pager: Arc<P>,
    bootstrap_key: PhysicalKey,
    manifest_key: PhysicalKey,
    // Interior mutability for concurrent reads/writes.
    manifest: RwLock<Manifest>,
    // field_id -> (column_index_pkey, decoded)
    colindex_cache: RwLock<FxHashMap<LogicalFieldId, (PhysicalKey, ColumnIndex)>>,

    // serialize writers to keep ColumnIndex/Manifest consistent
    append_lock: Mutex<()>,

    // pager-hit metrics (counts only) via atomics
    io_batches: AtomicUsize,
    io_get_raw_ops: AtomicUsize,
    io_get_typed_ops: AtomicUsize,
    io_put_raw_ops: AtomicUsize,
    io_put_typed_ops: AtomicUsize,
    io_free_ops: AtomicUsize,

    // New: simple in-memory map of columnar chunks per field (u64 prototype path).
    columnar_chunks_u64: RwLock<FxHashMap<LogicalFieldId, Vec<PhysicalKey>>>,
}

impl<P: Pager> ColumnStore<P> {
    /// Idempotent: open an existing store, or create bootstrap + empty manifest.
    /// Safe to call from multiple threads/processes that share the same Pager.
    pub fn open(pager: Arc<P>) -> Self {
        let manifest_key = {
            let resp = pager
                .batch_get(&[BatchGet::Typed {
                    key: BOOTSTRAP_PKEY,
                    kind: TypedKind::Bootstrap,
                }])
                .expect("pager get bootstrap");

            match &resp[0] {
                // Existing store
                GetResult::Typed {
                    value: TypedValue::Bootstrap(b),
                    ..
                } => b.manifest_physical_key,

                // Fresh store
                GetResult::Missing { .. } => {
                    // Ask the pager for a fresh Manifest key.
                    let mani = pager.alloc_many(1).expect("alloc manifest key")[0];

                    let empty = Manifest {
                        columns: Vec::new(),
                    };

                    // Write Manifest first, then Bootstrap pointing to it.
                    pager
                        .batch_put(&[
                            BatchPut::Typed {
                                key: mani,
                                value: TypedValue::Manifest(empty.clone()),
                            },
                            BatchPut::Typed {
                                key: BOOTSTRAP_PKEY, // 0
                                value: TypedValue::Bootstrap(Bootstrap {
                                    manifest_physical_key: mani,
                                }),
                            },
                        ])
                        .expect("write bootstrap+manifest");

                    mani
                }

                _ => panic!("corrupted bootstrap at key {}", BOOTSTRAP_PKEY),
            }
        };

        // Load manifest as you already do…
        let resp = pager
            .batch_get(&[BatchGet::Typed {
                key: manifest_key,
                kind: TypedKind::Manifest,
            }])
            .expect("pager get manifest");
        let manifest = match &resp[0] {
            GetResult::Typed {
                value: TypedValue::Manifest(m),
                ..
            } => m.clone(),
            GetResult::Missing { .. } => panic!("manifest missing at {}", manifest_key),
            _ => panic!("invalid manifest at {}", manifest_key),
        };

        ColumnStore {
            pager,
            bootstrap_key: BOOTSTRAP_PKEY,
            manifest_key,
            manifest: RwLock::new(manifest),
            colindex_cache: RwLock::new(FxHashMap::default()),
            append_lock: Mutex::new(()),
            io_batches: AtomicUsize::new(0),
            io_get_raw_ops: AtomicUsize::new(0),
            io_get_typed_ops: AtomicUsize::new(0),
            io_put_raw_ops: AtomicUsize::new(0),
            io_put_typed_ops: AtomicUsize::new(0),
            io_free_ops: AtomicUsize::new(0),
            columnar_chunks_u64: RwLock::new(FxHashMap::default()),
        }
    }

    // Helper to route batch PUTs through here to bump metrics in one place.
    fn do_puts(&self, puts: Vec<BatchPut>) {
        if puts.is_empty() {
            return;
        }
        // update counters before the call
        self.io_batches.fetch_add(1, Ordering::Relaxed);
        for p in &puts {
            match p {
                BatchPut::Raw { .. } => {
                    self.io_put_raw_ops.fetch_add(1, Ordering::Relaxed);
                }
                BatchPut::Typed { .. } => {
                    self.io_put_typed_ops.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        // ignore IO errors here for brevity, mirroring previous behavior
        let _ = self.pager.batch_put(&puts);
    }

    // Helper to route batch GETs through here to bump metrics in one place.
    fn do_gets(&self, gets: Vec<BatchGet>) -> Vec<GetResult<P::Blob>> {
        if gets.is_empty() {
            return Vec::new();
        }
        self.io_batches.fetch_add(1, Ordering::Relaxed);
        for g in &gets {
            match g {
                BatchGet::Raw { .. } => {
                    self.io_get_raw_ops.fetch_add(1, Ordering::Relaxed);
                }
                BatchGet::Typed { .. } => {
                    self.io_get_typed_ops.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        self.pager.batch_get(&gets).unwrap_or_default()
    }

    // TODO: Implement with GC
    // Helper to route batch Frees through here to bump metrics in one place.
    // fn do_frees(&self, keys: &[PhysicalKey]) {
    //     if keys.is_empty() {
    //         return;
    //     }
    //     self.io_batches.fetch_add(1, Ordering::Relaxed);
    //     self.io_free_ops.fetch_add(keys.len(), Ordering::Relaxed);
    //     let _ = self.pager.free_many(keys);
    // }

    /// Access current metrics (counts of batch/ops).
    pub fn io_stats(&self) -> metrics::IoStats {
        metrics::IoStats {
            batches: self.io_batches.load(Ordering::Relaxed),
            get_raw_ops: self.io_get_raw_ops.load(Ordering::Relaxed),
            get_typed_ops: self.io_get_typed_ops.load(Ordering::Relaxed),
            put_raw_ops: self.io_put_raw_ops.load(Ordering::Relaxed),
            put_typed_ops: self.io_put_typed_ops.load(Ordering::Relaxed),
            free_ops: self.io_free_ops.load(Ordering::Relaxed),
        }
    }

    /// Reset metrics to zero.
    pub fn reset_io_stats(&self) {
        self.io_batches.store(0, Ordering::Relaxed);
        self.io_get_raw_ops.store(0, Ordering::Relaxed);
        self.io_get_typed_ops.store(0, Ordering::Relaxed);
        self.io_put_raw_ops.store(0, Ordering::Relaxed);
        self.io_put_typed_ops.store(0, Ordering::Relaxed);
        self.io_free_ops.store(0, Ordering::Relaxed);
    }
}

/* ------------------------- Columnar (u64 prototype) ------------------------- */
use crate::column_store::columnar::{write_u64_chunk, get_chunk_blob, ChunkHeader};

impl<P: Pager> ColumnStore<P> {
    /// Append a dense u64 chunk (columnar, little-endian stripe) for `field_id`.
    /// Returns the PhysicalKey of the chunk.
    pub fn append_u64_chunk(&self, field_id: LogicalFieldId, values: &[u64]) -> PhysicalKey {
        let pk = self.pager.alloc_many(1).expect("alloc pk")[0];
        write_u64_chunk(self.pager.as_ref(), pk, values, 16_384, 0).expect("write u64 chunk");
        {
            let mut m = self.columnar_chunks_u64.write().unwrap();
            m.entry(field_id).or_default().push(pk);
        }
        pk
    }

    /// Iterator over &[u64] vectors for a column's chunks (u64 prototype path).
    pub fn scan_u64_vectors<'a>(&'a self, field_id: LogicalFieldId) -> U64VectorScan<'a, P> {
        let keys = {
            let m = self.columnar_chunks_u64.read().unwrap();
            m.get(&field_id).cloned().unwrap_or_default()
        };
        U64VectorScan::new(&self.pager, keys)
    }

    /// Iterator yielding full contiguous u64 stripes per chunk (fastest path).
    /// Returns one `(Blob, Range)` per chunk covering the entire value stripe.
    pub fn scan_u64_full_chunks<'a>(&'a self, field_id: LogicalFieldId) -> U64ChunkScan<'a, P> {
        let keys = {
            let m = self.columnar_chunks_u64.read().unwrap();
            m.get(&field_id).cloned().unwrap_or_default()
        };
        U64ChunkScan::new(&self.pager, keys)
    }

    /// Iterator over u64 vectors with options: bounds, direction, vector_len override.
    pub fn scan_u64_with<'a>(&'a self, field_id: LogicalFieldId, opts: ScanOptions) -> U64Scan<'a, P> {
        let mut keys = {
            let m = self.columnar_chunks_u64.read().unwrap();
            m.get(&field_id).cloned().unwrap_or_default()
        };
        if let Direction::Desc = opts.direction { keys.reverse(); }
        U64Scan::new(&self.pager, keys, opts)
    }
}

pub struct U64VectorScan<'a, P: Pager> {
    pager: &'a Arc<P>,
    keys: Vec<PhysicalKey>,
    cur_idx: usize,
    cur_blob: Option<P::Blob>,
    cur_off_base: usize,
    cur_total_bytes: usize,
    cur_vector_len: usize,
    off_bytes: usize,
}

impl<'a, P: Pager> U64VectorScan<'a, P> {
    fn new(pager: &'a Arc<P>, keys: Vec<PhysicalKey>) -> Self {
        Self { pager, keys, cur_idx: 0, cur_blob: None, cur_off_base: 0, cur_total_bytes: 0, cur_vector_len: 0, off_bytes: 0 }
    }
}

impl<'a, P: Pager> Iterator for U64VectorScan<'a, P> {
    type Item = (P::Blob, core::ops::Range<usize>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(blob) = &self.cur_blob {
                if self.off_bytes < self.cur_total_bytes {
                    let take_vals = core::cmp::min(self.cur_vector_len, (self.cur_total_bytes - self.off_bytes) / 8);
                    let start = self.cur_off_base + self.off_bytes;
                    let end = start + take_vals * 8;
                    self.off_bytes += take_vals * 8;
                    return Some((blob.clone(), start..end));
                }
                // advance to next chunk
                self.cur_blob = None;
                self.off_bytes = 0;
            }
            if self.cur_idx >= self.keys.len() { return None; }
            let key = self.keys[self.cur_idx];
            self.cur_idx += 1;
            let blob = get_chunk_blob(self.pager.as_ref(), key)?;
            let bytes = blob.as_ref();
            let (hdr, off) = ChunkHeader::decode(bytes)?;
            if hdr.kind != 3 || hdr.width != 8 || hdr.endian != 1 { return None; }
            let need = hdr.row_count as usize * 8;
            if bytes.len() < off + need { return None; }
            self.cur_blob = Some(blob);
            self.cur_off_base = off;
            self.cur_total_bytes = need;
            self.cur_vector_len = hdr.vector_len as usize;
            self.off_bytes = 0;
        }
    }
}

/// Option-driven scanner over u64 stripes supporting direction and bounds.
pub struct U64Scan<'a, P: Pager> {
    pager: &'a Arc<P>,
    keys: Vec<PhysicalKey>,
    opts: ScanOptions,
    cur_idx: usize,
    cur_blob: Option<P::Blob>,
    cur_off_base: usize,
    in_chunk_start_bytes: usize,
    in_chunk_end_bytes: usize,
    cur_vector_len: usize,
    next_off: usize,
}

impl<'a, P: Pager> U64Scan<'a, P> {
    fn new(pager: &'a Arc<P>, keys: Vec<PhysicalKey>, opts: ScanOptions) -> Self {
        Self {
            pager,
            keys,
            opts,
            cur_idx: 0,
            cur_blob: None,
            cur_off_base: 0,
            in_chunk_start_bytes: 0,
            in_chunk_end_bytes: 0,
            cur_vector_len: 0,
            next_off: 0,
        }
    }
}

impl<'a, P: Pager> Iterator for U64Scan<'a, P> {
    type Item = (P::Blob, core::ops::Range<usize>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(blob) = &self.cur_blob {
                // Produce next vector slice according to direction within [start,end)
                if self.in_chunk_start_bytes < self.in_chunk_end_bytes {
                    let vlen_vals = if self.cur_vector_len > 0 { self.cur_vector_len } else { 16_384 };
                    let step_bytes = vlen_vals * 8;
                    match self.opts.direction {
                        Direction::Asc => {
                            if self.next_off < self.in_chunk_end_bytes {
                                let start = self.next_off;
                                let end = core::cmp::min(self.in_chunk_end_bytes, start + step_bytes);
                                self.next_off = end;
                                return Some((blob.clone(), start..end));
                            }
                        }
                        Direction::Desc => {
                            if self.next_off > self.in_chunk_start_bytes {
                                let end = self.next_off;
                                let start = end.saturating_sub(step_bytes).max(self.in_chunk_start_bytes);
                                self.next_off = start;
                                return Some((blob.clone(), start..end));
                            }
                        }
                    }
                }
                // advance to next chunk
                self.cur_blob = None;
            }

            if self.cur_idx >= self.keys.len() { return None; }
            let key = self.keys[self.cur_idx];
            self.cur_idx += 1;
            let blob = super::columnar::get_chunk_blob(self.pager.as_ref(), key)?;
            let bytes = blob.as_ref();
            let (hdr, off) = super::columnar::ChunkHeader::decode(bytes)?;
            if hdr.kind != 3 || hdr.width != 8 || hdr.endian != 1 { return None; }
            let total_bytes = hdr.row_count as usize * 8;
            if bytes.len() < off + total_bytes { return None; }

            // Bounds handling
            let (start_row, end_row) = match self.opts.bounds {
                Bounds::None => (0usize, hdr.row_count as usize),
                Bounds::Rows { start_row, end_row } => {
                    let s = start_row.unwrap_or(0).min(hdr.row_count as usize);
                    let e = end_row.unwrap_or(hdr.row_count as usize).min(hdr.row_count as usize);
                    (s.min(e), e)
                }
                Bounds::Values { min, max } => {
                    // prune whole chunk if disjoint
                    if let Some(mi) = min { if hdr.max_val_u64 < mi { continue; } }
                    if let Some(ma) = max { if hdr.min_val_u64 > ma { continue; } }
                    (0usize, hdr.row_count as usize)
                }
            };

            self.cur_blob = Some(blob);
            self.cur_off_base = off;
            self.in_chunk_start_bytes = off + start_row * 8;
            self.in_chunk_end_bytes = off + end_row * 8;
            self.cur_vector_len = if self.opts.vector_len == 0 { hdr.vector_len as usize } else { self.opts.vector_len };
            self.next_off = match self.opts.direction { Direction::Asc => self.in_chunk_start_bytes, Direction::Desc => self.in_chunk_end_bytes };
        }
    }
}

/// Full-chunk scanner: yields one full stripe per chunk as a zero-copy slice range.
pub struct U64ChunkScan<'a, P: Pager> {
    pager: &'a Arc<P>,
    keys: Vec<PhysicalKey>,
    cur_idx: usize,
}

impl<'a, P: Pager> U64ChunkScan<'a, P> {
    fn new(pager: &'a Arc<P>, keys: Vec<PhysicalKey>) -> Self {
        Self { pager, keys, cur_idx: 0 }
    }
}

impl<'a, P: Pager> Iterator for U64ChunkScan<'a, P> {
    type Item = (P::Blob, core::ops::Range<usize>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_idx >= self.keys.len() { return None; }
        let key = self.keys[self.cur_idx];
        self.cur_idx += 1;
        let blob = super::columnar::get_chunk_blob(self.pager.as_ref(), key)?;
        let bytes = blob.as_ref();
        let (hdr, off) = super::columnar::ChunkHeader::decode(bytes)?;
        if hdr.kind != 3 || hdr.width != 8 || hdr.endian != 1 { return None; }
        let need = hdr.row_count as usize * 8;
        if bytes.len() < off + need { return None; }
        Some((blob, off..off + need))
    }
}


// ----------------------------- tests -----------------------------

#[cfg(all(test, feature = "legacy_api"))]
mod tests {
    use super::*;

    // Use the unified in-memory pager from the pager module.
    use crate::bounds::ValueBound;
    use crate::codecs::big_endian::{u64_be_array, u64_be_vec};
    use crate::column_index::IndexSegmentRef;
    use crate::storage::pager::MemPager;
    use crate::types::{AppendOptions, Put, ValueMode};
    use std::borrow::Cow;

    #[test]
    fn put_get_fixed_auto() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);

        let put = Put {
            field_id: 10,
            items: vec![
                (b"k3".into(), b"VVVVVVV3".into()),
                (b"k1".into(), b"VVVVVVV1".into()),
                (b"k2".into(), b"VVVVVVV2".into()),
                // duplicate key, last wins
                (b"k1".into(), b"NEWVVVV1".into()),
            ],
        };

        store.append_many(vec![put], AppendOptions::default());

        // batch get across columns (here only field 10)
        let got = store.get_many(vec![(10, vec![b"k1".into(), b"k2".into(), b"kX".into()])]);

        // got[0] corresponds to field 10's query vector
        assert_eq!(got[0][0].as_deref().unwrap(), b"NEWVVVV1");
        assert_eq!(got[0][1].as_deref().unwrap(), b"VVVVVVV2");
        assert!(got[0][2].is_none());

        // sanity: we should have issued at least one batch (append + get)
        let stats = store.io_stats();
        assert!(stats.batches >= 2);
        assert!(stats.put_raw_ops > 0 || stats.put_typed_ops > 0);
        assert!(stats.get_raw_ops > 0 || stats.get_typed_ops > 0);
    }

    #[test]
    fn put_get_variable_auto_with_chunking() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);

        // Values of different sizes force variable layout.
        let mut items = Vec::new();
        for i in 0..1000u32 {
            let k = format!("key{:04}", i).into_bytes().into();
            let v = vec![b'A' + (i % 26) as u8; (i % 17 + 1) as usize].into(); // 1..17
            items.push((k, v));
        }

        let opts = AppendOptions {
            mode: ValueMode::Auto,
            segment_max_entries: 200, // small to force multiple segments
            segment_max_bytes: 2_000, // also keep segments small
            last_write_wins_in_batch: true,
        };

        store.append_many(
            vec![Put {
                field_id: 77,
                items,
            }],
            opts,
        );

        // spot-check a few keys
        let q = vec![
            b"key0000".to_vec(),
            b"key0001".to_vec(),
            b"key0199".to_vec(),
            b"key0200".to_vec(),
            b"key0999".to_vec(),
            b"nope".to_vec(),
        ];

        store.reset_io_stats(); // isolate the get path
        let got = store.get_many(vec![(77, q.clone())]);

        // recompute expected lengths
        let expect_len = |s: &str| -> usize {
            let i: ByteWidth = s[3..].parse().unwrap();
            (i % 17 + 1) as usize
        };

        assert_eq!(got[0][0].as_deref().unwrap().len(), expect_len("key0000"));
        assert_eq!(got[0][1].as_deref().unwrap().len(), expect_len("key0001"));
        assert_eq!(got[0][2].as_deref().unwrap().len(), expect_len("key0199"));
        assert_eq!(got[0][3].as_deref().unwrap().len(), expect_len("key0200"));
        assert_eq!(got[0][4].as_deref().unwrap().len(), expect_len("key0999"));
        assert!(got[0][5].is_none());

        // sanity: get_many should have exactly 1 batch for segments+data,
        // plus 0 or 1 for initial ColumnIndex fills (depending on cache)
        let s = store.io_stats();
        assert!(s.batches >= 1);
        assert!(s.get_raw_ops > 0 || s.get_typed_ops > 0);
    }

    #[test]
    fn force_fixed_validation() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);
        let items = vec![
            (b"a".into(), vec![1u8; 4].into()),
            (b"b".into(), vec![2u8; 4].into()),
            (b"c".into(), vec![3u8; 4].into()),
        ];
        let opts = AppendOptions {
            mode: ValueMode::ForceFixed(4),
            ..Default::default()
        };
        store.append_many(vec![Put { field_id: 5, items }], opts);

        let got = store.get_many(vec![(5, vec![b"a".to_vec(), b"b".to_vec(), b"z".to_vec()])]);

        assert_eq!(got[0][0].as_deref().unwrap(), &[1u8; 4]);
        assert_eq!(got[0][1].as_deref().unwrap(), &[2u8; 4]);
        assert!(got[0][2].is_none());
    }
    #[test]
    fn last_write_wins_true_dedups_within_batch() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);

        // Three items, with a duplicate for key "k": last is "ZZ".
        let fid = 42;
        let put = Put {
            field_id: fid,
            items: vec![
                (b"k".into(), b"AA".into()),
                (b"x".into(), b"BB".into()),
                (b"k".into(), b"ZZ".into()),
            ],
        };

        let opts = AppendOptions {
            mode: ValueMode::ForceFixed(2), // keep sizes simple
            segment_max_entries: 1024,      // ensure a single segment
            last_write_wins_in_batch: true, // <- dedup ON
            ..Default::default()
        };

        store.append_many(vec![put], opts);

        // Read back the two keys we wrote; "k" should be the LAST value ("ZZ").
        let got = store.get_many(vec![(fid, vec![b"k".into(), b"x".into()])]);
        assert_eq!(got[0][0].as_deref().unwrap(), b"ZZ");
        assert_eq!(got[0][1].as_deref().unwrap(), b"BB");

        // Verify the segment entry count reflects deduplication (2 items).
        let total_entries_for_fid: usize = store
            .describe_storage()
            .into_iter()
            .map(|n| match n.kind {
                StorageKind::IndexSegment {
                    field_id,
                    n_entries,
                    ..
                } if field_id == fid => n_entries as usize,
                _ => 0,
            })
            .sum();
        assert_eq!(total_entries_for_fid, 2);
    }

    #[test]
    fn last_write_wins_false_keeps_dups_within_batch() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);

        // Same three items; duplicates for "k" are KEPT.
        let fid = 43;
        let put = Put {
            field_id: fid,
            items: vec![
                (b"k".into(), b"AA".into()),
                (b"x".into(), b"BB".into()),
                (b"k".into(), b"ZZ".into()),
            ],
        };

        let opts = AppendOptions {
            mode: ValueMode::ForceFixed(2),
            segment_max_entries: 1024,
            last_write_wins_in_batch: false, // <- dedup OFF
            ..Default::default()
        };

        store.append_many(vec![put], opts);

        // Reads for "k" will return one of the duplicate values.
        // (Order among equal keys after sort is not guaranteed.)
        let got = store.get_many(vec![(fid, vec![b"k".into(), b"x".into()])]);
        let v_k = got[0][0].as_deref().unwrap();
        assert!(
            v_k == b"AA" || v_k == b"ZZ",
            "expected one of the duplicate values, got {:?}",
            v_k
        );
        assert_eq!(got[0][1].as_deref().unwrap(), b"BB");

        // Verify the segment entry count reflects the duplicates (3 items).
        let total_entries_for_fid: usize = store
            .describe_storage()
            .into_iter()
            .map(|n| match n.kind {
                StorageKind::IndexSegment {
                    field_id,
                    n_entries,
                    ..
                } if field_id == fid => n_entries as usize,
                _ => 0,
            })
            .sum();
        assert_eq!(total_entries_for_fid, 3);
    }

    #[test]
    fn key_based_segment_pruning_uses_min_max_and_is_inclusive() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);
        let fid = 1234;

        // Two disjoint segments for same column:
        //   seg A covers [0..100)      (keys 0..99)
        //   seg B covers [200..300)    (keys 200..299)
        // Make values fixed 4B to keep things simple.

        // Build segment A
        let items_a = {
            const BB4: &[u8] = &[0xAA, 0, 0, 0];
            (0u64..100u64)
                .map(|k| (Cow::Owned(u64_be_array(k).to_vec()), Cow::Borrowed(BB4)))
                .collect()
        };

        store.append_many(
            vec![Put {
                field_id: fid,
                items: items_a,
            }],
            AppendOptions {
                mode: ValueMode::ForceFixed(4),
                segment_max_entries: 10_000,
                segment_max_bytes: 1_000_000,
                last_write_wins_in_batch: true,
            },
        );

        // Build segment B (newest)

        let items_b = {
            const BB4: &[u8] = &[0xBB, 0, 0, 0];
            (200u64..300u64)
                .map(|k| (Cow::Owned(u64_be_array(k).to_vec()), Cow::Borrowed(BB4)))
                .collect()
        };

        store.append_many(
            vec![Put {
                field_id: fid,
                items: items_b,
            }],
            AppendOptions {
                mode: ValueMode::ForceFixed(4),
                segment_max_entries: 10_000,
                segment_max_bytes: 1_000_000,
                last_write_wins_in_batch: true,
            },
        );

        // ------------- Query hits ONLY seg B ----------------
        store.reset_io_stats();
        let got = store.get_many(vec![(fid, vec![u64_be_vec(250)])]);
        assert_eq!(got[0][0].as_deref().unwrap(), &[0xBB, 0, 0, 0]);

        // Because of pruning, we should fetch exactly 1 IndexSegment (typed) and 1 data blob (raw)
        let s = store.io_stats();
        assert_eq!(
            s.get_typed_ops, 1,
            "should load only the matching index segment"
        );
        assert_eq!(s.get_raw_ops, 1, "should load only one data blob");

        // ------------- Inclusivity: min & max are hits -------
        store.reset_io_stats();
        let got = store.get_many(vec![(fid, vec![u64_be_vec(200), u64_be_vec(299)])]);
        assert_eq!(got[0][0].as_deref().unwrap(), &[0xBB, 0, 0, 0]); // min
        assert_eq!(got[0][1].as_deref().unwrap(), &[0xBB, 0, 0, 0]); // max
        let s = store.io_stats();
        assert_eq!(s.get_typed_ops, 1);
        assert_eq!(s.get_raw_ops, 1);

        // ------------- Query hits ONLY seg A ----------------
        store.reset_io_stats();
        let got = store.get_many(vec![(fid, vec![u64_be_vec(5)])]);
        assert_eq!(got[0][0].as_deref().unwrap(), &[0xAA, 0, 0, 0]);
        let s = store.io_stats();
        assert_eq!(s.get_typed_ops, 1);
        assert_eq!(s.get_raw_ops, 1);
    }

    #[test]
    fn value_min_max_recorded_fixed() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);
        let fid = 5050;

        // fixed 4B values with easy min/max
        let items = vec![
            (b"k1".into(), vec![9, 9, 9, 9].into()),
            (b"k2".into(), vec![0, 0, 0, 0].into()), // min
            (b"k3".into(), vec![255, 255, 255, 1].into()), // max (lexicographic)
        ];

        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            AppendOptions {
                mode: ValueMode::ForceFixed(4),
                ..Default::default()
            },
        );

        // Clone the newest ref while the lock is held
        let sref: IndexSegmentRef = {
            let ci = store.colindex_cache.read().unwrap();
            let (_pk, colidx) = ci.get(&fid).expect("colindex");
            colidx.segments[0].clone()
        };

        assert_eq!(sref.value_min, Some(ValueBound::from_bytes(&[0, 0, 0, 0])));
        assert_eq!(
            sref.value_max,
            Some(ValueBound::from_bytes(&[255, 255, 255, 1]))
        );
    }

    #[test]
    fn value_min_max_recorded_variable() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);
        let fid = 6060;

        // variable-length values; lexicographic min/max over raw bytes
        let items = vec![
            (b"a".into(), b"wolf".into()),
            (b"b".into(), b"ant".into()), // min ("ant" < "wolf" < "zebra")
            (b"c".into(), b"zebra".into()), // max
        ];

        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            AppendOptions {
                mode: ValueMode::ForceVariable,
                ..Default::default()
            },
        );

        // Clone the newest ref while the lock is held
        let sref: IndexSegmentRef = {
            let ci = store.colindex_cache.read().unwrap();
            let (_pk, colidx) = ci.get(&fid).expect("colindex");
            colidx.segments[0].clone()
        };

        assert_eq!(sref.value_min, Some(ValueBound::from_bytes(b"ant")));
        assert_eq!(sref.value_max, Some(ValueBound::from_bytes(b"zebra")));
    }

    #[test]
    fn value_bounds_do_not_duplicate_huge_value() {
        use crate::constants::VALUE_BOUND_MAX;

        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);
        let fid = 7070;

        // Single huge value (1 MiB)
        let huge = vec![42u8; 1_048_576];
        store.append_many(
            vec![Put {
                field_id: fid,
                items: vec![(b"k".into(), huge.into())],
            }],
            AppendOptions {
                mode: ValueMode::ForceVariable,
                segment_max_entries: 1_000_000,
                segment_max_bytes: usize::MAX,
                last_write_wins_in_batch: true,
            },
        );

        // Take ownership of the newest ref and seg pk while locked
        let (seg_pk, sref): (PhysicalKey, IndexSegmentRef) = {
            let ci = store.colindex_cache.read().unwrap();
            let (_pk, colidx) = ci.get(&fid).expect("colindex");
            let sref = colidx.segments[0].clone();
            (sref.index_physical_key, sref)
        };

        // both min and max are the same single huge value
        let vmin = sref.value_min.as_ref().expect("value_min");
        let vmax = sref.value_max.as_ref().expect("value_max");

        assert_eq!(vmin.total_len, 1_048_576);
        assert_eq!(vmax.total_len, 1_048_576);
        assert!(vmin.is_truncated());
        assert!(vmax.is_truncated());
        assert_eq!(vmin.prefix.len(), VALUE_BOUND_MAX);
        assert_eq!(vmax.prefix.len(), VALUE_BOUND_MAX);
        assert_eq!(vmin.prefix, vmax.prefix);

        // prove the index-segment blob on disk is tiny vs the 1 MiB data blob
        let nodes = store.describe_storage();
        let seg_node_size = nodes
            .into_iter()
            .find(|n| n.pk == seg_pk)
            .expect("segment node")
            .stored_len;

        assert!(
            seg_node_size < 32 * 1024,
            "segment blob unexpectedly large: {}",
            seg_node_size
        );
    }

    #[test]
    fn delete_many_and_get() {
        let p = Arc::new(MemPager::default());
        let store = ColumnStore::open(p);
        let fid = 101;

        // 1. Write some initial data
        let put = Put {
            field_id: fid,
            items: vec![
                (b"k1".into(), b"val1".into()),
                (b"k2".into(), b"val2".into()),
                (b"k3".into(), b"val3".into()),
            ],
        };
        store.append_many(vec![put], AppendOptions::default());

        // 2. Verify initial data
        let got1 = store.get_many(vec![(
            fid,
            vec![b"k1".to_vec(), b"k2".to_vec(), b"k3".to_vec()],
        )]);
        assert_eq!(got1[0][0].as_deref().unwrap(), b"val1");
        assert_eq!(got1[0][1].as_deref().unwrap(), b"val2");
        assert_eq!(got1[0][2].as_deref().unwrap(), b"val3");

        // 3. Delete k2 and a non-existent key k4
        store.delete_many(vec![(fid, vec![b"k2".to_vec(), b"k4".to_vec()])]);

        // 4. Verify deletion
        let got2 = store.get_many(vec![(
            fid,
            vec![b"k1".to_vec(), b"k2".to_vec(), b"k3".to_vec()],
        )]);
        assert_eq!(
            got2[0][0].as_deref().unwrap(),
            b"val1",
            "k1 should still exist"
        );
        assert!(got2[0][1].is_none(), "k2 should be deleted");
        assert_eq!(
            got2[0][2].as_deref().unwrap(),
            b"val3",
            "k3 should still exist"
        );

        // 5. Appending a new value for k2 should bring it back
        let put2 = Put {
            field_id: fid,
            items: vec![(b"k2".into(), b"new_val2".into())],
        };
        store.append_many(vec![put2], AppendOptions::default());

        let got3 = store.get_many(vec![(fid, vec![b"k1".to_vec(), b"k2".to_vec()])]);
        assert_eq!(got3[0][0].as_deref().unwrap(), b"val1");
        assert_eq!(
            got3[0][1].as_deref().unwrap(),
            b"new_val2",
            "k2 should have new value"
        );
    }
}
