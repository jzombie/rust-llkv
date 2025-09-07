// High-level append/query API on top of pager + index modules.
use crate::index::{
    Bootstrap, ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef, Manifest, ValueBound,
};
use crate::layout::{IndexLayoutInfo, KeyLayout, ValueLayout};
use crate::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::types::{
    ByteLen, ByteOffset, ByteWidth, IndexEntryCount, LogicalFieldId, LogicalKeyBytes, PhysicalKey,
    TypedKind, TypedValue,
};
use crate::views::ValueSlice;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;
use std::fmt::Write;
use std::sync::{
    RwLock,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Debug)]
pub struct StorageNode {
    pub pk: PhysicalKey,
    pub stored_len: usize,
    pub kind: StorageKind,
}

#[derive(Clone, Debug)]
pub enum StorageKind {
    Bootstrap,
    Manifest {
        column_count: usize,
    },
    ColumnIndex {
        field_id: LogicalFieldId,
        n_segments: usize,
    },
    IndexSegment {
        field_id: LogicalFieldId,
        n_entries: IndexEntryCount,
        layout: IndexLayoutInfo,
        data_pkey: PhysicalKey,
        owner_colindex_pk: PhysicalKey,
    },
    DataBlob {
        owner_index_pk: PhysicalKey,
    },
}

// ----------------------------- write API -----------------------------

pub struct Put {
    pub field_id: LogicalFieldId,
    pub items: Vec<(LogicalKeyBytes, Vec<u8>)>, // unordered; duplicates allowed (last wins)
}

#[derive(Clone, Copy, Debug)]
pub enum ValueMode {
    Auto,
    ForceFixed(ByteWidth),
    ForceVariable,
}

#[derive(Clone, Debug)]
pub struct AppendOptions {
    pub mode: ValueMode,
    pub segment_max_entries: usize,
    pub segment_max_bytes: usize, // data payload budget per segment
    pub last_write_wins_in_batch: bool,
}

impl Default for AppendOptions {
    fn default() -> Self {
        Self {
            mode: ValueMode::Auto,
            segment_max_entries: 65_536,
            segment_max_bytes: 8 * 1024 * 1024,
            last_write_wins_in_batch: true,
        }
    }
}

// ----------------------------- metrics -----------------------------

/// Minimal pager-hit metrics collected inside ColumnStore. This counts how
/// many *requests* we send to the pager (not bytes). If you need byte totals,
/// you can extend this to compute sizes for Raw puts/gets and (optionally)
/// encoded sizes for Typed values.
#[derive(Clone, Debug, Default)]
pub struct IoStats {
    pub batches: usize,       // number of times we called Pager::batch_* (get/put)
    pub get_raw_ops: usize,   // number of Raw gets requested
    pub get_typed_ops: usize, // number of Typed gets requested
    pub put_raw_ops: usize,   // number of Raw puts requested
    pub put_typed_ops: usize, // number of Typed puts requested
    pub free_ops: usize,      // number of physical keys freed
}

// ----------------------------- ColumnStore -----------------------------

pub struct ColumnStore<'p, P: Pager> {
    pager: &'p P,
    bootstrap_key: PhysicalKey,
    manifest_key: PhysicalKey,
    // Interior mutability for concurrent reads/writes.
    manifest: RwLock<Manifest>,
    // field_id -> (column_index_pkey, decoded)
    colindex_cache: RwLock<FxHashMap<LogicalFieldId, (PhysicalKey, ColumnIndex)>>,
    // pager-hit metrics (counts only) via atomics
    io_batches: AtomicUsize,
    io_get_raw_ops: AtomicUsize,
    io_get_typed_ops: AtomicUsize,
    io_put_raw_ops: AtomicUsize,
    io_put_typed_ops: AtomicUsize,
    io_free_ops: AtomicUsize,
}

impl<'p, P: Pager> ColumnStore<'p, P> {
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

    // Helper to route batch Frees through here to bump metrics in one place.
    fn do_frees(&self, keys: &[PhysicalKey]) {
        if keys.is_empty() {
            return;
        }
        self.io_batches.fetch_add(1, Ordering::Relaxed);
        self.io_free_ops.fetch_add(keys.len(), Ordering::Relaxed);
        let _ = self.pager.free_many(keys);
    }

    /// Access current metrics (counts of batch/ops).
    pub fn io_stats(&self) -> IoStats {
        IoStats {
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

    // Create fresh store (bootstrap->manifest, empty manifest).
    pub fn init_empty(pager: &'p P) -> Self {
        let bootstrap_key: PhysicalKey = 0;
        let manifest_key = pager.alloc_many(1).unwrap()[0];
        let manifest = Manifest {
            columns: Vec::new(),
        };

        // Write Manifest and Bootstrap in a single batch.
        // (Not counted in ColumnStore metrics since the store isn't constructed yet.)
        let _ = pager.batch_put(&[
            BatchPut::Typed {
                key: manifest_key,
                value: TypedValue::Manifest(manifest.clone()),
            },
            BatchPut::Typed {
                key: bootstrap_key,
                value: TypedValue::Bootstrap(Bootstrap {
                    manifest_physical_key: manifest_key,
                }),
            },
        ]);

        ColumnStore {
            pager,
            bootstrap_key,
            manifest_key,
            manifest: RwLock::new(manifest),
            colindex_cache: RwLock::new(FxHashMap::with_hasher(Default::default())),
            io_batches: AtomicUsize::new(0),
            io_get_raw_ops: AtomicUsize::new(0),
            io_get_typed_ops: AtomicUsize::new(0),
            io_put_raw_ops: AtomicUsize::new(0),
            io_put_typed_ops: AtomicUsize::new(0),
            io_free_ops: AtomicUsize::new(0),
        }
    }

    // Open existing store (bootstrap(0) -> manifest).
    pub fn open(pager: &'p P) -> Self {
        let bootstrap_key: PhysicalKey = 0;

        // Get Bootstrap (not counted; ColumnStore not constructed yet)
        let resp = pager
            .batch_get(&[BatchGet::Typed {
                key: bootstrap_key,
                kind: TypedKind::Bootstrap,
            }])
            .unwrap_or_default();
        let boot = match &resp[0] {
            GetResult::Typed {
                value: TypedValue::Bootstrap(b),
                ..
            } => b.clone(),
            _ => panic!("missing bootstrap at key 0"),
        };
        let manifest_key = boot.manifest_physical_key;

        // Get Manifest (not counted; ColumnStore not constructed yet)
        let resp = pager
            .batch_get(&[BatchGet::Typed {
                key: manifest_key,
                kind: TypedKind::Manifest,
            }])
            .unwrap_or_default();
        let manifest = match &resp[0] {
            GetResult::Typed {
                value: TypedValue::Manifest(m),
                ..
            } => m.clone(),
            _ => panic!("missing manifest"),
        };

        ColumnStore {
            pager,
            bootstrap_key,
            manifest_key,
            manifest: RwLock::new(manifest),
            colindex_cache: RwLock::new(FxHashMap::with_hasher(Default::default())),
            io_batches: AtomicUsize::new(0),
            io_get_raw_ops: AtomicUsize::new(0),
            io_get_typed_ops: AtomicUsize::new(0),
            io_put_raw_ops: AtomicUsize::new(0),
            io_put_typed_ops: AtomicUsize::new(0),
            io_free_ops: AtomicUsize::new(0),
        }
    }

    // Single entrypoint for writing. Many columns, each with unordered items.
    // Auto-chooses fixed vs variable, chunks to segments, writes everything in batches.
    pub fn append_many(&self, puts: Vec<Put>, opts: AppendOptions) {
        if puts.is_empty() {
            return;
        }

        // Normalize per column: last-write-wins (optional), then sort by logical key.
        // Also decide value layout for this batch (fixed width or variable).
        #[derive(Clone)]
        enum PlannedLayout {
            Fixed { width: ByteWidth },
            Variable, // offsets will be computed per segment
        }
        struct PlannedChunk {
            field_id: LogicalFieldId,
            keys_sorted: Vec<LogicalKeyBytes>,
            values: Vec<Vec<u8>>, // aligned to keys
            layout: PlannedLayout,
        }

        let mut planned_chunks: Vec<PlannedChunk> = Vec::new();

        for mut put in puts {
            if put.items.is_empty() {
                continue;
            }

            if opts.last_write_wins_in_batch {
                // last wins: overwrite by key
                let mut last: FxHashMap<LogicalKeyBytes, Vec<u8>> =
                    FxHashMap::with_capacity_and_hasher(put.items.len(), Default::default());
                for (k, v) in put.items {
                    last.insert(k, v);
                }
                put.items = last.into_iter().collect();
            }

            // sort by key
            put.items.sort_by(|a, b| a.0.cmp(&b.0));

            // decide layout
            let layout = match opts.mode {
                ValueMode::ForceFixed(w) => {
                    assert!(w > 0, "fixed width must be > 0");
                    for (_, v) in &put.items {
                        assert!(
                            v.len() == w as usize,
                            "ForceFixed width mismatch: expected {}, got {}",
                            w,
                            v.len()
                        );
                    }
                    PlannedLayout::Fixed { width: w }
                }
                ValueMode::ForceVariable => PlannedLayout::Variable,
                ValueMode::Auto => {
                    let w = put.items[0].1.len();
                    if w > 0 && put.items.iter().all(|(_, v)| v.len() == w) {
                        PlannedLayout::Fixed {
                            width: w as ByteWidth,
                        }
                    } else {
                        PlannedLayout::Variable
                    }
                }
            };

            // chunk by thresholds
            let mut i = 0usize;
            while i < put.items.len() {
                let remain = put.items.len() - i;
                let take_by_entries = core::cmp::min(remain, opts.segment_max_entries);

                let take = match layout {
                    PlannedLayout::Fixed { width } => {
                        // entries are uniform; respect both entry and byte budgets
                        let max_entries_by_bytes = if width == 0 {
                            take_by_entries
                        } else {
                            (opts.segment_max_bytes / (width as usize)).max(1)
                        };
                        core::cmp::min(take_by_entries, max_entries_by_bytes.max(1))
                    }
                    PlannedLayout::Variable => {
                        // grow until adding next would exceed bytes or entries
                        let mut acc = 0usize;
                        let mut cnt = 0usize;
                        for j in i..(i + take_by_entries) {
                            let sz = put.items[j].1.len();
                            if cnt > 0 && acc + sz > opts.segment_max_bytes {
                                break;
                            }
                            acc += sz;
                            cnt += 1;
                        }
                        if cnt == 0 {
                            // single huge value; take 1 anyway
                            1
                        } else {
                            cnt
                        }
                    }
                };

                let end = i + take;
                let mut keys_sorted = Vec::with_capacity(take);
                let mut vals = Vec::with_capacity(take);
                for (k, v) in put.items[i..end].iter() {
                    keys_sorted.push(k.clone());
                    vals.push(v.clone());
                }

                planned_chunks.push(PlannedChunk {
                    field_id: put.field_id,
                    keys_sorted,
                    values: vals,
                    layout: layout.clone(),
                });

                i = end;
            }
        }

        if planned_chunks.is_empty() {
            return;
        }

        // Allocate physical keys for all data blobs and index blobs in one go.
        let n_segs = planned_chunks.len();
        let mut pkeys = self.pager.alloc_many(n_segs * 2).unwrap();
        // first half used for data blobs, second half for index blobs
        let data_keys: Vec<PhysicalKey> = pkeys.drain(0..n_segs).collect();
        let index_keys: Vec<PhysicalKey> = pkeys.drain(0..n_segs).collect();

        // Build concrete IndexSegments and data blobs; group by column to update ColumnIndex.
        let mut data_puts: Vec<(PhysicalKey, Vec<u8>)> = Vec::with_capacity(n_segs);
        let mut index_puts: Vec<(PhysicalKey, IndexSegment)> = Vec::with_capacity(n_segs);

        // Prepare map: field_id -> (col_index_pkey, ColumnIndex)
        let mut need_manifest_update = false;
        let mut ensure_column_loaded: FxHashSet<LogicalFieldId> = FxHashSet::default();
        // track which ColumnIndex pkeys were actually modified this call.
        let mut touched_colindex_pkeys: FxHashSet<PhysicalKey> = FxHashSet::default();

        // Bring column indexes we will touch into cache (batch-read where possible)
        for chunk in &planned_chunks {
            ensure_column_loaded.insert(chunk.field_id);
        }

        let missing: Vec<LogicalFieldId> = {
            let cache = self.colindex_cache.read().unwrap();
            ensure_column_loaded
                .iter()
                .filter(|fid| !cache.contains_key(fid))
                .copied()
                .collect()
        };

        if !missing.is_empty() {
            // find entries in manifest
            let lookups: Vec<(LogicalFieldId, PhysicalKey)> = {
                let man = self.manifest.read().unwrap();
                let mut v = Vec::new();
                for fid in &missing {
                    if let Some(entry) = man.columns.iter().find(|e| e.field_id == *fid) {
                        v.push((*fid, entry.column_index_physical_key));
                    }
                }
                v
            };

            if !lookups.is_empty() {
                let gets: Vec<BatchGet> = lookups
                    .iter()
                    .map(|(_, pk)| BatchGet::Typed {
                        key: *pk,
                        kind: TypedKind::ColumnIndex,
                    })
                    .collect();
                let resp = self.do_gets(gets);

                let mut got_vec: Vec<ColumnIndex> = Vec::with_capacity(lookups.len());
                for gr in resp {
                    match gr {
                        GetResult::Typed {
                            value: TypedValue::ColumnIndex(ci),
                            ..
                        } => got_vec.push(ci),
                        _ => panic!("ColumnIndex missing"),
                    }
                }

                let mut cache = self.colindex_cache.write().unwrap();
                for (i, (fid, k)) in lookups.into_iter().enumerate() {
                    cache.insert(fid, (k, got_vec[i].clone()));
                }
            }
        }

        // For any columns still missing, create fresh ColumnIndex and manifest entry.
        {
            let mut cache = self.colindex_cache.write().unwrap();
            let mut man = self.manifest.write().unwrap();

            for fid in &missing {
                if !cache.contains_key(fid) {
                    let col_index_pkey = self.pager.alloc_many(1).unwrap()[0];
                    let col_index = ColumnIndex {
                        field_id: *fid,
                        segments: Vec::new(),
                    };
                    cache.insert(*fid, (col_index_pkey, col_index));
                    man.columns.push(ColumnEntry {
                        field_id: *fid,
                        column_index_physical_key: col_index_pkey,
                    });
                    need_manifest_update = true;
                }
            }
        }

        // Now materialize each planned chunk into data blob + IndexSegment, and update ColumnIndex.
        for (i, chunk) in planned_chunks.into_iter().enumerate() {
            let data_pkey = data_keys[i];
            let index_pkey = index_keys[i];

            // compute value bounds once per segment
            let (vmin, vmax) = ValueBound::min_max_bounds(&chunk.values);

            // build data blob (+ value offsets for variable)
            let (data_blob, seg, n_entries) = match chunk.layout {
                PlannedLayout::Fixed { width } => {
                    // values blob
                    let mut blob = Vec::with_capacity(chunk.values.len() * (width as usize));
                    for v in &chunk.values {
                        debug_assert_eq!(v.len(), width as usize);
                        blob.extend_from_slice(v);
                    }
                    // index segment (keys packed + fixed value layout)
                    let seg =
                        IndexSegment::build_fixed(data_pkey, chunk.keys_sorted.clone(), width);
                    (blob, seg, chunk.values.len() as IndexEntryCount)
                }
                PlannedLayout::Variable => {
                    // values blob
                    let mut blob = Vec::new();
                    let mut sizes: Vec<ByteLen> = Vec::with_capacity(chunk.values.len());
                    for v in &chunk.values {
                        blob.extend_from_slice(v);
                        sizes.push(v.len() as ByteLen);
                    }
                    // index segment (keys packed + var value layout)
                    let seg = IndexSegment::build_var(data_pkey, chunk.keys_sorted.clone(), &sizes);
                    (blob, seg, chunk.values.len() as IndexEntryCount)
                }
            };

            // queue writes
            data_puts.push((data_pkey, data_blob));
            index_puts.push((index_pkey, seg.clone()));

            // derive logical key span from already-sorted keys (first/last)
            let logical_key_min = chunk
                .keys_sorted
                .first()
                .cloned()
                .expect("non-empty segment");
            let logical_key_max = chunk
                .keys_sorted
                .last()
                .cloned()
                .expect("non-empty segment");

            // update ColumnIndex (newest-first: push_front semantics)
            {
                let mut cache = self.colindex_cache.write().unwrap();
                let (col_pkey, col_index) = cache
                    .get_mut(&chunk.field_id)
                    .expect("column index present");

                col_index.segments.insert(
                    0,
                    IndexSegmentRef {
                        index_physical_key: index_pkey,
                        data_physical_key: seg.data_physical_key,
                        logical_key_min,
                        logical_key_max,
                        value_min: Some(vmin.clone()),
                        value_max: Some(vmax.clone()),
                        n_entries,
                    },
                );

                // mark this ColumnIndex as touched so we persist only these (ingest perf fix)
                touched_colindex_pkeys.insert(*col_pkey);
            }
        }

        // Persist: data blobs, index segments, column indexes, manifest (if needed).
        // Combine everything into a SINGLE batch.
        let mut puts_batch: Vec<BatchPut> = Vec::new();

        if !data_puts.is_empty() {
            for (k, v) in data_puts {
                puts_batch.push(BatchPut::Raw { key: k, bytes: v });
            }
        }
        if !index_puts.is_empty() {
            for (k, seg) in index_puts {
                puts_batch.push(BatchPut::Typed {
                    key: k,
                    value: TypedValue::IndexSegment(seg),
                });
            }
        }

        // Only write ColumnIndex blobs that were actually touched this call.
        if !touched_colindex_pkeys.is_empty() {
            let cache = self.colindex_cache.read().unwrap();
            for (_fid, (pk, ci)) in cache.iter() {
                if touched_colindex_pkeys.contains(pk) {
                    puts_batch.push(BatchPut::Typed {
                        key: *pk,
                        value: TypedValue::ColumnIndex(ci.clone()),
                    });
                }
            }
        }

        if need_manifest_update {
            let man = self.manifest.read().unwrap();
            puts_batch.push(BatchPut::Typed {
                key: self.manifest_key,
                value: TypedValue::Manifest(man.clone()),
            });
        }

        if !puts_batch.is_empty() {
            #[cfg(debug_assertions)]
            {
                let mut seen = FxHashSet::default();
                for p in &puts_batch {
                    let k = match p {
                        BatchPut::Raw { key, .. } | BatchPut::Typed { key, .. } => *key,
                    };
                    assert!(seen.insert(k), "duplicate PUT for key {}", k);
                }
            }
            self.do_puts(puts_batch);
        }
    }

    /// Logically delete keys by appending tombstones (zero-length values).
    /// We force variable layout so size=0 entries are valid.
    pub fn delete_many(&self, items: Vec<(LogicalFieldId, Vec<LogicalKeyBytes>)>) {
        if items.is_empty() {
            return;
        }

        // Convert to Put batches with empty value bytes.
        let puts: Vec<Put> = items
            .into_iter()
            .map(|(fid, keys)| Put {
                field_id: fid,
                items: keys.into_iter().map(|k| (k, Vec::<u8>::new())).collect(),
            })
            .collect();

        // Force variable so 0-byte values are allowed; keep LWW in-batch.
        let mut opts = AppendOptions::default();
        opts.mode = ValueMode::ForceVariable;
        opts.last_write_wins_in_batch = true;

        self.append_many(puts, opts);
    }

    // ----------------------------- read API -----------------------------

    /// Batched point lookups across many columns.
    /// Each item is (field_id, keys). Output is aligned per input.
    ///
    /// I/O pattern (whole batch):
    ///   1) batch(ColumnIndex gets) for any missing columns (once)
    ///   2) batch(IndexSegment typed + DataBlob raw gets) (once)
    ///
    /// Newest-first shadowing: for each key we pick the first segment whose
    /// [min,max] covers it.
    pub fn get_many(
        &self,
        items: Vec<(LogicalFieldId, Vec<LogicalKeyBytes>)>,
    ) -> Vec<Vec<Option<ValueSlice<P::Blob>>>> {
        if items.is_empty() {
            return Vec::new();
        }

        // Pre-size outputs: one vector per input, one slot per key.
        let mut results: Vec<Vec<Option<ValueSlice<P::Blob>>>> =
            items.iter().map(|(_, ks)| vec![None; ks.len()]).collect();

        // Track which queries have been resolved (found value or tombstone).
        // Key: (query_index, key_index)
        let mut resolved_queries: FxHashSet<(usize, usize)> = FxHashSet::default();

        // -------- ensure ColumnIndex in cache for all referenced fields --------
        let mut need_fids: FxHashSet<LogicalFieldId> = FxHashSet::default();
        for (fid, _) in &items {
            need_fids.insert(*fid);
        }

        let mut to_load: Vec<(LogicalFieldId, PhysicalKey)> = Vec::new();
        {
            let cache = self.colindex_cache.read().unwrap();
            let man = self.manifest.read().unwrap();

            for fid in need_fids {
                if cache.contains_key(&fid) {
                    continue;
                }
                if let Some(entry) = man.columns.iter().find(|e| e.field_id == fid) {
                    to_load.push((fid, entry.column_index_physical_key));
                }
            }
        }

        if !to_load.is_empty() {
            let gets: Vec<BatchGet> = to_load
                .iter()
                .map(|(_, pk)| BatchGet::Typed {
                    key: *pk,
                    kind: TypedKind::ColumnIndex,
                })
                .collect();
            let resp = self.do_gets(gets);

            let mut got_vec: Vec<ColumnIndex> = Vec::with_capacity(to_load.len());
            for gr in resp {
                match gr {
                    GetResult::Typed {
                        value: TypedValue::ColumnIndex(ci),
                        ..
                    } => got_vec.push(ci),
                    _ => panic!("ColumnIndex missing"),
                }
            }

            let mut cache = self.colindex_cache.write().unwrap();
            for (i, (fid, pk)) in to_load.into_iter().enumerate() {
                cache.insert(fid, (pk, got_vec[i].clone()));
            }
        }

        // -------- routing: segment -> (data_pk, [(item_i, key_j), ...]) --------
        // We also compute a "recency rank" per segment pk so we can process newest-first globally.
        let mut per_seg: FxHashMap<PhysicalKey, (PhysicalKey, Vec<(usize, usize)>)> =
            FxHashMap::default();
        let mut seg_rank: FxHashMap<PhysicalKey, usize> = FxHashMap::default();

        {
            let cache = self.colindex_cache.read().unwrap();

            for (qi, (fid, keys)) in items.iter().enumerate() {
                let col_index = match cache.get(fid) {
                    Some((_, ci)) => ci,
                    None => continue, // unknown field => all None for that entry
                };

                // Record recency ranks for this column's segments (0 == newest).
                for (rank, segref) in col_index.segments.iter().enumerate() {
                    seg_rank
                        .entry(segref.index_physical_key)
                        .and_modify(|r| *r = (*r).min(rank))
                        .or_insert(rank);
                }

                for (kj, k) in keys.iter().enumerate() {
                    // Route this key to ALL segments whose min/max covers it.
                    // We'll decide which one wins later by processing order (newest-first),
                    // which preserves overwrite/tombstone semantics while allowing fallback
                    // to older segments if the newest covering segment doesn't contain the key.
                    for segref in &col_index.segments {
                        if segref.logical_key_min.as_slice() <= k.as_slice()
                            && k.as_slice() <= segref.logical_key_max.as_slice()
                        {
                            match per_seg.entry(segref.index_physical_key) {
                                Entry::Occupied(mut e) => {
                                    e.get_mut().1.push((qi, kj));
                                }
                                Entry::Vacant(e) => {
                                    e.insert((segref.data_physical_key, vec![(qi, kj)]));
                                }
                            }
                        }
                    }
                }
            }
        }

        if per_seg.is_empty() {
            return results;
        }

        // Build a processing order of segments: strictly newest-first using the recorded ranks.
        let mut seg_order: Vec<PhysicalKey> = per_seg.keys().copied().collect();
        seg_order.sort_by_key(|pk| *seg_rank.get(pk).unwrap_or(&usize::MAX));

        // -------- single batched load for segments AND data blobs -------------
        let mut gets: Vec<BatchGet> = Vec::with_capacity(seg_order.len() * 2);
        for pk in &seg_order {
            gets.push(BatchGet::Typed {
                key: *pk,
                kind: TypedKind::IndexSegment,
            });
        }
        let mut seen_data: FxHashSet<PhysicalKey> = FxHashSet::default();
        for pk in &seg_order {
            let (data_pk, _) = per_seg.get(pk).unwrap();
            // Defensive: skip impossible/sentinel data keys.
            if *data_pk != u64::MAX && seen_data.insert(*data_pk) {
                gets.push(BatchGet::Raw { key: *data_pk });
            }
        }
        let resp = self.do_gets(gets);

        let mut seg_map: FxHashMap<PhysicalKey, IndexSegment> = FxHashMap::default();
        let mut data_map: FxHashMap<PhysicalKey, P::Blob> = FxHashMap::default();
        for gr in resp {
            match gr {
                GetResult::Typed {
                    key,
                    value: TypedValue::IndexSegment(seg),
                } => {
                    seg_map.insert(key, seg);
                }
                GetResult::Raw { key, bytes } => {
                    data_map.insert(key, bytes);
                }
                _ => {}
            }
        }

        // -------- execution: search segments newest-first and slice values ----
        for seg_pk in seg_order {
            let (data_pk, pairs) = per_seg.remove(&seg_pk).expect("segment present in per_seg");
            let seg = seg_map
                .get(&seg_pk)
                .expect("loaded segment missing from seg_map");

            for (qi, kj) in pairs {
                // Skip if already satisfied by a newer segment (value or tombstone).
                if resolved_queries.contains(&(qi, kj)) {
                    continue;
                }

                let target = items[qi].1[kj].as_slice();
                if let Some(pos) = KeyLayout::binary_search_key_with_layout(
                    &seg.logical_key_bytes,
                    &seg.key_layout,
                    seg.n_entries as usize,
                    target,
                ) {
                    match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => {
                            // Fixed-width tombstone is only defensive (ForceFixed(0) is rejected).
                            if *width == 0 {
                                resolved_queries.insert((qi, kj));
                                continue;
                            }
                            if let Some(data_blob) = data_map.get(&data_pk) {
                                let w = *width as usize;
                                let a = (pos * w) as ByteOffset;
                                let b = a + w as ByteOffset;
                                results[qi][kj] = Some(ValueSlice {
                                    data: data_blob.clone(),
                                    start: a,
                                    end: b,
                                });
                            }
                            resolved_queries.insert((qi, kj));
                        }
                        ValueLayout::Variable { value_offsets } => {
                            let a = value_offsets[pos] as ByteOffset;
                            let b = value_offsets[pos + 1] as ByteOffset;

                            if a == b {
                                // zero-length value => tombstone
                                resolved_queries.insert((qi, kj));
                                continue;
                            }

                            if let Some(data_blob) = data_map.get(&data_pk) {
                                results[qi][kj] = Some(ValueSlice {
                                    data: data_blob.clone(),
                                    start: a,
                                    end: b,
                                });
                            }
                            resolved_queries.insert((qi, kj));
                        }
                    }
                }
                // else: not in this (newer) segment, keep looking in older ones
            }
        }

        results
    }
}

impl<'p, P: Pager> ColumnStore<'p, P> {
    /// Scans the manifest -> ColumnIndex -> IndexSegments -> Data blobs,
    /// returns one entry per physical key with sizes and relationships.
    /// Batch-only I/O.
    pub fn describe_storage(&self) -> Vec<StorageNode> {
        let mut out: Vec<StorageNode> = Vec::new();

        // --- bootstrap + manifest (raw sizes)
        let bootstrap_key: PhysicalKey = self.bootstrap_key;
        let header_keys = vec![bootstrap_key, self.manifest_key];

        // Fetch raw bytes for bootstrap + manifest in one batch.
        let gets = header_keys
            .iter()
            .map(|k| BatchGet::Raw { key: *k })
            .collect::<Vec<_>>();
        let resp = self.do_gets(gets);

        let mut raw_len_map: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
        for gr in resp {
            if let GetResult::Raw { key, bytes } = gr {
                raw_len_map.insert(key, bytes.as_ref().len());
            }
        }

        // bootstrap
        out.push(StorageNode {
            pk: bootstrap_key,
            stored_len: *raw_len_map.get(&bootstrap_key).unwrap_or(&0),
            kind: StorageKind::Bootstrap,
        });
        // manifest
        let column_count = { self.manifest.read().unwrap().columns.len() };
        out.push(StorageNode {
            pk: self.manifest_key,
            stored_len: *raw_len_map.get(&self.manifest_key).unwrap_or(&0),
            kind: StorageKind::Manifest { column_count },
        });

        // --- column indexes (typed + raw in a single pass)
        let col_index_pks: Vec<PhysicalKey> = {
            let man = self.manifest.read().unwrap();
            man.columns
                .iter()
                .map(|c| c.column_index_physical_key)
                .collect()
        };

        let mut colindex_nodes: Vec<StorageNode> = Vec::new();
        let mut field_for_colindex: FxHashMap<PhysicalKey, LogicalFieldId> = FxHashMap::default();
        let mut all_seg_index_pks: Vec<PhysicalKey> = Vec::new();
        let mut seg_owner_colindex: FxHashMap<PhysicalKey, (LogicalFieldId, PhysicalKey)> =
            FxHashMap::default();

        if !col_index_pks.is_empty() {
            // Combined batch: raw + typed for each ColumnIndex key
            let mut gets: Vec<BatchGet> = Vec::with_capacity(col_index_pks.len() * 2);
            for pk in &col_index_pks {
                gets.push(BatchGet::Raw { key: *pk });
                gets.push(BatchGet::Typed {
                    key: *pk,
                    kind: TypedKind::ColumnIndex,
                });
            }
            let resp = self.do_gets(gets);

            let mut colindex_raw_len: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
            let mut colindices_by_pk: FxHashMap<PhysicalKey, ColumnIndex> = FxHashMap::default();

            for gr in resp {
                match gr {
                    GetResult::Raw { key, bytes } => {
                        colindex_raw_len.insert(key, bytes.as_ref().len());
                    }
                    GetResult::Typed {
                        key,
                        value: TypedValue::ColumnIndex(ci),
                    } => {
                        colindices_by_pk.insert(key, ci);
                    }
                    _ => {}
                }
            }

            for pk in &col_index_pks {
                if let Some(ci) = colindices_by_pk.get(pk) {
                    field_for_colindex.insert(*pk, ci.field_id);
                    colindex_nodes.push(StorageNode {
                        pk: *pk,
                        stored_len: *colindex_raw_len.get(pk).unwrap_or(&0),
                        kind: StorageKind::ColumnIndex {
                            field_id: ci.field_id,
                            n_segments: ci.segments.len() as usize,
                        },
                    });

                    // collect segment index pkeys and owners
                    for sref in &ci.segments {
                        all_seg_index_pks.push(sref.index_physical_key);
                        seg_owner_colindex.insert(sref.index_physical_key, (ci.field_id, *pk));
                    }
                }
            }
        }

        out.extend(colindex_nodes);

        // --- index segments (typed + raw) and discover data pkeys
        let mut data_pkeys: Vec<PhysicalKey> = Vec::new();
        let mut owner_for_data: FxHashMap<PhysicalKey, PhysicalKey> = FxHashMap::default(); // data_pkey -> index_segment_pk
        let mut seg_nodes: Vec<StorageNode> = Vec::new();

        if !all_seg_index_pks.is_empty() {
            // Combined batch: raw + typed for each IndexSegment key
            let mut gets: Vec<BatchGet> = Vec::with_capacity(all_seg_index_pks.len() * 2);
            for pk in &all_seg_index_pks {
                gets.push(BatchGet::Raw { key: *pk });
                gets.push(BatchGet::Typed {
                    key: *pk,
                    kind: TypedKind::IndexSegment,
                });
            }
            let resp = self.do_gets(gets);

            let mut seg_raw_len: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
            let mut segs_by_pk: FxHashMap<PhysicalKey, IndexSegment> = FxHashMap::default();

            for gr in resp {
                match gr {
                    GetResult::Raw { key, bytes } => {
                        seg_raw_len.insert(key, bytes.as_ref().len());
                    }
                    GetResult::Typed {
                        key,
                        value: TypedValue::IndexSegment(seg),
                    } => {
                        segs_by_pk.insert(key, seg);
                    }
                    _ => {}
                }
            }

            for pk in &all_seg_index_pks {
                if let Some(seg) = segs_by_pk.get(pk) {
                    let (field_id, owner_colindex_pk) =
                        seg_owner_colindex.get(pk).cloned().unwrap();

                    // book-keep data blob for later
                    data_pkeys.push(seg.data_physical_key);
                    owner_for_data.insert(seg.data_physical_key, *pk);

                    // compute layout info
                    let key_bytes = seg.logical_key_bytes.len();
                    let key_offs_bytes = match &seg.key_layout {
                        KeyLayout::FixedWidth { .. } => 0,
                        KeyLayout::Variable { key_offsets } => {
                            key_offsets.len() * std::mem::size_of::<ByteOffset>()
                        }
                    };
                    let (kind, fixed_width, value_meta_bytes) = match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => {
                            ("fixed", Some(*width), std::mem::size_of::<ByteWidth>())
                        }
                        ValueLayout::Variable { value_offsets } => (
                            "variable",
                            None,
                            value_offsets.len() * std::mem::size_of::<ByteOffset>(),
                        ),
                    };

                    let layout = IndexLayoutInfo {
                        kind,
                        fixed_width,
                        key_bytes,
                        key_offs_bytes,
                        value_meta_bytes,
                    };

                    seg_nodes.push(StorageNode {
                        pk: *pk,
                        stored_len: *seg_raw_len.get(pk).unwrap_or(&0),
                        kind: StorageKind::IndexSegment {
                            field_id,
                            n_entries: seg.n_entries,
                            layout,
                            data_pkey: seg.data_physical_key,
                            owner_colindex_pk,
                        },
                    });
                }
            }
        }

        out.extend(seg_nodes);

        // --- data blobs (raw only)
        if !data_pkeys.is_empty() {
            let gets = data_pkeys
                .iter()
                .map(|k| BatchGet::Raw { key: *k })
                .collect::<Vec<_>>();
            let resp = self.do_gets(gets);

            let mut data_len: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
            for gr in resp {
                if let GetResult::Raw { key, bytes } = gr {
                    data_len.insert(key, bytes.as_ref().len());
                }
            }

            for dpk in data_pkeys {
                let owner = *owner_for_data.get(&dpk).unwrap();
                out.push(StorageNode {
                    pk: dpk,
                    stored_len: *data_len.get(&dpk).unwrap_or(&0),
                    kind: StorageKind::DataBlob {
                        owner_index_pk: owner,
                    },
                });
            }
        }

        // deterministic order
        out.sort_by_key(|n| n.pk);
        out
    }

    /// Renders a compact ASCII table of the current storage layout.
    /// (bytes column moved before details to keep the table aligned)
    pub fn render_storage_ascii(&self) -> String {
        let nodes = self.describe_storage();
        let mut s = String::new();

        // Header: phys_key | kind | field | bytes | details
        let header = format!(
            "{:<10} {:<12} {:<9} {:>10}  {}",
            "phys_key", "kind", "field", "bytes", "details"
        );
        let _ = writeln!(&mut s, "{header}");

        // Divider length covers the fixed-width columns (details is free-width at the end)
        let _ = writeln!(&mut s, "{}", "-".repeat(header.len()));

        for n in nodes {
            match n.kind {
                StorageKind::Bootstrap => {
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "bootstrap", "-", n.stored_len, "-"
                    );
                }
                StorageKind::Manifest { column_count } => {
                    let det = format!("columns={}", column_count);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "manifest", "-", n.stored_len, det
                    );
                }
                StorageKind::ColumnIndex {
                    field_id,
                    n_segments,
                } => {
                    let det = format!("segments={}", n_segments);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "col_index", field_id, n.stored_len, det
                    );
                }
                StorageKind::IndexSegment {
                    field_id,
                    n_entries,
                    layout,
                    data_pkey,
                    owner_colindex_pk,
                } => {
                    let det = match layout.fixed_width {
                        Some(w) => format!(
                            "entries={} layout=fixed({}) key_bytes={} key_offs={} val_meta={} data_pk={} colidx_pk={}",
                            n_entries,
                            w,
                            layout.key_bytes,
                            layout.key_offs_bytes,
                            layout.value_meta_bytes,
                            data_pkey,
                            owner_colindex_pk
                        ),
                        None => format!(
                            "entries={} layout=variable key_bytes={} key_offs={} val_meta={} data_pk={} colidx_pk={}",
                            n_entries,
                            layout.key_bytes,
                            layout.key_offs_bytes,
                            layout.value_meta_bytes,
                            data_pkey,
                            owner_colindex_pk
                        ),
                    };
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "idx_segment", field_id, n.stored_len, det
                    );
                }
                StorageKind::DataBlob { owner_index_pk } => {
                    let det = format!("owner_idx_pk={}", owner_index_pk);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "data_blob", "-", n.stored_len, det
                    );
                }
            }
        }
        s
    }

    /// Renders a Graphviz DOT graph showing edges:
    /// bootstrap -> manifest -> ColumnIndex -> IndexSegment -> DataBlob
    pub fn render_storage_dot(&self) -> String {
        let nodes = self.describe_storage();

        // index nodes by pk for quick lookup
        let mut map: FxHashMap<PhysicalKey, &StorageNode> = FxHashMap::default();
        for n in &nodes {
            map.insert(n.pk, n);
        }

        let mut s = String::new();
        let _ = writeln!(&mut s, "digraph storage {{");
        let _ = writeln!(&mut s, "  node [shape=box, fontname=\"monospace\"];");

        // emit nodes
        for n in &nodes {
            match &n.kind {
                StorageKind::Bootstrap => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"Bootstrap pk={} bytes={}\"];",
                        n.pk, n.pk, n.stored_len
                    );
                    // edge to manifest
                    let _ = writeln!(&mut s, "  n{} -> n{};", n.pk, self.manifest_key);
                }
                StorageKind::Manifest { column_count } => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"Manifest pk={} columns={} bytes={}\"];",
                        n.pk, n.pk, column_count, n.stored_len
                    );
                    // edges to column indexes
                    for e in &self.manifest.read().unwrap().columns {
                        let _ =
                            writeln!(&mut s, "  n{} -> n{};", n.pk, e.column_index_physical_key);
                    }
                }
                StorageKind::ColumnIndex {
                    field_id,
                    n_segments,
                } => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"ColumnIndex pk={} field={} segs={} bytes={}\"];",
                        n.pk, n.pk, field_id, n_segments, n.stored_len
                    );
                }
                StorageKind::IndexSegment {
                    field_id,
                    n_entries,
                    layout,
                    data_pkey,
                    owner_colindex_pk,
                } => {
                    let lay = match layout.fixed_width {
                        Some(w) => format!("fixed({})", w),
                        None => "variable".to_string(),
                    };
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"IndexSegment pk={} field={} entries={} layout={} idx_bytes={} (key_bytes={}, key_offs={}, val_meta={})\"];",
                        n.pk,
                        n.pk,
                        field_id,
                        n_entries,
                        lay,
                        n.stored_len,
                        layout.key_bytes,
                        layout.key_offs_bytes,
                        layout.value_meta_bytes
                    );
                    // owner colindex and data edge
                    let _ = writeln!(&mut s, "  n{} -> n{};", owner_colindex_pk, n.pk);
                    let _ = writeln!(&mut s, "  n{} -> n{};", n.pk, data_pkey);
                }
                StorageKind::DataBlob { owner_index_pk } => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"DataBlob pk={} bytes={}\"];",
                        n.pk, n.pk, n.stored_len
                    );
                    let _ = writeln!(&mut s, "  n{} -> n{};", owner_index_pk, n.pk);
                }
            }
        }

        let _ = writeln!(&mut s, "}}");
        s
    }
}

// ----------------------------- tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Use the unified in-memory pager from the pager module.
    use crate::pager::MemPager;

    #[test]
    fn put_get_fixed_auto() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);

        let put = Put {
            field_id: 10,
            items: vec![
                (b"k3".to_vec(), b"VVVVVVV3".to_vec()),
                (b"k1".to_vec(), b"VVVVVVV1".to_vec()),
                (b"k2".to_vec(), b"VVVVVVV2".to_vec()),
                // duplicate key, last wins
                (b"k1".to_vec(), b"NEWVVVV1".to_vec()),
            ],
        };

        store.append_many(vec![put], AppendOptions::default());

        // batch get across columns (here only field 10)
        let got = store.get_many(vec![(
            10,
            vec![b"k1".to_vec(), b"k2".to_vec(), b"kX".to_vec()],
        )]);

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
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);

        // Values of different sizes force variable layout.
        let mut items = Vec::new();
        for i in 0..1000u32 {
            let k = format!("key{:04}", i).into_bytes();
            let v = vec![b'A' + (i % 26) as u8; (i % 17 + 1) as usize]; // 1..17
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
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let items = vec![
            (b"a".to_vec(), vec![1u8; 4]),
            (b"b".to_vec(), vec![2u8; 4]),
            (b"c".to_vec(), vec![3u8; 4]),
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
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);

        // Three items, with a duplicate for key "k": last is "ZZ".
        let fid = 42;
        let put = Put {
            field_id: fid,
            items: vec![
                (b"k".to_vec(), b"AA".to_vec()),
                (b"x".to_vec(), b"BB".to_vec()),
                (b"k".to_vec(), b"ZZ".to_vec()),
            ],
        };

        let mut opts = AppendOptions::default();
        opts.mode = ValueMode::ForceFixed(2); // keep sizes simple
        opts.segment_max_entries = 1024; // ensure a single segment
        opts.last_write_wins_in_batch = true; // <- dedup ON

        store.append_many(vec![put], opts);

        // Read back the two keys we wrote; "k" should be the LAST value ("ZZ").
        let got = store.get_many(vec![(fid, vec![b"k".to_vec(), b"x".to_vec()])]);
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
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);

        // Same three items; duplicates for "k" are KEPT.
        let fid = 43;
        let put = Put {
            field_id: fid,
            items: vec![
                (b"k".to_vec(), b"AA".to_vec()),
                (b"x".to_vec(), b"BB".to_vec()),
                (b"k".to_vec(), b"ZZ".to_vec()),
            ],
        };

        let mut opts = AppendOptions::default();
        opts.mode = ValueMode::ForceFixed(2);
        opts.segment_max_entries = 1024; // keep to a single segment
        opts.last_write_wins_in_batch = false; // <- dedup OFF

        store.append_many(vec![put], opts);

        // Reads for "k" will return one of the duplicate values.
        // (Order among equal keys after sort is not guaranteed.)
        let got = store.get_many(vec![(fid, vec![b"k".to_vec(), b"x".to_vec()])]);
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
        fn be_key(v: u64) -> Vec<u8> {
            v.to_be_bytes().to_vec()
        }

        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 1234;

        // Two disjoint segments for same column:
        //   seg A covers [0..100)      (keys 0..99)
        //   seg B covers [200..300)    (keys 200..299)
        // Make values fixed 4B to keep things simple.

        // Build segment A
        let items_a: Vec<(Vec<u8>, Vec<u8>)> = (0u64..100u64)
            .map(|k| (be_key(k), vec![0xAA, 0, 0, 0]))
            .collect();
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
        let items_b: Vec<(Vec<u8>, Vec<u8>)> = (200u64..300u64)
            .map(|k| (be_key(k), vec![0xBB, 0, 0, 0]))
            .collect();
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
        let got = store.get_many(vec![(fid, vec![be_key(250)])]);
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
        let got = store.get_many(vec![(fid, vec![be_key(200), be_key(299)])]);
        assert_eq!(got[0][0].as_deref().unwrap(), &[0xBB, 0, 0, 0]); // min
        assert_eq!(got[0][1].as_deref().unwrap(), &[0xBB, 0, 0, 0]); // max
        let s = store.io_stats();
        assert_eq!(s.get_typed_ops, 1);
        assert_eq!(s.get_raw_ops, 1);

        // ------------- Query hits ONLY seg A ----------------
        store.reset_io_stats();
        let got = store.get_many(vec![(fid, vec![be_key(5)])]);
        assert_eq!(got[0][0].as_deref().unwrap(), &[0xAA, 0, 0, 0]);
        let s = store.io_stats();
        assert_eq!(s.get_typed_ops, 1);
        assert_eq!(s.get_raw_ops, 1);
    }

    #[test]
    fn value_min_max_recorded_fixed() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 5050;

        // fixed 4B values with easy min/max
        let items = vec![
            (b"k1".to_vec(), vec![9, 9, 9, 9]),
            (b"k2".to_vec(), vec![0, 0, 0, 0]),       // min
            (b"k3".to_vec(), vec![255, 255, 255, 1]), // max (lexicographic)
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
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 6060;

        // variable-length values; lexicographic min/max over raw bytes
        let items = vec![
            (b"a".to_vec(), b"wolf".to_vec()),
            (b"b".to_vec(), b"ant".to_vec()), // min ("ant" < "wolf" < "zebra")
            (b"c".to_vec(), b"zebra".to_vec()), // max
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

        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 7070;

        // Single huge value (1 MiB)
        let huge = vec![42u8; 1_048_576];
        store.append_many(
            vec![Put {
                field_id: fid,
                items: vec![(b"k".to_vec(), huge.clone())],
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
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 101;

        // 1. Write some initial data
        let put = Put {
            field_id: fid,
            items: vec![
                (b"k1".to_vec(), b"val1".to_vec()),
                (b"k2".to_vec(), b"val2".to_vec()),
                (b"k3".to_vec(), b"val3".to_vec()),
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
        // NOTE: This test will FAIL until get_many is updated to handle tombstones.
        // The current implementation of get_many does not check for tombstone segments
        // (segments with value width 0) and will still find the old data.
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
            items: vec![(b"k2".to_vec(), b"new_val2".to_vec())],
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
