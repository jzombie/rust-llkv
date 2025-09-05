// High-level append/query API on top of pager + index modules.
use crate::index::{
    Bootstrap, ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef, KeyLayout, Manifest,
    ValueLayout,
};
use crate::pager::{
    BatchGet, BatchPut, BatchRequest, BatchResponse, GetResult, Pager, TypedKind, TypedValue,
};
use crate::types::{IndexEntryCount, LogicalFieldId, LogicalKeyBytes, PhysicalKey};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::min;
use std::collections::hash_map::Entry;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct IndexLayoutInfo {
    pub kind: &'static str,       // "fixed" or "variable" (value layout)
    pub fixed_width: Option<u32>, // when value layout is fixed
    // TODO: Rename to indicate *logical* and *len*?
    pub key_bytes: usize,        // logical_key_bytes.len()
    pub key_offs_bytes: usize, // if KeyLayout::Variable: key_offsets.len() * sizeof(IndexEntryCount); else 0
    pub value_meta_bytes: usize, // Variable: value_offsets.len()*sizeof(IndexEntryCount), Fixed: 4 (width)
}

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

// ----------------------------- helpers -----------------------------

fn slice_key_by_layout<'a>(bytes: &'a [u8], layout: &'a KeyLayout, i: usize) -> &'a [u8] {
    match layout {
        KeyLayout::FixedWidth { width } => {
            let w = *width as usize;
            let a = i * w;
            let b = a + w;
            &bytes[a..b]
        }
        KeyLayout::Variable { key_offsets } => {
            let a = key_offsets[i] as usize;
            let b = key_offsets[i + 1] as usize;
            &bytes[a..b]
        }
    }
}

fn binary_search_key_with_layout(
    bytes: &[u8],
    layout: &KeyLayout,
    n_entries: usize,
    target: &[u8],
) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = n_entries; // exclusive
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = slice_key_by_layout(bytes, layout, mid);
        match k.cmp(target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}

// ----------------------------- write API -----------------------------

pub struct Put {
    pub field_id: LogicalFieldId,
    pub items: Vec<(LogicalKeyBytes, Vec<u8>)>, // unordered; duplicates allowed (last wins)
}

#[derive(Clone, Copy, Debug)]
pub enum ValueMode {
    Auto,
    ForceFixed(u32),
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
    pub batches: usize,       // number of times we called Pager::batch
    pub get_raw_ops: usize,   // number of Raw gets requested
    pub get_typed_ops: usize, // number of Typed gets requested
    pub put_raw_ops: usize,   // number of Raw puts requested
    pub put_typed_ops: usize, // number of Typed puts requested
}

impl IoStats {
    pub fn reset(&mut self) {
        *self = IoStats::default();
    }
}

// ----------------------------- ColumnStore -----------------------------

pub struct ColumnStore<'p, P: Pager> {
    pager: &'p mut P,
    bootstrap_key: PhysicalKey,
    manifest_key: PhysicalKey,
    manifest: Manifest,
    // field_id -> (column_index_pkey, decoded)
    colindex_cache: FxHashMap<LogicalFieldId, (PhysicalKey, ColumnIndex)>,
    // pager-hit metrics (counts only)
    io_stats: IoStats,
}

impl<'p, P: Pager> ColumnStore<'p, P> {
    // Helper to route all batch calls through here so we bump metrics in one place.
    fn do_batch<'a>(&'a mut self, req: BatchRequest) -> BatchResponse<'a> {
        // update counters before the call
        self.io_stats.batches += 1;
        for p in &req.puts {
            match p {
                BatchPut::Raw { .. } => self.io_stats.put_raw_ops += 1,
                BatchPut::Typed { .. } => self.io_stats.put_typed_ops += 1,
            }
        }
        for g in &req.gets {
            match g {
                BatchGet::Raw { .. } => self.io_stats.get_raw_ops += 1,
                BatchGet::Typed { .. } => self.io_stats.get_typed_ops += 1,
            }
        }
        self.pager.batch(&req)
    }

    /// Access current metrics (counts of batch/ops).
    pub fn io_stats(&self) -> &IoStats {
        &self.io_stats
    }

    /// Reset metrics to zero.
    pub fn reset_io_stats(&mut self) {
        self.io_stats.reset();
    }

    // Create fresh store (bootstrap->manifest, empty manifest).
    pub fn init_empty(pager: &'p mut P) -> Self {
        let bootstrap_key: PhysicalKey = 0;
        let manifest_key = pager.alloc_many(1)[0];
        let manifest = Manifest {
            columns: Vec::new(),
        };

        // Write Manifest and Bootstrap in a single batch.
        // (Not counted in ColumnStore metrics since the store isn't constructed yet.)
        let _ = pager.batch(&BatchRequest {
            puts: vec![
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
            ],
            gets: vec![],
        });

        ColumnStore {
            pager,
            bootstrap_key,
            manifest_key,
            manifest,
            colindex_cache: FxHashMap::with_hasher(Default::default()),
            io_stats: IoStats::default(),
        }
    }

    // Open existing store (bootstrap(0) -> manifest).
    pub fn open(pager: &'p mut P) -> Self {
        let bootstrap_key: PhysicalKey = 0;

        // Get Bootstrap (not counted; ColumnStore not constructed yet)
        let resp = pager.batch(&BatchRequest {
            puts: vec![],
            gets: vec![BatchGet::Typed {
                key: bootstrap_key,
                kind: TypedKind::Bootstrap,
            }],
        });
        let boot = match &resp.get_results[0] {
            GetResult::Typed {
                value: TypedValue::Bootstrap(b),
                ..
            } => b.clone(),
            _ => panic!("missing bootstrap at key 0"),
        };
        let manifest_key = boot.manifest_physical_key;

        // Get Manifest (not counted; ColumnStore not constructed yet)
        let resp = pager.batch(&BatchRequest {
            puts: vec![],
            gets: vec![BatchGet::Typed {
                key: manifest_key,
                kind: TypedKind::Manifest,
            }],
        });
        let manifest = match &resp.get_results[0] {
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
            manifest,
            colindex_cache: FxHashMap::with_hasher(Default::default()),
            io_stats: IoStats::default(),
        }
    }

    // Single entrypoint for writing. Many columns, each with unordered items.
    // Auto-chooses fixed vs variable, chunks to segments, writes everything in batches.
    pub fn append_many(&mut self, puts: Vec<Put>, opts: AppendOptions) {
        if puts.is_empty() {
            return;
        }

        // Normalize per column: last-write-wins (optional), then sort by logical key.
        // Also decide value layout for this batch (fixed width or variable).
        #[derive(Clone)]
        enum PlannedLayout {
            Fixed { width: u32 },
            Variable, // offsets will be computed per segment
        }
        struct PlannedChunk {
            field_id: LogicalFieldId,
            keys_sorted: Vec<LogicalKeyBytes>,
            values: Vec<Vec<u8>>, // aligned to keys
            layout: PlannedLayout,
        }

        let mut planned_chunks: Vec<PlannedChunk> = Vec::new();

        for put in puts {
            if put.items.is_empty() {
                continue;
            }
            let mut items = put.items;

            if opts.last_write_wins_in_batch {
                // last wins: overwrite by key
                let mut last: FxHashMap<LogicalKeyBytes, Vec<u8>> =
                    FxHashMap::with_capacity_and_hasher(items.len(), Default::default());
                for (k, v) in items {
                    last.insert(k, v);
                }
                items = last.into_iter().collect();
            }

            // sort by key
            items.sort_by(|a, b| a.0.cmp(&b.0));

            // decide layout
            let layout = match opts.mode {
                ValueMode::ForceFixed(w) => {
                    assert!(w > 0, "fixed width must be > 0");
                    for (_, v) in &items {
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
                    let w = items[0].1.len();
                    if w > 0 && items.iter().all(|(_, v)| v.len() == w) {
                        PlannedLayout::Fixed { width: w as u32 }
                    } else {
                        PlannedLayout::Variable
                    }
                }
            };

            // chunk by thresholds
            let mut i = 0usize;
            while i < items.len() {
                let remain = items.len() - i;
                let take_by_entries = min(remain, opts.segment_max_entries);

                let take = match layout {
                    PlannedLayout::Fixed { width } => {
                        // entries are uniform; respect both entry and byte budgets
                        let max_entries_by_bytes = if width == 0 {
                            take_by_entries
                        } else {
                            (opts.segment_max_bytes / (width as usize)).max(1)
                        };
                        min(take_by_entries, max_entries_by_bytes.max(1))
                    }
                    PlannedLayout::Variable => {
                        // grow until adding next would exceed bytes or entries
                        let mut acc = 0usize;
                        let mut cnt = 0usize;
                        for j in i..(i + take_by_entries) {
                            let sz = items[j].1.len();
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
                for (k, v) in items[i..end].iter() {
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
        let mut pkeys = self.pager.alloc_many(n_segs * 2);
        // first half used for data blobs, second half for index blobs
        let data_keys: Vec<PhysicalKey> = pkeys.drain(0..n_segs).collect();
        let index_keys: Vec<PhysicalKey> = pkeys.drain(0..n_segs).collect();

        // Build concrete IndexSegments and data blobs; group by column to update ColumnIndex.
        let mut data_puts: Vec<(PhysicalKey, Vec<u8>)> = Vec::with_capacity(n_segs);
        let mut index_puts: Vec<(PhysicalKey, IndexSegment)> = Vec::with_capacity(n_segs);

        // Prepare map: field_id -> (col_index_pkey, ColumnIndex)
        let mut need_manifest_update = false;
        let mut ensure_column_loaded: FxHashSet<LogicalFieldId> =
            FxHashSet::with_hasher(Default::default());
        // NEW: track which ColumnIndex pkeys were actually modified this call.
        let mut touched_colindex_pkeys: FxHashSet<PhysicalKey> =
            FxHashSet::with_hasher(Default::default());

        // Bring column indexes we will touch into cache (batch-read where possible)
        for chunk in &planned_chunks {
            ensure_column_loaded.insert(chunk.field_id);
        }
        let missing: Vec<LogicalFieldId> = ensure_column_loaded
            .iter()
            .filter(|fid| !self.colindex_cache.contains_key(fid))
            .copied()
            .collect();

        if !missing.is_empty() {
            // find entries in manifest
            let mut lookups: Vec<(LogicalFieldId, PhysicalKey)> = Vec::new();
            for fid in &missing {
                if let Some(entry) = self.manifest.columns.iter().find(|e| e.field_id == *fid) {
                    lookups.push((*fid, entry.column_index_physical_key));
                }
            }
            if !lookups.is_empty() {
                let gets: Vec<BatchGet> = lookups
                    .iter()
                    .map(|(_, pk)| BatchGet::Typed {
                        key: *pk,
                        kind: TypedKind::ColumnIndex,
                    })
                    .collect();
                let resp = self.do_batch(BatchRequest { puts: vec![], gets });

                let mut got_vec: Vec<ColumnIndex> = Vec::with_capacity(lookups.len());
                for gr in resp.get_results {
                    match gr {
                        GetResult::Typed {
                            value: TypedValue::ColumnIndex(ci),
                            ..
                        } => got_vec.push(ci),
                        _ => panic!("ColumnIndex missing"),
                    }
                }

                for (i, (fid, k)) in lookups.into_iter().enumerate() {
                    self.colindex_cache.insert(fid, (k, got_vec[i].clone()));
                }
            }
        }

        // For any columns still missing, create fresh ColumnIndex and manifest entry.
        for fid in &missing {
            if !self.colindex_cache.contains_key(fid) {
                let col_index_pkey = self.pager.alloc_many(1)[0];
                let col_index = ColumnIndex {
                    field_id: *fid,
                    segments: Vec::new(),
                };
                self.colindex_cache
                    .insert(*fid, (col_index_pkey, col_index));
                // add to manifest (in-memory)
                self.manifest.columns.push(ColumnEntry {
                    field_id: *fid,
                    column_index_physical_key: col_index_pkey,
                });
                need_manifest_update = true;
            }
        }

        // Now materialize each planned chunk into data blob + IndexSegment, and update ColumnIndex.
        for (i, chunk) in planned_chunks.into_iter().enumerate() {
            let data_pkey = data_keys[i];
            let index_pkey = index_keys[i];

            // build data blob (+ value offsets for variable)
            let (data_blob, seg, n_entries) = match chunk.layout {
                PlannedLayout::Fixed { width } => {
                    // values blob
                    let mut blob = Vec::with_capacity(chunk.values.len() * (width as usize));
                    for v in &chunk.values {
                        debug_assert_eq!(v.len(), width as usize);
                        blob.extend_from_slice(v);
                    }
                    // index segment (lets the index module compute key layout & min/max)
                    let seg =
                        IndexSegment::build_fixed(data_pkey, chunk.keys_sorted.clone(), width);
                    (blob, seg, chunk.values.len() as IndexEntryCount)
                }
                PlannedLayout::Variable => {
                    // values blob
                    let mut blob = Vec::new();
                    let mut sizes: Vec<u32> = Vec::with_capacity(chunk.values.len());
                    for v in &chunk.values {
                        blob.extend_from_slice(v);
                        sizes.push(v.len() as u32);
                    }
                    // index segment
                    let seg = IndexSegment::build_var(data_pkey, chunk.keys_sorted.clone(), &sizes);
                    (blob, seg, chunk.values.len() as IndexEntryCount)
                }
            };

            // queue writes
            data_puts.push((data_pkey, data_blob));
            index_puts.push((index_pkey, seg.clone()));

            // update ColumnIndex (newest-first: push_front semantics)
            let (col_pkey, col_index) = self
                .colindex_cache
                .get_mut(&chunk.field_id)
                .expect("column index present");

            col_index.segments.insert(
                0,
                IndexSegmentRef {
                    index_physical_key: index_pkey,
                    data_physical_key: seg.data_physical_key,
                    logical_key_min: seg.logical_key_min.clone(),
                    logical_key_max: seg.logical_key_max.clone(),
                    n_entries,
                },
            );

            // mark this ColumnIndex as touched so we persist only these (ingest perf fix)
            touched_colindex_pkeys.insert(*col_pkey);
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
            for (_fid, (pk, ci)) in self.colindex_cache.iter() {
                if touched_colindex_pkeys.contains(pk) {
                    puts_batch.push(BatchPut::Typed {
                        key: *pk,
                        value: TypedValue::ColumnIndex(ci.clone()),
                    });
                }
            }
        }

        if need_manifest_update {
            puts_batch.push(BatchPut::Typed {
                key: self.manifest_key,
                value: TypedValue::Manifest(self.manifest.clone()),
            });
        }

        if !puts_batch.is_empty() {
            // Assert we're using storage efficiently
            #[cfg(debug_assertions)]
            {
                let mut seen = FxHashSet::with_hasher(Default::default());
                for p in &puts_batch {
                    let k = match p {
                        BatchPut::Raw { key, .. } => *key,
                        BatchPut::Typed { key, .. } => *key,
                    };
                    assert!(seen.insert(k), "duplicate PUT for key {}", k);
                }
            }

            let _ = self.do_batch(BatchRequest {
                puts: puts_batch,
                gets: vec![],
            });
        }
    }

    // ----------------------------- read API -----------------------------

    /// Batched point lookups across many columns, zero-copy.
    /// Each item is (field_id, keys). Output is aligned per input, where
    /// each value is an `Option<&[u8]>` slice into prefetched data blobs.
    ///
    /// I/O pattern (whole batch):
    ///   1) batch(ColumnIndex gets) for any missing columns (once)
    ///   2) batch(IndexSegment typed + DataBlob raw gets) (once)
    ///
    /// Newest-first shadowing: for each key we pick the first segment whose
    /// [min,max] covers it.
    ///
    /// Lifetimes tie returned slices to `&'s mut self`.
    pub fn get_many<'s>(
        &'s mut self,
        items: Vec<(LogicalFieldId, Vec<LogicalKeyBytes>)>,
    ) -> Vec<Vec<Option<&'s [u8]>>> {
        if items.is_empty() {
            return Vec::new();
        }

        // Pre-size outputs: one vector per input, one slot per key.
        let mut results: Vec<Vec<Option<&'s [u8]>>> =
            items.iter().map(|(_, ks)| vec![None; ks.len()]).collect();

        // -------- ensure ColumnIndex in cache for all referenced fields --------
        let mut need_fids: FxHashSet<LogicalFieldId> = FxHashSet::with_hasher(Default::default());
        for (fid, _) in &items {
            need_fids.insert(*fid);
        }

        let mut to_load: Vec<(LogicalFieldId, PhysicalKey)> = Vec::new();
        for fid in need_fids {
            if self.colindex_cache.contains_key(&fid) {
                continue;
            }
            if let Some(entry) = self.manifest.columns.iter().find(|e| e.field_id == fid) {
                to_load.push((fid, entry.column_index_physical_key));
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
            let resp = self.do_batch(BatchRequest { puts: vec![], gets });

            let mut got_vec: Vec<ColumnIndex> = Vec::with_capacity(to_load.len());
            for gr in resp.get_results {
                match gr {
                    GetResult::Typed {
                        value: TypedValue::ColumnIndex(ci),
                        ..
                    } => got_vec.push(ci),
                    _ => panic!("ColumnIndex missing"),
                }
            }

            for (i, (fid, pk)) in to_load.into_iter().enumerate() {
                self.colindex_cache.insert(fid, (pk, got_vec[i].clone()));
            }
        }

        // -------- routing: segment -> (data_pk, [(item_i, key_j), ...]) --------
        let mut per_seg: FxHashMap<PhysicalKey, (PhysicalKey, Vec<(usize, usize)>)> =
            FxHashMap::with_hasher(Default::default());

        for (qi, (fid, keys)) in items.iter().enumerate() {
            let col_index = match self.colindex_cache.get(fid) {
                Some((_, ci)) => ci,
                None => continue, // unknown field => all None for that entry
            };

            for (kj, k) in keys.iter().enumerate() {
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
                        break;
                    }
                }
            }
        }

        if per_seg.is_empty() {
            return results;
        }

        // -------- single batched load for segments AND data blobs -------------
        // Build one get list containing typed IndexSegments and raw Data blobs.
        let mut gets: Vec<BatchGet> = Vec::with_capacity(per_seg.len() * 2);

        // 1) typed load of all needed index segments
        let seg_keys: Vec<PhysicalKey> = per_seg.keys().copied().collect();
        for pk in &seg_keys {
            gets.push(BatchGet::Typed {
                key: *pk,
                kind: TypedKind::IndexSegment,
            });
        }

        // 2) raw load of all needed data blobs (dedup)
        let mut data_keys: Vec<PhysicalKey> = Vec::with_capacity(per_seg.len());
        let mut seen_data: FxHashSet<PhysicalKey> = FxHashSet::with_hasher(Default::default());
        for v in per_seg.values() {
            let data_pk = v.0;
            if seen_data.insert(data_pk) {
                data_keys.push(data_pk);
            }
        }
        for pk in &data_keys {
            gets.push(BatchGet::Raw { key: *pk });
        }

        let resp = self.do_batch(BatchRequest { puts: vec![], gets });

        // Partition response into typed segments and raw data slices.
        let mut seg_map: FxHashMap<PhysicalKey, IndexSegment> =
            FxHashMap::with_capacity_and_hasher(seg_keys.len(), Default::default());
        let mut data_map: FxHashMap<PhysicalKey, &'s [u8]> =
            FxHashMap::with_capacity_and_hasher(data_keys.len(), Default::default());

        for gr in resp.get_results {
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

        // -------- execution: binary search inside segments and slice values ----
        for (seg_pk, (data_pk, pairs)) in per_seg {
            let seg = seg_map
                .get(&seg_pk)
                .expect("loaded segment missing from seg_map");
            let data_blob = data_map
                .get(&data_pk)
                .expect("loaded data blob missing from data_map");

            for (qi, kj) in pairs {
                let target = items[qi].1[kj].as_slice();
                if let Some(pos) = binary_search_key_with_layout(
                    &seg.logical_key_bytes,
                    &seg.key_layout,
                    seg.n_entries as usize,
                    target,
                ) {
                    match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => {
                            let w = *width as usize;
                            let a = pos * w;
                            let b = a + w;
                            results[qi][kj] = Some(&data_blob[a..b]);
                        }
                        ValueLayout::Variable { value_offsets } => {
                            let a = value_offsets[pos] as usize;
                            let b = value_offsets[pos + 1] as usize;
                            results[qi][kj] = Some(&data_blob[a..b]);
                        }
                    }
                }
            }
        }

        results
    }
}

impl<'p, P: Pager> ColumnStore<'p, P> {
    /// Scans the manifest -> ColumnIndex -> IndexSegments -> Data blobs,
    /// returns one entry per physical key with sizes and relationships.
    /// Batch-only I/O.
    pub fn describe_storage(&mut self) -> Vec<StorageNode> {
        let mut out: Vec<StorageNode> = Vec::new();

        // --- bootstrap + manifest (raw sizes)
        let bootstrap_key: PhysicalKey = self.bootstrap_key;
        let header_keys = vec![bootstrap_key, self.manifest_key];

        // Fetch raw bytes for bootstrap + manifest in one batch.
        let gets = header_keys
            .iter()
            .map(|k| BatchGet::Raw { key: *k })
            .collect::<Vec<_>>();
        let resp = self.do_batch(BatchRequest { puts: vec![], gets });

        let mut raw_len_map: FxHashMap<PhysicalKey, usize> =
            FxHashMap::with_hasher(Default::default());
        for gr in resp.get_results {
            if let GetResult::Raw { key, bytes } = gr {
                raw_len_map.insert(key, bytes.len());
            }
        }

        // bootstrap
        out.push(StorageNode {
            pk: bootstrap_key,
            stored_len: *raw_len_map.get(&bootstrap_key).unwrap_or(&0),
            kind: StorageKind::Bootstrap,
        });
        // manifest
        out.push(StorageNode {
            pk: self.manifest_key,
            stored_len: *raw_len_map.get(&self.manifest_key).unwrap_or(&0),
            kind: StorageKind::Manifest {
                column_count: self.manifest.columns.len(),
            },
        });

        // --- column indexes (typed + raw in a single pass)
        let col_index_pks: Vec<PhysicalKey> = self
            .manifest
            .columns
            .iter()
            .map(|c| c.column_index_physical_key)
            .collect();

        let mut colindex_nodes: Vec<StorageNode> = Vec::new();
        let mut field_for_colindex: FxHashMap<PhysicalKey, LogicalFieldId> =
            FxHashMap::with_hasher(Default::default());
        let mut all_seg_index_pks: Vec<PhysicalKey> = Vec::new();
        let mut seg_owner_colindex: FxHashMap<PhysicalKey, (LogicalFieldId, PhysicalKey)> =
            FxHashMap::with_hasher(Default::default());

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
            let resp = self.do_batch(BatchRequest { puts: vec![], gets });

            let mut colindex_raw_len: FxHashMap<PhysicalKey, usize> =
                FxHashMap::with_hasher(Default::default());
            let mut colindices_by_pk: FxHashMap<PhysicalKey, ColumnIndex> =
                FxHashMap::with_hasher(Default::default());

            for gr in resp.get_results {
                match gr {
                    GetResult::Raw { key, bytes } => {
                        colindex_raw_len.insert(key, bytes.len());
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
        let mut owner_for_data: FxHashMap<PhysicalKey, PhysicalKey> =
            FxHashMap::with_hasher(Default::default()); // data_pkey -> index_segment_pk
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
            let resp = self.do_batch(BatchRequest { puts: vec![], gets });

            let mut seg_raw_len: FxHashMap<PhysicalKey, usize> =
                FxHashMap::with_hasher(Default::default());
            let mut segs_by_pk: FxHashMap<PhysicalKey, IndexSegment> =
                FxHashMap::with_hasher(Default::default());

            for gr in resp.get_results {
                match gr {
                    GetResult::Raw { key, bytes } => {
                        seg_raw_len.insert(key, bytes.len());
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
                            key_offsets.len() * std::mem::size_of::<IndexEntryCount>()
                        }
                    };
                    let (kind, fixed_width, value_meta_bytes) = match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => {
                            ("fixed", Some(*width), std::mem::size_of::<u32>())
                        }
                        ValueLayout::Variable { value_offsets } => (
                            "variable",
                            None,
                            value_offsets.len() * std::mem::size_of::<IndexEntryCount>(),
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
            let resp = self.do_batch(BatchRequest { puts: vec![], gets });

            let mut data_len: FxHashMap<PhysicalKey, usize> =
                FxHashMap::with_hasher(Default::default());
            for gr in resp.get_results {
                if let GetResult::Raw { key, bytes } = gr {
                    data_len.insert(key, bytes.len());
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
    pub fn render_storage_ascii(&mut self) -> String {
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
    pub fn render_storage_dot(&mut self) -> String {
        let nodes = self.describe_storage();

        // index nodes by pk for quick lookup
        let mut map: FxHashMap<PhysicalKey, &StorageNode> =
            FxHashMap::with_hasher(Default::default());
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
                    for e in &self.manifest.columns {
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
        let mut p = MemPager::default();
        let mut store = ColumnStore::init_empty(&mut p);

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

        // zero-copy batch get across columns (here only field 10)
        let got = store.get_many(vec![(
            10,
            vec![b"k1".to_vec(), b"k2".to_vec(), b"kX".to_vec()],
        )]);

        // got[0] corresponds to field 10's query vector
        assert_eq!(got[0][0].unwrap(), b"NEWVVVV1");
        assert_eq!(got[0][1].unwrap(), b"VVVVVVV2");
        assert!(got[0][2].is_none());

        // sanity: we should have issued at least one batch (append + get)
        let stats = store.io_stats().clone();
        assert!(stats.batches >= 2);
        assert!(stats.put_raw_ops > 0 || stats.put_typed_ops > 0);
        assert!(stats.get_raw_ops > 0 || stats.get_typed_ops > 0);
    }

    #[test]
    fn put_get_variable_auto_with_chunking() {
        let mut p = MemPager::default();
        let mut store = ColumnStore::init_empty(&mut p);

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
            let i: u32 = s[3..].parse().unwrap();
            (i % 17 + 1) as usize
        };

        assert_eq!(got[0][0].unwrap().len(), expect_len("key0000"));
        assert_eq!(got[0][1].unwrap().len(), expect_len("key0001"));
        assert_eq!(got[0][2].unwrap().len(), expect_len("key0199"));
        assert_eq!(got[0][3].unwrap().len(), expect_len("key0200"));
        assert_eq!(got[0][4].unwrap().len(), expect_len("key0999"));
        assert!(got[0][5].is_none());

        // sanity: get_many should have exactly 1 batch for segments+data,
        // plus 0 or 1 for initial ColumnIndex fills (depending on cache)
        let s = store.io_stats();
        assert!(s.batches >= 1);
        assert!(s.get_raw_ops > 0 || s.get_typed_ops > 0);
    }

    #[test]
    fn force_fixed_validation() {
        let mut p = MemPager::default();
        let mut store = ColumnStore::init_empty(&mut p);
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

        assert_eq!(got[0][0].unwrap(), &[1u8; 4]);
        assert_eq!(got[0][1].unwrap(), &[2u8; 4]);
        assert!(got[0][2].is_none());
    }
}
