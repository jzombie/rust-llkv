// High-level append/query API on top of pager + index modules.

use std::cmp::min;
use std::collections::{HashMap, HashSet};

pub mod index;
pub mod pager;
pub mod types;

use crate::index::{
    Bootstrap, ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef, Manifest, ValueLayout,
};
use crate::pager::Pager;
use crate::types::{IndexEntryCount, LogicalFieldId, LogicalKeyBytes, PhysicalKey};

use std::fmt::Write as _;

#[derive(Clone, Debug)]
pub struct IndexLayoutInfo {
    pub kind: &'static str,       // "fixed" or "variable"
    pub fixed_width: Option<u32>, // when fixed
    // TODO: Rename to indicate *logical* and *len*?
    pub key_bytes: usize,        // logical_key_bytes.len()
    pub key_offs_bytes: usize,   // logical_key_offsets.len() * 4
    pub value_meta_bytes: usize, // Variable: value_offsets.len()*4, Fixed: 4 (width)
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

fn concat_keys_and_offsets_sorted(
    keys_sorted: &[LogicalKeyBytes],
) -> (Vec<u8>, Vec<IndexEntryCount>) {
    let mut bytes = Vec::new();
    let mut offs = Vec::with_capacity(keys_sorted.len() + 1);
    let mut acc: IndexEntryCount = 0;
    offs.push(acc);
    for k in keys_sorted {
        bytes.extend_from_slice(k);
        acc += k.len() as IndexEntryCount;
        offs.push(acc);
    }
    (bytes, offs)
}

fn slice_key<'a>(bytes: &'a [u8], offs: &'a [IndexEntryCount], i: usize) -> &'a [u8] {
    let a = offs[i] as usize;
    let b = offs[i + 1] as usize;
    &bytes[a..b]
}

fn binary_search_key(bytes: &[u8], offs: &[IndexEntryCount], target: &[u8]) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = offs.len() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = slice_key(bytes, offs, mid);
        match k.cmp(target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid),
        }
    }
    // final check
    if lo < offs.len() - 1 && slice_key(bytes, offs, lo) == target {
        Some(lo)
    } else {
        None
    }
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

// ----------------------------- ColumnStore -----------------------------

pub struct ColumnStore<'p, P: Pager> {
    pager: &'p mut P,
    bootstrap_key: PhysicalKey,
    manifest_key: PhysicalKey,
    manifest: Manifest,
    // field_id -> (column_index_pkey, decoded)
    colindex_cache: HashMap<LogicalFieldId, (PhysicalKey, ColumnIndex)>,
}

impl<'p, P: Pager> ColumnStore<'p, P> {
    // Create fresh store (bootstrap->manifest, empty manifest).
    pub fn init_empty(pager: &'p mut P) -> Self {
        let bootstrap_key: PhysicalKey = 0;
        let manifest_key = pager.alloc_many(1)[0];
        let manifest = Manifest {
            columns: Vec::new(),
        };
        pager.batch_put_typed::<Manifest>(&[(manifest_key, manifest.clone())]);
        pager.batch_put_typed::<Bootstrap>(&[(
            bootstrap_key,
            Bootstrap {
                manifest_physical_key: manifest_key,
            },
        )]);
        ColumnStore {
            pager,
            bootstrap_key,
            manifest_key,
            manifest,
            colindex_cache: HashMap::new(),
        }
    }

    // Open existing store (bootstrap(0) -> manifest).
    pub fn open(pager: &'p mut P) -> Self {
        let bootstrap_key: PhysicalKey = 0;
        let b: Vec<Bootstrap> = pager.batch_get_typed::<Bootstrap>(&[bootstrap_key]);
        let boot = b.into_iter().next().expect("missing bootstrap at key 0");
        let manifest_key = boot.manifest_physical_key;
        let m: Vec<Manifest> = pager.batch_get_typed::<Manifest>(&[manifest_key]);
        let manifest = m.into_iter().next().expect("missing manifest");
        ColumnStore {
            pager,
            bootstrap_key,
            manifest_key,
            manifest,
            colindex_cache: HashMap::new(),
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
                let mut last: HashMap<LogicalKeyBytes, Vec<u8>> =
                    HashMap::with_capacity(items.len());
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
        let mut ensure_column_loaded: HashSet<LogicalFieldId> = HashSet::new();

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
                let keys: Vec<PhysicalKey> = lookups.iter().map(|(_, k)| *k).collect();
                let got: Vec<ColumnIndex> = self.pager.batch_get_typed::<ColumnIndex>(&keys);
                for (i, (fid, k)) in lookups.into_iter().enumerate() {
                    self.colindex_cache.insert(fid, (k, got[i].clone()));
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

            // build data blob (+ offsets if variable)
            let (data_blob, layout_enum, n_entries) = match chunk.layout {
                PlannedLayout::Fixed { width } => {
                    let mut blob = Vec::with_capacity(chunk.values.len() * (width as usize));
                    for v in &chunk.values {
                        debug_assert_eq!(v.len(), width as usize);
                        blob.extend_from_slice(v);
                    }
                    (
                        blob,
                        ValueLayout::FixedWidth { width },
                        chunk.values.len() as IndexEntryCount,
                    )
                }
                PlannedLayout::Variable => {
                    let mut blob = Vec::new();
                    let mut off: Vec<IndexEntryCount> = Vec::with_capacity(chunk.values.len() + 1);
                    let mut acc: IndexEntryCount = 0;
                    off.push(0);
                    for v in &chunk.values {
                        blob.extend_from_slice(v);
                        acc += v.len() as IndexEntryCount;
                        off.push(acc);
                    }
                    (
                        blob,
                        ValueLayout::Variable { value_offsets: off },
                        chunk.values.len() as IndexEntryCount,
                    )
                }
            };

            // Build key storage
            let (key_bytes, key_offs) = concat_keys_and_offsets_sorted(&chunk.keys_sorted);
            let logical_key_min = chunk.keys_sorted.first().cloned().unwrap().to_vec();
            let logical_key_max = chunk.keys_sorted.last().cloned().unwrap().to_vec();

            // Assemble IndexSegment
            let seg = IndexSegment {
                data_physical_key: data_pkey,
                n_entries,
                logical_key_bytes: key_bytes,
                logical_key_offsets: key_offs,
                value_layout: match layout_enum {
                    ValueLayout::FixedWidth { width } => ValueLayout::FixedWidth { width },
                    ValueLayout::Variable { value_offsets } => {
                        ValueLayout::Variable { value_offsets }
                    }
                },
                logical_key_min,
                logical_key_max,
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
                    n_entries: seg.n_entries,
                },
            );

            // write updated ColumnIndex after loop (batched)
            let _ = col_pkey; // kept for later batch write
        }

        // Persist: data blobs, index segments, column indexes, manifest (if needed).
        if !data_puts.is_empty() {
            self.pager.batch_put_raw(&data_puts);
        }
        if !index_puts.is_empty() {
            self.pager.batch_put_typed::<IndexSegment>(&index_puts);
        }

        if !self.colindex_cache.is_empty() {
            // batch write all updated column indexes
            let mut colindex_puts: Vec<(PhysicalKey, ColumnIndex)> =
                Vec::with_capacity(self.colindex_cache.len());
            for (_fid, (pk, ci)) in self.colindex_cache.iter() {
                colindex_puts.push((*pk, ci.clone()));
            }
            self.pager.batch_put_typed::<ColumnIndex>(&colindex_puts);
        }

        if need_manifest_update {
            self.pager
                .batch_put_typed::<Manifest>(&[(self.manifest_key, self.manifest.clone())]);
        }
    }

    // ----------------------------- read API -----------------------------

    // TODO: Make zero-copy
    // Batched point lookup: values for a single column and a list of logical keys.
    // Returns values in the same order as requested (None if missing).
    pub fn get_in_column(
        &mut self,
        field_id: LogicalFieldId,
        keys: Vec<LogicalKeyBytes>,
    ) -> Vec<Option<Vec<u8>>> {
        if keys.is_empty() {
            return Vec::new();
        }

        // resolve column index (cache or load)
        let (col_index_pkey, col_index) = match self.colindex_cache.get(&field_id) {
            Some((pk, ci)) => (*pk, ci.clone()),
            None => {
                // look up in manifest
                let entry = match self
                    .manifest
                    .columns
                    .iter()
                    .find(|e| e.field_id == field_id)
                {
                    Some(e) => e.clone(),
                    None => {
                        // column is unknown
                        return vec![None; keys.len()];
                    }
                };
                let got: Vec<ColumnIndex> = self
                    .pager
                    .batch_get_typed::<ColumnIndex>(&[entry.column_index_physical_key]);
                let ci = got.into_iter().next().expect("ColumnIndex missing");
                self.colindex_cache
                    .insert(field_id, (entry.column_index_physical_key, ci.clone()));
                (entry.column_index_physical_key, ci)
            }
        };
        let _ = col_index_pkey;

        // Decide candidate segments directly from ColumnIndex (newest-first),
        // and record both index_pk and data_pk so we can prefetch data blobs
        // WITHOUT decoding segments first.
        let mut per_seg: HashMap<PhysicalKey, Vec<usize>> = HashMap::new();
        let mut seg_to_data: HashMap<PhysicalKey, PhysicalKey> = HashMap::new();

        for (qi, k) in keys.iter().enumerate() {
            for segref in &col_index.segments {
                if segref.logical_key_min.as_slice() <= k.as_slice()
                    && k.as_slice() <= segref.logical_key_max.as_slice()
                {
                    per_seg
                        .entry(segref.index_physical_key)
                        .or_default()
                        .push(qi);
                    // map index -> data for the selected segment
                    seg_to_data.insert(segref.index_physical_key, segref.data_physical_key);
                    break; // newest-first: stop at first covering segment
                }
            }
        }

        // If nothing matched, return all None quickly.
        let needed_seg_keys: Vec<PhysicalKey> = per_seg.keys().copied().collect();
        if needed_seg_keys.is_empty() {
            return vec![None; keys.len()];
        }

        // Prefetch data blobs in one batch directly from the column index refs.
        // Build data key list from the exact segments we will open.
        let mut data_keys: Vec<PhysicalKey> = needed_seg_keys
            .iter()
            .filter_map(|idx_pk| seg_to_data.get(idx_pk).copied())
            .collect();
        // de-dup in case multiple queries hit the same segment
        {
            let mut seen = HashSet::with_capacity(data_keys.len());
            data_keys.retain(|k| seen.insert(*k));
        }
        let data_blobs_slices: Vec<&[u8]> = self.pager.batch_get_raw(&data_keys);
        let mut data_map: HashMap<PhysicalKey, &[u8]> = HashMap::with_capacity(data_keys.len());
        for (i, pk) in data_keys.iter().enumerate() {
            data_map.insert(*pk, data_blobs_slices[i]);
        }

        // Load all needed index segments in a single batched typed read.
        let segments: Vec<IndexSegment> =
            self.pager.batch_get_typed::<IndexSegment>(&needed_seg_keys);

        // Map index_physical_key -> loaded segment
        let mut seg_map: HashMap<PhysicalKey, IndexSegment> =
            HashMap::with_capacity(segments.len());
        for (i, pk) in needed_seg_keys.iter().enumerate() {
            seg_map.insert(*pk, segments[i].clone());
        }

        // answer vector
        let mut out: Vec<Option<Vec<u8>>> = vec![None; keys.len()];

        // For each segment: binary search the keys it owns; then slice from data blob
        for (seg_pk, qidxs) in per_seg {
            let seg = seg_map.get(&seg_pk).unwrap();
            let data_pk = *seg_to_data.get(&seg_pk).unwrap();
            let data_blob = data_map.get(&data_pk).unwrap();

            for qi in qidxs {
                let target = keys[qi].as_slice();
                if let Some(pos) =
                    binary_search_key(&seg.logical_key_bytes, &seg.logical_key_offsets, target)
                {
                    match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => {
                            let w = *width as usize;
                            let a = pos * w;
                            let b = a + w;
                            out[qi] = Some(data_blob[a..b].to_vec());
                        }
                        ValueLayout::Variable { value_offsets } => {
                            let a = value_offsets[pos] as usize;
                            let b = value_offsets[pos + 1] as usize;
                            out[qi] = Some(data_blob[a..b].to_vec());
                        }
                    }
                } // else None (not found inside)
            }
        }

        out
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
        let mut header_keys = vec![bootstrap_key, self.manifest_key];
        let header_raw = self.pager.batch_get_raw(&header_keys);
        // bootstrap
        out.push(StorageNode {
            pk: bootstrap_key,
            stored_len: header_raw[0].len(),
            kind: StorageKind::Bootstrap,
        });
        // manifest
        out.push(StorageNode {
            pk: self.manifest_key,
            stored_len: header_raw[1].len(),
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
        let mut field_for_colindex: HashMap<PhysicalKey, LogicalFieldId> = HashMap::new();
        let mut all_seg_index_pks: Vec<PhysicalKey> = Vec::new();
        let mut seg_owner_colindex: HashMap<PhysicalKey, (LogicalFieldId, PhysicalKey)> =
            HashMap::new();

        if !col_index_pks.is_empty() {
            let colindex_raw = self.pager.batch_get_raw(&col_index_pks);
            let colindices: Vec<ColumnIndex> =
                self.pager.batch_get_typed::<ColumnIndex>(&col_index_pks);

            for (i, ci) in colindices.into_iter().enumerate() {
                let pk = col_index_pks[i];
                field_for_colindex.insert(pk, ci.field_id);
                colindex_nodes.push(StorageNode {
                    pk,
                    stored_len: colindex_raw[i].len(),
                    kind: StorageKind::ColumnIndex {
                        field_id: ci.field_id,
                        n_segments: ci.segments.len() as usize,
                    },
                });

                // collect segment index pkeys and owners
                for sref in ci.segments {
                    all_seg_index_pks.push(sref.index_physical_key);
                    seg_owner_colindex.insert(sref.index_physical_key, (ci.field_id, pk));
                }
            }
        }

        out.extend(colindex_nodes);

        // --- index segments (typed + raw) and discover data pkeys
        let mut data_pkeys: Vec<PhysicalKey> = Vec::new();
        let mut owner_for_data: HashMap<PhysicalKey, PhysicalKey> = HashMap::new(); // data_pkey -> index_segment_pk
        let mut seg_nodes: Vec<StorageNode> = Vec::new();

        if !all_seg_index_pks.is_empty() {
            let seg_raw = self.pager.batch_get_raw(&all_seg_index_pks);
            let segs: Vec<IndexSegment> = self
                .pager
                .batch_get_typed::<IndexSegment>(&all_seg_index_pks);

            for (i, seg) in segs.into_iter().enumerate() {
                let seg_pk = all_seg_index_pks[i];
                let (field_id, owner_colindex_pk) =
                    seg_owner_colindex.get(&seg_pk).cloned().unwrap();

                // book-keep data blob for later
                data_pkeys.push(seg.data_physical_key);
                owner_for_data.insert(seg.data_physical_key, seg_pk);

                // compute layout info
                let key_bytes = seg.logical_key_bytes.len();
                let key_offs_bytes =
                    seg.logical_key_offsets.len() * std::mem::size_of::<IndexEntryCount>();
                let (kind, fixed_width, value_meta_bytes) = match &seg.value_layout {
                    ValueLayout::FixedWidth { width } => {
                        ("fixed", Some(*width), std::mem::size_of::<u32>())
                    }
                    ValueLayout::Variable { value_offsets } => (
                        "variable",
                        None,
                        value_offsets.len() * std::mem::size_of::<u32>(),
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
                    pk: seg_pk,
                    stored_len: seg_raw[i].len(),
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

        out.extend(seg_nodes);

        // --- data blobs (raw only)
        if !data_pkeys.is_empty() {
            let data_raw = self.pager.batch_get_raw(&data_pkeys);
            for (i, dpk) in data_pkeys.iter().enumerate() {
                let owner = *owner_for_data.get(dpk).unwrap();
                out.push(StorageNode {
                    pk: *dpk,
                    stored_len: data_raw[i].len(),
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
    pub fn render_storage_ascii(&mut self) -> String {
        let nodes = self.describe_storage();
        let mut s = String::new();
        let _ = writeln!(
            &mut s,
            "{:<10} {:<12} {:<9} {:<40} {:>8}",
            "phys_key", "kind", "field", "details", "bytes"
        );
        let _ = writeln!(
            &mut s,
            "{}",
            "-".repeat(10 + 1 + 12 + 1 + 9 + 1 + 40 + 1 + 8)
        );

        for n in nodes {
            match n.kind {
                StorageKind::Bootstrap => {
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:<40} {:>8}",
                        n.pk, "bootstrap", "-", "-", n.stored_len
                    );
                }
                StorageKind::Manifest { column_count } => {
                    let det = format!("columns={}", column_count);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:<40} {:>8}",
                        n.pk, "manifest", "-", det, n.stored_len
                    );
                }
                StorageKind::ColumnIndex {
                    field_id,
                    n_segments,
                } => {
                    let det = format!("segments={}", n_segments);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:<40} {:>8}",
                        n.pk, "col_index", field_id, det, n.stored_len
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
                        "{:<10} {:<12} {:<9} {:<40} {:>8}",
                        n.pk, "idx_segment", field_id, det, n.stored_len
                    );
                }
                StorageKind::DataBlob { owner_index_pk } => {
                    let det = format!("owner_idx_pk={}", owner_index_pk);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:<40} {:>8}",
                        n.pk, "data_blob", "-", det, n.stored_len
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
        let mut map: HashMap<PhysicalKey, &StorageNode> = HashMap::new();
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
    use bitcode::{Decode, Encode};
    use std::collections::HashMap;

    // in-memory pager for tests
    struct MemPager {
        map: HashMap<PhysicalKey, Vec<u8>>,
        next: PhysicalKey,
    }

    impl Default for MemPager {
        fn default() -> Self {
            Self {
                map: HashMap::new(),
                next: 1, // reserve 0 for bootstrap
            }
        }
    }

    impl Pager for MemPager {
        fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey> {
            let start = self.next;
            self.next += n as u64;
            (0..n).map(|i| start + i as u64).collect()
        }
        fn batch_put_raw(&mut self, items: &[(PhysicalKey, Vec<u8>)]) {
            for (k, v) in items {
                self.map.insert(*k, v.clone());
            }
        }
        fn batch_get_raw<'a>(&'a self, keys: &[PhysicalKey]) -> Vec<&'a [u8]> {
            keys.iter()
                .map(|k| self.map.get(k).expect("missing key").as_slice())
                .collect()
        }
        fn batch_put_typed<T: Encode>(&mut self, items: &[(PhysicalKey, T)]) {
            let enc: Vec<(PhysicalKey, Vec<u8>)> = items
                .iter()
                .map(|(k, v)| (*k, bitcode::encode(v)))
                .collect();
            self.batch_put_raw(&enc);
        }
        fn batch_get_typed<T>(&self, keys: &[PhysicalKey]) -> Vec<T>
        where
            for<'a> T: Decode<'a>,
        {
            self.batch_get_raw(keys)
                .into_iter()
                .map(|b| bitcode::decode(b).expect("decode fail"))
                .collect()
        }
    }

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

        let got = store.get_in_column(10, vec![b"k1".to_vec(), b"k2".to_vec(), b"kX".to_vec()]);
        assert_eq!(got[0].as_ref().unwrap(), b"NEWVVVV1");
        assert_eq!(got[1].as_ref().unwrap(), b"VVVVVVV2");
        assert!(got[2].is_none());
    }

    #[test]
    fn put_get_variable_auto_with_chunking() {
        let mut p = MemPager::default();
        let mut store = ColumnStore::init_empty(&mut p);

        // Values of different sizes force variable layout.
        let mut items = Vec::new();
        for i in 0..1000u32 {
            let k = format!("key{:04}", i).into_bytes();
            let v = vec![b'A' + (i % 26) as u8; (i % 17 + 1) as usize]; // variable 1..17
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
        let got = store.get_in_column(77, q.clone());

        // recompute expected lengths
        let expect_len = |s: &str| -> usize {
            let i: u32 = s[3..].parse().unwrap();
            (i % 17 + 1) as usize
        };

        assert_eq!(got[0].as_ref().unwrap().len(), expect_len("key0000"));
        assert_eq!(got[1].as_ref().unwrap().len(), expect_len("key0001"));
        assert_eq!(got[2].as_ref().unwrap().len(), expect_len("key0199"));
        assert_eq!(got[3].as_ref().unwrap().len(), expect_len("key0200"));
        assert_eq!(got[4].as_ref().unwrap().len(), expect_len("key0999"));
        assert!(got[5].is_none());
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

        let got = store.get_in_column(5, vec![b"a".to_vec(), b"b".to_vec(), b"z".to_vec()]);
        assert_eq!(got[0].as_ref().unwrap(), &[1u8; 4]);
        assert_eq!(got[1].as_ref().unwrap(), &[2u8; 4]);
        assert!(got[2].is_none());
    }
}
