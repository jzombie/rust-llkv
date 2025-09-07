use super::ColumnStore;
use crate::bounds::ValueBound;
use crate::index::{ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::types::{
    ByteLen, ByteWidth, IndexEntryCount, LogicalFieldId, LogicalKeyBytes, PhysicalKey, TypedKind,
    TypedValue,
};
use rustc_hash::{FxHashMap, FxHashSet};

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

impl<'p, P: Pager> ColumnStore<'p, P> {
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
        let opts = AppendOptions {
            mode: ValueMode::ForceVariable,
            last_write_wins_in_batch: true,
            ..Default::default()
        };

        self.append_many(puts, opts);
    }
}
