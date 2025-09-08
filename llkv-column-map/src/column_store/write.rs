use super::ColumnStore;
use crate::bounds::ValueBound;
use crate::column_index::{
    ColumnEntry, ColumnIndex, IndexSegment, IndexSegmentRef, ValueDirL2, ValueIndex,
};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::types::{
    AppendOptions, ByteLen, ByteWidth, IndexEntryCount, LogicalFieldId, LogicalKeyBytes,
    PhysicalKey, Put, TypedKind, TypedValue, ValueMode,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;

// Normalize per column: last-write-wins (optional), then sort by logical key.
// Also decide value layout for this batch (fixed width or variable).
#[derive(Clone)]
enum PlannedWriteLayout {
    Fixed { width: ByteWidth },
    Variable, // offsets will be computed per segment
}
struct PlannedWriteChunk {
    field_id: LogicalFieldId,
    keys_sorted: Vec<LogicalKeyBytes>,
    values: Vec<Vec<u8>>, // aligned to keys
    layout: PlannedWriteLayout,
}

impl<'p, P: Pager> ColumnStore<'p, P> {
    // TODO: Return `Result` type
    // Single entrypoint for writing. Many columns, each with unordered items.
    // Auto-chooses fixed vs variable, chunks to segments, writes everything in batches.
    /// Single entrypoint for writing. Many columns, each with unordered items.
    /// Auto-chooses fixed vs variable, chunks to segments, writes everything
    /// in batches.
    pub fn append_many(&self, puts: Vec<Put>, opts: AppendOptions) {
        if puts.is_empty() {
            return;
        }

        let mut planned_chunks: Vec<PlannedWriteChunk> = Vec::new();
        for mut put in puts {
            if put.items.is_empty() {
                continue;
            }

            if opts.last_write_wins_in_batch {
                let mut last: FxHashMap<&[u8], Cow<[u8]>> =
                    FxHashMap::with_capacity_and_hasher(put.items.len(), Default::default());
                for (k, v) in put.items.iter() {
                    last.insert(k.as_ref(), v.clone());
                }
                put.items = last
                    .into_iter()
                    .map(|(k, v)| (Cow::Owned(k.to_vec()), v))
                    .collect();
            }

            // sort by key
            put.items.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));

            let (mut keys_sorted, mut values): (Vec<Vec<u8>>, Vec<Vec<u8>>) = put
                .items
                .into_iter()
                .map(|(k, v)| (k.into_owned(), v.into_owned()))
                .unzip();

            // decide layout
            let layout = match opts.mode {
                ValueMode::ForceFixed(w) => {
                    assert!(w > 0, "fixed width must be > 0");
                    for v in &values {
                        assert!(
                            v.len() == w as usize,
                            "ForceFixed width mismatch: expected {}, got {}",
                            w,
                            v.len()
                        );
                    }
                    PlannedWriteLayout::Fixed { width: w }
                }
                ValueMode::ForceVariable => PlannedWriteLayout::Variable,
                ValueMode::Auto => {
                    if !values.is_empty() {
                        let w = values[0].len();
                        if w > 0 && values.iter().all(|v| v.len() == w) {
                            PlannedWriteLayout::Fixed {
                                width: w as ByteWidth,
                            }
                        } else {
                            PlannedWriteLayout::Variable
                        }
                    } else {
                        PlannedWriteLayout::Variable // Default for empty
                    }
                }
            };

            // Chunk the data by draining vectors instead of slicing and cloning.
            while !keys_sorted.is_empty() {
                let remain = keys_sorted.len();
                let take_by_entries = core::cmp::min(remain, opts.segment_max_entries);

                let take = match layout {
                    PlannedWriteLayout::Fixed { width } => {
                        // entries are uniform; respect both entry and byte budgets
                        let max_entries_by_bytes = if width == 0 {
                            take_by_entries
                        } else {
                            (opts.segment_max_bytes / (width as usize)).max(1)
                        };
                        core::cmp::min(take_by_entries, max_entries_by_bytes.max(1))
                    }
                    PlannedWriteLayout::Variable => {
                        // grow until adding next would exceed bytes or entries
                        let mut acc = 0usize;
                        let mut cnt = 0usize;
                        for v in values.iter().take(take_by_entries) {
                            let sz = v.len();
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

                let chunk_keys: Vec<Vec<u8>> = keys_sorted.drain(..take).collect();
                let chunk_values: Vec<Vec<u8>> = values.drain(..take).collect();

                planned_chunks.push(PlannedWriteChunk {
                    field_id: put.field_id,
                    keys_sorted: chunk_keys,
                    values: chunk_values,
                    layout: layout.clone(),
                });
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

        // Build concrete IndexSegments and data blobs; group by column to update
        // ColumnIndex.
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

        // For any columns still missing, create fresh ColumnIndex and manifest
        // entry.
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

        // Now materialize each planned chunk into data blob + IndexSegment, and
        // update ColumnIndex.
        for (i, chunk) in planned_chunks.into_iter().enumerate() {
            let data_pkey = data_keys[i];
            let index_pkey = index_keys[i];

            // compute value bounds once per segment
            let (vmin, vmax) = ValueBound::min_max_bounds(&chunk.values);

            // build data blob (+ value offsets for variable)
            let (data_blob, mut seg, n_entries) = match chunk.layout {
                PlannedWriteLayout::Fixed { width } => {
                    // values blob
                    let blob = chunk.values.concat();
                    // index segment (keys packed + fixed value layout)
                    let seg = IndexSegment::build_fixed(data_pkey, &chunk.keys_sorted, width);
                    (blob, seg, chunk.values.len() as IndexEntryCount)
                }
                PlannedWriteLayout::Variable => {
                    // values blob
                    let mut blob = Vec::new();
                    let mut sizes: Vec<ByteLen> = Vec::with_capacity(chunk.values.len());
                    for v in &chunk.values {
                        blob.extend_from_slice(v);
                        sizes.push(v.len() as ByteLen);
                    }
                    // index segment (keys packed + var value layout)
                    let seg = IndexSegment::build_var(data_pkey, &chunk.keys_sorted, &sizes);
                    (blob, seg, chunk.values.len() as IndexEntryCount)
                }
            };

            // Compute value index from the in-memory values for this segment.
            let hot_threshold = 4096; // tune later; promotes big L1 buckets
            let (value_order, l1_dir, l2_pairs) =
                Self::build_value_index_from_values(&chunk.values, hot_threshold);
            let value_index = ValueIndex {
                value_order: value_order
                    .into_iter()
                    .map(|v| v as IndexEntryCount)
                    .collect(),
                l1_dir: l1_dir.into_iter().map(|v| v as IndexEntryCount).collect(),
                l2_dirs: l2_pairs
                    .into_iter()
                    .map(|(first_byte, dir257)| ValueDirL2 {
                        first_byte,
                        dir257: dir257.into_iter().map(|v| v as IndexEntryCount).collect(),
                    })
                    .collect(),
            };
            seg.value_index = Some(value_index);

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

                // mark this ColumnIndex as touched so we persist only these
                // (ingest perf fix)
                touched_colindex_pkeys.insert(*col_pkey);
            }
        }

        // Persist: data blobs, index segments, column indexes, manifest (if needed).
        // Combine everything into a SINGLE batch.
        let mut puts_batch: Vec<BatchPut> = Vec::new();
        for (k, v) in data_puts {
            puts_batch.push(BatchPut::Raw { key: k, bytes: v });
        }
        for (k, seg) in index_puts {
            puts_batch.push(BatchPut::Typed {
                key: k,
                value: TypedValue::IndexSegment(seg),
            });
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
                    assert!(seen.insert(k), "duplicate PUT for key {k}");
                }
            }
            self.do_puts(puts_batch);
        }
    }

    // TODO: Return `Result` type
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
                items: keys
                    .into_iter()
                    .map(|k| (Cow::Owned(k), Cow::Borrowed(&[] as &[u8])))
                    .collect(),
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

    /// Build value_order + l1/l2 directories from in-memory `values`.
    /// `hot_threshold`: promote a first-byte bucket to L2 if its span >= this.
    fn build_value_index_from_values(
        values: &[Vec<u8>], // TODO: More efficient input?
        hot_threshold: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<(u8, Vec<u32>)>) {
        let n = values.len();
        let mut idx: Vec<u32> = (0..n as u32).collect();

        // Stable sort by raw bytes (lexicographic).
        idx.sort_by(|&a, &b| values[a as usize].cmp(&values[b as usize]));

        // L1: 257 absolute offsets.
        let mut l1 = vec![0u32; 257];
        let mut cur = 0usize;
        for b0 in 0..256usize {
            while cur < n
                && values[idx[cur] as usize]
                    .get(0)
                    .map(|v| *v as usize)
                    .unwrap_or(0)
                    == b0
            {
                cur += 1;
            }
            l1[b0 + 1] = cur as u32;
        }

        // L2: sparse second-byte split for hot first-byte buckets.
        let mut l2: Vec<(u8, Vec<u32>)> = Vec::new();
        for b0 in 0..256usize {
            let start = l1[b0] as usize;
            let end = l1[b0 + 1] as usize;
            let len = end - start;
            if len >= hot_threshold {
                let mut dir = vec![0u32; 257];
                let mut cur2 = start;
                for b1 in 0..256usize {
                    while cur2 < end {
                        let s = &values[idx[cur2] as usize];
                        let v1 = if s.len() >= 2 { s[1] as usize } else { 0 };
                        if v1 != b1 {
                            break;
                        }
                        cur2 += 1;
                    }
                    dir[b1 + 1] = (cur2 - start) as u32;
                }
                l2.push((b0 as u8, dir));
            }
        }

        (idx, l1, l2)
    }
}
