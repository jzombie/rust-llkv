use super::ColumnStore;
use crate::column_index::{ColumnIndex, IndexSegment};
use crate::layout::{KeyLayout, ValueLayout};
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::types::{
    ByteOffset, LogicalFieldId, LogicalKeyBytes, PhysicalKey, TypedKind, TypedValue,
};
use crate::views::ValueSlice;
use std::collections::hash_map::Entry;

use rustc_hash::{FxHashMap, FxHashSet};

impl<'p, P: Pager> ColumnStore<'p, P> {
    // TODO: Return `Result` type
    /// Batched point lookups across many columns.
    /// Each item is (field_id, keys). Output is aligned per input.
    ///
    /// I/O pattern (whole batch):
    ///   1) batch(ColumnIndex gets) for any missing columns (once)
    ///   2) batch(IndexSegment typed + DataBlob raw gets) (once)
    ///
    /// Newest-first shadowing: for each key we pick the first segment whose
    /// \[min,max\] covers it.
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
