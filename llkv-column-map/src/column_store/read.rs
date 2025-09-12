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

impl<P: Pager> ColumnStore<P> {
    // TODO: Return `Result` type
    /// Convenience: shared-keyset API that uses the single core.
    pub fn get_many_projected<'a>(
        &self,
        fids: &[LogicalFieldId],
        keys: &'a [&'a [u8]],
    ) -> Vec<Vec<Option<ValueSlice<P::Blob>>>> {
        self.get_many_core(fids, keys)
    }

    // TODO: Return `Result` type
    /// Public API: accepts arbitrary (fid, keys) groups. Internally
    /// normalizes to a single deduped keyset and calls the single core.
    pub fn get_many(
        &self,
        items: Vec<(LogicalFieldId, Vec<LogicalKeyBytes>)>,
    ) -> Vec<Vec<Option<ValueSlice<P::Blob>>>> {
        if items.is_empty() {
            return Vec::new();
        }

        // Build unified (deduped) keyset and a mapping from each original key
        // to its unified index.
        let mut unified_keys: Vec<&[u8]> = Vec::new();
        let mut key_to_idx: FxHashMap<&[u8], usize> = FxHashMap::default();

        // Keep fids unique for the core call, but remember mapping for rebuild.
        let mut unique_fids: Vec<LogicalFieldId> = Vec::new();
        let mut fid_to_u: FxHashMap<LogicalFieldId, usize> = FxHashMap::default();

        for (fid, keys) in &items {
            if !fid_to_u.contains_key(fid) {
                let u = unique_fids.len();
                unique_fids.push(*fid);
                fid_to_u.insert(*fid, u);
            }
            for k in keys {
                let ks = k.as_slice();
                if let Entry::Vacant(e) = key_to_idx.entry(ks) {
                    let u = unified_keys.len();
                    unified_keys.push(ks);
                    e.insert(u);
                }
            }
        }

        // Execute the single core path once for all columns and unified keys.
        let core = self.get_many_core(&unique_fids, &unified_keys);

        // Rebuild outputs in the original (fid, keys) grouping/shape.
        let mut out: Vec<Vec<Option<ValueSlice<P::Blob>>>> =
            items.iter().map(|(_, ks)| vec![None; ks.len()]).collect();

        for (qi, (fid, keys)) in items.iter().enumerate() {
            let fi = match fid_to_u.get(fid) {
                Some(i) => *i,
                None => {
                    // Shouldn't happen (we inserted above), but keep safe.
                    continue;
                }
            };
            for (kj, k) in keys.iter().enumerate() {
                if let Some(&uk) = key_to_idx.get(k.as_slice()) {
                    out[qi][kj] = core[fi][uk].clone();
                }
            }
        }

        out
    }

    // TODO: Return `Result` type
    /// Core: single codepath that executes a batched point-lookup for
    /// a set of columns (`fids`) over a single shared keyset (`keys`).
    fn get_many_core(
        &self,
        fids: &[LogicalFieldId],
        keys: &[&[u8]],
    ) -> Vec<Vec<Option<ValueSlice<P::Blob>>>> {
        if fids.is_empty() || keys.is_empty() {
            return (0..fids.len()).map(|_| Vec::new()).collect();
        }

        // Output: one vec per fid, one slot per key.
        let mut results: Vec<Vec<Option<ValueSlice<P::Blob>>>> =
            (0..fids.len()).map(|_| vec![None; keys.len()]).collect();

        // (fid_index, key_index) pairs that have been satisfied (value or tombstone).
        let mut resolved: FxHashSet<(usize, usize)> = FxHashSet::default();

        // -------- ensure ColumnIndex in cache for all referenced fields --------
        let mut to_load: Vec<(LogicalFieldId, PhysicalKey)> = Vec::new();
        {
            let cache = self.colindex_cache.read().unwrap();
            let man = self.manifest.read().unwrap();

            for &fid in fids {
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

        // -------- routing: segment -> (data_pk, [(fid_i, key_j), ...]) --------
        // and record a global newest-first rank per segment pk.
        let mut per_seg: FxHashMap<PhysicalKey, (PhysicalKey, Vec<(usize, usize)>)> =
            FxHashMap::default();
        let mut seg_rank: FxHashMap<PhysicalKey, usize> = FxHashMap::default();

        {
            let cache = self.colindex_cache.read().unwrap();

            for (fi, fid) in fids.iter().enumerate() {
                let col_index = match cache.get(fid) {
                    Some((_, ci)) => ci,
                    None => continue, // unknown field => remains all None
                };

                for (rank, segref) in col_index.segments.iter().enumerate() {
                    seg_rank
                        .entry(segref.index_physical_key)
                        .and_modify(|r| *r = (*r).min(rank))
                        .or_insert(rank);
                }

                for (kj, &k) in keys.iter().enumerate() {
                    for segref in &col_index.segments {
                        if segref.logical_key_min.as_slice() <= k
                            && k <= segref.logical_key_max.as_slice()
                        {
                            match per_seg.entry(segref.index_physical_key) {
                                Entry::Occupied(mut e) => e.get_mut().1.push((fi, kj)),
                                Entry::Vacant(e) => {
                                    e.insert((segref.data_physical_key, vec![(fi, kj)]));
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

        // Newest-first across all segments.
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

            for (fi, kj) in pairs {
                if resolved.contains(&(fi, kj)) {
                    continue;
                }

                let target = keys[kj];
                if let Some(pos) = KeyLayout::binary_search_key_with_layout(
                    &seg.logical_key_bytes,
                    &seg.key_layout,
                    seg.n_entries as usize,
                    target,
                ) {
                    match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => {
                            if *width == 0 {
                                // tombstone
                                resolved.insert((fi, kj));
                                continue;
                            }
                            if let Some(data_blob) = data_map.get(&data_pk) {
                                let w = *width as usize;
                                let a = (pos * w) as ByteOffset;
                                let b = a + w as ByteOffset;
                                results[fi][kj] = Some(ValueSlice {
                                    data: data_blob.clone(),
                                    start: a,
                                    end: b,
                                });
                            }
                            resolved.insert((fi, kj));
                        }
                        ValueLayout::Variable { value_offsets } => {
                            let a = value_offsets[pos] as ByteOffset;
                            let b = value_offsets[pos + 1] as ByteOffset;

                            if a == b {
                                // TODO: If these are tombstones, how do we store potential null values?  Skip them entirely?
                                // tombstone
                                resolved.insert((fi, kj));
                                continue;
                            }

                            if let Some(data_blob) = data_map.get(&data_pk) {
                                results[fi][kj] = Some(ValueSlice {
                                    data: data_blob.clone(),
                                    start: a,
                                    end: b,
                                });
                            }
                            resolved.insert((fi, kj));
                        }
                    }
                }
            }
        }

        results
    }
}
