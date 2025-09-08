//! Value-ordered range scan with strict LWW and frame predicate.
//!
//! Iterator uses per-segment value_order + L1/L2 directories to seed
//! one head per candidate segment, and a 2-tier radix PQ to pop the
//! next value across segments. Strict LWW is enforced by probing any
//! newer segment (by recency rank) for membership of the same key.
//!
//! - Newer segments are batch-loaded as IndexSegment (no data blobs).
//! - Active segments (value producers) load index + data blobs once.
//! - Frame predicate is evaluated on encoded values (zero copy).

use crate::bounds::ValueBound;
use crate::column_index::{IndexSegment, ValueIndex};
use crate::layout::{KeyLayout, ValueLayout};
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::types::{LogicalFieldId, PhysicalKey, TypedKind, TypedValue};
use crate::views::ValueSlice;

use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::Bound;
use std::sync::Arc;

/// Direction for value-ordered scans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Reverse,
}

/// Conflict resolution across generations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictPolicy {
    /// Newest write wins: suppress older versions of the same key.
    LWW,
    /// First (oldest) write wins: suppress newer versions of the same key.
    FWW,
}

/// Frame predicate type over encoded **values** (head vs current).
pub type FramePred = Arc<dyn Fn(&[u8], &[u8]) -> bool + Send + Sync>;

/// Scan options for value-ordered range scans.
pub struct ValueScanOpts<'a> {
    pub dir: Direction,
    pub lo: Bound<&'a [u8]>,
    pub hi: Bound<&'a [u8]>,
    /// Optional prefix to prune quickly (value-side).
    pub prefix: Option<&'a [u8]>,
    /// First 2 bytes are the bucket id. Keep at 2 for now.
    pub bucket_prefix_len: usize,
    /// Head tag width in bytes used for fast compares.
    pub head_tag_len: usize,
    /// Optional end-of-frame predicate on encoded **values**.
    pub frame_predicate: Option<FramePred>,
}

impl<'a> Default for ValueScanOpts<'a> {
    fn default() -> Self {
        Self {
            dir: Direction::Forward,
            lo: Bound::Unbounded,
            hi: Bound::Unbounded,
            prefix: None,
            bucket_prefix_len: 2,
            head_tag_len: 8,
            frame_predicate: None,
        }
    }
}

/// Public item: zero-copy value slice + owned key bytes.
pub struct ValueScanItem<B> {
    pub key: Vec<u8>,
    pub value: ValueSlice<B>,
}

/// Internal: context per active segment.
struct SegCtx<P: Pager> {
    _seg_pk: PhysicalKey,
    _data_pk: PhysicalKey,
    seg: IndexSegment,
    data: P::Blob,
    /// Newest-first order rank (0 == newest).
    rank: usize,
    /// Window in value_index.value_order: [begin, end).
    begin: usize,
    end: usize,
}

/// Minimal metadata for shadow (index-only) membership checks.
struct ShadowRef {
    pk: PhysicalKey,
    rank: usize,
}

/// The iterator. Owns the PQ and segment cursors.
pub struct ValueScan<P: Pager> {
    segs: Vec<SegCtx<P>>,
    pq: RadixPq<P>,
    _field_id: LogicalFieldId,
    _lo: Bound<Vec<u8>>,
    _hi: Bound<Vec<u8>>,
    reverse: bool,
    frame_pred: Option<FramePred>,           // evaluated on VALUE bytes
    frame_head: Option<ValueSlice<P::Blob>>, // first emitted VALUE slice (zero-copy)
    halted: bool,
    /// Shadow refs grouped by rank for conflict evaluation.
    shadow_by_rank: Vec<Vec<ShadowRef>>,
    /// Prefetched index blobs for shadow refs (no data blobs).
    shadow_seg_map: FxHashMap<PhysicalKey, IndexSegment>,
    /// PQ tuning (copied from opts).
    bucket_prefix_len: usize,
    head_tag_len: usize,
    /// Conflict policy.
    policy: ConflictPolicy,
}

/// Min-heap per bucket with a 65,536-bit tier-1 directory.
/// We keep one BinaryHeap<Reverse<Node>> per active bucket.
/// Node carries a `reverse` flag that flips compare semantics.
struct RadixPq<P: Pager> {
    bitset: [u64; 1024], // 1024 * 64 = 65,536 bits
    buckets: FxHashMap<u16, BinaryHeap<std::cmp::Reverse<Node<P>>>>,
}

impl<P: Pager> RadixPq<P> {
    fn new() -> Self {
        Self {
            bitset: [0u64; 1024],
            buckets: FxHashMap::default(),
        }
    }

    fn push(&mut self, node: Node<P>) {
        let b = node.bucket;
        let idx = (b / 64) as usize;
        let bit = 1u64 << (b % 64);
        self.buckets
            .entry(b)
            .or_insert_with(BinaryHeap::new)
            .push(std::cmp::Reverse(node));
        self.bitset[idx] |= bit;
    }

    fn pop(&mut self, reverse: bool) -> Option<Node<P>> {
        let bi = self.find_bucket(reverse)?;
        let heap = self.buckets.get_mut(&bi)?;
        let out = heap.pop()?.0;
        if heap.is_empty() {
            let idx = (bi / 64) as usize;
            let bit = 1u64 << (bi % 64);
            self.bitset[idx] &= !bit;
            self.buckets.remove(&bi);
        }
        Some(out)
    }

    fn find_bucket(&self, reverse: bool) -> Option<u16> {
        if !reverse {
            for (i, word) in self.bitset.iter().enumerate() {
                if *word == 0 {
                    continue;
                }
                let t = word.trailing_zeros() as u16;
                return Some(((i as u16) * 64) + t);
            }
        } else {
            for (i, word) in self.bitset.iter().enumerate().rev() {
                if *word == 0 {
                    continue;
                }
                let lz = word.leading_zeros() as u16;
                let bit = 63u16.saturating_sub(lz);
                return Some(((i as u16) * 64) + bit);
            }
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

/// PQ node: one active head (segment cursor).
#[derive(Clone)]
struct Node<P: Pager> {
    bucket: u16,
    head_tag: u64,
    /// Newest-first tie-breaker: lower rank means newer.
    rank: usize,
    seg_idx: usize,
    pos: usize,
    val: ValueSlice<P::Blob>,
    reverse: bool,
}

impl<P: Pager> Node<P> {
    #[inline]
    fn bytes<'a>(&'a self) -> &'a [u8] {
        let a = self.val.start as usize;
        let b = self.val.end as usize;
        &self.val.data.as_ref()[a..b]
    }
}

/// Ordering toggles by `reverse` for value fields. Rank is always asc
/// (newer first) on ties.
impl<P: Pager> Eq for Node<P> {}
impl<P: Pager> PartialEq for Node<P> {
    fn eq(&self, other: &Self) -> bool {
        self.bucket == other.bucket
            && self.head_tag == other.head_tag
            && self.bytes() == other.bytes()
            && self.rank == other.rank
            && self.pos == other.pos
            && self.reverse == other.reverse
    }
}
impl<P: Pager> Ord for Node<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap takes the greatest; we wrap with Reverse,
        // so this defines ascending for Forward and descending for Reverse.
        let mut ord = if !self.reverse {
            self.bucket.cmp(&other.bucket)
        } else {
            other.bucket.cmp(&self.bucket)
        };
        if ord != Ordering::Equal {
            return ord;
        }

        ord = if !self.reverse {
            self.head_tag.cmp(&other.head_tag)
        } else {
            other.head_tag.cmp(&self.head_tag)
        };
        if ord != Ordering::Equal {
            return ord;
        }

        ord = if !self.reverse {
            self.bytes().cmp(other.bytes())
        } else {
            other.bytes().cmp(self.bytes())
        };
        if ord != Ordering::Equal {
            return ord;
        }

        // Newer (smaller rank) first on perfect byte ties.
        ord = self.rank.cmp(&other.rank);
        if ord != Ordering::Equal {
            return ord;
        }

        // Stable within-segment ordering.
        self.pos.cmp(&other.pos)
    }
}
impl<P: Pager> PartialOrd for Node<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Pager> super::ColumnStore<'_, P> {
    /// LWW-enforced, value-ordered scan over [lo, hi).
    pub fn scan_values_lww(
        &self,
        field_id: LogicalFieldId,
        opts: ValueScanOpts<'_>,
    ) -> Result<ValueScan<P>, ScanError> {
        ValueScan::new(self, field_id, opts, ConflictPolicy::LWW)
    }

    /// FWW-enforced, value-ordered scan over [lo, hi).
    pub fn scan_values_fww(
        &self,
        field_id: LogicalFieldId,
        opts: ValueScanOpts<'_>,
    ) -> Result<ValueScan<P>, ScanError> {
        ValueScan::new(self, field_id, opts, ConflictPolicy::FWW)
    }

    /// Generic: explicit conflict policy.
    pub fn scan_values_with_policy(
        &self,
        field_id: LogicalFieldId,
        opts: ValueScanOpts<'_>,
        policy: ConflictPolicy,
    ) -> Result<ValueScan<P>, ScanError> {
        ValueScan::new(self, field_id, opts, policy)
    }
}

#[derive(Debug)]
pub enum ScanError {
    ColumnMissing(LogicalFieldId),
    NoActiveSegments,
    ValueIndexMissing(PhysicalKey),
    Storage(String),
}

impl<P: Pager> ValueScan<P> {
    pub fn new(
        col: &super::ColumnStore<'_, P>,
        field_id: LogicalFieldId,
        opts: ValueScanOpts<'_>,
        policy: ConflictPolicy,
    ) -> Result<Self, ScanError> {
        // -------- ensure ColumnIndex in cache ----------
        let colindex = {
            let cache = col.colindex_cache.read().unwrap();
            if let Some((_pk, ci)) = cache.get(&field_id) {
                ci.clone()
            } else {
                let man = col.manifest.read().unwrap();
                let ent = man
                    .columns
                    .iter()
                    .find(|e| e.field_id == field_id)
                    .ok_or(ScanError::ColumnMissing(field_id))?;
                drop(cache);
                let resp = col.do_gets(vec![BatchGet::Typed {
                    key: ent.column_index_physical_key,
                    kind: TypedKind::ColumnIndex,
                }]);
                let mut got = None;
                for gr in resp {
                    if let GetResult::Typed {
                        value: TypedValue::ColumnIndex(ci),
                        ..
                    } = gr
                    {
                        got = Some(ci);
                    }
                }
                let ci = got.ok_or_else(|| ScanError::Storage("ColumnIndex missing".into()))?;
                let mut cache_w = col.colindex_cache.write().unwrap();
                cache_w.insert(field_id, (ent.column_index_physical_key, ci.clone()));
                ci
            }
        };

        // -------- prune segments by value_min..value_max ----------
        let (act_refs, seg_rank) = {
            let mut v = Vec::new();
            let mut ranks = FxHashMap::default();
            for (rank, r) in colindex.segments.iter().enumerate() {
                if overlap_bounds(
                    &opts.lo,
                    &opts.hi,
                    r.value_min.as_ref(),
                    r.value_max.as_ref(),
                ) {
                    v.push(r.clone());
                    ranks.insert(r.index_physical_key, rank);
                }
            }
            (v, ranks)
        };
        if act_refs.is_empty() {
            return Err(ScanError::NoActiveSegments);
        }

        // ---- active rank stats ----
        let total_ranks = colindex.segments.len();
        let mut active_ranks: Vec<usize> = Vec::with_capacity(act_refs.len());
        for r in &act_refs {
            active_ranks.push(*seg_rank.get(&r.index_physical_key).unwrap_or(&usize::MAX));
        }
        let min_active_rank = *active_ranks.iter().min().unwrap_or(&0);
        let max_active_rank = *active_ranks.iter().max().unwrap_or(&0);

        // ---- active logical-key spans (borrowed; no clones) ----
        let mut active_spans: Vec<(&[u8], &[u8])> = Vec::new();
        for r in &act_refs {
            active_spans.push((r.logical_key_min.as_slice(), r.logical_key_max.as_slice()));
        }

        // ---- compute shadow refs depending on policy ----
        let mut shadow_refs_all: Vec<ShadowRef> = Vec::new();
        for (rank, r) in colindex.segments.iter().enumerate() {
            let needed = match policy {
                // rank 0 is newest; include strictly newer than some active
                ConflictPolicy::LWW => rank < max_active_rank,
                // include strictly older than some active
                ConflictPolicy::FWW => rank > min_active_rank,
            };
            if !needed {
                continue;
            }
            if overlaps_any(
                r.logical_key_min.as_slice(),
                r.logical_key_max.as_slice(),
                &active_spans,
            ) {
                shadow_refs_all.push(ShadowRef {
                    pk: r.index_physical_key,
                    rank,
                });
            }
        }

        // ---- build gets: active (index+data) + shadow (index-only) ----
        let mut gets: Vec<BatchGet> =
            Vec::with_capacity(act_refs.len() * 2 + shadow_refs_all.len());
        let mut active_index_pks = FxHashSet::default();
        let mut active_data_pks = FxHashSet::default();
        for r in &act_refs {
            active_index_pks.insert(r.index_physical_key);
            active_data_pks.insert(r.data_physical_key);
            gets.push(BatchGet::Typed {
                key: r.index_physical_key,
                kind: TypedKind::IndexSegment,
            });
            gets.push(BatchGet::Raw {
                key: r.data_physical_key,
            });
        }
        {
            let mut seen = FxHashSet::default();
            for rf in &shadow_refs_all {
                if seen.insert(rf.pk) && !active_index_pks.contains(&rf.pk) {
                    gets.push(BatchGet::Typed {
                        key: rf.pk,
                        kind: TypedKind::IndexSegment,
                    });
                }
            }
        }

        // ---- fetch everything in one go ----
        let mut resp = col.do_gets(gets);

        let mut seg_map: FxHashMap<PhysicalKey, IndexSegment> = FxHashMap::default();
        let mut data_map: FxHashMap<PhysicalKey, P::Blob> = FxHashMap::default();
        let mut shadow_seg_map: FxHashMap<PhysicalKey, IndexSegment> = FxHashMap::default();

        for gr in resp.drain(..) {
            match gr {
                GetResult::Typed {
                    key,
                    value: TypedValue::IndexSegment(seg),
                } => {
                    if active_index_pks.contains(&key) {
                        seg_map.insert(key, seg);
                    } else {
                        shadow_seg_map.insert(key, seg);
                    }
                }
                GetResult::Raw { key, bytes } => {
                    if active_data_pks.contains(&key) {
                        data_map.insert(key, bytes);
                    }
                }
                _ => {}
            }
        }

        // ---- index shadow refs by rank (length = total_ranks) ----
        let mut shadow_by_rank: Vec<Vec<ShadowRef>> = Vec::new();
        shadow_by_rank.resize_with(total_ranks, Vec::new);
        for rf in shadow_refs_all {
            shadow_by_rank[rf.rank].push(rf);
        }

        // -------- build cursors and PQ heads ----------
        let mut segs: Vec<SegCtx<P>> = Vec::with_capacity(act_refs.len());
        let mut pq = RadixPq::new();

        for r in act_refs {
            let seg = seg_map
                .remove(&r.index_physical_key)
                .ok_or_else(|| ScanError::Storage("segment blob missing".into()))?;
            let data = data_map
                .remove(&r.data_physical_key)
                .ok_or_else(|| ScanError::Storage("data blob missing".into()))?;

            let vix = match &seg.value_index {
                Some(v) => v,
                None => return Err(ScanError::ValueIndexMissing(r.index_physical_key)),
            };

            // Compute [begin, end) in value_order for [lo, hi).
            let (begin, end) = range_in_value_index::<P>(&seg, vix, &data, &opts.lo, &opts.hi);
            if begin >= end {
                continue;
            }

            let rank = *seg_rank.get(&r.index_physical_key).unwrap_or(&usize::MAX);

            let seg_idx = segs.len();
            segs.push(SegCtx {
                _seg_pk: r.index_physical_key,
                _data_pk: r.data_physical_key,
                seg,
                data,
                rank,
                begin,
                end,
            });

            // Head depends on direction.
            let head_pos = if matches!(opts.dir, Direction::Reverse) {
                end - 1
            } else {
                begin
            };

            // Push head node.
            let (seg_ref, data_ref) = {
                let s = &segs[seg_idx].seg;
                let d = &segs[seg_idx].data;
                (s, d)
            };
            let val = slice_value::<P>(seg_ref, data_ref, head_pos);
            let (bucket, tag) = {
                let a = val.start as usize;
                let b = val.end as usize;
                let bytes = &val.data.as_ref()[a..b];
                bucket_and_tag(bytes, opts.bucket_prefix_len, opts.head_tag_len)
            };
            pq.push(Node {
                bucket,
                head_tag: tag,
                rank,
                seg_idx,
                pos: head_pos,
                val,
                reverse: matches!(opts.dir, Direction::Reverse),
            });
        }

        if pq.is_empty() {
            return Err(ScanError::NoActiveSegments);
        }

        Ok(Self {
            segs,
            pq,
            _field_id: field_id,
            _lo: clone_bound_bytes(&opts.lo),
            _hi: clone_bound_bytes(&opts.hi),
            reverse: matches!(opts.dir, Direction::Reverse),
            frame_pred: opts.frame_predicate.clone(),
            frame_head: None,
            halted: false,
            shadow_by_rank,
            shadow_seg_map,
            bucket_prefix_len: opts.bucket_prefix_len,
            head_tag_len: opts.head_tag_len,
            policy,
        })
    }

    #[inline]
    fn key_in_index(seg: &IndexSegment, key: &[u8]) -> bool {
        KeyLayout::binary_search_key_with_layout(
            &seg.logical_key_bytes,
            &seg.key_layout,
            seg.n_entries as usize,
            key,
        )
        .is_some()
    }

    /// Is this `key@rank` shadowed by the chosen policy?
    fn is_shadowed(&self, key: &[u8], rank: usize) -> bool {
        // 1) Active segments we already hold (index+data).
        for sc in &self.segs {
            let rel = sc.rank as isize - rank as isize;
            let dominates = match self.policy {
                ConflictPolicy::LWW => rel < 0, // newer dominates
                ConflictPolicy::FWW => rel > 0, // older dominates
            };
            if dominates && Self::key_in_index(&sc.seg, key) {
                return true;
            }
        }

        // 2) Shadow segments (index-only).
        match self.policy {
            ConflictPolicy::LWW => {
                // Check strictly newer ranks [0 .. rank)
                for r in 0..rank {
                    for rf in &self.shadow_by_rank[r] {
                        if let Some(seg) = self.shadow_seg_map.get(&rf.pk) {
                            if Self::key_in_index(seg, key) {
                                return true;
                            }
                        }
                    }
                }
            }
            ConflictPolicy::FWW => {
                // Check strictly older ranks (rank+1 ..)
                for r in (rank + 1)..self.shadow_by_rank.len() {
                    for rf in &self.shadow_by_rank[r] {
                        if let Some(seg) = self.shadow_seg_map.get(&rf.pk) {
                            if Self::key_in_index(seg, key) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

impl<P: Pager> Iterator for ValueScan<P> {
    type Item = ValueScanItem<P::Blob>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.halted {
            return None;
        }

        loop {
            let node = self.pq.pop(self.reverse)?;
            let seg_idx = node.seg_idx;

            // ---- short immutable borrow to slice key and plan next head ----
            let (key_owned, rank, next_pos_opt, next_val_opt) = {
                let sc = &self.segs[seg_idx];
                let row_pos = row_pos_at(&sc.seg, node.pos);
                let key_slice = KeyLayout::slice_key_by_layout(
                    &sc.seg.logical_key_bytes,
                    &sc.seg.key_layout,
                    row_pos,
                );
                let key_owned = key_slice.to_vec();

                let begin = sc.begin;
                let end = sc.end;
                let next_pos_opt = next_pos(node.pos, begin, end, self.reverse);
                let next_val_opt = next_pos_opt.map(|np| slice_value::<P>(&sc.seg, &sc.data, np));
                (key_owned, sc.rank, next_pos_opt, next_val_opt)
            };
            // ---- immutable borrow dropped here ----

            // Conflict policy: skip if shadowed.
            if self.is_shadowed(key_owned.as_slice(), rank) {
                if let (Some(np), Some(val)) = (next_pos_opt, next_val_opt) {
                    let (bkt, tag) = {
                        let a = val.start as usize;
                        let b = val.end as usize;
                        let bytes = &val.data.as_ref()[a..b];
                        bucket_and_tag(bytes, self.bucket_prefix_len, self.head_tag_len)
                    };
                    self.pq.push(Node {
                        bucket: bkt,
                        head_tag: tag,
                        rank,
                        seg_idx,
                        pos: np,
                        val,
                        reverse: self.reverse,
                    });
                }
                continue;
            }

            // Frame predicate on VALUE bytes (head vs current).
            if let Some(pred) = &self.frame_pred {
                if let Some(ref head_slice) = self.frame_head {
                    let ha = head_slice.start as usize;
                    let hb = head_slice.end as usize;
                    let head = &head_slice.data.as_ref()[ha..hb];
                    let ca = node.val.start as usize;
                    let cb = node.val.end as usize;
                    let cur = &node.val.data.as_ref()[ca..cb];
                    if !(pred)(head, cur) {
                        self.halted = true;
                        return None;
                    }
                } else {
                    // initialize head to first emitted value
                    self.frame_head = Some(node.val.clone());
                }
            } else if self.frame_head.is_none() {
                // keep consistent semantics even if predicate added later
                self.frame_head = Some(node.val.clone());
            }

            // Accept: advance this segment by pushing its next head, if any.
            if let (Some(np), Some(val)) = (next_pos_opt, next_val_opt) {
                let (bkt, tag) = {
                    let a = val.start as usize;
                    let b = val.end as usize;
                    let bytes = &val.data.as_ref()[a..b];
                    bucket_and_tag(bytes, self.bucket_prefix_len, self.head_tag_len)
                };
                self.pq.push(Node {
                    bucket: bkt,
                    head_tag: tag,
                    rank,
                    seg_idx,
                    pos: np,
                    val,
                    reverse: self.reverse,
                });
            }

            // Yield current row. Values are zero-copy; key is one owned copy.
            return Some(ValueScanItem {
                key: key_owned,
                value: node.val,
            });
        }
    }
}

// -------- helpers: direction, bounds, seeding, slicing, compare -----

#[inline]
fn next_pos(cur: usize, begin: usize, end: usize, reverse: bool) -> Option<usize> {
    if !reverse {
        let np = cur + 1;
        if np < end { Some(np) } else { None }
    } else if cur > begin {
        Some(cur - 1)
    } else {
        None
    }
}

fn overlaps_any(a_min: &[u8], a_max: &[u8], spans: &[(&[u8], &[u8])]) -> bool {
    for (b_min, b_max) in spans {
        if !(a_max < *b_min || a_min > *b_max) {
            return true;
        }
    }
    false
}

fn overlap_bounds(
    lo: &Bound<&[u8]>,
    hi: &Bound<&[u8]>,
    s_lo: Option<&ValueBound>,
    s_hi: Option<&ValueBound>,
) -> bool {
    // If any bound is missing, keep the segment (cannot safely prune).
    let (slo, shi) = match (s_lo, s_hi) {
        (Some(a), Some(b)) => (a, b),
        _ => return true,
    };

    // If either bound is truncated, keep it (conservative).
    if slo.is_truncated() || shi.is_truncated() {
        return true;
    }

    let slo_bytes = slo.prefix.as_slice();
    let shi_bytes = shi.prefix.as_slice();

    // [seg_lo, seg_hi] vs [lo, hi) intersection check.
    let lower_ok = match lo {
        Bound::Unbounded => true,
        Bound::Included(x) => shi_bytes >= *x,
        Bound::Excluded(x) => shi_bytes > *x,
    };
    if !lower_ok {
        return false;
    }

    let upper_ok = match hi {
        Bound::Unbounded => true,
        Bound::Included(x) => slo_bytes <= *x,
        Bound::Excluded(x) => slo_bytes < *x,
    };
    lower_ok && upper_ok
}

fn range_in_value_index<P: Pager>(
    seg: &IndexSegment,
    vix: &ValueIndex,
    data: &P::Blob,
    lo: &Bound<&[u8]>,
    hi: &Bound<&[u8]>,
) -> (usize, usize) {
    let n = seg.n_entries as usize;
    if n == 0 {
        return (0, 0);
    }
    // lower bound
    let begin = match lo {
        Bound::Unbounded => 0usize,
        Bound::Included(x) | Bound::Excluded(x) => lower_bound_by_value::<P>(seg, vix, data, x),
    };
    // upper bound (exclusive)
    let end = match hi {
        Bound::Unbounded => n,
        Bound::Included(x) => upper_bound_by_value::<P>(seg, vix, data, x, true),
        Bound::Excluded(x) => upper_bound_by_value::<P>(seg, vix, data, x, false),
    };
    (begin.min(n), end.min(n))
}

fn lower_bound_by_value<P: Pager>(
    seg: &IndexSegment,
    vix: &ValueIndex,
    data: &P::Blob,
    probe: &[u8],
) -> usize {
    // Narrow by L1/L2, then binary-search the subrange.
    let (mut lo, mut hi) = l1_l2_window(vix, probe);
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let pos = row_pos_at(seg, mid);
        let cmp = cmp_value_at::<P>(seg, data, pos, probe);
        if cmp == Ordering::Less {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn upper_bound_by_value<P: Pager>(
    seg: &IndexSegment,
    vix: &ValueIndex,
    data: &P::Blob,
    probe: &[u8],
    include_equal: bool,
) -> usize {
    // If include_equal, we want first index > probe; else >= probe.
    let (mut lo, mut hi) = l1_l2_window(vix, probe);
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let pos = row_pos_at(seg, mid);
        let cmp = cmp_value_at::<P>(seg, data, pos, probe);
        let go_right = if include_equal {
            cmp != Ordering::Greater // <=
        } else {
            cmp == Ordering::Less // <
        };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn l1_l2_window(vix: &ValueIndex, probe: &[u8]) -> (usize, usize) {
    let b0 = probe.get(0).copied().unwrap_or(0) as usize;
    let start = vix.l1_dir[b0] as usize;
    let end = vix.l1_dir[b0 + 1] as usize;
    if start >= end {
        return (start, end);
    }
    // Optional L2 split
    if let Some(dir) = vix.l2_dirs.iter().find(|d| d.first_byte as usize == b0) {
        let b1 = probe.get(1).copied().unwrap_or(0) as usize;
        let off0 = dir.dir257[b1] as usize;
        let off1 = dir.dir257[b1 + 1] as usize;
        return (start + off0, start + off1);
    }
    (start, end)
}

fn row_pos_at(seg: &IndexSegment, rank: usize) -> usize {
    seg.value_index
        .as_ref()
        .expect("value_index required")
        .value_order[rank] as usize
}

fn slice_value<P: Pager>(seg: &IndexSegment, data: &P::Blob, pos: usize) -> ValueSlice<P::Blob> {
    match &seg.value_layout {
        ValueLayout::FixedWidth { width } => {
            let w = *width as usize;
            let a = (pos * w) as u32;
            let b = a + (w as u32);
            ValueSlice {
                data: data.clone(),
                start: a,
                end: b,
            }
        }
        ValueLayout::Variable { value_offsets } => {
            let a = value_offsets[pos];
            let b = value_offsets[pos + 1];
            ValueSlice {
                data: data.clone(),
                start: a,
                end: b,
            }
        }
    }
}

fn cmp_value_at<P: Pager>(
    seg: &IndexSegment,
    data: &P::Blob,
    pos: usize,
    probe: &[u8],
) -> Ordering {
    let s = slice_value::<P>(seg, data, pos);
    let a = s.start as usize;
    let b = s.end as usize;
    let bytes = &s.data.as_ref()[a..b];
    bytes.cmp(probe)
}

fn bucket_and_tag(v: &[u8], bucket_prefix_len: usize, head_tag_len: usize) -> (u16, u64) {
    // bucket = first 2 bytes big-endian; zeros if missing
    let b0 = *v.get(0).unwrap_or(&0) as u16;
    let b1 = *v.get(1).unwrap_or(&0) as u16;
    let bucket = ((b0 << 8) | b1) as u16;

    // tag = next up to 8 bytes big-endian after the bucket prefix
    let mut tag: u64 = 0;
    let start = bucket_prefix_len;
    let end = (start + head_tag_len).min(v.len());
    for i in start..end {
        tag = (tag << 8) | (v[i] as u64);
    }
    if end - start < head_tag_len {
        tag <<= 8 * (head_tag_len - (end - start));
    }
    (bucket, tag)
}

/// Clone `Bound<&[u8]>` into owned `Bound<Vec<u8>>` for storage.
fn clone_bound_bytes(b: &Bound<&[u8]>) -> Bound<Vec<u8>> {
    match b {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(x) => Bound::Included((*x).to_vec()),
        Bound::Excluded(x) => Bound::Excluded((*x).to_vec()),
    }
}

// ----------------------------- tests -----------------------------
#[cfg(test)]
mod value_scan_tests {
    use super::*;
    use crate::ColumnStore;
    use crate::codecs::big_endian::u64_be_array;
    use crate::storage::pager::MemPager;
    use crate::types::{AppendOptions, Put, ValueMode};
    use std::collections::{BTreeMap, BTreeSet};

    // TODO: Dedupe
    fn be64_vec(x: u64) -> Vec<u8> {
        u64_be_array(x).to_vec()
    }
    // TODO: Dedupe
    fn key_bytes(i: u32) -> Vec<u8> {
        format!("k{:06}", i).into_bytes()
    }
    // TODO: Dedupe
    fn parse_key_u32(k: &[u8]) -> u32 {
        // "k000123" -> 123
        std::str::from_utf8(k)
            .ok()
            .and_then(|s| s[1..].parse::<u32>().ok())
            .expect("key parse")
    }
    // TODO: Dedupe
    fn parse_be64(v: &[u8]) -> u64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&v[..8]);
        u64::from_be_bytes(a)
    }

    /// Build three overlapping generations of data for the same column:
    ///   gen1 (oldest): keys [0..1000), value = 1000 + i
    ///   gen2 (newer) : keys [200..1200), value = 200_000 + i
    ///   gen3 (newest): keys [600..1400), value = 300_000 + i
    ///
    /// With LWW, the winner per key i is:
    ///   i in [0,200)     -> gen1
    ///   i in [200,600)   -> gen2
    ///   i in [600,1400)  -> gen3
    fn seed_three_generations<P: Pager>(store: &ColumnStore<P>, fid: LogicalFieldId) {
        let opts = AppendOptions {
            mode: ValueMode::ForceFixed(8),
            // small limits to force multiple physical segments
            segment_max_entries: 128,
            segment_max_bytes: 1 << 14,
            last_write_wins_in_batch: true,
        };

        // gen1
        let mut items = Vec::with_capacity(1000);
        for i in 0u32..1000 {
            items.push((key_bytes(i).into(), be64_vec(1000 + i as u64).into()));
        }
        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            opts.clone(),
        );

        // gen2
        let mut items = Vec::with_capacity(1000);
        for i in 200u32..1200 {
            items.push((key_bytes(i).into(), be64_vec(200_000 + i as u64).into()));
        }
        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            opts.clone(),
        );

        // gen3
        let mut items = Vec::with_capacity(800);
        for i in 600u32..1400 {
            items.push((key_bytes(i).into(), be64_vec(300_000 + i as u64).into()));
        }
        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            opts,
        );
    }

    /// End-to-end: forward scan over full range, assert LWW winners and uniqueness.
    #[test]
    fn scan_values_lww_forward_big_multi_segment() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p); // bootstrap+manifest, empty store
        let fid = 42u32;

        seed_three_generations(&store, fid);

        // full-range value scan
        let it = store
            .scan_values_lww(
                fid,
                ValueScanOpts {
                    dir: Direction::Forward,
                    lo: Bound::Unbounded,
                    hi: Bound::Unbounded,
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        // Accumulate winners (key -> value)
        let mut by_key: BTreeMap<u32, u64> = BTreeMap::new();
        for item in it {
            let k = parse_key_u32(&item.key);
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            let v = parse_be64(&item.value.data.as_ref()[a..b]);
            // Expect uniqueness (shadowing applied)
            assert!(by_key.insert(k, v).is_none(), "duplicate key {k} in scan");
        }

        // Expected domain of surviving keys is [0..1400)
        assert_eq!(by_key.len(), 1400, "unique keys after LWW");

        // spot-check winners come from the right generation
        // [0,200) -> gen1
        assert_eq!(by_key[&0], 1000 + 0);
        assert_eq!(by_key[&199], 1000 + 199);
        // [200,600) -> gen2
        assert_eq!(by_key[&200], 200_000 + 200);
        assert_eq!(by_key[&599], 200_000 + 599);
        // [600,1400) -> gen3
        assert_eq!(by_key[&600], 300_000 + 600);
        assert_eq!(by_key[&1399], 300_000 + 1399);
    }

    /// Narrow value window: ensure [lo, hi) (value-space) is honored.
    #[test]
    fn scan_values_lww_value_window_slice() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 7u32;
        seed_three_generations(&store, fid);

        // Choose a window that lands entirely in gen2's value band.
        // gen2 encodes as 200_000 + i, i in [200,1200)
        // Grab values [200_250 .. 200_260) => i in [250..260)
        let lo = be64_vec(200_250);
        let hi = be64_vec(200_260);

        let it = store
            .scan_values_lww(
                fid,
                ValueScanOpts {
                    dir: Direction::Forward,
                    lo: Bound::Included(&lo),
                    hi: Bound::Excluded(&hi),
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        let mut keys: BTreeSet<u32> = BTreeSet::new();
        let mut vals: BTreeSet<u64> = BTreeSet::new();

        for item in it {
            let k = parse_key_u32(&item.key);
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            let v = parse_be64(&item.value.data.as_ref()[a..b]);
            keys.insert(k);
            vals.insert(v);
        }

        // In that slice we expect 10 items (i = 250..259), if shadowing is applied.
        assert_eq!(keys.len(), 10, "value-window result size");
        // ensure keys are the right ones
        let expected: BTreeSet<u32> = (250u32..260u32).collect();
        assert_eq!(keys, expected);
        // ensure values match exactly
        let expected_vals: BTreeSet<u64> = (200_250u64..200_260u64).collect();
        assert_eq!(vals, expected_vals);
    }

    /// Reverse scan order contract (LWW).
    #[test]
    fn scan_values_lww_reverse_order_contract() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 99u32;
        seed_three_generations(&store, fid);

        let it = store
            .scan_values_lww(
                fid,
                ValueScanOpts {
                    dir: Direction::Reverse,
                    lo: Bound::Unbounded,
                    hi: Bound::Unbounded,
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        let mut prev: Option<u64> = None;
        for item in it {
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            let v = parse_be64(&item.value.data.as_ref()[a..b]);
            if let Some(pv) = prev {
                assert!(v <= pv, "reverse order expected: {} <= {}", v, pv);
            }
            prev = Some(v);
        }
    }

    /// FWW semantics: oldest wins.
    #[test]
    fn scan_values_fww_forward_contract() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 5u32;
        seed_three_generations(&store, fid);

        let it = store
            .scan_values_fww(
                fid,
                ValueScanOpts {
                    dir: Direction::Forward,
                    lo: Bound::Unbounded,
                    hi: Bound::Unbounded,
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        let mut by_key: BTreeMap<u32, u64> = BTreeMap::new();
        for item in it {
            let k = parse_key_u32(&item.key);
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            let v = parse_be64(&item.value.data.as_ref()[a..b]);
            assert!(by_key.insert(k, v).is_none(), "duplicate key {}", k);
        }

        // Oldest wins expectation:
        // [0,1000) -> gen1; [1000,1200) -> gen2; [1200,1400) -> gen3
        assert_eq!(by_key[&0], 1000 + 0);
        assert_eq!(by_key[&999], 1000 + 999);
        assert_eq!(by_key[&1000], 200_000 + 1000);
        assert_eq!(by_key[&1199], 200_000 + 1199);
        assert_eq!(by_key[&1200], 300_000 + 1200);
        assert_eq!(by_key[&1399], 300_000 + 1399);
    }

    /// Frame predicate/windowing contract (on values).
    #[test]
    fn scan_values_lww_frame_predicate_contract() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 12u32;
        seed_three_generations(&store, fid);

        let cap = 64u64;
        let it = store
            .scan_values_lww(
                fid,
                ValueScanOpts {
                    dir: Direction::Forward,
                    lo: Bound::Unbounded,
                    hi: Bound::Unbounded,
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: Some(Arc::new(move |head, cur| {
                        let hv = parse_be64(head);
                        let cv = parse_be64(cur);
                        cv - hv <= cap
                    })),
                },
            )
            .expect("iterator");

        let vals: Vec<u64> = it
            .map(|x| {
                let a = x.value.start as usize;
                let b = x.value.end as usize;
                parse_be64(&x.value.data.as_ref()[a..b])
            })
            .collect();

        assert!(!vals.is_empty());
        if let Some(&head) = vals.first() {
            assert!(vals.iter().all(|&v| v - head <= cap));
        }
    }

    #[test]
    fn scan_values_fww_reverse_order_contract() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 77u32;
        seed_three_generations(&store, fid);

        let it = store
            .scan_values_fww(
                fid,
                ValueScanOpts {
                    dir: Direction::Reverse,
                    lo: Bound::Unbounded,
                    hi: Bound::Unbounded,
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        let mut seen_keys = BTreeSet::new();
        let mut prev: Option<u64> = None;
        let mut first: Option<u64> = None;
        let mut count = 0usize;

        for item in it {
            // keys unique under FWW
            let k = parse_key_u32(&item.key);
            assert!(seen_keys.insert(k), "duplicate key {}", k);

            // values non-increasing in reverse
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            let v = parse_be64(&item.value.data.as_ref()[a..b]);

            if first.is_none() {
                first = Some(v);
            }
            if let Some(pv) = prev {
                assert!(v <= pv, "reverse order expected: {} <= {}", v, pv);
            }
            prev = Some(v);
            count += 1;
        }

        // exactly one winner per key (0..1400)
        assert_eq!(count, 1400, "FWW should keep exactly one value per key");

        // endpoints for reverse order under FWW:
        //   first (max)  -> gen3@1399 = 300_000 + 1399
        //   last  (min)  -> gen1@0    = 1000 + 0
        assert_eq!(first.unwrap(), 300_000 + 1399);
        assert_eq!(prev.unwrap(), 1000 + 0);
    }

    #[test]
    fn scan_values_lww_reverse_windowed_gen3() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 101u32;
        seed_three_generations(&store, fid);

        // Window fully inside gen3 (i = 700..710)
        let lo = be64_vec(300_700);
        let hi = be64_vec(300_710);

        let it = store
            .scan_values_lww(
                fid,
                ValueScanOpts {
                    dir: Direction::Reverse,
                    lo: Bound::Included(&lo),
                    hi: Bound::Excluded(&hi),
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        let mut vals: Vec<u64> = Vec::new();
        let mut keys: BTreeSet<u32> = BTreeSet::new();
        for item in it {
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            vals.push(parse_be64(&item.value.data.as_ref()[a..b]));
            keys.insert(parse_key_u32(&item.key));
        }

        // reverse monotonic in window
        assert!(vals.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(vals.len(), 10);

        // keys are exactly 700..709
        let expected: BTreeSet<u32> = (700u32..710u32).collect();
        assert_eq!(keys, expected);

        // endpoints
        assert_eq!(vals.first().copied().unwrap(), 300_709);
        assert_eq!(vals.last().copied().unwrap(), 300_700);
    }

    #[test]
    fn scan_values_fww_forward_windowed_gen1_positive() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 102u32;
        seed_three_generations(&store, fid);

        // Window fully inside gen1 (i = 250..260) -> winners are gen1 under FWW
        let lo = be64_vec(1000 + 250);
        let hi = be64_vec(1000 + 260);

        let it = store
            .scan_values_fww(
                fid,
                ValueScanOpts {
                    dir: Direction::Forward,
                    lo: Bound::Included(&lo),
                    hi: Bound::Excluded(&hi),
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        let mut vals: Vec<u64> = Vec::new();
        let mut keys: BTreeSet<u32> = BTreeSet::new();
        for item in it {
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            vals.push(parse_be64(&item.value.data.as_ref()[a..b]));
            keys.insert(parse_key_u32(&item.key));
        }

        assert_eq!(vals.len(), 10);
        assert!(vals.windows(2).all(|w| w[0] <= w[1])); // forward monotonic
        let expected_keys: BTreeSet<u32> = (250u32..260u32).collect();
        assert_eq!(keys, expected_keys);
        assert_eq!(vals.first().copied().unwrap(), 1000 + 250);
        assert_eq!(vals.last().copied().unwrap(), 1000 + 259);
    }

    #[test]
    fn scan_values_fww_forward_windowed_gen2_strict_suppression() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 103u32;
        seed_three_generations(&store, fid);

        // Window fully inside gen2’s value band (i = 250..260).
        // Under strict FWW, the winners for those keys are in gen1,
        // whose values (1000+i) lie OUTSIDE this window -> expect empty.
        let lo = be64_vec(200_250);
        let hi = be64_vec(200_260);

        let it = store
            .scan_values_fww(
                fid,
                ValueScanOpts {
                    dir: Direction::Forward,
                    lo: Bound::Included(&lo),
                    hi: Bound::Excluded(&hi),
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        assert_eq!(
            it.count(),
            0,
            "strict FWW suppresses newer rewrites in-window"
        );
    }

    #[test]
    fn scan_values_fww_reverse_windowed_gen1() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 104u32;
        seed_three_generations(&store, fid);

        // Window at the high end of gen1 (i = 990..1000)
        let lo = be64_vec(1000 + 990);
        let hi = be64_vec(1000 + 1000);

        let it = store
            .scan_values_fww(
                fid,
                ValueScanOpts {
                    dir: Direction::Reverse,
                    lo: Bound::Included(&lo),
                    hi: Bound::Excluded(&hi),
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator");

        let mut vals: Vec<u64> = Vec::new();
        let mut keys: Vec<u32> = Vec::new();
        for item in it {
            let a = item.value.start as usize;
            let b = item.value.end as usize;
            vals.push(parse_be64(&item.value.data.as_ref()[a..b]));
            keys.push(parse_key_u32(&item.key));
        }

        // reverse monotonic & correct endpoints
        assert_eq!(vals.len(), 10);
        assert!(vals.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(vals.first().copied().unwrap(), 1000 + 999);
        assert_eq!(vals.last().copied().unwrap(), 1000 + 990);

        // exact keys
        let expected: Vec<u32> = (990u32..1000u32).rev().collect();
        assert_eq!(keys, expected);
    }
}
