//! Value-ordered range scan with strict LWW and frame predicate.
//!
//! Iterator uses per-segment value_order + L1/L2 directories to seed
//! one head per candidate segment, and a 2-tier radix PQ to pop the
//! next value across segments. Strict LWW is enforced by probing any
//! newer segment (by recency rank) for membership of the same key.
//!
//! - Newer segments are batch-loaded as IndexSegment (no data blobs).
//! - Active segments (value producers) load index + data blobs once.
//! - Frame predicate is evaluated on encoded keys (zero copy).

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

/// Frame predicate type over encoded keys.
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
    /// Optional end-of-frame predicate on encoded keys.
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

/// Minimal metadata for newer segments used by LWW membership.
struct LwwRef {
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
    frame_pred: Option<FramePred>,
    frame_head: Option<Vec<u8>>,
    halted: bool,
    /// Strict LWW: newer segment refs grouped by rank (0 newest).
    newer_by_rank: Vec<Vec<LwwRef>>,
    /// Newer segments' index blobs (no data blobs).
    newer_seg_map: FxHashMap<PhysicalKey, IndexSegment>,
    /// PQ tuning (copied from opts).
    bucket_prefix_len: usize,
    head_tag_len: usize,
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
        // BinaryHeap takes the greatest; we always wrap with Reverse,
        // so this `cmp` defines "ascending" for forward and "descending"
        // for reverse by flipping comparisons at the value levels only.
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
    /// NOTE: requires segments to have `value_index`.
    pub fn scan_values_lww(
        &self,
        field_id: LogicalFieldId,
        opts: ValueScanOpts<'_>,
    ) -> Result<ValueScan<P>, ScanError> {
        ValueScan::new(self, field_id, opts)
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

        // Collect sets to classify typed results.
        let mut active_index_pks: FxHashSet<PhysicalKey> = FxHashSet::default();
        let mut active_data_pks: FxHashSet<PhysicalKey> = FxHashSet::default();
        for r in &act_refs {
            active_index_pks.insert(r.index_physical_key);
            active_data_pks.insert(r.data_physical_key);
        }

        // Collect index+data gets for active segments.
        let mut gets: Vec<BatchGet> = Vec::with_capacity(act_refs.len() * 2);
        for r in &act_refs {
            gets.push(BatchGet::Typed {
                key: r.index_physical_key,
                kind: TypedKind::IndexSegment,
            });
            gets.push(BatchGet::Raw {
                key: r.data_physical_key,
            });
        }

        // -------- determine newer segments needed for strict LWW ----
        let max_active_rank = act_refs
            .iter()
            .map(|r| *seg_rank.get(&r.index_physical_key).unwrap_or(&0))
            .max()
            .unwrap_or(0);

        // Build a list of active logical key spans (borrowed; no clones).
        let mut active_spans: Vec<(&[u8], &[u8])> = Vec::new();
        for r in &act_refs {
            active_spans.push((r.logical_key_min.as_slice(), r.logical_key_max.as_slice()));
        }

        let mut newer_refs_all: Vec<LwwRef> = Vec::new();
        for (rank, r) in colindex.segments.iter().enumerate() {
            if rank >= max_active_rank {
                continue;
            }
            if overlaps_any(
                r.logical_key_min.as_slice(),
                r.logical_key_max.as_slice(),
                &active_spans,
            ) {
                newer_refs_all.push(LwwRef {
                    pk: r.index_physical_key,
                    rank,
                });
            }
        }

        // Batch-load index blobs for all these newer segments (typed only).
        let mut lww_gets: Vec<BatchGet> = Vec::new();
        {
            let mut seen = FxHashSet::default();
            for rf in &newer_refs_all {
                if seen.insert(rf.pk) && !active_index_pks.contains(&rf.pk) {
                    lww_gets.push(BatchGet::Typed {
                        key: rf.pk,
                        kind: TypedKind::IndexSegment,
                    });
                }
            }
        }

        // Execute both batches in one call.
        let mut resp = col.do_gets([gets, lww_gets].concat());

        let mut seg_map: FxHashMap<PhysicalKey, IndexSegment> = FxHashMap::default();
        let mut data_map: FxHashMap<PhysicalKey, P::Blob> = FxHashMap::default();
        let mut newer_seg_map: FxHashMap<PhysicalKey, IndexSegment> = FxHashMap::default();

        for gr in resp.drain(..) {
            match gr {
                GetResult::Typed {
                    key,
                    value: TypedValue::IndexSegment(seg),
                } => {
                    if active_index_pks.contains(&key) {
                        seg_map.insert(key, seg);
                    } else {
                        newer_seg_map.insert(key, seg);
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

        // Build newer_by_rank without requiring Clone on Vec<LwwRef>.
        let mut newer_by_rank: Vec<Vec<LwwRef>> = Vec::new();
        newer_by_rank.resize_with(max_active_rank, Vec::new);
        for rf in newer_refs_all {
            newer_by_rank[rf.rank].push(rf);
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
            newer_by_rank,
            newer_seg_map,
            bucket_prefix_len: opts.bucket_prefix_len,
            head_tag_len: opts.head_tag_len,
        })
    }

    /// Strict LWW: is `key` present in any newer segment (< cur_rank)?
    #[inline]
    fn lww_shadowed(&self, key: &[u8], cur_rank: usize) -> bool {
        let upto = cur_rank.min(self.newer_by_rank.len());
        for r in 0..upto {
            for rf in &self.newer_by_rank[r] {
                if let Some(seg) = self.newer_seg_map.get(&rf.pk) {
                    if let Some(_pos) = KeyLayout::binary_search_key_with_layout(
                        &seg.logical_key_bytes,
                        &seg.key_layout,
                        seg.n_entries as usize,
                        key,
                    ) {
                        return true;
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
            // Pop one head.
            let node = self.pq.pop(self.reverse)?;
            let seg_idx = node.seg_idx;

            // ----- Immutable borrow in a tight scope -----
            // Slice encoded key zero-copy and make the one required owned copy.
            let (key_owned, rank, begin, end) = {
                let sc = &self.segs[seg_idx];
                let row_pos = row_pos_at(&sc.seg, node.pos);
                let key_slice = KeyLayout::slice_key_by_layout(
                    &sc.seg.logical_key_bytes,
                    &sc.seg.key_layout,
                    row_pos,
                );
                (key_slice.to_vec(), sc.rank, sc.begin, sc.end)
            };
            // ----- Immutable borrow of segs is now dropped -----

            // Strict LWW: skip if any newer segment has this key.
            if self.lww_shadowed(key_owned.as_slice(), rank) {
                // Advance this segment head and continue (no mutable borrow).
                if let Some(np) = next_pos(node.pos, begin, end, self.reverse) {
                    let sc = &self.segs[seg_idx];
                    let val = slice_value::<P>(&sc.seg, &sc.data, np);
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

            // Frame predicate on encoded keys.
            if let Some(pred) = &self.frame_pred {
                if let Some(head) = &self.frame_head {
                    if !pred(head.as_slice(), key_owned.as_slice()) {
                        self.halted = true;
                        return None;
                    }
                } else {
                    // First yield establishes the frame head. This is the only
                    // place we do an extra copy (first row only).
                    self.frame_head = Some(key_owned.clone());
                }
            }

            // Advance this segment head and push next node (still no mutable
            // borrow needed; cursor lives in Node).
            if let Some(np) = next_pos(node.pos, begin, end, self.reverse) {
                let sc = &self.segs[seg_idx];
                let val = slice_value::<P>(&sc.seg, &sc.data, np);
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

            // Yield the row. Values are zero-copy; keys are one owned copy.
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
    } else {
        if cur > begin { Some(cur - 1) } else { None }
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
            cmp != Ordering::Greater
        } else {
            cmp == Ordering::Less
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
