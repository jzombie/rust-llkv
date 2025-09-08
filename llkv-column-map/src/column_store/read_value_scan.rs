//! Value-ordered range scan with strict LWW hook points.
//!
//! Iterator uses per-segment value_order + L1/L2 directories to seed
//! one head per candidate segment, and a 2-tier radix PQ to pop the
//! next smallest value across segments. LWW gate is a placeholder;
//! wire in newest[key] before enabling in prod.

use crate::bounds::ValueBound;
use crate::column_index::{IndexSegment, ValueIndex};
use crate::layout::{KeyLayout, ValueLayout};
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::types::{LogicalFieldId, PhysicalKey, TypedKind, TypedValue};
use crate::views::ValueSlice;

use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Bound;

/// Scan options for value-ordered range scans.
pub struct ValueScanOpts<'a> {
    pub lo: Bound<&'a [u8]>,
    pub hi: Bound<&'a [u8]>,
    /// Optional prefix to prune quickly (value-side).
    pub prefix: Option<&'a [u8]>,
    /// First 2 bytes are the bucket id. Keep at 2 for now.
    pub bucket_prefix_len: usize,
    /// Head tag width in bytes used for fast compares (8 or 16 typical).
    pub head_tag_len: usize,
}

impl<'a> Default for ValueScanOpts<'a> {
    fn default() -> Self {
        Self {
            lo: Bound::Unbounded,
            hi: Bound::Unbounded,
            prefix: None,
            bucket_prefix_len: 2,
            head_tag_len: 8,
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
    seg_pk: PhysicalKey,
    data_pk: PhysicalKey,
    seg: IndexSegment,
    data: P::Blob,
    /// Newest-first order rank (0 == newest).
    rank: usize,
    /// [pos, end) in seg.value_index.value_order for this query window.
    pos: usize,
    end: usize,
}

/// The iterator. Owns the PQ and segment cursors.
pub struct ValueScan<P: Pager> {
    segs: Vec<SegCtx<P>>,
    pq: RadixPq<P>,
    field_id: LogicalFieldId,
    /// Bounds copied for short-circuit checks.
    lo: Bound<Vec<u8>>,
    hi: Bound<Vec<u8>>,
    /// Enable/disable LWW while wiring. Keep true in prod.
    enforce_lww: bool,
}

/// Min-heap over active heads using a 2-tier radix design:
/// Tier 1: 16-bit bucket id with a bitset (65,536 buckets).
/// Tier 2: per-bucket heap ordered by (head_tag, full bytes, rank, pos).
struct RadixPq<P: Pager> {
    bitset: [u64; 1024], // 1024 * 64 = 65,536 bits
    buckets: HashMap<u16, SmallHeap<P>>,
}

impl<P: Pager> RadixPq<P> {
    fn new() -> Self {
        Self {
            bitset: [0u64; 1024],
            buckets: HashMap::new(),
        }
    }

    fn push(&mut self, node: Node<P>) {
        let b = node.bucket;
        let idx = (b / 64) as usize;
        let bit = 1u64 << (b % 64);
        self.buckets
            .entry(b)
            .or_insert_with(SmallHeap::new)
            .push(node);
        self.bitset[idx] |= bit;
    }

    fn pop(&mut self) -> Option<Node<P>> {
        let bi = self.find_lowest_bucket()?;
        let heap = self.buckets.get_mut(&bi)?;
        let out = heap.pop()?;
        if heap.is_empty() {
            let idx = (bi / 64) as usize;
            let bit = 1u64 << (bi % 64);
            self.bitset[idx] &= !bit;
            self.buckets.remove(&bi);
        }
        Some(out)
    }

    fn find_lowest_bucket(&self) -> Option<u16> {
        for (i, word) in self.bitset.iter().enumerate() {
            if *word == 0 {
                continue;
            }
            let t = word.trailing_zeros() as u16;
            return Some(((i as u16) * 64) + t);
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

/// Small per-bucket heap using BinaryHeap with Reverse nodes.
struct SmallHeap<P: Pager>(BinaryHeap<std::cmp::Reverse<Node<P>>>);

impl<P: Pager> SmallHeap<P> {
    fn new() -> Self {
        Self(BinaryHeap::new())
    }
    fn push(&mut self, n: Node<P>) {
        self.0.push(std::cmp::Reverse(n));
    }
    fn pop(&mut self) -> Option<Node<P>> {
        self.0.pop().map(|r| r.0)
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
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
}

impl<P: Pager> Node<P> {
    #[inline]
    fn bytes<'a>(&'a self) -> &'a [u8] {
        let a = self.val.start as usize;
        let b = self.val.end as usize;
        &self.val.data.as_ref()[a..b]
    }
}

/// Order ascending by (bucket, head_tag, full bytes, rank asc, pos asc).
impl<P: Pager> Eq for Node<P> {}
impl<P: Pager> PartialEq for Node<P> {
    fn eq(&self, other: &Self) -> bool {
        self.bucket == other.bucket
            && self.head_tag == other.head_tag
            && self.bytes() == other.bytes()
            && self.rank == other.rank
            && self.pos == other.pos
    }
}
impl<P: Pager> Ord for Node<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we wrap with Reverse in SmallHeap.
        match self.bucket.cmp(&other.bucket) {
            Ordering::Equal => {}
            ord => return ord.reverse(),
        }
        match self.head_tag.cmp(&other.head_tag) {
            Ordering::Equal => {}
            ord => return ord.reverse(),
        }
        match self.bytes().cmp(other.bytes()) {
            Ordering::Equal => {}
            ord => return ord.reverse(),
        }
        match self.rank.cmp(&other.rank) {
            Ordering::Equal => {}
            ord => return ord, // newer (smaller) ranks come first
        }
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

        // -------- batch load segments + data blobs ----------
        let mut gets: Vec<BatchGet> = Vec::with_capacity(act_refs.len() * 2);
        let mut seen_data: FxHashSet<PhysicalKey> = FxHashSet::default();
        for r in &act_refs {
            gets.push(BatchGet::Typed {
                key: r.index_physical_key,
                kind: TypedKind::IndexSegment,
            });
            if seen_data.insert(r.data_physical_key) {
                gets.push(BatchGet::Raw {
                    key: r.data_physical_key,
                });
            }
        }
        let resp = col.do_gets(gets);

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

            // Compute [pos, end) in value_order for [lo, hi).
            let (pos, end) = range_in_value_index::<P>(&seg, vix, &data, &opts.lo, &opts.hi);

            if pos >= end {
                continue;
            }

            let rank = *seg_rank.get(&r.index_physical_key).unwrap_or(&usize::MAX);

            let seg_idx = segs.len();
            segs.push(SegCtx {
                seg_pk: r.index_physical_key,
                data_pk: r.data_physical_key,
                seg,
                data,
                rank,
                pos,
                end,
            });

            // Push the head node for this segment.
            let (seg_ref, data_ref) = {
                let s = &segs[seg_idx].seg;
                let d = &segs[seg_idx].data;
                (s, d)
            };
            let val = slice_value::<P>(seg_ref, data_ref, pos);
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
                pos,
                val,
            });
        }

        if pq.is_empty() {
            return Err(ScanError::NoActiveSegments);
        }

        Ok(Self {
            segs,
            pq,
            field_id,
            lo: clone_bound_bytes(&opts.lo),
            hi: clone_bound_bytes(&opts.hi),
            enforce_lww: true,
        })
    }
}

impl<P: Pager> Iterator for ValueScan<P> {
    type Item = ValueScanItem<P::Blob>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.pq.pop()?;
            let sc = &mut self.segs[node.seg_idx];

            // LWW gate (placeholder).
            if self.enforce_lww {
                let accept = true;
                if !accept {
                    sc.pos += 1;
                    if sc.pos < sc.end {
                        let val = slice_value::<P>(&sc.seg, &sc.data, sc.pos);
                        let (b, t) = {
                            let a = val.start as usize;
                            let b = val.end as usize;
                            let bytes = &val.data.as_ref()[a..b];
                            bucket_and_tag(bytes, 2, 8)
                        };
                        self.pq.push(Node {
                            bucket: b,
                            head_tag: t,
                            rank: sc.rank,
                            seg_idx: node.seg_idx,
                            pos: sc.pos,
                            val,
                        });
                    }
                    continue;
                }
            }

            // Prepare output (key + value).
            let row_pos = row_pos_at(&sc.seg, node.pos);
            let key = slice_key(&sc.seg, row_pos);

            // Advance this segment's cursor and push next head if any.
            sc.pos += 1;
            if sc.pos < sc.end {
                let val = slice_value::<P>(&sc.seg, &sc.data, sc.pos);
                let (b, t) = {
                    let a = val.start as usize;
                    let b2 = val.end as usize;
                    let bytes = &val.data.as_ref()[a..b2];
                    bucket_and_tag(bytes, 2, 8)
                };
                self.pq.push(Node {
                    bucket: b,
                    head_tag: t,
                    rank: sc.rank,
                    seg_idx: node.seg_idx,
                    pos: sc.pos,
                    val,
                });
            }

            return Some(ValueScanItem {
                key,
                value: node.val,
            });
        }
    }
}

// -------- helpers: bounds, seeding, slicing, compare --------

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
    let pos = match lo {
        Bound::Unbounded => 0usize,
        Bound::Included(x) | Bound::Excluded(x) => lower_bound_by_value::<P>(seg, vix, data, x),
    };
    // upper bound (exclusive)
    let end = match hi {
        Bound::Unbounded => n,
        Bound::Included(x) => upper_bound_by_value::<P>(seg, vix, data, x, true),
        Bound::Excluded(x) => upper_bound_by_value::<P>(seg, vix, data, x, false),
    };
    (pos.min(n), end.min(n))
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

/// Clone `Bound<&[u8]>` into owned `Bound<Vec<u8>>` for storage in the
/// iterator. Keeps API ergonomic at the callsite.
fn clone_bound_bytes(b: &Bound<&[u8]>) -> Bound<Vec<u8>> {
    match b {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(x) => Bound::Included((*x).to_vec()),
        Bound::Excluded(x) => Bound::Excluded((*x).to_vec()),
    }
}

/// Slice key bytes for the given row position (owned).
fn slice_key(seg: &IndexSegment, row_pos: usize) -> Vec<u8> {
    let s = KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, row_pos);
    s.to_vec()
}
