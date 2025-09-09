//! Value-/key-ordered range scan with strict LWW/FWW and optional frame predicate.
//!
//! Iterator uses per-segment order + L1/L2 directories (for value order) to seed
//! one head per candidate segment, and a 2-tier radix PQ to pop the next item
//! across segments. Strict conflict policy (LWW/FWW) is enforced by probing any
//! newer/older segment (by recency rank) for membership of the same key.
//!
//! Modes:
//! - OrderBy::Value  (default): preserves the previous behavior (value-ordered).
//! - OrderBy::Key               : “natural” logical key order (segment key order).
//!
//! - Newer/older segments for shadow checks are batch-loaded as IndexSegment
//!   (no data blobs).
//! - Active segments (producers) load index + data blobs once.
//! - Frame predicate is evaluated on encoded **values** (zero copy), regardless
//!   of the chosen ordering (value/key).

use crate::bounds::ValueBound;
use crate::column_index::{IndexSegment, ValueIndex};
use crate::layout::{KeyLayout, ValueLayout};
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::types::{LogicalFieldId, LogicalKeyBytes, PhysicalKey, TypedKind, TypedValue};
use crate::views::ValueSlice;

use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::Bound;
use std::sync::Arc;

/// Direction along the chosen ordering (value or key).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Reverse,
}

/// What to order the scan by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderBy {
    /// Order by encoded VALUE bytes (old behavior).
    Value,
    /// Order by LOGICAL KEY bytes (segment’s natural key order).
    Key,
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

/// Scan options.
pub struct ValueScanOpts<'a> {
    /// Direction along the chosen `order_by`.
    pub dir: Direction,
    /// Lower/upper bound in the chosen domain:
    /// - OrderBy::Value => VALUE bounds
    /// - OrderBy::Key   => KEY   bounds
    pub lo: Bound<&'a [u8]>,
    pub hi: Bound<&'a [u8]>,
    /// Optional prefix to prune quickly (domain-specific; not currently used).
    pub prefix: Option<&'a [u8]>,
    /// First 2 bytes are the bucket id. Keep at 2 for now.
    pub bucket_prefix_len: usize,
    /// Head tag width in bytes used for fast compares.
    pub head_tag_len: usize,
    /// Optional end-of-frame predicate on encoded **values**.
    pub frame_predicate: Option<FramePred>,
    /// Order by VALUE (default) or KEY (natural key order).
    pub order_by: OrderBy,
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
            order_by: OrderBy::Value,
        }
    }
}

/// Public item: zero-copy value slice + owned key bytes.
pub struct ValueScanItem<B> {
    pub key: LogicalKeyBytes,
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
    /// Window in the ACTIVE ORDER:
    /// - value order: range into value_index.value_order: [begin, end)
    /// - key   order: range into key rows (0..n_entries): [begin, end)
    begin: usize,
    end: usize,
    /// Cached logical-key bounds (for fast reject in membership probes).
    min_key: LogicalKeyBytes,
    max_key: LogicalKeyBytes,
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
    /// True if ordering is VALUE-based (else KEY-based).
    by_value: bool,
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

// -------- priority queue (radix buckets) --------

const BUCKETS: usize = 1 << 16; // 65_536 buckets for u16 ids [0..=65535]
const WORDS: usize = BUCKETS / 64;

struct HeapSlot<P: Pager> {
    bucket: u16,
    heap: BinaryHeap<std::cmp::Reverse<Node<P>>>,
}

/// A radix-bucketed **priority queue** specialized for byte-sorted streams (value or key).
struct RadixPq<P: Pager> {
    /// 65_536 one-bit entries, packed into 1024 u64 words.
    bitset: [u64; WORDS],
    min_word: usize,
    max_word: usize,

    /// Sparse set of active buckets -> per-bucket heap.
    slots: Vec<HeapSlot<P>>,
    /// Directory: bucket id -> slots index (u32::MAX means empty).
    slot_of_bucket: Vec<u32>, // len == BUCKETS
}

impl<P: Pager> RadixPq<P> {
    #[inline]
    fn new() -> Self {
        Self {
            bitset: [0u64; WORDS],
            min_word: usize::MAX,
            max_word: 0,
            slots: Vec::new(),
            // **Important**: 65_536 entries, not 65_535.
            slot_of_bucket: vec![u32::MAX; BUCKETS],
        }
    }

    #[inline]
    fn push(&mut self, node: Node<P>) {
        let b = node.bucket as usize;
        debug_assert!(b < BUCKETS);
        let word_idx = b >> 6;
        let bit_mask = 1u64 << (b & 63);

        // If this word was previously empty, update watermarks before setting the bit.
        if self.bitset[word_idx] == 0 {
            if self.min_word == usize::MAX || word_idx < self.min_word {
                self.min_word = word_idx;
            }
            if word_idx > self.max_word {
                self.max_word = word_idx;
            }
        }

        // Find or create the heap slot for this bucket.
        let slot_idx_u32 = self.slot_of_bucket[b];
        let idx = if slot_idx_u32 == u32::MAX {
            let idx = self.slots.len();
            self.slot_of_bucket[b] = idx as u32;
            self.slots.push(HeapSlot {
                bucket: node.bucket,
                heap: BinaryHeap::new(),
            });
            idx
        } else {
            slot_idx_u32 as usize
        };

        self.slots[idx].heap.push(std::cmp::Reverse(node));
        self.bitset[word_idx] |= bit_mask;
    }

    #[inline]
    fn pop(&mut self, reverse: bool) -> Option<Node<P>> {
        let bi = self.find_bucket(reverse)? as usize;
        debug_assert!(bi < BUCKETS);

        let slot_idx_u32 = self.slot_of_bucket[bi];
        debug_assert!(slot_idx_u32 != u32::MAX, "active bucket without slot");
        let idx = slot_idx_u32 as usize;

        let out = self.slots[idx].heap.pop()?.0;

        if self.slots[idx].heap.is_empty() {
            // Clear bit & directory entry, maintain watermarks.
            let word_idx = bi >> 6;
            let bit_mask = 1u64 << (bi & 63);
            self.bitset[word_idx] &= !bit_mask;
            self.slot_of_bucket[bi] = u32::MAX;

            // Remove empty slot with swap_remove; fix mapping of the moved one.
            let last = self.slots.len() - 1;
            if idx != last {
                self.slots.swap_remove(idx);
                let moved_bucket = self.slots[idx].bucket as usize;
                self.slot_of_bucket[moved_bucket] = idx as u32;
            } else {
                self.slots.pop();
            }

            // Adjust watermarks if the word became empty.
            if self.bitset[word_idx] == 0 {
                if word_idx == self.min_word {
                    let mut i = self.min_word;
                    while i <= self.max_word && self.bitset[i] == 0 {
                        i += 1;
                    }
                    if i > self.max_word {
                        self.min_word = usize::MAX;
                        self.max_word = 0;
                    } else {
                        self.min_word = i;
                    }
                }
                if self.min_word != usize::MAX && word_idx == self.max_word {
                    let mut i = self.max_word;
                    loop {
                        if self.bitset[i] != 0 {
                            self.max_word = i;
                            break;
                        }
                        if i == self.min_word {
                            // all empty
                            self.min_word = usize::MAX;
                            self.max_word = 0;
                            break;
                        }
                        i -= 1;
                    }
                }
            }
        }

        Some(out)
    }

    #[inline]
    fn find_bucket(&self, reverse: bool) -> Option<u16> {
        if self.min_word == usize::MAX {
            return None;
        }

        if !reverse {
            let mut i = self.min_word;
            while i <= self.max_word {
                let w = self.bitset[i];
                if w != 0 {
                    let t = w.trailing_zeros() as u16;
                    // (i * 64) + first-set-bit
                    return Some(((i as u16) << 6) | t);
                }
                i += 1;
            }
        } else {
            let mut i = self.max_word;
            loop {
                let w = self.bitset[i];
                if w != 0 {
                    let lz = w.leading_zeros() as u16;
                    let bit = 63u16.saturating_sub(lz);
                    return Some(((i as u16) << 6) | bit);
                }
                if i == self.min_word {
                    break;
                }
                i -= 1;
            }
        }
        None
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.min_word == usize::MAX
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
    /// Position in the ACTIVE ORDER:
    /// - value order: value-rank (index into value_index.value_order)
    /// - key   order: row position (0..n_entries)
    pos: usize,
    /// Zero-copy value slice (for returning and for frame predicate).
    val: ValueSlice<P::Blob>,
    /// When present, ordering is by KEY bytes; otherwise by VALUE bytes.
    key_ord: Option<Vec<u8>>,
    reverse: bool,
}

impl<P: Pager> Node<P> {
    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        if let Some(ref k) = self.key_ord {
            k.as_slice()
        } else {
            let a = self.val.start as usize;
            let b = self.val.end as usize;
            &self.val.data.as_ref()[a..b]
        }
    }
}

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
    /// LWW-enforced scan over [lo, hi) in the chosen domain (value/key).
    pub fn scan_values_lww(
        &self,
        field_id: LogicalFieldId,
        opts: ValueScanOpts<'_>,
    ) -> Result<ValueScan<P>, ScanError> {
        ValueScan::new(self, field_id, opts, ConflictPolicy::LWW)
    }

    /// FWW-enforced scan over [lo, hi) in the chosen domain (value/key).
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
        let by_value = matches!(opts.order_by, OrderBy::Value);

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

        // -------- choose candidate segments by bounds ----------
        let (act_refs, seg_rank) = {
            let mut v = Vec::new();
            let mut ranks = FxHashMap::default();
            for (rank, r) in colindex.segments.iter().enumerate() {
                let keep = if by_value {
                    overlap_bounds(
                        &opts.lo,
                        &opts.hi,
                        r.value_min.as_ref(),
                        r.value_max.as_ref(),
                    )
                } else {
                    overlap_bounds_key(
                        &opts.lo,
                        &opts.hi,
                        r.logical_key_min.as_slice(),
                        r.logical_key_max.as_slice(),
                    )
                };
                if keep {
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
                ConflictPolicy::LWW => rank < max_active_rank, // check strictly newer
                ConflictPolicy::FWW => rank > min_active_rank, // check strictly older
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

            let (begin, end) = if by_value {
                let vix = match &seg.value_index {
                    Some(v) => v,
                    None => return Err(ScanError::ValueIndexMissing(r.index_physical_key)),
                };
                range_in_value_index::<P>(&seg, vix, &data, &opts.lo, &opts.hi)
            } else {
                range_in_key_index(&seg, &opts.lo, &opts.hi)
            };
            if begin >= end {
                continue;
            }

            let rank = *seg_rank.get(&r.index_physical_key).unwrap_or(&usize::MAX);

            // --- cache min/max logical key for this segment (fast reject) ---
            let min_key =
                KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, 0).to_vec();
            let max_key = KeyLayout::slice_key_by_layout(
                &seg.logical_key_bytes,
                &seg.key_layout,
                seg.n_entries as usize - 1,
            )
            .to_vec();

            let seg_idx = segs.len();
            segs.push(SegCtx {
                _seg_pk: r.index_physical_key,
                _data_pk: r.data_physical_key,
                seg,
                data,
                rank,
                begin,
                end,
                min_key,
                max_key,
            });

            // Head depends on direction.
            let head_pos = if matches!(opts.dir, Direction::Reverse) {
                end - 1
            } else {
                begin
            };

            // Build first node for this segment.
            let (seg_ref, data_ref) = {
                let s = &segs[seg_idx].seg;
                let d = &segs[seg_idx].data;
                (s, d)
            };
            // Row position depends on ordering.
            let head_row = if by_value {
                row_pos_at(seg_ref, head_pos)
            } else {
                head_pos
            };
            let val = slice_value::<P>(seg_ref, data_ref, head_row);

            // Ordering fields (bucket/tag) + optional key bytes.
            let (bucket, tag, key_ord) = if by_value {
                let (bkt, tg) = bucket_and_tag_from_value_slice::<P>(
                    &val,
                    opts.bucket_prefix_len,
                    opts.head_tag_len,
                );
                (bkt, tg, None)
            } else {
                let key_slice = KeyLayout::slice_key_by_layout(
                    &seg_ref.logical_key_bytes,
                    &seg_ref.key_layout,
                    head_row,
                );
                let kb = key_slice.to_vec();
                let (bkt, tg) =
                    bucket_and_tag_bytes_fast(&kb, opts.bucket_prefix_len, opts.head_tag_len);
                (bkt, tg, Some(kb))
            };

            pq.push(Node {
                bucket,
                head_tag: tag,
                rank,
                seg_idx,
                pos: head_pos,
                val,
                key_ord,
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
            by_value,
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

    /// Fast bound check using borrowed min/max from the segment.
    #[inline]
    fn key_within_seg_bounds(seg: &IndexSegment, key: &[u8]) -> bool {
        let min = KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, 0);
        let max = KeyLayout::slice_key_by_layout(
            &seg.logical_key_bytes,
            &seg.key_layout,
            seg.n_entries as usize - 1,
        );
        !(key < min || key > max)
    }

    /// Is this `key@rank` shadowed by the chosen policy?
    fn is_shadowed(&self, key: &[u8], rank: usize) -> bool {
        // 1) Check against already-loaded active segments (index+data).
        for sc in &self.segs {
            let rel = sc.rank as isize - rank as isize;
            let dominates = match self.policy {
                ConflictPolicy::LWW => rel < 0, // newer dominates
                ConflictPolicy::FWW => rel > 0, // older dominates
            };
            if !dominates {
                continue;
            }
            if key < sc.min_key.as_slice() || key > sc.max_key.as_slice() {
                continue;
            }
            if Self::key_in_index(&sc.seg, key) {
                return true;
            }
        }

        // 2) Check index-only shadow segments (prefetched).
        match self.policy {
            ConflictPolicy::LWW => {
                for r in 0..rank {
                    for rf in &self.shadow_by_rank[r] {
                        if let Some(seg) = self.shadow_seg_map.get(&rf.pk) {
                            if !Self::key_within_seg_bounds(seg, key) {
                                continue;
                            }
                            if Self::key_in_index(seg, key) {
                                return true;
                            }
                        }
                    }
                }
            }
            ConflictPolicy::FWW => {
                for r in (rank + 1)..self.shadow_by_rank.len() {
                    for rf in &self.shadow_by_rank[r] {
                        if let Some(seg) = self.shadow_seg_map.get(&rf.pk) {
                            if !Self::key_within_seg_bounds(seg, key) {
                                continue;
                            }
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

            // 0) Early frame check (value bytes only) — avoid extra work if we’ll halt.
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
                }
            }
            // Initialize frame head lazily (semantics preserved).
            if self.frame_head.is_none() {
                self.frame_head = Some(node.val.clone());
            }

            // 1) Borrow the segment briefly to get:
            //    - current row position (in the active ordering)
            //    - a borrowed key slice for membership checks
            //    - the next position + its value (and key bytes if key-ordered)
            //    - whether this row is shadowed (under policy)
            let (rank, next_pos_opt, next_val_opt, next_key_for_order, shadowed, key_owned_if_emit) = {
                let sc = &self.segs[seg_idx];

                // Row position depends on active ordering.
                let row_pos = if self.by_value {
                    row_pos_at(&sc.seg, node.pos)
                } else {
                    node.pos
                };

                // Borrowed key slice for membership checks.
                let key_slice = KeyLayout::slice_key_by_layout(
                    &sc.seg.logical_key_bytes,
                    &sc.seg.key_layout,
                    row_pos,
                );

                // Compute next cursor position in active window.
                let next_pos_opt = next_pos(node.pos, sc.begin, sc.end, self.reverse);

                // Prepare next value (and next key bytes if key-ordered) now,
                // while we still have the borrow. Values are zero-copy.
                let (next_val_opt, next_key_for_order) = next_pos_opt.map_or((None, None), |np| {
                    let next_row = if self.by_value {
                        row_pos_at(&sc.seg, np)
                    } else {
                        np
                    };
                    let sval = slice_value::<P>(&sc.seg, &sc.data, next_row);
                    let skey_ord = if self.by_value {
                        None
                    } else {
                        Some(
                            KeyLayout::slice_key_by_layout(
                                &sc.seg.logical_key_bytes,
                                &sc.seg.key_layout,
                                next_row,
                            )
                            .to_vec(),
                        )
                    };
                    (Some(sval), skey_ord)
                });

                // Shadow check uses the borrowed slice; no allocation yet.
                let shadowed = self.is_shadowed(key_slice, sc.rank);

                // Only allocate the owned key if we are going to emit.
                let key_owned_if_emit = if shadowed {
                    None
                } else {
                    Some(key_slice.to_vec())
                };

                (
                    sc.rank,
                    next_pos_opt,
                    next_val_opt,
                    next_key_for_order,
                    shadowed,
                    key_owned_if_emit,
                )
            }; // <-- borrow of self.segs ends here

            // Helper to push the next head if present (value- or key-ordered).
            let mut push_next = |np: usize, val: ValueSlice<P::Blob>| {
                if self.by_value {
                    let (bkt, tag) = bucket_and_tag_from_value_slice::<P>(
                        &val,
                        self.bucket_prefix_len,
                        self.head_tag_len,
                    );
                    self.pq.push(Node {
                        bucket: bkt,
                        head_tag: tag,
                        rank,
                        seg_idx,
                        pos: np,
                        val,
                        key_ord: None,
                        reverse: self.reverse,
                    });
                } else {
                    let kb = next_key_for_order
                        .as_ref()
                        .expect("prepared key bytes for key order")
                        .clone();
                    push_next_keyordered(
                        &mut self.pq,
                        rank,
                        seg_idx,
                        np,
                        val,
                        kb,
                        self.reverse,
                        self.bucket_prefix_len,
                        self.head_tag_len,
                    );
                }
            };

            // 2) If shadowed, just advance this segment and continue.
            if shadowed {
                if let (Some(np), Some(v)) = (next_pos_opt, next_val_opt) {
                    push_next(np, v);
                }
                continue;
            }

            // 3) Accepted: advance this segment before yielding.
            if let (Some(np), Some(v)) = (next_pos_opt, next_val_opt) {
                push_next(np, v);
            }

            // 4) Yield the current row (value zero-copy; key owned).
            let key_owned = key_owned_if_emit.expect("owned key prepared for emission");
            return Some(ValueScanItem {
                key: key_owned,
                value: node.val,
            });
        }
    }
}

// -------- helpers: direction, bounds, seeding, slicing, compare -----

#[inline(always)]
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

fn overlap_bounds_key(lo: &Bound<&[u8]>, hi: &Bound<&[u8]>, s_lo: &[u8], s_hi: &[u8]) -> bool {
    let lower_ok = match lo {
        Bound::Unbounded => true,
        Bound::Included(x) => s_hi >= *x,
        Bound::Excluded(x) => s_hi > *x,
    };
    if !lower_ok {
        return false;
    }
    let upper_ok = match hi {
        Bound::Unbounded => true,
        Bound::Included(x) => s_lo <= *x,
        Bound::Excluded(x) => s_lo < *x,
    };
    lower_ok && upper_ok
}

#[inline(always)]
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

#[inline(always)]
fn range_in_key_index(seg: &IndexSegment, lo: &Bound<&[u8]>, hi: &Bound<&[u8]>) -> (usize, usize) {
    let n = seg.n_entries as usize;
    if n == 0 {
        return (0, 0);
    }
    let begin = match lo {
        Bound::Unbounded => 0usize,
        Bound::Included(x) | Bound::Excluded(x) => key_lower_bound(seg, x),
    };
    let end = match hi {
        Bound::Unbounded => n,
        Bound::Included(x) => key_upper_bound(seg, x, true),
        Bound::Excluded(x) => key_upper_bound(seg, x, false),
    };
    (begin.min(n), end.min(n))
}

#[inline(always)]
fn key_lower_bound(seg: &IndexSegment, probe: &[u8]) -> usize {
    let n = seg.n_entries as usize;
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let k = KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, mid);
        if k < probe {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline(always)]
fn key_upper_bound(seg: &IndexSegment, probe: &[u8], include_equal: bool) -> usize {
    let n = seg.n_entries as usize;
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let k = KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, mid);
        let go_right = if include_equal {
            !(k > probe)
        } else {
            k < probe
        };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline(always)]
fn lower_bound_by_value<P: Pager>(
    seg: &IndexSegment,
    vix: &ValueIndex,
    data: &P::Blob,
    probe: &[u8],
) -> usize {
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

#[inline(always)]
fn upper_bound_by_value<P: Pager>(
    seg: &IndexSegment,
    vix: &ValueIndex,
    data: &P::Blob,
    probe: &[u8],
    include_equal: bool,
) -> usize {
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

#[inline(always)]
fn l1_l2_window(vix: &ValueIndex, probe: &[u8]) -> (usize, usize) {
    let b0 = probe.first().copied().unwrap_or(0) as usize;
    let start = vix.l1_dir[b0] as usize;
    let end = vix.l1_dir[b0 + 1] as usize;
    if start >= end {
        return (start, end);
    }
    if let Some(dir) = vix.l2_dirs.iter().find(|d| d.first_byte as usize == b0) {
        let b1 = probe.get(1).copied().unwrap_or(0) as usize;
        let off0 = dir.dir257[b1] as usize;
        let off1 = dir.dir257[b1 + 1] as usize;
        return (start + off0, start + off1);
    }
    (start, end)
}

#[inline(always)]
fn row_pos_at(seg: &IndexSegment, rank: usize) -> usize {
    seg.value_index
        .as_ref()
        .expect("value_index required")
        .value_order[rank] as usize
}

#[inline(always)]
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

#[inline(always)]
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
    let b0 = *v.first().unwrap_or(&0) as u16;
    let b1 = *v.get(1).unwrap_or(&0) as u16;
    let bucket = (b0 << 8) | b1;

    // tag = next up to head_tag_len bytes big-endian after the bucket prefix
    let mut tag: u64 = 0;
    let start = bucket_prefix_len;
    let end = (start + head_tag_len).min(v.len());
    for &x in &v[start..end] {
        tag = (tag << 8) | (x as u64);
    }
    if end - start < head_tag_len {
        tag <<= 8 * (head_tag_len - (end - start));
    }
    (bucket, tag)
}

// --------- hot-path specializations & cold helper ----------

/// Fast path for (bucket_prefix_len == 2 && head_tag_len == 8) on VALUE slices.
#[inline(always)]
fn bucket_and_tag_from_value_slice<P: Pager>(
    v: &ValueSlice<P::Blob>,
    bucket_prefix_len: usize,
    head_tag_len: usize,
) -> (u16, u64) {
    if bucket_prefix_len == 2 && head_tag_len == 8 {
        let bytes = &v.data.as_ref()[v.start as usize..v.end as usize];
        let b0 = *bytes.get(0).unwrap_or(&0) as u16;
        let b1 = *bytes.get(1).unwrap_or(&0) as u16;
        let bucket = (b0 << 8) | b1;
        let mut tag: u64 = 0;
        let mut i = 2usize;
        let end = (2 + 8).min(bytes.len());
        while i < end {
            tag = (tag << 8) | (bytes[i] as u64);
            i += 1;
        }
        tag <<= 8 * (2 + 8 - end);
        (bucket, tag)
    } else {
        let a = v.start as usize;
        let b = v.end as usize;
        bucket_and_tag(&v.data.as_ref()[a..b], bucket_prefix_len, head_tag_len)
    }
}

/// Fast path for (bucket_prefix_len == 2 && head_tag_len == 8) on KEY bytes.
#[inline(always)]
fn bucket_and_tag_bytes_fast(
    v: &[u8],
    bucket_prefix_len: usize,
    head_tag_len: usize,
) -> (u16, u64) {
    if bucket_prefix_len == 2 && head_tag_len == 8 {
        let b0 = *v.get(0).unwrap_or(&0) as u16;
        let b1 = *v.get(1).unwrap_or(&0) as u16;
        let bucket = (b0 << 8) | b1;
        let mut tag: u64 = 0;
        let mut i = 2usize;
        let end = (2 + 8).min(v.len());
        while i < end {
            tag = (tag << 8) | (v[i] as u64);
            i += 1;
        }
        tag <<= 8 * (2 + 8 - end);
        (bucket, tag)
    } else {
        bucket_and_tag(v, bucket_prefix_len, head_tag_len)
    }
}

/// Keep the key-ordered reseed path out of the hot icache.
#[cold]
#[inline(never)]
fn push_next_keyordered<P: Pager>(
    pq: &mut RadixPq<P>,
    rank: usize,
    seg_idx: usize,
    pos: usize,
    val: ValueSlice<P::Blob>,
    kb: Vec<u8>,
    reverse: bool,
    bucket_prefix_len: usize,
    head_tag_len: usize,
) {
    let (bkt, tag) = bucket_and_tag_bytes_fast(&kb, bucket_prefix_len, head_tag_len);
    pq.push(Node {
        bucket: bkt,
        head_tag: tag,
        rank,
        seg_idx,
        pos,
        val,
        key_ord: Some(kb),
        reverse,
    });
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
                    order_by: OrderBy::Value,
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
        assert_eq!(by_key[&0], 1000);
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
                    order_by: OrderBy::Value,
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
                    order_by: OrderBy::Value,
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
                    order_by: OrderBy::Value,
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
        assert_eq!(by_key[&0], 1000);
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
                    order_by: OrderBy::Value,
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
                    order_by: OrderBy::Value,
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
        //   last  (min)  -> gen1@0    = 1000
        assert_eq!(first.unwrap(), 300_000 + 1399);
        assert_eq!(prev.unwrap(), 1000);
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
                    order_by: OrderBy::Value,
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
                    order_by: OrderBy::Value,
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
                    order_by: OrderBy::Value,
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
                    order_by: OrderBy::Value,
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

    /// LWW + key-ordered, reverse: keys must be descending.
    #[test]
    fn scan_values_lww_key_order_reverse() {
        let p = MemPager::default();
        let store = ColumnStore::init_empty(&p);
        let fid = 4343u32;

        seed_three_generations(&store, fid);

        let it = store
            .scan_values_lww(
                fid,
                ValueScanOpts {
                    order_by: OrderBy::Key,
                    dir: Direction::Reverse,
                    lo: Bound::Unbounded,
                    hi: Bound::Unbounded,
                    prefix: None,
                    bucket_prefix_len: 2,
                    head_tag_len: 8,
                    frame_predicate: None,
                },
            )
            .expect("iterator (key-ordered, reverse)");

        let mut last_key: Option<u32> = None;
        let mut count = 0usize;

        for item in it {
            let k = parse_key_u32(&item.key);
            if let Some(prev) = last_key {
                assert!(
                    k <= prev,
                    "keys must be non-increasing in reverse key order"
                );
            }
            last_key = Some(k);
            count += 1;
        }

        assert_eq!(count, 1400, "one winner per key in reverse key order");
        assert_eq!(
            last_key.unwrap(),
            0,
            "last yielded key should be the minimum"
        );
    }
}
