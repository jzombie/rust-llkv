//! Ultra-fast k-way merge over pre-sorted chunks (brand-new API).
//!
//! Design goals:
//! - Pre-apply per-chunk UInt32 perms to build contiguous sorted buffers
//!   (best locality; one take per chunk).
//! - Binary heap + run coalescing: one pop/push per emitted run, not per
//!   element.
//! - Typed runs: return references to typed arrays + (start,len). Caller
//!   can scan values directly without constructing new Arrow arrays.
//! - Explicit sort semantics via SortOptions.
//! - Ready to add more primitive types by one macro line.
//!
//! Extend notes:
//! - To emit row_id with values, add a second sorted array per chunk
//!   (UInt64Array) and include it in Run enums. The merge logic does not
//!   change.

use crate::error::{Error, Result};
use crate::serialization::deserialize_array;
use crate::store::descriptor::ChunkMetadata;
use crate::types::PhysicalKey;

use arrow::array::{Array, ArrayRef, Int32Array, UInt32Array, UInt64Array};
use arrow::compute;
use arrow::datatypes::DataType;

use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

// ----------------------------- Sort options -------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SortOptions {
    /// true => descending. false => ascending.
    pub descending: bool,
    /// true => nulls first. false => nulls last.
    pub nulls_first: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            descending: false,
            nulls_first: true,
        }
    }
}

// ------------------------------ Key & heap --------------------------------

#[derive(Clone, Copy, Debug)]
struct Key<K: Ord + Copy> {
    null_rank: u8, // 0 or 1 (precomputed by nulls_first)
    val: K,        // ignored if both sides are null
    flip: bool,    // true => reverse value ordering
}

impl<K: Ord + Copy> PartialEq for Key<K> {
    fn eq(&self, other: &Self) -> bool {
        self.null_rank == other.null_rank && self.val == other.val
    }
}
impl<K: Ord + Copy> Eq for Key<K> {}

impl<K: Ord + Copy> PartialOrd for Key<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord + Copy> Ord for Key<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        let nr = self.null_rank.cmp(&other.null_rank);
        if nr != Ordering::Equal {
            return nr;
        }
        let v = self.val.cmp(&other.val);
        if self.flip { v.reverse() } else { v }
    }
}

#[derive(Clone, Copy, Debug)]
struct HeapItem<K: Ord + Copy> {
    cursor_idx: usize,
    sorted_idx: usize, // index in the chunk's sorted payload
    key: Key<K>,
}

impl<K: Ord + Copy> PartialEq for HeapItem<K> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.cursor_idx == other.cursor_idx
    }
}
impl<K: Ord + Copy> Eq for HeapItem<K> {}

impl<K: Ord + Copy> PartialOrd for HeapItem<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord + Copy> Ord for HeapItem<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| self.cursor_idx.cmp(&other.cursor_idx))
    }
}

// ------------------------------- Cursors ----------------------------------

struct Cursor<A> {
    /// Sorted payload for this chunk (contiguous).
    sorted_any: ArrayRef,
    /// Typed view to load keys cheaply.
    sorted: A,
    /// Rows in chunk.
    len: usize,
}

// --------------------------- Public run types -----------------------------

pub enum Run<'a> {
    U64 {
        arr: &'a UInt64Array,
        start: usize,
        len: usize,
    },
    I32 {
        arr: &'a Int32Array,
        start: usize,
        len: usize,
    },
}

impl<'a> Run<'a> {
    /// Total values in this run.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Run::U64 { len, .. } => *len,
            Run::I32 { len, .. } => *len,
        }
    }
    /// True if run has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// -------------------- Binary search helpers for clamping ------------------

#[inline]
fn lower_bound_u64(arr: &UInt64Array, mut lo: usize, mut hi: usize, key: u64) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if arr.value(mid) < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn upper_bound_u64(arr: &UInt64Array, mut lo: usize, mut hi: usize, key: u64) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if arr.value(mid) <= key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn lower_bound_i32(arr: &Int32Array, mut lo: usize, mut hi: usize, key: i32) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if arr.value(mid) < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn upper_bound_i32(arr: &Int32Array, mut lo: usize, mut hi: usize, key: i32) -> usize {
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if arr.value(mid) <= key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// ------------------- Monomorphized merge (primitive T) --------------------

macro_rules! impl_merge_for_primitive {
    ($Name:ident,
     $ArrTy:ty,
     $KeyTy:ty,
     $RunVariant:ident,
     $lower_fn:ident,
     $upper_fn:ident) => {
        pub struct $Name {
            cursors: Vec<Cursor<$ArrTy>>,
            heap: BinaryHeap<Reverse<HeapItem<$KeyTy>>>,
            /// Cached nulls_first as null rank function.
            nulls_first: bool,
            /// Flip value order for descending.
            flip: bool,
            /// Optional inclusive lower/upper bounds on values.
            bound_lo: Option<$KeyTy>,
            bound_hi: Option<$KeyTy>,
            /// Keep blobs alive (zero-copy backing).
            _blobs: FxHashMap<PhysicalKey, EntryHandle>,
        }

        impl $Name {
            /// Build from chunk metadata and blobs containing:
            /// - each chunk's data payload
            /// - each chunk's UInt32 permutation
            pub fn build(
                metas: &[ChunkMetadata],
                blobs: FxHashMap<PhysicalKey, EntryHandle>,
                opts: SortOptions,
            ) -> Result<Self> {
                let mut cursors = Vec::with_capacity(metas.len());

                for m in metas {
                    if m.row_count == 0 {
                        continue;
                    }
                    // Load data.
                    let data_blob = blobs.get(&m.chunk_pk).ok_or(Error::NotFound)?.clone();
                    let data_any = deserialize_array(data_blob)?;

                    // Load perm.
                    let perm_blob = blobs
                        .get(&m.value_order_perm_pk)
                        .ok_or(Error::NotFound)?
                        .clone();
                    let perm_any = deserialize_array(perm_blob)?;
                    let perm = perm_any
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .ok_or_else(|| Error::Internal("perm not u32".into()))?;

                    // Build contiguous sorted payload once.
                    let sorted_any = compute::take(&data_any, perm, None)?;
                    let sorted_typed = sorted_any
                        .as_any()
                        .downcast_ref::<$ArrTy>()
                        .ok_or_else(|| Error::Internal("typed downcast failed".into()))?
                        .clone();

                    cursors.push(Cursor::<$ArrTy> {
                        sorted_any: sorted_any.clone(),
                        sorted: sorted_typed,
                        len: sorted_any.len(),
                    });
                }

                // Seed heap with first row from each cursor.
                let mut heap: BinaryHeap<Reverse<HeapItem<$KeyTy>>> = BinaryHeap::new();

                let nulls_first = opts.nulls_first;
                let flip = opts.descending;

                for (i, c) in cursors.iter().enumerate() {
                    if c.len == 0 {
                        continue;
                    }
                    let k = Self::key_at(c, 0, nulls_first, flip);
                    heap.push(Reverse(HeapItem {
                        cursor_idx: i,
                        sorted_idx: 0,
                        key: k,
                    }));
                }

                Ok(Self {
                    cursors,
                    heap,
                    nulls_first,
                    flip,
                    bound_lo: None,
                    bound_hi: None,
                    _blobs: blobs,
                })
            }

            /// Return the next run from the same chunk, maximally coalesced.
            /// When the heap is empty, returns None.
            pub fn next_run<'a>(&'a mut self) -> Option<Run<'a>> {
                // Keep popping until we can emit a non-empty (clamped) run.
                loop {
                    let Reverse(top) = self.heap.pop()?;

                    // Threshold is the best of the others (peek).
                    let thr = self.heap.peek().map(|x| x.0.key);

                    let cidx = top.cursor_idx;
                    let start = top.sorted_idx;

                    let c = &self.cursors[cidx];
                    let len = c.len;

                    // Advance within the winner while it stays <= threshold.
                    let mut end = start + 1;

                    // Fast paths:
                    // - If heap is empty, we can take the whole tail.
                    if thr.is_none() {
                        end = len;
                    } else {
                        let thr_key = thr.unwrap();
                        while end < len {
                            let k = Self::key_at(c, end, self.nulls_first, self.flip);
                            if k > thr_key {
                                break;
                            }
                            end += 1;
                        }
                    }

                    // Reinsert cursor with its next position (if any).
                    if end < len {
                        let k = Self::key_at(c, end, self.nulls_first, self.flip);
                        self.heap.push(Reverse(HeapItem {
                            cursor_idx: cidx,
                            sorted_idx: end,
                            key: k,
                        }));
                    }

                    // Clamp [s,e) by optional bounds using typed array.
                    let arr = &c.sorted;
                    let mut s = start;
                    let mut e = end;

                    if let Some(lo) = self.bound_lo {
                        s = $lower_fn(arr, s, e, lo);
                        if s >= e {
                            // Entire run < lo; try next.
                            continue;
                        }
                    }
                    if let Some(hi) = self.bound_hi {
                        // If first value already > hi, discard this run.
                        if arr.value(s) > hi {
                            continue;
                        }
                        e = $upper_fn(arr, s, e, hi);
                        if s >= e {
                            continue;
                        }
                    }

                    let c = &self.cursors[cidx];
                    return Some(Run::$RunVariant {
                        arr: &c.sorted,
                        start: s,
                        len: e - s,
                    });
                }
            }

            #[inline(always)]
            fn key_at(
                c: &Cursor<$ArrTy>,
                idx: usize,
                nulls_first: bool,
                flip: bool,
            ) -> Key<$KeyTy> {
                let is_null = c.sorted.is_null(idx);
                let null_rank = if nulls_first {
                    if is_null { 0 } else { 1 }
                } else {
                    if is_null { 1 } else { 0 }
                };
                let v = if is_null {
                    <$KeyTy as Default>::default()
                } else {
                    c.sorted.value(idx) as $KeyTy
                };
                Key {
                    null_rank,
                    val: v,
                    flip,
                }
            }

            /// Total rows across all chunks (after sorting).
            pub fn total_rows(&self) -> usize {
                self.cursors.iter().map(|c| c.len).sum()
            }

            /// Number of active cursors (non-empty chunks).
            pub fn num_cursors(&self) -> usize {
                self.cursors.len()
            }

            /// True when the merge is fully drained.
            pub fn is_empty(&self) -> bool {
                self.heap.is_empty()
            }
        }
    };
}

// Instantiate for the types you care about today.
// Add more by repeating lines below (e.g., Int64Array/i64).
impl_merge_for_primitive!(
    MergeU64,
    UInt64Array,
    u64,
    U64,
    lower_bound_u64,
    upper_bound_u64
);
impl_merge_for_primitive!(
    MergeI32,
    Int32Array,
    i32,
    I32,
    lower_bound_i32,
    upper_bound_i32
);

// ------------------------------ Type dispatch -----------------------------

pub enum SortedMerge {
    U64(MergeU64),
    I32(MergeI32),
}

impl SortedMerge {
    /// Build a new merge from chunk metadata and blobs. You must pass
    /// the same blobs map that contains each chunk's data and its perm.
    pub fn build(
        metas: &[ChunkMetadata],
        blobs: FxHashMap<PhysicalKey, EntryHandle>,
        opts: SortOptions,
    ) -> Result<Self> {
        // Find first non-empty to choose the type.
        let first = match metas.iter().find(|m| m.row_count > 0) {
            Some(m) => m,
            None => {
                // Empty input. Default to u64 variant with zero cursors.
                // Callers should check `is_empty()` anyway.
                return Ok(SortedMerge::U64(MergeU64::build(&[], blobs, opts)?));
            }
        };

        let first_blob = blobs.get(&first.chunk_pk).ok_or(Error::NotFound)?.clone();
        let first_any = deserialize_array(first_blob)?;

        match first_any.data_type() {
            DataType::UInt64 => {
                let it = MergeU64::build(metas, blobs, opts)?;
                Ok(SortedMerge::U64(it))
            }
            DataType::Int32 => {
                let it = MergeI32::build(metas, blobs, opts)?;
                Ok(SortedMerge::I32(it))
            }
            other => Err(Error::Internal(format!(
                "Unsupported sort type {:?}",
                other
            ))),
        }
    }

    /// Drain one coalesced run. Returns None when finished.
    pub fn next_run<'a>(&'a mut self) -> Option<Run<'a>> {
        match self {
            SortedMerge::U64(m) => m.next_run(),
            SortedMerge::I32(m) => m.next_run(),
        }
    }

    /// Set inclusive value bounds for u64 streams. No-op for other types.
    #[inline]
    pub fn with_u64_bounds(mut self, lo: Option<u64>, hi: Option<u64>) -> Self {
        if let SortedMerge::U64(m) = &mut self {
            m.bound_lo = lo;
            m.bound_hi = hi;
        }
        self
    }

    /// Set inclusive value bounds for i32 streams. No-op for others.
    #[inline]
    pub fn with_i32_bounds(mut self, lo: Option<i32>, hi: Option<i32>) -> Self {
        if let SortedMerge::I32(m) = &mut self {
            m.bound_lo = lo;
            m.bound_hi = hi;
        }
        self
    }

    pub fn total_rows(&self) -> usize {
        match self {
            SortedMerge::U64(m) => m.total_rows(),
            SortedMerge::I32(m) => m.total_rows(),
        }
    }

    pub fn num_cursors(&self) -> usize {
        match self {
            SortedMerge::U64(m) => m.num_cursors(),
            SortedMerge::I32(m) => m.num_cursors(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            SortedMerge::U64(m) => m.is_empty(),
            SortedMerge::I32(m) => m.is_empty(),
        }
    }
}
