//! Ultra-fast k-way merge over pre-sorted chunks and a simple unsorted
//! streaming iterator.
//!
//! Design goals for sorted merge:
//! - Pre-apply per-chunk UInt32 perms to build contiguous sorted buffers.
//! - Binary heap + run coalescing: one pop/push per emitted run, not per
//!   element.
//! - Typed runs: return references to typed arrays + (start,len). Caller
//!   scans values directly without constructing new Arrow arrays.
//! - Explicit sort semantics via SortOptions (descending supported).
//! - Bounds via std::ops::Bound, clamped with generic binary search.
//!
//! Notes:
//! - Nulls are not stored (they are treated as deletes), so no null
//!   checks in the hot path.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Array, PrimitiveArray, UInt32Array, UInt64Array};
use arrow::compute;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Int32Type, UInt64Type};

use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;

use crate::error::{Error, Result};
use crate::serialization::deserialize_array;
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::store::ColumnStore;
use crate::store::descriptor::{ChunkMetadata, DescriptorIterator};
use crate::types::{LogicalFieldId, PhysicalKey};

// ----------------------------- Sort options -------------------------------

/// SortOptions is kept source-compatible. We ignore nulls_* at runtime,
/// because nulls are not stored (deleted at ingest).
#[derive(Clone, Copy, Debug)]
pub struct SortOptions {
    /// true => descending, false => ascending.
    pub descending: bool,
    /// Kept for API compatibility (not used on hot path).
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
    val: K,
    flip: bool, // true => reverse value ordering
}

impl<K: Ord + Copy> PartialEq for Key<K> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
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
    /// Typed view to load keys cheaply (already permuted/sorted).
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

// ------------------ Generic binary search (primitive T) -------------------

#[inline(always)]
fn lower_bound_prim<T>(
    arr: &PrimitiveArray<T>,
    mut lo: usize,
    mut hi: usize,
    key: T::Native,
) -> usize
where
    T: ArrowPrimitiveType,
    T::Native: Ord + Copy,
{
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

#[inline(always)]
fn upper_bound_prim<T>(
    arr: &PrimitiveArray<T>,
    mut lo: usize,
    mut hi: usize,
    key: T::Native,
) -> usize
where
    T: ArrowPrimitiveType,
    T::Native: Ord + Copy,
{
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
     $PrimTy:ty,
     $KeyTy:ty,
     $RunVariant:ident) => {
        pub struct $Name {
            cursors: Vec<Cursor<$ArrTy>>,
            heap: BinaryHeap<Reverse<HeapItem<$KeyTy>>>,
            /// Reverse value order for descending.
            flip: bool,
            /// Inclusive/exclusive bounds (per stream).
            bound_lo: Bound<$KeyTy>,
            bound_hi: Bound<$KeyTy>,
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

                    // Load perm (must exist for sorted scan).
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
                        sorted: sorted_typed,
                        len: sorted_any.len(),
                    });
                }

                // Seed heap with first row from each cursor.
                let mut heap: BinaryHeap<Reverse<HeapItem<$KeyTy>>> = BinaryHeap::new();

                let flip = opts.descending;

                for (i, c) in cursors.iter().enumerate() {
                    if c.len == 0 {
                        continue;
                    }
                    let k = Self::key_at(c, 0, flip);
                    heap.push(Reverse(HeapItem {
                        cursor_idx: i,
                        sorted_idx: 0,
                        key: k,
                    }));
                }

                Ok(Self {
                    cursors,
                    heap,
                    flip,
                    bound_lo: Bound::Unbounded,
                    bound_hi: Bound::Unbounded,
                    _blobs: blobs,
                })
            }

            /// Set bounds (builder-style).
            #[inline]
            pub fn with_bounds(mut self, lo: Bound<$KeyTy>, hi: Bound<$KeyTy>) -> Self {
                self.bound_lo = lo;
                self.bound_hi = hi;
                self
            }

            /// Set bounds (mut-style).
            #[inline]
            pub fn set_bounds(&mut self, lo: Bound<$KeyTy>, hi: Bound<$KeyTy>) {
                self.bound_lo = lo;
                self.bound_hi = hi;
            }

            /// Drain one coalesced run (clamped by bounds). Returns None
            /// when fully drained.
            pub fn next_run<'a>(&'a mut self) -> Option<Run<'a>> {
                loop {
                    let Reverse(top) = self.heap.pop()?;

                    // Threshold is the best of the others (peek).
                    let thr = self.heap.peek().map(|x| x.0.key);

                    let cidx = top.cursor_idx;
                    let start = top.sorted_idx;

                    let c = &self.cursors[cidx];
                    let len = c.len;

                    // Advance within the winner while it stays <= thr.
                    let mut end = start + 1;
                    if let Some(thr_key) = thr {
                        while end < len {
                            let k = Self::key_at(c, end, self.flip);
                            if k > thr_key {
                                break;
                            }
                            end += 1;
                        }
                    } else {
                        // Heap empty => take the whole tail.
                        end = len;
                    }

                    // Reinsert cursor with its next position (if any).
                    if end < len {
                        let k = Self::key_at(c, end, self.flip);
                        self.heap.push(Reverse(HeapItem {
                            cursor_idx: cidx,
                            sorted_idx: end,
                            key: k,
                        }));
                    }

                    // Clamp [start,end) by bounds using typed array.
                    let arr = &c.sorted;
                    let mut s = start;
                    let mut e = end;

                    // Lower bound clamp.
                    match self.bound_lo {
                        Bound::Unbounded => {}
                        Bound::Included(lo) => {
                            s = lower_bound_prim::<$PrimTy>(arr, s, e, lo);
                            if s >= e {
                                continue;
                            }
                        }
                        Bound::Excluded(lo) => {
                            s = upper_bound_prim::<$PrimTy>(arr, s, e, lo);
                            if s >= e {
                                continue;
                            }
                        }
                    }

                    // Upper bound clamp.
                    match self.bound_hi {
                        Bound::Unbounded => {}
                        Bound::Included(hi) => {
                            if arr.value(s) > hi {
                                continue;
                            }
                            e = upper_bound_prim::<$PrimTy>(arr, s, e, hi);
                            if s >= e {
                                continue;
                            }
                        }
                        Bound::Excluded(hi) => {
                            if arr.value(s) >= hi {
                                continue;
                            }
                            e = lower_bound_prim::<$PrimTy>(arr, s, e, hi);
                            if s >= e {
                                continue;
                            }
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
            fn key_at(c: &Cursor<$ArrTy>, idx: usize, flip: bool) -> Key<$KeyTy> {
                let v = c.sorted.value(idx) as $KeyTy;
                Key { val: v, flip }
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
impl_merge_for_primitive!(MergeU64, UInt64Array, UInt64Type, u64, U64);
impl_merge_for_primitive!(MergeI32, Int32Array, Int32Type, i32, I32);

// ------------------------------ Type dispatch -----------------------------

pub enum SortedMerge {
    U64(MergeU64),
    I32(MergeI32),
}

/// Type-erased bound so call sites don't need matches.
#[derive(Clone, Debug)]
pub enum BoundValue {
    U64(Bound<u64>),
    I32(Bound<i32>),
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

    /// Builder: type-erased bounds (no match soup at call sites).
    /// Returns an error if the bound type doesn't match the stream.
    #[inline]
    pub fn with_bounds(self, lo: BoundValue, hi: BoundValue) -> Result<Self> {
        match (self, lo, hi) {
            (SortedMerge::U64(mut m), BoundValue::U64(lo), BoundValue::U64(hi)) => {
                m.set_bounds(lo, hi);
                Ok(SortedMerge::U64(m))
            }
            (SortedMerge::I32(mut m), BoundValue::I32(lo), BoundValue::I32(hi)) => {
                m.set_bounds(lo, hi);
                Ok(SortedMerge::I32(m))
            }
            _ => Err(Error::Internal("bound type mismatch".into())),
        }
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

// ------------------------------ Unsorted scan -----------------------------

pub struct UnsortedScan<P: Pager<Blob = EntryHandle>> {
    pager: Arc<P>,
    metas: Vec<ChunkMetadata>,
    idx: usize,
}

impl<P> Iterator for UnsortedScan<P>
where
    P: Pager<Blob = EntryHandle>,
{
    type Item = Result<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.metas.len() {
            return None;
        }
        let m = self.metas[self.idx];
        self.idx += 1;

        let res = self.pager.batch_get(&[BatchGet::Raw { key: m.chunk_pk }]);
        let blob = match res {
            Ok(mut v) => match v.pop() {
                Some(GetResult::Raw { bytes, .. }) => bytes,
                _ => return Some(Err(Error::NotFound)),
            },
            Err(e) => return Some(Err(e)),
        };
        let arr = deserialize_array(blob);
        Some(arr)
    }
}

// ------ ColumnStore shims: scan() and sorted variants (back-compat + ergonomic) ---------

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle>,
{
    /// Unsorted streaming scan over the column's chunks.
    pub fn scan(&self, field_id: LogicalFieldId) -> Result<UnsortedScan<P>> {
        // Look up descriptor pk
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;

        // Load ColumnDescriptor to get head_page_pk
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // Walk the descriptor pages to gather metas
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk) {
            let meta = m?;
            if meta.row_count > 0 {
                metas.push(meta);
            }
        }

        Ok(UnsortedScan {
            pager: Arc::clone(&self.pager),
            metas,
            idx: 0,
        })
    }

    /// Sorted scan via k-way merge over pre-sorted chunks. Requires that
    /// `create_sort_index(field_id)` has been called (perms must exist).
    pub fn scan_sorted(&self, field_id: LogicalFieldId) -> Result<SortedMerge> {
        // Look up descriptor pk
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;

        // Load ColumnDescriptor to get head_page_pk
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let desc = crate::store::descriptor::ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // Gather metas from the page chain (require perm for sorted scans)
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk) {
            let meta = m?;
            if meta.row_count == 0 {
                continue;
            }
            if meta.value_order_perm_pk == 0 {
                return Err(Error::NotFound);
            }
            metas.push(meta);
        }

        if metas.is_empty() {
            // Build an empty iterator with correct variant
            return SortedMerge::build(&[], FxHashMap::default(), SortOptions::default());
        }

        // Batch get all data and perm blobs in one go.
        let mut gets: Vec<BatchGet> = Vec::with_capacity(metas.len() * 2);
        for m in &metas {
            gets.push(BatchGet::Raw { key: m.chunk_pk });
            gets.push(BatchGet::Raw {
                key: m.value_order_perm_pk,
            });
        }
        let results = self.pager.batch_get(&gets)?;

        let mut blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results {
            match r {
                GetResult::Raw { key, bytes } => {
                    blobs.insert(key, bytes);
                }
                _ => return Err(Error::NotFound),
            }
        }

        SortedMerge::build(&metas, blobs, SortOptions::default())
    }

    /// Ergonomic: sorted scan + apply type-erased bounds in one call.
    pub fn scan_sorted_with_bounds(
        &self,
        field_id: LogicalFieldId,
        lo: BoundValue,
        hi: BoundValue,
    ) -> Result<SortedMerge> {
        let it = self.scan_sorted(field_id)?;
        it.with_bounds(lo, hi)
    }
}
