//! K-way merge over pre-sorted chunks with monomorphized hot path.
//!
//! Design:
//! - One-time type dispatch at construction.
//! - Hot loop compares native keys (no u128 widen).
//! - Returned items remain `ArrayRef` 1-row slices (API unchanged).
//!
//! Note on copies: `compute::take` materializes a sorted view per chunk.
//! For fixed-width types this is a single contiguous copy per chunk.
//! No `Vec<u8>` cloning of underlying blob pages occurs here; buffers
//! remain Arc-backed and zero-copy from your EntryHandle.

use super::zero_offset;
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

// -----------------------------------------------------------------------------
// Generic heap item keyed by a native scalar. Used with Reverse for min-heap.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct HeapItem<K: Ord + Copy> {
    cursor_idx: usize,
    data_idx: usize,
    key: K,
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

// -----------------------------------------------------------------------------
// Typed cursor holding both Any view (for slicing) and typed array for keys.
// We store the typed array by value (cheap clone). Underlying buffers remain
// Arc-backed; no deep copy of payloads.
// -----------------------------------------------------------------------------

struct Cursor<A> {
    any: ArrayRef,
    typed: A,
    len: usize,
}

// -----------------------------------------------------------------------------
// Macro generates a monomorphized merge iterator for a given Arrow type.
// `ArrTy` is the Arrow array type; `KeyTy` is the comparable key type.
// -----------------------------------------------------------------------------

macro_rules! impl_merge_for_primitive {
    ($Name:ident, $ArrTy:ty, $KeyTy:ty) => {
        struct $Name {
            cursors: Vec<Cursor<$ArrTy>>,
            heap: BinaryHeap<Reverse<HeapItem<$KeyTy>>>,
            _blobs: FxHashMap<PhysicalKey, EntryHandle>,
        }

        impl $Name {
            fn try_new(
                metadata: Vec<ChunkMetadata>,
                blobs: FxHashMap<PhysicalKey, EntryHandle>,
            ) -> Result<Self> {
                let mut cursors = Vec::with_capacity(metadata.len());

                // Build typed, sorted cursors once per chunk.
                for m in &metadata {
                    if m.row_count == 0 {
                        continue;
                    }

                    let data_blob = blobs.get(&m.chunk_pk).ok_or(Error::NotFound)?.clone();
                    let data_any = deserialize_array(data_blob)?;

                    let perm_blob = blobs
                        .get(&m.value_order_perm_pk)
                        .ok_or(Error::NotFound)?
                        .clone();
                    let perm_any = deserialize_array(perm_blob)?;
                    let perm = perm_any
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .ok_or_else(|| Error::Internal("perm not u32".into()))?;

                    // Materialize sorted view for this chunk (contiguous copy).
                    let sorted_any = compute::take(&data_any, perm, None)?;
                    let sorted_any = zero_offset(&sorted_any);

                    // Cheap-clone typed array (shares buffers).
                    let typed = sorted_any
                        .as_any()
                        .downcast_ref::<$ArrTy>()
                        .ok_or_else(|| Error::Internal("typed downcast failed".into()))?
                        .clone();

                    cursors.push(Cursor::<$ArrTy> {
                        any: sorted_any.clone(),
                        typed,
                        len: sorted_any.len(),
                    });
                }

                // Seed heap with first element from each cursor.
                let mut heap: BinaryHeap<Reverse<HeapItem<$KeyTy>>> = BinaryHeap::new();

                for (i, c) in cursors.iter().enumerate() {
                    if c.len > 0 {
                        let k: $KeyTy = c.typed.value(0) as $KeyTy;
                        heap.push(Reverse(HeapItem {
                            cursor_idx: i,
                            data_idx: 0,
                            key: k,
                        }));
                    }
                }

                Ok(Self {
                    cursors,
                    heap,
                    _blobs: blobs,
                })
            }

            #[inline]
            fn next_inner(&mut self) -> Option<Result<ArrayRef>> {
                let Reverse(item) = self.heap.pop()?;

                let c = &self.cursors[item.cursor_idx];
                let out = c.any.slice(item.data_idx, 1);

                let next_idx = item.data_idx + 1;
                if next_idx < c.len {
                    let k: $KeyTy = c.typed.value(next_idx) as $KeyTy;
                    self.heap.push(Reverse(HeapItem {
                        cursor_idx: item.cursor_idx,
                        data_idx: next_idx,
                        key: k,
                    }));
                }

                Some(Ok(out))
            }
        }

        impl Iterator for $Name {
            type Item = Result<ArrayRef>;
            fn next(&mut self) -> Option<Self::Item> {
                self.next_inner()
            }
        }
    };
}

// Instantiate for currently supported types.
// Extend by adding more lines (e.g., Int64Array/i64, UInt32Array/u32).
impl_merge_for_primitive!(MergeU64, UInt64Array, u64);
impl_merge_for_primitive!(MergeI32, Int32Array, i32);

// -----------------------------------------------------------------------------
// Public facade: preserves original API and surface area.
// -----------------------------------------------------------------------------

enum MergeInner {
    U64(MergeU64),
    I32(MergeI32),
    Empty,
}

pub struct MergeSortedIterator {
    inner: MergeInner,
}

impl MergeSortedIterator {
    /// Construct a merge iterator over value-sorted chunks.
    ///
    /// `metadata` must refer to chunks where `value_order_perm_pk` is set.
    /// Keeps `blobs` alive to preserve zero-copy backing pages.
    pub fn try_new(
        metadata: Vec<ChunkMetadata>,
        blobs: FxHashMap<PhysicalKey, EntryHandle>,
    ) -> Result<Self> {
        // Find first non-empty chunk to detect the type.
        let first = match metadata.iter().find(|m| m.row_count > 0) {
            Some(m) => m,
            None => {
                return Ok(Self {
                    inner: MergeInner::Empty,
                });
            }
        };

        let first_blob = blobs.get(&first.chunk_pk).ok_or(Error::NotFound)?.clone();
        let first_any = deserialize_array(first_blob)?;

        let inner = match first_any.data_type() {
            DataType::UInt64 => MergeInner::U64(MergeU64::try_new(metadata, blobs)?),
            DataType::Int32 => MergeInner::I32(MergeI32::try_new(metadata, blobs)?),
            other => {
                return Err(Error::Internal(format!(
                    "Unsupported sort type {:?}",
                    other
                )));
            }
        };

        Ok(Self { inner })
    }
}

impl Iterator for MergeSortedIterator {
    type Item = Result<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            MergeInner::U64(it) => it.next(),
            MergeInner::I32(it) => it.next(),
            MergeInner::Empty => None,
        }
    }
}
