use super::zero_offset;
use crate::error::{Error, Result};
use crate::serialization::deserialize_array;
use crate::store::descriptor::ChunkMetadata;
use crate::types::PhysicalKey;
use arrow::array::Int32Array;
use arrow::array::{Array, ArrayRef, UInt32Array, UInt64Array};
use arrow::compute;
use arrow::datatypes::DataType;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// --- K-way merge implementation (EntryHandle-backed, zero-copy) ---

struct ChunkCursor {
    /// A fully sorted view of the chunk's data.
    data_sorted: ArrayRef,
    len: usize,
}

#[derive(Clone, Copy, Debug)]
struct HeapItem {
    cursor_idx: usize,
    /// The index into the `data_sorted` array for the current cursor.
    data_idx: usize,
    /// A normalized `u128` representation of the item's value, used for
    /// sorting across different numeric types in the min-heap.
    key_u128: u128,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key_u128 == other.key_u128 && self.cursor_idx == other.cursor_idx
    }
}
impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural ascending order by key, with a stable tie-breaker (cursor_idx).
        // This is the correct logic for use with `Reverse` to create a min-heap.
        self.key_u128
            .cmp(&other.key_u128)
            .then_with(|| self.cursor_idx.cmp(&other.cursor_idx))
    }
}

#[derive(Clone, Copy)]
enum KeyType {
    U64,
    I32,
}

/// Encodes different numeric types into a common `u128` space where their natural
/// sort order is preserved for the min-heap.
///
/// - **For `u64`**, this is a direct cast, as its sort order is already correct.
/// - **For `i32`**, a simple cast would break sorting due to two's complement
///   representation (e.g., `-1` would become a very large positive number).
///   To fix this, we shift the entire `i32` range to be non-negative by
///   subtracting `i32::MIN`. This ensures that if `a < b` in `i32`, their `u128`
///   representations also satisfy `a' < b'`.
#[inline]
fn encode_key_u128(arr: &ArrayRef, idx: usize, kind: KeyType) -> u128 {
    match kind {
        KeyType::U64 => {
            let a = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
            a.value(idx) as u128
        }
        KeyType::I32 => {
            let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
            (a.value(idx) as i128 - i32::MIN as i128) as u128
        }
    }
}

pub struct MergeSortedIterator {
    cursors: Vec<ChunkCursor>,
    heap: BinaryHeap<Reverse<HeapItem>>, // min-heap by key
    key_kind: KeyType,
    _blobs: FxHashMap<PhysicalKey, EntryHandle>, // Keeps memory alive
}

impl MergeSortedIterator {
    pub fn try_new(
        metadata: Vec<ChunkMetadata>,
        blobs: FxHashMap<PhysicalKey, EntryHandle>,
    ) -> Result<Self> {
        // Determine key type from the first non-empty chunk.
        let key_kind = {
            let first_meta = metadata.iter().find(|m| m.row_count > 0);
            if first_meta.is_none() {
                // Handle case where all chunks are empty
                return Ok(Self {
                    cursors: Vec::new(),
                    heap: BinaryHeap::new(),
                    key_kind: KeyType::U64, // Default, won't be used
                    _blobs: blobs,
                });
            }
            let data_blob = blobs
                .get(&first_meta.unwrap().chunk_pk)
                .ok_or(Error::NotFound)?
                .clone();
            let data_array = deserialize_array(data_blob)?;
            match data_array.data_type() {
                &DataType::UInt64 => KeyType::U64,
                &DataType::Int32 => KeyType::I32,
                other => {
                    return Err(Error::Internal(format!(
                        "Unsupported sort type {:?}",
                        other
                    )));
                }
            }
        };
        let mut cursors = Vec::with_capacity(metadata.len());
        for meta in &metadata {
            if meta.row_count == 0 {
                continue;
            }
            let data_blob = blobs.get(&meta.chunk_pk).ok_or(Error::NotFound)?.clone();
            let data_arr = deserialize_array(data_blob)?;
            let perm_blob = blobs
                .get(&meta.value_order_perm_pk)
                .ok_or(Error::NotFound)?
                .clone();
            let perm_any = deserialize_array(perm_blob)?;
            let perm = perm_any
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or(Error::Internal("perm not u32".into()))?;
            // Materialize the sorted view of the chunk's data.
            let sorted = compute::take(&data_arr, perm, None)?;
            cursors.push(ChunkCursor {
                data_sorted: zero_offset(&sorted),
                len: sorted.len(),
            });
        }

        // Seed the heap with the first element from each chunk.
        let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        for (i, cursor) in cursors.iter().enumerate() {
            if !cursor.data_sorted.is_empty() {
                let key = encode_key_u128(&cursor.data_sorted, 0, key_kind);
                heap.push(Reverse(HeapItem {
                    cursor_idx: i,
                    data_idx: 0,
                    key_u128: key,
                }));
            }
        }

        Ok(Self {
            cursors,
            heap,
            key_kind,
            _blobs: blobs,
        })
    }
}

impl Iterator for MergeSortedIterator {
    type Item = Result<ArrayRef>;
    fn next(&mut self) -> Option<Self::Item> {
        // Pop the item with the smallest key from the heap.
        match self.heap.pop() {
            Some(Reverse(item)) => {
                let cursor = &self.cursors[item.cursor_idx];
                // Create a 1-element slice to return.
                let output_slice = cursor.data_sorted.slice(item.data_idx, 1);
                // If there are more elements in this chunk, push the next one onto the heap.
                let next_data_idx = item.data_idx + 1;
                if next_data_idx < cursor.len {
                    let next_key =
                        encode_key_u128(&cursor.data_sorted, next_data_idx, self.key_kind);
                    self.heap.push(Reverse(HeapItem {
                        cursor_idx: item.cursor_idx,
                        data_idx: next_data_idx,
                        key_u128: next_key,
                    }));
                }

                Some(Ok(output_slice))
            }
            // If the heap is empty, the iteration is complete.
            None => None,
        }
    }
}
