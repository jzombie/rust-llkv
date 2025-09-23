//! Presence index: type and ops live here.

use super::{Index, IndexKind, IndexOps};
use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchPut, Pager};
use crate::store::descriptor::ChunkMetadata;
use crate::types::PhysicalKey;
use arrow::array::UInt64Array;
use arrow::compute::{SortColumn, lexsort_to_indices};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Public marker type for the presence index.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct PresenceIndex;

impl Index for PresenceIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Presence
    }
}

/// Concrete implementation of presence index operations.
#[derive(Default)]
pub struct PresenceIndexOps;

impl<P: Pager<Blob = EntryHandle>> IndexOps<P> for PresenceIndexOps {
    /// For the Presence index, this operates on the row ID chunk, not the value chunk.
    fn stage_update_for_new_chunk(
        &self,
        pager: &Arc<P>,
        _value_slice: &arrow::array::ArrayRef, // Presence index ignores the values
        rid_slice: &arrow::array::ArrayRef,
        _value_meta: &mut ChunkMetadata,
        rid_meta: &mut ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let rids = rid_slice
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("row_id array must be u64".into()))?;

        // Check if the row IDs are already sorted.
        let mut is_sorted = true;
        if !rids.is_empty() {
            let mut last = rids.value(0);
            for i in 1..rids.len() {
                let current = rids.value(i);
                if current < last {
                    is_sorted = false;
                    break;
                }
                last = current;
            }
        }

        if !is_sorted {
            let sort_col = SortColumn {
                values: rid_slice.clone(),
                options: None,
            };
            let perm_indices = lexsort_to_indices(&[sort_col], None)?;
            let perm_bytes = serialize_array(&perm_indices)?;
            let perm_pk = pager.alloc_many(1)?[0];
            puts.push(BatchPut::Raw {
                key: perm_pk,
                bytes: perm_bytes,
            });
            rid_meta.value_order_perm_pk = perm_pk;
        }
        Ok(())
    }

    /// The presence index is built on the shadow row_id column.
    fn stage_build_all(
        &self,
        pager: &Arc<P>,
        metas: &mut [ChunkMetadata],
        chunk_blobs: &rustc_hash::FxHashMap<PhysicalKey, P::Blob>,
    ) -> Result<Vec<BatchPut>> {
        let mut puts = Vec::new();
        for meta in metas.iter_mut() {
            // Only build if a permutation doesn't already exist.
            if meta.value_order_perm_pk == 0 {
                let blob = chunk_blobs
                    .get(&meta.chunk_pk)
                    .ok_or(Error::NotFound)?
                    .clone();
                let rid_arr = deserialize_array(blob)?;
                // Reuse the chunk logic to perform the build.
                self.stage_update_for_new_chunk(
                    pager,
                    &rid_arr, // value_slice is ignored, can pass anything
                    &rid_arr,
                    &mut ChunkMetadata::default(), // value_meta is ignored
                    meta,
                    &mut puts,
                )?;
            }
        }
        Ok(puts)
    }

    /// The presence index lives on the row_id column's metadata.
    fn stage_drop_index(
        &self,
        _pager: &Arc<P>,
        metas: &mut [ChunkMetadata],
    ) -> Result<Vec<PhysicalKey>> {
        let mut frees = Vec::new();
        for m in metas.iter_mut() {
            if m.value_order_perm_pk != 0 {
                frees.push(m.value_order_perm_pk);
                m.value_order_perm_pk = 0;
            }
        }
        Ok(frees)
    }
}
