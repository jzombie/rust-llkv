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
    fn stage_update_for_new_chunk(
        &self,
        pager: &Arc<P>,
        _value_slice: &arrow::array::ArrayRef,
        rid_slice: &arrow::array::ArrayRef,
        _value_meta: &mut ChunkMetadata,
        rid_meta: &mut ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let rids = rid_slice
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("row_id array must be u64".into()))?;

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

    /// Builds the presence index for a single existing row_id chunk.
    fn stage_build_for_chunk(
        &self,
        pager: &Arc<P>,
        meta: &mut ChunkMetadata,
        chunk_blob: &P::Blob,
    ) -> Result<Option<BatchPut>> {
        if meta.value_order_perm_pk != 0 {
            return Ok(None); // Permutation already exists
        }

        let rid_arr = deserialize_array(chunk_blob.clone())?;
        let rids = rid_arr.as_any().downcast_ref::<UInt64Array>().unwrap();

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
                values: rid_arr.clone(),
                options: None,
            };
            let perm_indices = lexsort_to_indices(&[sort_col], None)?;
            let perm_bytes = serialize_array(&perm_indices)?;
            let perm_pk = pager.alloc_many(1)?[0];
            meta.value_order_perm_pk = perm_pk;
            Ok(Some(BatchPut::Raw {
                key: perm_pk,
                bytes: perm_bytes,
            }))
        } else {
            Ok(None) // Chunk is already sorted, no permutation needed.
        }
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
