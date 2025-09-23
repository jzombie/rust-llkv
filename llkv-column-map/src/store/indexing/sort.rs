//! Sort index: type and ops live here.

use super::{Index, IndexKind, IndexOps};
use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchPut, Pager};
use crate::store::descriptor::ChunkMetadata;
use crate::types::PhysicalKey;
use arrow::compute::{SortColumn, lexsort_to_indices};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Public marker type for the sort index.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct SortIndex;

impl Index for SortIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Sort
    }
}

/// Concrete implementation of sort index operations.
#[derive(Default)]
pub struct SortIndexOps;

impl<P: Pager<Blob = EntryHandle>> IndexOps<P> for SortIndexOps {
    /// For the Sort index, this operates on the value chunk.
    fn stage_update_for_new_chunk(
        &self,
        pager: &Arc<P>,
        value_slice: &arrow::array::ArrayRef,
        _rid_slice: &arrow::array::ArrayRef, // Sort index ignores row IDs
        value_meta: &mut ChunkMetadata,
        _rid_meta: &mut ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let sort_col = SortColumn {
            values: value_slice.clone(),
            options: None,
        };
        let perm_indices = lexsort_to_indices(&[sort_col], None)?;
        let perm_bytes = serialize_array(&perm_indices)?;
        let perm_pk = pager.alloc_many(1)?[0];
        puts.push(BatchPut::Raw {
            key: perm_pk,
            bytes: perm_bytes,
        });
        value_meta.value_order_perm_pk = perm_pk;
        Ok(())
    }

    /// Builds the sort index for all provided data chunks.
    fn stage_build_all(
        &self,
        pager: &Arc<P>,
        metas: &mut [ChunkMetadata],
        chunk_blobs: &rustc_hash::FxHashMap<PhysicalKey, P::Blob>,
    ) -> Result<Vec<BatchPut>> {
        let mut puts = Vec::new();
        for meta in metas.iter_mut() {
            let blob = chunk_blobs
                .get(&meta.chunk_pk)
                .ok_or(Error::NotFound)?
                .clone();
            let arr = deserialize_array(blob)?;

            let sort_col = SortColumn {
                values: arr,
                options: None,
            };
            let idx = lexsort_to_indices(&[sort_col], None)?;
            let perm_bytes = serialize_array(&idx)?;
            let perm_pk = pager.alloc_many(1)?[0];

            puts.push(BatchPut::Raw {
                key: perm_pk,
                bytes: perm_bytes,
            });
            meta.value_order_perm_pk = perm_pk;
        }
        Ok(puts)
    }

    /// Clears sort permutations from the data column's metadata.
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
