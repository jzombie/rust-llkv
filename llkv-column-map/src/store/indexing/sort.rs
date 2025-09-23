//! Sort index: type and ops live here.

use super::{Index, IndexKind, IndexOps};
use crate::error::Result;
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
    fn stage_update_for_new_chunk(
        &self,
        pager: &Arc<P>,
        value_slice: &arrow::array::ArrayRef,
        _rid_slice: &arrow::array::ArrayRef,
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

    /// Builds the sort index for a single existing data chunk.
    fn stage_build_for_chunk(
        &self,
        pager: &Arc<P>,
        meta: &mut ChunkMetadata,
        chunk_blob: &P::Blob,
    ) -> Result<Option<BatchPut>> {
        let arr = deserialize_array(chunk_blob.clone())?;

        let sort_col = SortColumn {
            values: arr,
            options: None,
        };
        let idx = lexsort_to_indices(&[sort_col], None)?;
        let perm_bytes = serialize_array(&idx)?;
        let perm_pk = pager.alloc_many(1)?[0];

        meta.value_order_perm_pk = perm_pk;

        Ok(Some(BatchPut::Raw {
            key: perm_pk,
            bytes: perm_bytes,
        }))
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
