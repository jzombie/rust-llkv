//! Sort index: type and ops live here.

use super::{Index, IndexKind, IndexManager, IndexOps};
use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::{
    ColumnStore,
    catalog::ColumnCatalog,
    descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator},
};
use crate::types::{LogicalFieldId, PhysicalKey};
use arrow::compute::{SortColumn, lexsort_to_indices};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLockReadGuard};

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
    fn stage_registration(
        &self,
        index_manager: &IndexManager<P>,
        column_store: &ColumnStore<P>,
        _catalog: RwLockReadGuard<'_, ColumnCatalog>,
        _field_id: LogicalFieldId,
        descriptor: &mut ColumnDescriptor,
        descriptor_pk: PhysicalKey,
        puts: &mut Vec<BatchPut>,
        frees: &mut Vec<PhysicalKey>,
    ) -> Result<()> {
        // The Sort index operates on the main data column's chunks.
        // This vector will hold the modified metadata to rewrite the descriptor pages at the end.
        let mut modified_metas = Vec::new();

        // Iterate through the column's metadata one chunk at a time to keep memory usage low.
        for meta_result in
            DescriptorIterator::new(index_manager.pager.as_ref(), descriptor.head_page_pk)
        {
            let mut meta = meta_result?;
            // Fetch only the data for the current chunk.
            let chunk_blob = index_manager
                .pager
                .batch_get(&[BatchGet::Raw { key: meta.chunk_pk }])?
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?;
            // Call the specialist to build the index for this single chunk.
            // This modifies `meta` in-place and returns a `BatchPut` for any new permutation blob.
            if let Some(put) = <SortIndexOps as IndexOps<P>>::stage_build_for_chunk(
                &index_manager.sort_ops,
                &index_manager.pager,
                &mut meta,
                &chunk_blob,
            )? {
                puts.push(put);
            }
            modified_metas.push(meta);
        }

        // Logically register the index on the in-memory descriptor FIRST.
        index_manager.stage_index_registration(descriptor, IndexKind::Sort)?;

        // NOW rewrite the descriptor chain, which will serialize the updated descriptor and its pages.
        column_store.write_descriptor_chain(
            descriptor_pk,
            descriptor,
            &modified_metas,
            puts,
            frees,
        )?;

        Ok(())
    }

    /// Stages the full removal of an index from a column.
    fn stage_unregistration(
        &self,
        index_manager: &IndexManager<P>,
        column_store: &ColumnStore<P>,
        _catalog: RwLockReadGuard<'_, ColumnCatalog>,
        _field_id: LogicalFieldId,
        descriptor: &mut ColumnDescriptor,
        descriptor_pk: PhysicalKey,
        puts: &mut Vec<BatchPut>,
        frees: &mut Vec<PhysicalKey>,
    ) -> Result<()> {
        // The Sort index lives on the main data column.
        let mut metas: Vec<ChunkMetadata> =
            DescriptorIterator::new(index_manager.pager.as_ref(), descriptor.head_page_pk)
                .collect::<Result<_>>()?;

        // `stage_drop_index` modifies `metas` in-place (clearing perm_pk) and returns a list of keys to free.
        let sort_frees = <SortIndexOps as IndexOps<P>>::stage_drop_index(
            &index_manager.sort_ops,
            &index_manager.pager,
            &mut metas,
        )?;
        frees.extend(sort_frees);

        // Logically unregister the index first.
        index_manager.stage_index_unregistration(descriptor, IndexKind::Sort)?;

        // Then rewrite the descriptor chain, which will serialize the updated descriptor and its pages.
        column_store.write_descriptor_chain(descriptor_pk, descriptor, &metas, puts, frees)?;

        Ok(())
    }

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
