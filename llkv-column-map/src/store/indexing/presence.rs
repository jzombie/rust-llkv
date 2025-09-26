//! Presence index: type and ops live here.

use super::{Index, IndexKind, IndexManager, IndexOps};
use crate::store::{
    ColumnStore,
    catalog::ColumnCatalog,
    descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator},
    rowid_fid,
};
use crate::types::LogicalFieldId;
use arrow::array::UInt64Array;
use arrow::compute::{SortColumn, lexsort_to_indices};
use llkv_result::{Error, Result};
use llkv_storage::{
    pager::{BatchGet, BatchPut, GetResult, Pager},
    serialization::{deserialize_array, serialize_array},
    types::PhysicalKey,
};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLockReadGuard};

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
    fn stage_registration(
        &self,
        index_manager: &IndexManager<P>,
        column_store: &ColumnStore<P>,
        catalog: RwLockReadGuard<'_, ColumnCatalog>,
        field_id: LogicalFieldId,
        descriptor: &mut ColumnDescriptor,
        descriptor_pk: PhysicalKey,
        puts: &mut Vec<BatchPut>,
        frees: &mut Vec<PhysicalKey>,
    ) -> Result<()> {
        // The Presence index operates on the shadow row_id column.
        let rid_fid = rowid_fid(field_id);
        let rid_pk = *catalog.map.get(&rid_fid).ok_or(Error::NotFound)?;
        let rid_desc_blob = index_manager
            .pager
            .batch_get(&[BatchGet::Raw { key: rid_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut rid_desc = ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());

        let mut modified_rid_metas = Vec::new();
        for meta_result in
            DescriptorIterator::new(index_manager.pager.as_ref(), rid_desc.head_page_pk)
        {
            let mut meta = meta_result?;
            let chunk_blob = index_manager
                .pager
                .batch_get(&[BatchGet::Raw { key: meta.chunk_pk }])?
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?;
            if let Some(put) = <PresenceIndexOps as IndexOps<P>>::stage_build_for_chunk(
                &index_manager.presence_ops,
                &index_manager.pager,
                &mut meta,
                &chunk_blob,
            )? {
                puts.push(put);
            }
            modified_rid_metas.push(meta);
        }

        // Logically register the index on the main data descriptor.
        index_manager.stage_index_registration(descriptor, IndexKind::Presence)?;

        // Stage a write for the main descriptor, as it's now changed.
        puts.push(BatchPut::Raw {
            key: descriptor_pk,
            bytes: descriptor.to_le_bytes(),
        });
        // Separately, rewrite the descriptor chain for the shadow row_id column.
        column_store.write_descriptor_chain(
            rid_pk,
            &mut rid_desc,
            &modified_rid_metas,
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
        catalog: RwLockReadGuard<'_, ColumnCatalog>,
        field_id: LogicalFieldId,
        descriptor: &mut ColumnDescriptor,
        descriptor_pk: PhysicalKey,
        puts: &mut Vec<BatchPut>,
        frees: &mut Vec<PhysicalKey>,
    ) -> Result<()> {
        // The Presence index lives on the shadow row_id column.
        let rid_fid = rowid_fid(field_id);
        let rid_pk = *catalog.map.get(&rid_fid).ok_or(Error::NotFound)?;
        let rid_desc_blob = index_manager
            .pager
            .batch_get(&[BatchGet::Raw { key: rid_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut rid_desc = ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());

        let mut rid_metas: Vec<ChunkMetadata> =
            DescriptorIterator::new(index_manager.pager.as_ref(), rid_desc.head_page_pk)
                .collect::<Result<_>>()?;

        let presence_frees = <PresenceIndexOps as IndexOps<P>>::stage_drop_index(
            &index_manager.presence_ops,
            &index_manager.pager,
            &mut rid_metas,
        )?;
        frees.extend(presence_frees);

        // Logically unregister on the main descriptor first.
        index_manager.stage_index_unregistration(descriptor, IndexKind::Presence)?;

        // Stage the write for the main descriptor.
        puts.push(BatchPut::Raw {
            key: descriptor_pk,
            bytes: descriptor.to_le_bytes(),
        });

        // Then rewrite the now-modified shadow descriptor chain.
        column_store.write_descriptor_chain(rid_pk, &mut rid_desc, &rid_metas, puts, frees)?;

        Ok(())
    }

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
