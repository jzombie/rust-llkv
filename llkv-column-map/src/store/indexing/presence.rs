//! Presence index: type and ops live here.

use super::IndexKind;
use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::store::indexing::{Index, IndexOps, IndexUpdateHint};
use crate::store::rowid::rowid_fid; // Helper to get the shadow fid
use crate::types::LogicalFieldId;
use arrow::array::UInt64Array;
use arrow::compute::{SortColumn, lexsort_to_indices};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

/// Public marker type for the presence index.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct PresenceIndex;

impl PresenceIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for PresenceIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Presence
    }
}

/// Concrete implementation of presence index operations.
pub struct PresenceIndexOps;

impl<P> IndexOps<P> for PresenceIndexOps
where
    P: Pager<Blob = EntryHandle>,
{
    /// Builds the sorting permutation for all chunks in a column's shadow row_id column.
    fn build_all(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()> {
        let cat = catalog.read().unwrap();
        // 1. Get the field ID for the shadow row_id column.
        let rid_fid = rowid_fid(field);
        let descriptor_pk = *cat.map.get(&rid_fid).ok_or(Error::NotFound)?;
        drop(cat);

        // 2. Load the descriptor for the row_id column.
        let desc_blob = pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // 3. Gather metadata for all chunks.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            return Ok(());
        }

        // 4. Batch fetch all chunk blobs.
        let gets: Vec<BatchGet> = metas
            .iter()
            .map(|m| BatchGet::Raw { key: m.chunk_pk })
            .collect();
        let results = pager.batch_get(&gets)?;
        let chunks: FxHashMap<_, _> = results
            .into_iter()
            .filter_map(|r| match r {
                GetResult::Raw { key, bytes } => Some((key, bytes)),
                _ => None,
            })
            .collect();

        let mut puts: Vec<BatchPut> = Vec::new();

        // 5. For each chunk, compute and persist a sorting permutation if it's not already sorted.
        for meta in metas.iter_mut() {
            if meta.value_order_perm_pk != 0 {
                continue; // Skip if permutation already exists
            }
            let blob = chunks.get(&meta.chunk_pk).ok_or(Error::NotFound)?.clone();
            let arr = deserialize_array(blob)?;

            // Check if the row_id chunk is already sorted
            let rids = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
            let is_sorted = rids.values().windows(2).all(|w| w[0] <= w[1]);

            if !is_sorted {
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
        }

        // 6. Rewrite descriptor pages with updated metadata and commit writes.
        descriptor.rewrite_pages(Arc::clone(pager), descriptor_pk, &mut metas, &mut puts)?;
        if !puts.is_empty() {
            pager.batch_put(&puts)?;
        }
        Ok(())
    }

    fn update(
        &self,
        _pager: &Arc<P>,
        _catalog: &Arc<RwLock<ColumnCatalog>>,
        _field: LogicalFieldId,
        _hint: &IndexUpdateHint,
    ) -> Result<()> {
        // TODO: Implement incremental updates for changed chunks, similar to build_all
        // but filtered by the hint's changed_chunk_pks. For now, a full rebuild is a
        // safe fallback.
        self.build_all(_pager, _catalog, _field)
    }

    /// Clears all row_id sorting permutations for the given field.
    fn drop_index(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()> {
        let cat = catalog.read().unwrap();
        let rid_fid = rowid_fid(field);
        let descriptor_pk = *cat.map.get(&rid_fid).ok_or(Error::NotFound)?;
        drop(cat);

        let desc_blob = pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }

        let mut frees = Vec::new();
        for m in metas.iter_mut() {
            if m.value_order_perm_pk != 0 {
                frees.push(m.value_order_perm_pk);
                m.value_order_perm_pk = 0; // Clear the reference
            }
        }

        let mut puts: Vec<BatchPut> = Vec::new();
        descriptor.rewrite_pages(Arc::clone(pager), descriptor_pk, &mut metas, &mut puts)?;

        if !puts.is_empty() {
            pager.batch_put(&puts)?;
        }
        if !frees.is_empty() {
            pager.free_many(&frees)?;
        }
        Ok(())
    }
}
