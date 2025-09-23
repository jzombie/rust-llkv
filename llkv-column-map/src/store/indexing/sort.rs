//! Sort index: type and ops live here.

use super::IndexKind;
use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::store::indexing::{Index, IndexOps, IndexUpdateHint};
use crate::types::{LogicalFieldId, PhysicalKey};
use arrow::compute::{SortColumn, lexsort_to_indices};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

/// Public marker type for the sort index.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct SortIndex;

impl SortIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for SortIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Sort
    }
}

/// Concrete implementation of sort index operations.
///
/// This is a direct migration of the logic previously in
/// `core.rs::create_sort_index`, structured under the uniform
/// `IndexOps` trait.
pub struct SortIndexOps;

impl Default for SortIndexOps {
    fn default() -> Self {
        SortIndexOps
    }
}

impl<P> IndexOps<P> for SortIndexOps
where
    P: Pager<Blob = EntryHandle>,
{
    fn build_all(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()> {
        // Resolve descriptor pk for the data column.
        let cat = catalog.read().unwrap();
        let descriptor_pk = *cat.map.get(&field).ok_or(Error::NotFound)?;
        drop(cat);

        // Load descriptor.
        let desc_blob = pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // Gather all chunk metadata.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            return Ok(());
        }

        // Fetch all data chunks in one batch.
        let gets: Vec<BatchGet> = metas
            .iter()
            .map(|m| BatchGet::Raw { key: m.chunk_pk })
            .collect();
        let results = pager.batch_get(&gets)?;

        let mut chunks: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results {
            match r {
                GetResult::Raw { key, bytes } => {
                    chunks.insert(key, bytes);
                }
                GetResult::Missing { .. } => return Err(Error::NotFound),
            }
        }

        let mut puts: Vec<BatchPut> = Vec::new();

        // Compute and persist per-chunk permutations.
        for meta in metas.iter_mut() {
            let blob = chunks.get(&meta.chunk_pk).ok_or(Error::NotFound)?.clone();
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

        // Rewrite descriptor pages with updated metas.
        rewrite_descriptor_pages(pager, descriptor_pk, &mut descriptor, &mut metas, &mut puts)?;

        // Persist perms and page rewrites.
        if !puts.is_empty() {
            pager.batch_put(&puts)?;
        }
        Ok(())
    }

    fn update(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
        hint: &IndexUpdateHint,
    ) -> Result<()> {
        // If no hints or full rebuild allowed, just rebuild all.
        if hint.full_rebuild_ok || hint.changed_chunk_pks.is_empty() {
            return self.build_all(pager, catalog, field);
        }

        // Resolve descriptor pk and load descriptor.
        let cat = catalog.read().unwrap();
        let descriptor_pk = *cat.map.get(&field).ok_or(Error::NotFound)?;
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

        // Gather metas.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            return Ok(());
        }

        // Filter target chunks by the hint.
        let mut target: FxHashMap<PhysicalKey, ()> = FxHashMap::default();
        for k in hint.changed_chunk_pks.iter().copied() {
            target.insert(k, ());
        }

        let mut gets: Vec<BatchGet> = Vec::new();
        let mut idxs: Vec<usize> = Vec::new();
        for (i, m) in metas.iter().enumerate() {
            if target.contains_key(&m.chunk_pk) {
                gets.push(BatchGet::Raw { key: m.chunk_pk });
                idxs.push(i);
            }
        }
        if gets.is_empty() {
            return Ok(());
        }

        let results = pager.batch_get(&gets)?;
        let mut by_pk: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results {
            if let GetResult::Raw { key, bytes } = r {
                by_pk.insert(key, bytes);
            }
        }

        let mut puts: Vec<BatchPut> = Vec::new();

        // Recompute and write perms for targeted chunks.
        for &i in &idxs {
            let m = &mut metas[i];
            let blob = by_pk.get(&m.chunk_pk).ok_or(Error::NotFound)?.clone();
            let arr = deserialize_array(blob)?;

            let sort_col = SortColumn {
                values: arr,
                options: None,
            };
            let idx = lexsort_to_indices(&[sort_col], None)?;
            let perm_bytes = serialize_array(&idx)?;

            // Allocate if none; otherwise overwrite existing page.
            if m.value_order_perm_pk == 0 {
                m.value_order_perm_pk = pager.alloc_many(1)?[0];
            }
            puts.push(BatchPut::Raw {
                key: m.value_order_perm_pk,
                bytes: perm_bytes,
            });
        }

        // Rewrite descriptor pages to persist updated metas.
        rewrite_descriptor_pages(pager, descriptor_pk, &mut descriptor, &mut metas, &mut puts)?;

        if !puts.is_empty() {
            pager.batch_put(&puts)?;
        }
        Ok(())
    }

    fn drop_index(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()> {
        // Resolve descriptor pk and load descriptor.
        let cat = catalog.read().unwrap();
        let descriptor_pk = *cat.map.get(&field).ok_or(Error::NotFound)?;
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

        // Gather metas.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            return Ok(());
        }

        // Clear per-chunk permutation references.
        for m in metas.iter_mut() {
            m.value_order_perm_pk = 0;
        }

        let mut puts: Vec<BatchPut> = Vec::new();

        // Rewrite descriptor pages with cleared refs and persist.
        rewrite_descriptor_pages(pager, descriptor_pk, &mut descriptor, &mut metas, &mut puts)?;

        if !puts.is_empty() {
            pager.batch_put(&puts)?;
        }
        Ok(())
    }
}

/* ============================== helpers =============================== */

fn rewrite_descriptor_pages<P>(
    pager: &Arc<P>,
    descriptor_pk: PhysicalKey,
    descriptor: &mut ColumnDescriptor,
    metas: &mut [ChunkMetadata],
    puts: &mut Vec<BatchPut>,
) -> Result<()>
where
    P: Pager<Blob = EntryHandle>,
{
    let mut current_page_pk = descriptor.head_page_pk;
    let mut page_start_idx = 0usize;

    while current_page_pk != 0 {
        let page_blob = pager
            .batch_get(&[BatchGet::Raw {
                key: current_page_pk,
            }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?
            .as_ref()
            .to_vec();

        let header =
            DescriptorPageHeader::from_le_bytes(&page_blob[..DescriptorPageHeader::DISK_SIZE]);

        let n_on_page = header.entry_count as usize;
        let end_idx = page_start_idx + n_on_page;

        let mut new_page_data = Vec::new();
        for m in &metas[page_start_idx..end_idx] {
            new_page_data.extend_from_slice(&m.to_le_bytes());
        }

        let mut final_page_bytes =
            Vec::with_capacity(DescriptorPageHeader::DISK_SIZE + new_page_data.len());
        final_page_bytes.extend_from_slice(&header.to_le_bytes());
        final_page_bytes.extend_from_slice(&new_page_data);

        puts.push(BatchPut::Raw {
            key: current_page_pk,
            bytes: final_page_bytes,
        });

        current_page_pk = header.next_page_pk;
        page_start_idx = end_idx;
    }

    // Update totals on the descriptor and write it.
    let mut total_rows = 0u64;
    for m in metas.iter() {
        total_rows += m.row_count;
    }
    descriptor.total_row_count = total_rows;

    puts.push(BatchPut::Raw {
        key: descriptor_pk,
        bytes: descriptor.to_le_bytes(),
    });

    Ok(())
}
