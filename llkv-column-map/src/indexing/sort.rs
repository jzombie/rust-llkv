//! The sort index, implemented as value-sorted permutations.

use super::traits::{BackfillContext, Index, IndexPlan};
use crate::error::{Error, Result};
use crate::serialization::serialize_array;
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{ColumnDescriptor, DescriptorIterator};
use crate::types::{LogicalFieldId, Namespace};
use arrow::array::ArrayRef;
use arrow::compute::{SortColumn, lexsort_to_indices};
use simd_r_drive_entry_handle::EntryHandle;

/// A no-op index used for discovery of legacy sort permutations.
#[derive(Default)]
pub struct SortIndex;

impl SortIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for SortIndex {
    fn name(&self) -> &'static str {
        "sort"
    }

    /// Planning is a no-op because the sort index is currently created
    /// explicitly via `create_sort_index` and not on the append path.
    fn plan_append(
        &self,
        _data_for_chunk: &ArrayRef,
        _rids_for_chunk: &ArrayRef,
    ) -> Result<Option<IndexPlan>> {
        Ok(None)
    }

    /// A sort index is present if any chunk for a user data column has a perm key.
    fn discover(
        &self,
        pager: &dyn Pager<Blob = EntryHandle>,
        catalog: &ColumnCatalog,
    ) -> Result<Vec<LogicalFieldId>> {
        let mut found = Vec::new();
        for (&field_id, &descriptor_pk) in &catalog.map {
            if field_id.namespace() == Namespace::UserData {
                let desc_blob = pager
                    .batch_get(&[crate::storage::pager::BatchGet::Raw { key: descriptor_pk }])?
                    .pop()
                    .and_then(|r| match r {
                        crate::storage::pager::GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?;
                let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
                for m in DescriptorIterator::new(pager, descriptor.head_page_pk) {
                    if m?.value_order_perm_pk != 0 {
                        found.push(field_id);
                        break;
                    }
                }
            }
        }
        Ok(found)
    }

    fn backfill(&self, context: &mut BackfillContext<'_>, field_id: LogicalFieldId) -> Result<()> {
        let descriptor_pk = *context.catalog.map.get(&field_id).ok_or(Error::NotFound)?;

        let desc_blob = context
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let mut all_chunk_metadata = Vec::new();
        let meta_iter = DescriptorIterator::new(context.pager, descriptor.head_page_pk);
        for meta_result in meta_iter {
            all_chunk_metadata.push(meta_result?);
        }

        let data_chunk_keys: Vec<_> = all_chunk_metadata.iter().map(|m| m.chunk_pk).collect();
        let gets: Vec<_> = data_chunk_keys
            .into_iter()
            .map(|k| BatchGet::Raw { key: k })
            .collect();
        let results = context.pager.batch_get(&gets)?;

        let mut chunks_map = rustc_hash::FxHashMap::default();
        for result in results {
            match result {
                GetResult::Raw { key, bytes } => {
                    chunks_map.insert(key, bytes);
                }
                GetResult::Missing { .. } => return Err(Error::NotFound),
            }
        }

        for meta in all_chunk_metadata.iter_mut() {
            if meta.value_order_perm_pk == 0 {
                let chunk_blob = chunks_map.get(&meta.chunk_pk).ok_or(Error::NotFound)?;
                let chunk_array = crate::serialization::deserialize_array(chunk_blob.clone())?;

                let sort_column = SortColumn {
                    values: chunk_array,
                    options: None,
                };
                let indices = lexsort_to_indices(&[sort_column], None)?;
                let perm_bytes = serialize_array(&indices)?;
                let perm_pk = context.pager.alloc_many(1)?[0];
                context.puts.push(crate::storage::pager::BatchPut::Raw {
                    key: perm_pk,
                    bytes: perm_bytes,
                });
                meta.value_order_perm_pk = perm_pk;
            }
        }

        // Rewrite descriptor pages with updated perm keys.
        let mut current_page_pk = descriptor.head_page_pk;
        let mut page_start_chunk_idx = 0;
        while current_page_pk != 0 {
            let page_blob = context
                .pager
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
            let header = crate::store::descriptor::DescriptorPageHeader::from_le_bytes(
                &page_blob[..crate::store::descriptor::DescriptorPageHeader::DISK_SIZE],
            );
            let chunks_on_this_page = header.entry_count as usize;
            let page_end_chunk_idx = page_start_chunk_idx + chunks_on_this_page;
            let updated_chunks = &all_chunk_metadata[page_start_chunk_idx..page_end_chunk_idx];
            let new_page_data =
                crate::store::descriptor::build_descriptor_page_bytes(&header, updated_chunks);

            context.puts.push(crate::storage::pager::BatchPut::Raw {
                key: current_page_pk,
                bytes: new_page_data,
            });
            current_page_pk = header.next_page_pk;
            page_start_chunk_idx = page_end_chunk_idx;
        }

        Ok(())
    }
}
