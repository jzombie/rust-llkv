//! The presence index, implemented as a shadow column of row IDs.

use super::traits::{Index, IndexOpContext};
use crate::error::{Error, Result};
use crate::serialization::serialize_array;
use crate::storage::pager::{BatchGet, BatchPut, GetResult};
use crate::store::descriptor::{
    build_descriptor_page_bytes, ChunkMetadata, ColumnDescriptor, DescriptorIterator,
    DescriptorPageHeader,
};
use crate::store::{rowid_fid, zero_offset};
use crate::types::LogicalFieldId;
use arrow::array::{ArrayRef, UInt64Array};
use arrow::compute::{SortColumn, lexsort_to_indices};

/// An index that tracks the presence of rows in a column via a shadow row-id column.
#[derive(Default)]
pub struct PresenceIndex;

impl PresenceIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for PresenceIndex {
    fn name(&self) -> &'static str {
        "presence"
    }

    fn on_append(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        new_chunks: &[(ChunkMetadata, ArrayRef)],
    ) -> Result<()> {
        let rid_fid = rowid_fid(field_id);
        let _rid_descriptor_pk = *context
            .catalog
            .map
            .entry(rid_fid)
            .or_insert_with(|| context.pager.alloc_many(1).unwrap()[0]);

        // This logic would be more complex if we didn't create a new descriptor every time
        // but for now, this matches the old behavior of `load_descriptor_state`.
        let mut rid_descriptor = ColumnDescriptor::default();
        rid_descriptor.field_id = rid_fid;

        // Load existing descriptor (if any) and existing metas/pages.
        let mut existing_metas: Vec<ChunkMetadata> = Vec::new();
        let mut existing_pages: Vec<u64> = Vec::new();
        let mut rid_descriptor = {
            let desc_blob_opt = context
                .pager
                .batch_get(&[BatchGet::Raw { key: _rid_descriptor_pk }])?
                .pop();
            match desc_blob_opt {
                Some(GetResult::Raw { bytes, .. }) => ColumnDescriptor::from_le_bytes(bytes.as_ref()),
                _ => ColumnDescriptor::default(),
            }
        };
        rid_descriptor.field_id = rid_fid;

        // If there is an existing descriptor chain, load all metas and collect page keys.
        if rid_descriptor.head_page_pk != 0 {
            // Walk the page chain, collecting both metas and page keys.
            let mut pk = rid_descriptor.head_page_pk;
            while pk != 0 {
                let page_blob = context
                    .pager
                    .batch_get(&[BatchGet::Raw { key: pk }])?
                    .pop()
                    .and_then(|res| match res {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?;
                let blob = page_blob.as_ref();
                let hdr_sz = DescriptorPageHeader::DISK_SIZE;
                let header = DescriptorPageHeader::from_le_bytes(&blob[..hdr_sz]);
                existing_pages.push(pk);
                // Extract packed entries on this page
                for i in 0..(header.entry_count as usize) {
                    let off = hdr_sz + i * ChunkMetadata::DISK_SIZE;
                    let end = off + ChunkMetadata::DISK_SIZE;
                    let meta = ChunkMetadata::from_le_bytes(&blob[off..end]);
                    existing_metas.push(meta);
                }
                pk = header.next_page_pk;
            }
        }

        // Build new metas for all incoming rid chunks and write their blobs/perms.
        let mut new_metas: Vec<ChunkMetadata> = Vec::new();
        for (_data_meta, rids_clean) in new_chunks {
            let rid_norm = zero_offset(rids_clean);
            let rid_pk = context.pager.alloc_many(1)?[0];
            let rid_bytes = serialize_array(rid_norm.as_ref())?;
            context.puts.push(BatchPut::Raw {
                key: rid_pk,
                bytes: rid_bytes,
            });

            let rids = rid_norm
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id downcast".into()))?;

            let mut min = u64::MAX;
            let mut max = 0u64;
            let mut sorted = true;
            let mut last = 0u64;
            for i in 0..rids.len() {
                let v = rids.value(i);
                if i == 0 {
                    last = v;
                } else if v < last {
                    sorted = false;
                } else {
                    last = v;
                }
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }

            let mut rid_perm_pk = 0u64;
            if !sorted {
                let sort_col = SortColumn {
                    values: rid_norm.clone(),
                    options: None,
                };
                let rid_idx = lexsort_to_indices(&[sort_col], None)?;
                let perm_bytes = serialize_array(&rid_idx)?;
                rid_perm_pk = context.pager.alloc_many(1)?[0];
                context.puts.push(BatchPut::Raw {
                    key: rid_perm_pk,
                    bytes: perm_bytes,
                });
            }

            let rid_meta = ChunkMetadata {
                chunk_pk: rid_pk,
                value_order_perm_pk: rid_perm_pk,
                row_count: rids.len() as u64,
                serialized_bytes: rid_norm.get_array_memory_size() as u64,
                min_val_u64: if !rids.is_empty() { min } else { 0 },
                max_val_u64: if !rids.is_empty() { max } else { 0 },
            };
            new_metas.push(rid_meta);
        }

        // Rewrite descriptor pages by flattening all metas into a single page.
        // Reuse the first existing page if present; otherwise allocate a new one.
        let page_pk = if let Some(&first) = existing_pages.first() {
            first
        } else {
            context.pager.alloc_many(1)?[0]
        };

        let mut all_metas = existing_metas;
        all_metas.extend(new_metas.into_iter());

        let header = DescriptorPageHeader {
            next_page_pk: 0,
            entry_count: all_metas.len() as u32,
            _padding: [0; 4],
        };
        let page_bytes = build_descriptor_page_bytes(&header, &all_metas);
        context.puts.push(BatchPut::Raw {
            key: page_pk,
            bytes: page_bytes,
        });

        // Free any surplus existing pages beyond the first.
        if existing_pages.len() > 1 {
            context
                .frees
                .extend(existing_pages.iter().copied().skip(1));
        }

        // Update and persist the descriptor.
        rid_descriptor.head_page_pk = page_pk;
        rid_descriptor.tail_page_pk = page_pk;
        rid_descriptor.total_chunk_count = all_metas.len() as u64;
        rid_descriptor.total_row_count = all_metas.iter().map(|m| m.row_count).sum();
        context.puts.push(BatchPut::Raw {
            key: _rid_descriptor_pk,
            bytes: rid_descriptor.to_le_bytes(),
        });

        Ok(())
    }

    fn on_rewrite(
        &mut self,
        _context: &mut IndexOpContext<'_>,
        _field_id: LogicalFieldId,
        _rewritten_chunks: &[(ChunkMetadata, ArrayRef)],
    ) -> Result<()> {
        // TODO: Implement rewrite logic for presence index.
        // This involves fetching the corresponding row-id chunks, applying the same edits
        // (filtering, injecting), and then writing back the updated rid chunks and perms.
        Ok(())
    }

    fn on_compact(
        &mut self,
        _context: &mut IndexOpContext<'_>,
        _field_id: LogicalFieldId,
        _new_metas: &[ChunkMetadata],
    ) -> Result<()> {
        // TODO: Implement compaction logic for the presence index.
        // This would involve finding the corresponding runs of small rid chunks,
        // merging them, and rewriting the rid descriptor chain.
        Ok(())
    }

    fn backfill(
        &mut self,
        _context: &mut IndexOpContext<'_>,
        _field_id: LogicalFieldId,
    ) -> Result<()> {
        // TODO: Implement backfill.
        // This would scan the data column's row ids and build the entire presence
        // index from scratch.
        unimplemented!("backfill not yet supported for presence index")
    }
}
