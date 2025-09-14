// File: src/store/mod.rs
//! The main ColumnStore API.

use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchPut, Pager};
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::types::LogicalFieldId;
use arrow::array::{Array, ArrayRef};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

pub mod descriptor;

// Maximum number of ChunkMetadata entries we'll pack into a single descriptor page.
const DESCRIPTOR_ENTRIES_PER_PAGE: usize = 256;

pub struct ColumnStore<P: Pager> {
    pager: Arc<P>,
    // In a real implementation, you would have a cached ColumnCatalog here.
    // For this basic example, we'll keep it simple.
}

impl<P: Pager> ColumnStore<P> {
    /// Opens an existing store or creates a new one.
    pub fn open(pager: Arc<P>) -> Result<Self> {
        // In a real implementation, this would initialize the catalog if it doesn't exist.
        // For example, writing an empty catalog to CATALOG_ROOT_PKEY.
        Ok(Self { pager })
    }

    /// Appends a batch of data to the store.
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        // This is a simplified append. A real implementation would be more complex,
        // involving a write lock and careful catalog/descriptor management.
        let mut puts = Vec::new();
        let schema = batch.schema();

        for (i, array) in batch.columns().iter().enumerate() {
            let field = schema.field(i);
            let field_id = field
                .metadata()
                .get("field_id")
                .ok_or_else(|| Error::Internal("Missing field_id in metadata".to_string()))?
                .parse::<LogicalFieldId>()
                .unwrap();

            // 1. Serialize Arrow array and allocate a key for it.
            let chunk_pk = self.pager.alloc_many(1)?[0];
            let chunk_bytes = serialize_array(array.as_ref())?;

            puts.push(BatchPut::Raw {
                key: chunk_pk,
                bytes: chunk_bytes.clone(),
            });

            // 2. Create metadata for this new chunk.
            let chunk_meta = ChunkMetadata {
                chunk_pk,
                tombstone_pk: 0,
                value_order_perm_pk: 0,
                row_count: array.len() as u64,
                serialized_bytes: chunk_bytes.len() as u64,
                min_val_u64: 0, // TODO: Compute real min/max for pruning
                max_val_u64: u64::MAX,
            };

            // 3. Update the column descriptor chain.
            // This is the most complex part, involving a read-modify-write of the
            // tail descriptor page.
            self.append_chunk_metadata(field_id, chunk_meta, &mut puts)?;
        }

        self.pager.batch_put(&puts)?;
        Ok(())
    }

    /// Returns a stream of Arrow arrays for a given column.
    pub fn scan(&self, field_id: LogicalFieldId) -> Result<impl Iterator<Item = Result<ArrayRef>>> {
        // 1. Load the root descriptor for the column.
        // In a real system, you'd look this up in the catalog.
        // Here we'll assume a placeholder key.
        let descriptor_pk = field_id; // Placeholder!
        let desc_blob = self.pager.get_raw(descriptor_pk)?.ok_or(Error::NotFound)?;
        let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // 2. Create an iterator that walks the descriptor pages.
        let meta_iter = DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk);

        // 3. Create a final iterator that fetches and deserializes the Arrow arrays.
        let pager = self.pager.clone();
        let array_iter = meta_iter.map(move |meta_result| {
            let metadata = meta_result?;
            let array_blob = pager.get_raw(metadata.chunk_pk)?.ok_or(Error::NotFound)?;
            deserialize_array(array_blob.as_ref())
        });

        Ok(array_iter)
    }

    // A helper to encapsulate the logic of appending to the descriptor chain.
    fn append_chunk_metadata(
        &self,
        field_id: LogicalFieldId,
        meta: ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let descriptor_pk = field_id; // Placeholder for catalog lookup

        let mut descriptor = match self.pager.get_raw(descriptor_pk)? {
            Some(blob) => ColumnDescriptor::from_le_bytes(blob.as_ref()),
            None => {
                // First time we're seeing this column. Create a new descriptor and first page.
                let first_page_pk = self.pager.alloc_many(1)?[0];
                let new_desc = ColumnDescriptor {
                    field_id,
                    head_page_pk: first_page_pk,
                    tail_page_pk: first_page_pk,
                    ..Default::default()
                };

                let header = DescriptorPageHeader {
                    next_page_pk: 0,
                    entry_count: 0,
                    _padding: [0; 4],
                };
                let page_bytes = header.to_le_bytes().to_vec();
                puts.push(BatchPut::Raw {
                    key: first_page_pk,
                    bytes: page_bytes,
                });
                new_desc
            }
        };

        let tail_page_pk = descriptor.tail_page_pk;
        let mut tail_page_bytes = self
            .pager
            .get_raw(tail_page_pk)?
            .ok_or(Error::NotFound)?
            .as_ref()
            .to_vec();

        let header = DescriptorPageHeader::from_le_bytes(
            &tail_page_bytes[..DescriptorPageHeader::DISK_SIZE],
        );

        if (header.entry_count as usize) < DESCRIPTOR_ENTRIES_PER_PAGE {
            // There's space. Append to the existing tail page.
            let mut new_header = header;
            new_header.entry_count += 1;
            tail_page_bytes[..DescriptorPageHeader::DISK_SIZE]
                .copy_from_slice(&new_header.to_le_bytes());
            tail_page_bytes.extend_from_slice(&meta.to_le_bytes());
            puts.push(BatchPut::Raw {
                key: tail_page_pk,
                bytes: tail_page_bytes,
            });
        } else {
            // Page is full. Allocate a new page and link it.
            let new_tail_pk = self.pager.alloc_many(1)?[0];
            let mut old_tail_header = header;
            old_tail_header.next_page_pk = new_tail_pk;
            tail_page_bytes[..DescriptorPageHeader::DISK_SIZE]
                .copy_from_slice(&old_tail_header.to_le_bytes());
            puts.push(BatchPut::Raw {
                key: tail_page_pk,
                bytes: tail_page_bytes,
            });
            let new_header = DescriptorPageHeader {
                next_page_pk: 0,
                entry_count: 1,
                _padding: [0; 4],
            };
            let mut new_page_bytes = new_header.to_le_bytes().to_vec();
            new_page_bytes.extend_from_slice(&meta.to_le_bytes());

            puts.push(BatchPut::Raw {
                key: new_tail_pk,
                bytes: new_page_bytes,
            });
            descriptor.tail_page_pk = new_tail_pk;
        }

        descriptor.total_row_count += meta.row_count;
        descriptor.total_chunk_count += 1;
        puts.push(BatchPut::Raw {
            key: descriptor_pk,
            bytes: descriptor.to_le_bytes(),
        });
        Ok(())
    }
}
