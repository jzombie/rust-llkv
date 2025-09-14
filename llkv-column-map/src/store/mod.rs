// File: src/store/mod.rs
//! The main ColumnStore API.

use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchPut, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::types::{CATALOG_ROOT_PKEY, LogicalFieldId, PhysicalKey};
use arrow::array::{Array, ArrayRef};
use arrow::record_batch::RecordBatch;
use std::sync::{Arc, RwLock};

pub mod catalog;
pub mod descriptor;

// Maximum number of ChunkMetadata entries we'll pack into a single descriptor page.
const DESCRIPTOR_ENTRIES_PER_PAGE: usize = 256;

pub struct ColumnStore<P: Pager> {
    pager: Arc<P>,
    catalog: RwLock<ColumnCatalog>,
}

impl<P: Pager> ColumnStore<P> {
    /// Opens an existing store or creates a new one.
    pub fn open(pager: Arc<P>) -> Result<Self> {
        let catalog = match pager.get_raw(CATALOG_ROOT_PKEY)? {
            Some(blob) => ColumnCatalog::from_bytes(blob.as_ref())?,
            None => ColumnCatalog::default(), // New store, empty catalog
        };
        Ok(Self {
            pager,
            catalog: RwLock::new(catalog),
        })
    }

    /// Appends a batch of data to the store.
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        let mut puts = Vec::new();
        let schema = batch.schema();

        // This flag will track if we need to save the catalog at the end.
        let mut catalog_dirty = false;
        // Take a write lock on the catalog for the duration of the append.
        let mut catalog = self.catalog.write().unwrap();

        for (i, array) in batch.columns().iter().enumerate() {
            let field = schema.field(i);
            let field_id = field
                .metadata()
                .get("field_id")
                .ok_or_else(|| Error::Internal("Missing field_id in metadata".to_string()))?
                .parse::<LogicalFieldId>()
                .unwrap();

            let chunk_pk = self.pager.alloc_many(1)?[0];
            let chunk_bytes = serialize_array(array.as_ref())?;

            puts.push(BatchPut::Raw {
                key: chunk_pk,
                bytes: chunk_bytes.clone(),
            });

            let chunk_meta = ChunkMetadata {
                chunk_pk,
                tombstone_pk: 0,
                value_order_perm_pk: 0,
                row_count: array.len() as u64,
                serialized_bytes: chunk_bytes.len() as u64,
                min_val_u64: 0,
                max_val_u64: u64::MAX,
            };

            // Look up the descriptor key in the catalog.
            let descriptor_pk = match catalog.map.get(&field_id) {
                Some(&pk) => pk,
                None => {
                    // This is a new column. Allocate a key for its descriptor.
                    let new_descriptor_pk = self.pager.alloc_many(1)?[0];
                    catalog.map.insert(field_id, new_descriptor_pk);
                    catalog_dirty = true;
                    new_descriptor_pk
                }
            };

            self.append_chunk_metadata(descriptor_pk, field_id, chunk_meta, &mut puts)?;
        }

        // If we added new columns, save the updated catalog.
        if catalog_dirty {
            puts.push(BatchPut::Raw {
                key: CATALOG_ROOT_PKEY,
                bytes: catalog.to_bytes(),
            });
        }

        self.pager.batch_put(&puts)?;
        Ok(())
    }

    /// Returns a stream of Arrow arrays for a given column.
    pub fn scan(&self, field_id: LogicalFieldId) -> Result<impl Iterator<Item = Result<ArrayRef>>> {
        // Take a read lock on the catalog to perform the lookup.
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = match catalog.map.get(&field_id) {
            Some(&pk) => pk,
            // If the column doesn't exist in the catalog, return an empty iterator.
            None => {
                return Ok(
                    Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Result<ArrayRef>>>
                );
            }
        };

        let desc_blob = self.pager.get_raw(descriptor_pk)?.ok_or(Error::NotFound)?;
        let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let meta_iter = DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk);

        let pager = self.pager.clone();
        let array_iter = meta_iter.map(move |meta_result| {
            let metadata = meta_result?;
            let array_blob = pager.get_raw(metadata.chunk_pk)?.ok_or(Error::NotFound)?;
            deserialize_array(array_blob.as_ref())
        });

        // Box the iterator to erase its concrete type for the return signature.
        Ok(Box::new(array_iter) as Box<dyn Iterator<Item = Result<ArrayRef>>>)
    }

    // A helper to encapsulate the logic of appending to the descriptor chain.
    fn append_chunk_metadata(
        &self,
        descriptor_pk: PhysicalKey,
        field_id: LogicalFieldId,
        meta: ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        match self.pager.get_raw(descriptor_pk)? {
            Some(blob) => {
                // Logic for an EXISTING column
                let mut descriptor = ColumnDescriptor::from_le_bytes(blob.as_ref());
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
            }
            None => {
                // Logic for a NEW column
                let first_page_pk = self.pager.alloc_many(1)?[0];
                let descriptor = ColumnDescriptor {
                    field_id,
                    head_page_pk: first_page_pk,
                    tail_page_pk: first_page_pk,
                    total_row_count: meta.row_count,
                    total_chunk_count: 1,
                };
                let header = DescriptorPageHeader {
                    next_page_pk: 0,
                    entry_count: 1,
                    _padding: [0; 4],
                };
                let mut page_bytes = header.to_le_bytes().to_vec();
                page_bytes.extend_from_slice(&meta.to_le_bytes());
                puts.push(BatchPut::Raw {
                    key: first_page_pk,
                    bytes: page_bytes,
                });
                puts.push(BatchPut::Raw {
                    key: descriptor_pk,
                    bytes: descriptor.to_le_bytes(),
                });
            }
        }
        Ok(())
    }
}
