// File: src/store/mod.rs
//! The main ColumnStore API.

use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::types::{CATALOG_ROOT_PKEY, LogicalFieldId, PhysicalKey};
use arrow::array::{Array, ArrayRef, BooleanArray, Int32Array, UInt32Array, UInt64Array};
use arrow::compute::{self, SortColumn, lexsort_to_indices};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use roaring::RoaringBitmap;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::BitOrAssign;
use std::sync::{Arc, RwLock};

pub mod catalog;
pub mod descriptor;

const DESCRIPTOR_ENTRIES_PER_PAGE: usize = 256;

pub struct ColumnStore<P: Pager> {
    pager: Arc<P>,
    catalog: RwLock<ColumnCatalog>,
}

impl<P: Pager> ColumnStore<P> {
    pub fn open(pager: Arc<P>) -> Result<Self> {
        let catalog = match pager.get_raw(CATALOG_ROOT_PKEY)? {
            Some(blob) => ColumnCatalog::from_bytes(blob.as_ref())?,
            None => ColumnCatalog::default(),
        };
        Ok(Self {
            pager,
            catalog: RwLock::new(catalog),
        })
    }

    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        let mut puts = Vec::new();
        let schema = batch.schema();
        let mut catalog_dirty = false;
        let mut catalog = self.catalog.write().unwrap();

        for (i, array) in batch.columns().iter().enumerate() {
            let field = schema.field(i);
            let field_id = field
                .metadata()
                .get("field_id")
                .ok_or_else(|| Error::Internal("Missing field_id".into()))?
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

            let descriptor_pk = *catalog.map.entry(field_id).or_insert_with(|| {
                catalog_dirty = true;
                self.pager.alloc_many(1).unwrap()[0]
            });
            self.append_chunk_metadata(descriptor_pk, field_id, chunk_meta, &mut puts)?;
        }

        if catalog_dirty {
            puts.push(BatchPut::Raw {
                key: CATALOG_ROOT_PKEY,
                bytes: catalog.to_bytes(),
            });
        }
        self.pager.batch_put(&puts)?;
        Ok(())
    }

    pub fn delete_rows(
        &self,
        field_id: LogicalFieldId,
        rows_to_delete: &RoaringBitmap,
    ) -> Result<()> {
        if rows_to_delete.is_empty() {
            return Ok(());
        }

        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self.pager.get_raw(descriptor_pk)?.ok_or(Error::NotFound)?;
        let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let mut all_chunk_metadata = Vec::new();
        let meta_iter = DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk);
        for meta_result in meta_iter {
            all_chunk_metadata.push(meta_result?);
        }

        let mut puts = Vec::new();
        let mut cumulative_rows = 0;

        for meta in all_chunk_metadata.iter_mut() {
            let chunk_row_start = cumulative_rows;
            let chunk_row_end = chunk_row_start + meta.row_count;

            let deletes_in_chunk: RoaringBitmap = rows_to_delete
                .iter()
                .filter(|&row_idx| {
                    (row_idx as u64) >= chunk_row_start && (row_idx as u64) < chunk_row_end
                })
                .map(|row_idx| (row_idx as u64 - chunk_row_start) as u32)
                .collect();

            if !deletes_in_chunk.is_empty() {
                let mut existing_tombstone = if meta.tombstone_pk != 0 {
                    let tombstone_blob = self
                        .pager
                        .get_raw(meta.tombstone_pk)?
                        .ok_or(Error::NotFound)?;
                    RoaringBitmap::deserialize_from(tombstone_blob.as_ref())
                        .map_err(|e| Error::Io(e))?
                } else {
                    RoaringBitmap::new()
                };

                existing_tombstone.bitor_assign(&deletes_in_chunk);

                let mut tombstone_bytes = Vec::new();
                existing_tombstone.serialize_into(&mut tombstone_bytes)?;
                let new_tombstone_pk = self.pager.alloc_many(1)?[0];
                puts.push(BatchPut::Raw {
                    key: new_tombstone_pk,
                    bytes: tombstone_bytes,
                });

                meta.tombstone_pk = new_tombstone_pk;
            }
            cumulative_rows = chunk_row_end;
        }

        let mut current_page_pk = descriptor.head_page_pk;
        let mut page_start_chunk_idx = 0;
        while current_page_pk != 0 {
            let mut page_blob = self
                .pager
                .get_raw(current_page_pk)?
                .ok_or(Error::NotFound)?
                .as_ref()
                .to_vec();
            let header =
                DescriptorPageHeader::from_le_bytes(&page_blob[..DescriptorPageHeader::DISK_SIZE]);

            let chunks_on_this_page = header.entry_count as usize;
            let page_end_chunk_idx = page_start_chunk_idx + chunks_on_this_page;

            let updated_chunks = &all_chunk_metadata[page_start_chunk_idx..page_end_chunk_idx];
            let mut new_page_data = Vec::new();
            for chunk_meta in updated_chunks {
                new_page_data.extend_from_slice(&chunk_meta.to_le_bytes());
            }

            page_blob.truncate(DescriptorPageHeader::DISK_SIZE);
            page_blob.extend_from_slice(&new_page_data);

            puts.push(BatchPut::Raw {
                key: current_page_pk,
                bytes: page_blob,
            });
            current_page_pk = header.next_page_pk;
            page_start_chunk_idx = page_end_chunk_idx;
        }

        self.pager.batch_put(&puts)?;
        Ok(())
    }

    pub fn scan(
        &self,
        field_id: LogicalFieldId,
    ) -> Result<Box<dyn Iterator<Item = Result<ArrayRef>>>> {
        let catalog = self.catalog.read().unwrap();
        let &descriptor_pk = match catalog.map.get(&field_id) {
            Some(pk) => pk,
            None => return Ok(Box::new(std::iter::empty())),
        };

        let desc_blob = self.pager.get_raw(descriptor_pk)?.ok_or(Error::NotFound)?;
        let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let mut all_chunk_metadata = Vec::new();
        let meta_iter = DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk);
        for meta_result in meta_iter {
            all_chunk_metadata.push(meta_result?);
        }

        if all_chunk_metadata.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        let mut gets = Vec::new();
        for meta in &all_chunk_metadata {
            gets.push(BatchGet::Raw { key: meta.chunk_pk });
            if meta.tombstone_pk != 0 {
                gets.push(BatchGet::Raw {
                    key: meta.tombstone_pk,
                });
            }
        }
        let get_results = self.pager.batch_get(&gets)?;

        let mut blobs_map = FxHashMap::default();
        for result in get_results {
            if let GetResult::Raw { key, bytes } = result {
                blobs_map.insert(key, bytes);
            }
        }

        let result_iterator = all_chunk_metadata.into_iter().map(move |meta| {
            let array_blob = blobs_map.get(&meta.chunk_pk).ok_or(Error::NotFound)?;
            let array = deserialize_array(array_blob.as_ref())?;

            if meta.tombstone_pk != 0 {
                let tombstone_blob = blobs_map.get(&meta.tombstone_pk).ok_or(Error::NotFound)?;
                let bitmap = RoaringBitmap::deserialize_from(tombstone_blob.as_ref())
                    .map_err(|e| Error::Io(e))?;
                let filter_mask = BooleanArray::from_iter(
                    (0..array.len()).map(|i| Some(!bitmap.contains(i as u32))),
                );
                Ok(compute::filter(&array, &filter_mask)?)
            } else {
                Ok(array)
            }
        });

        Ok(Box::new(result_iterator))
    }

    pub fn create_sort_index(&self, field_id: LogicalFieldId) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self.pager.get_raw(descriptor_pk)?.ok_or(Error::NotFound)?;
        let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let mut all_chunk_metadata = Vec::new();
        let meta_iter = DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk);
        for meta_result in meta_iter {
            all_chunk_metadata.push(meta_result?);
        }

        let data_chunk_keys: Vec<PhysicalKey> =
            all_chunk_metadata.iter().map(|m| m.chunk_pk).collect();
        let gets: Vec<BatchGet> = data_chunk_keys
            .into_iter()
            .map(|k| BatchGet::Raw { key: k })
            .collect();
        let results = self.pager.batch_get(&gets)?;

        let mut chunks_map = FxHashMap::default();
        for result in results {
            if let GetResult::Raw { key, bytes } = result {
                chunks_map.insert(key, bytes);
            }
        }

        let mut puts = Vec::new();
        for meta in all_chunk_metadata.iter_mut() {
            let chunk_blob = chunks_map.get(&meta.chunk_pk).ok_or(Error::NotFound)?;
            let chunk_array = deserialize_array(chunk_blob.as_ref())?;

            let sort_column = SortColumn {
                values: chunk_array,
                options: None,
            };
            let indices = lexsort_to_indices(&[sort_column], None)?;
            let perm_bytes = serialize_array(&indices)?;
            let perm_pk = self.pager.alloc_many(1)?[0];

            puts.push(BatchPut::Raw {
                key: perm_pk,
                bytes: perm_bytes,
            });
            meta.value_order_perm_pk = perm_pk;
        }

        let mut current_page_pk = descriptor.head_page_pk;
        let mut page_start_chunk_idx = 0;
        while current_page_pk != 0 {
            let mut page_blob = self
                .pager
                .get_raw(current_page_pk)?
                .ok_or(Error::NotFound)?
                .as_ref()
                .to_vec();
            let header =
                DescriptorPageHeader::from_le_bytes(&page_blob[..DescriptorPageHeader::DISK_SIZE]);
            let chunks_on_this_page = header.entry_count as usize;
            let page_end_chunk_idx = page_start_chunk_idx + chunks_on_this_page;
            let updated_chunks = &all_chunk_metadata[page_start_chunk_idx..page_end_chunk_idx];

            let mut new_page_data = Vec::new();
            for chunk_meta in updated_chunks {
                new_page_data.extend_from_slice(&chunk_meta.to_le_bytes());
            }

            page_blob.truncate(DescriptorPageHeader::DISK_SIZE);
            page_blob.extend_from_slice(&new_page_data);

            puts.push(BatchPut::Raw {
                key: current_page_pk,
                bytes: page_blob,
            });
            current_page_pk = header.next_page_pk;
            page_start_chunk_idx = page_end_chunk_idx;
        }

        self.pager.batch_put(&puts)?;
        Ok(())
    }

    pub fn scan_sorted(&self, field_id: LogicalFieldId) -> Result<MergeSortedIterator<P::Blob>> {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self.pager.get_raw(descriptor_pk)?.ok_or(Error::NotFound)?;
        let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let mut all_chunk_metadata = Vec::new();
        let meta_iter = DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk);
        for meta_result in meta_iter {
            all_chunk_metadata.push(meta_result?);
        }

        let mut gets = Vec::new();
        for meta in &all_chunk_metadata {
            gets.push(BatchGet::Raw { key: meta.chunk_pk });
            if meta.value_order_perm_pk != 0 {
                gets.push(BatchGet::Raw {
                    key: meta.value_order_perm_pk,
                });
            } else {
                return Err(Error::Internal(format!(
                    "Chunk {} is not sorted",
                    meta.chunk_pk
                )));
            }
        }
        let get_results = self.pager.batch_get(&gets)?;

        let mut blobs_map = FxHashMap::default();
        for result in get_results {
            if let GetResult::Raw { key, bytes } = result {
                blobs_map.insert(key, bytes);
            }
        }

        MergeSortedIterator::try_new(all_chunk_metadata, blobs_map)
    }

    fn append_chunk_metadata(
        &self,
        descriptor_pk: PhysicalKey,
        field_id: LogicalFieldId,
        meta: ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        match self.pager.get_raw(descriptor_pk)? {
            Some(blob) => {
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

// --- K-way merge implementation ---

struct ChunkCursor {
    perm_indices: Arc<UInt32Array>,
    data_array: ArrayRef,
    pos: usize,
}

struct HeapItem {
    cursor_idx: usize,
    value: ArrayRef,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // This implementation is now generalized to handle multiple primitive types.
        let ordering = match self.value.data_type() {
            &DataType::UInt64 => {
                let self_val = self
                    .value
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0);
                let other_val = other
                    .value
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0);
                self_val.cmp(&other_val)
            }
            &DataType::Int32 => {
                let self_val = self
                    .value
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .value(0);
                let other_val = other
                    .value
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .value(0);
                self_val.cmp(&other_val)
            }
            _ => {
                panic!("Unsupported sort type in HeapItem::cmp. This needs to be generalized.");
            }
        };
        // Reversed for min-heap behavior.
        ordering.reverse()
    }
}

pub struct MergeSortedIterator<B: AsRef<[u8]>> {
    cursors: Vec<ChunkCursor>,
    heap: BinaryHeap<HeapItem>,
    _blobs: FxHashMap<PhysicalKey, B>,
}

impl<B: AsRef<[u8]>> MergeSortedIterator<B> {
    fn try_new(metadata: Vec<ChunkMetadata>, blobs: FxHashMap<PhysicalKey, B>) -> Result<Self> {
        let mut cursors = Vec::with_capacity(metadata.len());
        for meta in &metadata {
            let data_blob = blobs.get(&meta.chunk_pk).ok_or(Error::NotFound)?;
            let perm_blob = blobs
                .get(&meta.value_order_perm_pk)
                .ok_or(Error::NotFound)?;

            let data_array = deserialize_array(data_blob.as_ref())?;
            let perm_array = deserialize_array(perm_blob.as_ref())?;
            let perm_indices = perm_array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Error::Internal("Invalid permutation index type".into()))?;

            cursors.push(ChunkCursor {
                perm_indices: Arc::new(perm_indices.clone()),
                data_array,
                pos: 0,
            });
        }

        let mut heap = BinaryHeap::new();
        for (i, cursor) in cursors.iter_mut().enumerate() {
            if let Some(value) = Self::get_value_at_cursor(cursor) {
                heap.push(HeapItem {
                    cursor_idx: i,
                    value,
                });
            }
        }

        Ok(Self {
            cursors,
            heap,
            _blobs: blobs,
        })
    }

    fn get_value_at_cursor(cursor: &ChunkCursor) -> Option<ArrayRef> {
        if cursor.pos < cursor.perm_indices.len() {
            let data_idx = cursor.perm_indices.value(cursor.pos) as usize;
            Some(cursor.data_array.slice(data_idx, 1))
        } else {
            None
        }
    }
}

impl<B: AsRef<[u8]>> Iterator for MergeSortedIterator<B> {
    type Item = Result<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.heap.pop() {
            Some(item) => {
                let cursor = &mut self.cursors[item.cursor_idx];
                cursor.pos += 1;

                if let Some(next_value) = Self::get_value_at_cursor(cursor) {
                    self.heap.push(HeapItem {
                        cursor_idx: item.cursor_idx,
                        value: next_value,
                    });
                }
                Some(Ok(item.value))
            }
            None => None,
        }
    }
}
