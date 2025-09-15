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
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, RwLock};

pub mod catalog;
pub mod descriptor;

const DESCRIPTOR_ENTRIES_PER_PAGE: usize = 256;

/// High bit is reserved for per-field shadow row_id columns.
const ROW_ID_TAG: u64 = 1u64 << 63;

fn rowid_fid(fid: LogicalFieldId) -> LogicalFieldId {
    fid | ROW_ID_TAG
}

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

    /// Append with automatic LWW (by row_id) via in-place rewrites.
    ///
    /// Requirements:
    /// - Batch must contain a non-nullable UInt64 "row_id" column.
    /// - "row_id" in a single batch must be strictly increasing
    ///   (sorted asc) and contain no duplicates. This avoids any
    ///   per-batch hash/dedup work and keeps ingest hot.
    /// - For each data column (with "field_id" metadata), we:
    ///   1) rewrite any existing chunks to remove rows whose per-field
    ///      shadow row_ids match incoming row_ids (LWW),
    ///   2) append the new data chunk, and
    ///   3) append a shadow row_id chunk for this field.
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        let schema = batch.schema();

        // Locate row_id.
        let mut row_id_idx: Option<usize> = None;
        for (i, f) in schema.fields().iter().enumerate() {
            if f.name() == "row_id" {
                row_id_idx = Some(i);
                break;
            }
        }
        let row_id_idx = row_id_idx.ok_or_else(|| Error::Internal("row_id required".into()))?;
        let row_id_field = schema.field(row_id_idx);
        if row_id_field.data_type() != &DataType::UInt64 {
            return Err(Error::Internal("row_id must be UInt64".into()));
        }
        if row_id_field.is_nullable() {
            return Err(Error::Internal("row_id must be non-null".into()));
        }
        let row_id_arr = batch
            .column(row_id_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("row_id downcast failed".into()))?;

        // CHECK: enforce strictly increasing, unique row_id in this batch.
        // O(1) extra space, single pass, no hashing/sorting.
        for i in 1..row_id_arr.len() {
            if row_id_arr.value(i - 1) >= row_id_arr.value(i) {
                return Err(Error::InvalidArgumentError(
                    "row_id in batch must be strictly increasing \
                     and unique"
                        .into(),
                ));
            }
        }

        // Build set for matches (used only to detect overlaps with old
        // chunks). We keep it compact and reuse it across fields.
        let mut incoming_ids = FxHashSet::default();
        incoming_ids.reserve(row_id_arr.len());
        for i in 0..row_id_arr.len() {
            incoming_ids.insert(row_id_arr.value(i));
        }

        let mut catalog_dirty = false;
        let mut catalog = self.catalog.write().unwrap();

        // Phase 1: LWW in-place rewrites on any overlapping old chunks.
        let mut puts_rewrites: Vec<BatchPut> = Vec::new();

        // Keep the data columns we will append after rewrites are staged.
        let mut data_cols: Vec<(LogicalFieldId, ArrayRef)> = Vec::new();
        for (i, array) in batch.columns().iter().enumerate() {
            if i == row_id_idx {
                continue;
            }
            let field = schema.field(i);
            let field_id = field
                .metadata()
                .get("field_id")
                .ok_or_else(|| Error::Internal("Missing field_id".into()))?
                .parse::<LogicalFieldId>()
                .map_err(|_| Error::Internal("bad field_id".into()))?;

            self.lww_rewrite_for_field(&mut catalog, field_id, &incoming_ids, &mut puts_rewrites)?;

            data_cols.push((field_id, Arc::clone(array)));
        }

        if !puts_rewrites.is_empty() {
            self.pager.batch_put(&puts_rewrites)?;
        }

        // Phase 2: Append new data chunks + per-field shadow row_id chunks.
        let row_id_bytes = serialize_array(row_id_arr)?;
        let mut puts_appends: Vec<BatchPut> = Vec::new();

        for (field_id, array) in data_cols.into_iter() {
            // data chunk
            let chunk_pk = self.pager.alloc_many(1)?[0];
            let chunk_bytes = serialize_array(array.as_ref())?;
            puts_appends.push(BatchPut::Raw {
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
            self.append_chunk_metadata(descriptor_pk, field_id, chunk_meta, &mut puts_appends)?;

            // shadow row_id chunk for this field
            let rid_fid = rowid_fid(field_id);
            let rid_chunk_pk = self.pager.alloc_many(1)?[0];
            puts_appends.push(BatchPut::Raw {
                key: rid_chunk_pk,
                bytes: row_id_bytes.clone(),
            });

            let rid_meta = ChunkMetadata {
                chunk_pk: rid_chunk_pk,
                tombstone_pk: 0,
                value_order_perm_pk: 0,
                row_count: row_id_arr.len() as u64,
                serialized_bytes: row_id_bytes.len() as u64,
                min_val_u64: 0,
                max_val_u64: u64::MAX,
            };

            let rid_descriptor_pk = *catalog.map.entry(rid_fid).or_insert_with(|| {
                catalog_dirty = true;
                self.pager.alloc_many(1).unwrap()[0]
            });
            self.append_chunk_metadata(rid_descriptor_pk, rid_fid, rid_meta, &mut puts_appends)?;
        }

        if catalog_dirty {
            puts_appends.push(BatchPut::Raw {
                key: CATALOG_ROOT_PKEY,
                bytes: catalog.to_bytes(),
            });
        }

        if !puts_appends.is_empty() {
            self.pager.batch_put(&puts_appends)?;
        }
        Ok(())
    }

    /// LWW via in-place rewrite for a single field.
    fn lww_rewrite_for_field(
        &self,
        catalog: &mut ColumnCatalog,
        field_id: LogicalFieldId,
        incoming_ids: &FxHashSet<u64>,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let desc_pk_data = match catalog.map.get(&field_id) {
            Some(pk) => *pk,
            None => return Ok(()),
        };
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = match catalog.map.get(&rid_fid) {
            Some(pk) => *pk,
            None => return Ok(()),
        };

        let desc_blob_data = self.pager.get_raw(desc_pk_data)?.ok_or(Error::NotFound)?;
        let mut descriptor_data = ColumnDescriptor::from_le_bytes(desc_blob_data.as_ref());

        let desc_blob_rid = self.pager.get_raw(desc_pk_rid)?.ok_or(Error::NotFound)?;
        let mut descriptor_rid = ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref());

        let mut metas_data: Vec<ChunkMetadata> = Vec::new();
        let mut metas_rid: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor_data.head_page_pk) {
            metas_data.push(m?);
        }
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor_rid.head_page_pk) {
            metas_rid.push(m?);
        }
        let n = metas_data.len().min(metas_rid.len());
        if n == 0 {
            return Ok(());
        }

        // Detect which chunks have any of the incoming row_ids.
        let mut hit_idxs: Vec<usize> = Vec::new();
        {
            let mut gets_rid = Vec::with_capacity(n);
            for i in 0..n {
                gets_rid.push(BatchGet::Raw {
                    key: metas_rid[i].chunk_pk,
                });
            }
            let rid_results = self.pager.batch_get(&gets_rid)?;
            let mut rid_blobs: FxHashMap<PhysicalKey, P::Blob> = FxHashMap::default();
            for r in rid_results {
                if let GetResult::Raw { key, bytes } = r {
                    rid_blobs.insert(key, bytes);
                }
            }
            for i in 0..n {
                let rid_blob = rid_blobs
                    .get(&metas_rid[i].chunk_pk)
                    .ok_or(Error::NotFound)?;
                let rid_arr = deserialize_array(rid_blob.as_ref())?;
                let rid_arr = rid_arr
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("row_id chunk type mismatch".into()))?;
                let mut any = false;
                for j in 0..rid_arr.len() {
                    if incoming_ids.contains(&rid_arr.value(j)) {
                        any = true;
                        break;
                    }
                }
                if any {
                    hit_idxs.push(i);
                }
            }
        }
        if hit_idxs.is_empty() {
            return Ok(());
        }

        // Fetch data + rid blobs for the hit chunks once.
        let mut gets = Vec::with_capacity(hit_idxs.len() * 2);
        for &i in &hit_idxs {
            gets.push(BatchGet::Raw {
                key: metas_data[i].chunk_pk,
            });
            gets.push(BatchGet::Raw {
                key: metas_rid[i].chunk_pk,
            });
        }
        let results = self.pager.batch_get(&gets)?;
        let mut blob_map: FxHashMap<PhysicalKey, P::Blob> = FxHashMap::default();
        for r in results {
            if let GetResult::Raw { key, bytes } = r {
                blob_map.insert(key, bytes);
            }
        }

        // Rewrite each hit chunk in place.
        for &i in &hit_idxs {
            // Keep mask from row_id shadow.
            let rid_blob = blob_map
                .get(&metas_rid[i].chunk_pk)
                .ok_or(Error::NotFound)?;
            let rid_arr_any = deserialize_array(rid_blob.as_ref())?;
            let rid_arr = rid_arr_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id chunk type mismatch".into()))?;
            let keep = BooleanArray::from_iter(
                (0..rid_arr.len()).map(|j| Some(!incoming_ids.contains(&rid_arr.value(j)))),
            );

            // Data rewrite.
            let data_blob = blob_map
                .get(&metas_data[i].chunk_pk)
                .ok_or(Error::NotFound)?;
            let data_arr = deserialize_array(data_blob.as_ref())?;
            let data_f = compute::filter(&data_arr, &keep)?;
            let data_bytes = serialize_array(&data_f)?;
            puts.push(BatchPut::Raw {
                key: metas_data[i].chunk_pk,
                bytes: data_bytes.clone(),
            });

            // Row_id shadow rewrite.
            let rid_f = compute::filter(&rid_arr_any, &keep)?;
            let rid_bytes = serialize_array(&rid_f)?;
            puts.push(BatchPut::Raw {
                key: metas_rid[i].chunk_pk,
                bytes: rid_bytes.clone(),
            });

            // Update metas; drop tombstone; recompute perm if present.
            metas_data[i].row_count = data_f.len() as u64;
            metas_data[i].serialized_bytes = data_bytes.len() as u64;
            metas_data[i].tombstone_pk = 0;

            metas_rid[i].row_count = rid_f.len() as u64;
            metas_rid[i].serialized_bytes = rid_bytes.len() as u64;

            if metas_data[i].value_order_perm_pk != 0 {
                let sort_column = SortColumn {
                    values: data_f.clone(),
                    options: None,
                };
                let indices = lexsort_to_indices(&[sort_column], None)?;
                let perm_bytes = serialize_array(&indices)?;
                puts.push(BatchPut::Raw {
                    key: metas_data[i].value_order_perm_pk,
                    bytes: perm_bytes,
                });
            }
        }

        // Persist descriptor page updates for data and row_id shadow.
        self.rewrite_descriptor_pages(desc_pk_data, &mut descriptor_data, &mut metas_data, puts)?;
        self.rewrite_descriptor_pages(desc_pk_rid, &mut descriptor_rid, &mut metas_rid, puts)?;

        Ok(())
    }

    /// Rewrite all descriptor pages for a column and update totals.
    fn rewrite_descriptor_pages(
        &self,
        descriptor_pk: PhysicalKey,
        descriptor: &mut ColumnDescriptor,
        metas: &mut [ChunkMetadata],
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let mut current_page_pk = descriptor.head_page_pk;
        let mut page_start_idx = 0usize;

        while current_page_pk != 0 {
            let mut page_blob = self
                .pager
                .get_raw(current_page_pk)?
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

            page_blob.truncate(DescriptorPageHeader::DISK_SIZE);
            page_blob.extend_from_slice(&new_page_data);

            puts.push(BatchPut::Raw {
                key: current_page_pk,
                bytes: page_blob,
            });

            current_page_pk = header.next_page_pk;
            page_start_idx = end_idx;
        }

        // Update descriptor totals.
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

    /// Explicit delete by global row indexes -> in-place chunk rewrite.
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
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // Gather metas.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            return Ok(());
        }

        // If we have a shadow row_id column, rewrite it too.
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = catalog.map.get(&rid_fid).copied();

        let mut metas_rid: Vec<ChunkMetadata> = Vec::new();
        let mut descriptor_rid: Option<ColumnDescriptor> = None;
        if let Some(pk) = desc_pk_rid {
            let desc_blob_rid = self.pager.get_raw(pk)?.ok_or(Error::NotFound)?;
            descriptor_rid = Some(ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref()));
            for m in DescriptorIterator::new(
                self.pager.as_ref(),
                descriptor_rid.as_ref().unwrap().head_page_pk,
            ) {
                metas_rid.push(m?);
            }
        }

        // Work chunk-by-chunk and rewrite in place where needed.
        let mut puts = Vec::new();
        let mut cum_rows = 0u64;

        for (i, meta) in metas.iter_mut().enumerate() {
            let start = cum_rows;
            let end = start + meta.row_count;

            // Relative delete set for this chunk.
            let mut del_set = FxHashSet::default();
            for idx in rows_to_delete.iter() {
                let idx = idx as u64;
                if idx >= start && idx < end {
                    del_set.insert((idx - start) as u32);
                }
            }
            if del_set.is_empty() {
                cum_rows = end;
                continue;
            }

            // Keep mask.
            let keep = BooleanArray::from_iter(
                (0..meta.row_count as usize).map(|j| Some(!del_set.contains(&(j as u32)))),
            );

            // Data rewrite.
            let data_blob = self.pager.get_raw(meta.chunk_pk)?.ok_or(Error::NotFound)?;
            let data_arr = deserialize_array(data_blob.as_ref())?;
            let data_f = compute::filter(&data_arr, &keep)?;
            let data_bytes = serialize_array(&data_f)?;
            puts.push(BatchPut::Raw {
                key: meta.chunk_pk,
                bytes: data_bytes.clone(),
            });

            meta.row_count = data_f.len() as u64;
            meta.serialized_bytes = data_bytes.len() as u64;
            meta.tombstone_pk = 0;

            // Shadow row_id rewrite if present.
            if let (Some(_pk), Some(rm)) = (desc_pk_rid, metas_rid.get_mut(i)) {
                let rid_blob = self.pager.get_raw(rm.chunk_pk)?.ok_or(Error::NotFound)?;
                let rid_arr = deserialize_array(rid_blob.as_ref())?;
                let rid_f = compute::filter(&rid_arr, &keep)?;
                let rid_bytes = serialize_array(&rid_f)?;
                puts.push(BatchPut::Raw {
                    key: rm.chunk_pk,
                    bytes: rid_bytes.clone(),
                });

                rm.row_count = rid_f.len() as u64;
                rm.serialized_bytes = rid_bytes.len() as u64;
                rm.tombstone_pk = 0;
            }

            // Rewrite permutation if present.
            if meta.value_order_perm_pk != 0 {
                let sort_column = SortColumn {
                    values: data_f.clone(),
                    options: None,
                };
                let indices = lexsort_to_indices(&[sort_column], None)?;
                let perm_bytes = serialize_array(&indices)?;
                puts.push(BatchPut::Raw {
                    key: meta.value_order_perm_pk,
                    bytes: perm_bytes,
                });
            }

            cum_rows = end;
        }

        // Persist descriptor updates for data and row_id shadow.
        self.rewrite_descriptor_pages(descriptor_pk, &mut descriptor, &mut metas, &mut puts)?;
        if let (Some(pk), Some(mut d_rid)) = (desc_pk_rid, descriptor_rid) {
            self.rewrite_descriptor_pages(pk, &mut d_rid, &mut metas_rid, &mut puts)?;
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
                other_val.cmp(&self_val).reverse().reverse(); // same
                self_val.cmp(&other_val)
            }
            _ => {
                panic!(
                    "Unsupported sort type in HeapItem::cmp. \
                     This needs to be generalized."
                );
            }
        };
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
