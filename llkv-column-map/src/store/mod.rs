//! The main ColumnStore API.
//!
//! Automatic coalescing (no tombstones):
//! - On append, for fixed-width types, we merge a prefix of the incoming
//!   data into the current tail chunk up to ~TARGET_BYTES and rewrite the
//!   tail in place (data and shadow row_id). The remainder is appended as
//!   one new chunk (no tiny-slice splitting). No tombstones.
//! - Deletes already do full in-place overwrites with a keep mask.
//!
//! Sort index handling:
//! - If a chunk has a stored permutation, we rebuild it whenever that
//!   chunk is rewritten (e.g., tail merge or delete rewrite).

use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::types::{CATALOG_ROOT_PKEY, LogicalFieldId, PhysicalKey};

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int32Array, UInt32Array, UInt64Array, make_array,
};
use arrow::compute::{self, SortColumn, lexsort_to_indices};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use roaring::RoaringBitmap;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
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

/// Desired serialized size of a data chunk (~65 KiB).
const TARGET_BYTES: usize = 65 * 1024;

/// Below this threshold, try merging tail with incoming data.
const MIN_BYTES: usize = TARGET_BYTES / 2;

/// Bytes-per-value for fixed-width primitives. Boolean as 1 byte here.
#[inline]
fn fixed_width_bpv(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Int8 | DataType::UInt8 => Some(1),
        DataType::Int16 | DataType::UInt16 => Some(2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => Some(4),
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => Some(8),
        DataType::Boolean => Some(1),
        _ => None,
    }
}

/// Normalize an array to offset==0 without copying payload buffers.
#[inline]
fn zero_offset(arr: &ArrayRef) -> ArrayRef {
    if arr.offset() == 0 {
        return arr.clone();
    }
    let data = arr.to_data();
    let sliced = data.slice(arr.offset(), arr.len());
    make_array(sliced)
}

/// Concatenate two arrays (Arrow does the right thing).
#[inline]
fn concat_arrays(a: &ArrayRef, b: &ArrayRef) -> Result<ArrayRef> {
    Ok(arrow::compute::concat(&[a.as_ref(), b.as_ref()])?)
}

pub struct ColumnStore<P: Pager> {
    pager: Arc<P>,
    catalog: RwLock<ColumnCatalog>,
}

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle>,
{
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

    /// Append with automatic LWW (by row_id) and tail coalescing.
    ///
    /// Requirements:
    /// - Batch contains non-nullable UInt64 column "row_id".
    /// - For each data column (with "field_id" metadata):
    ///   1) In-place rewrite to drop rows whose per-field shadow row_ids
    ///      match incoming IDs (LWW, no tombstones).
    ///   2) Coalesce into the tail by bytes (fixed-width types only).
    ///   3) Append remainder data and aligned shadow row_ids as one chunk.
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        let schema = batch.schema();

        // Locate "row_id".
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

        // Incoming IDs for LWW.
        let mut incoming_ids = FxHashSet::default();
        incoming_ids.reserve(row_id_arr.len());
        for i in 0..row_id_arr.len() {
            incoming_ids.insert(row_id_arr.value(i));
        }

        let mut catalog_dirty = false;
        let mut catalog = self.catalog.write().unwrap();

        // Phase 1: LWW rewrites.
        let mut puts_rewrites: Vec<BatchPut> = Vec::new();
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

        // Phase 2: coalesce tail and append remainder (single chunk).
        let mut puts_appends: Vec<BatchPut> = Vec::new();

        for (field_id, array) in data_cols.into_iter() {
            let (data_remaining, rid_remaining) = self.coalesce_with_tail_bytes(
                &mut catalog,
                field_id,
                &array,
                row_id_arr,
                &mut puts_appends,
            )?;

            if data_remaining.len() == 0 {
                continue;
            }

            // Append remainder as one chunk, with aligned row_ids.
            let rows = data_remaining.len();
            let rid_slice_any = rid_remaining.slice(0, rows);
            let rid_slice = rid_slice_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id type mismatch".into()))?;

            // Data chunk
            let chunk_pk = self.pager.alloc_many(1)?[0];
            let data_norm = zero_offset(&data_remaining);
            let chunk_bytes = serialize_array(data_norm.as_ref())?;
            puts_appends.push(BatchPut::Raw {
                key: chunk_pk,
                bytes: chunk_bytes.clone(),
            });

            let chunk_meta = ChunkMetadata {
                chunk_pk,
                tombstone_pk: 0,
                value_order_perm_pk: 0,
                row_count: rows as u64,
                serialized_bytes: chunk_bytes.len() as u64,
                min_val_u64: 0,
                max_val_u64: u64::MAX,
            };

            let descriptor_pk = *catalog.map.entry(field_id).or_insert_with(|| {
                catalog_dirty = true;
                self.pager.alloc_many(1).unwrap()[0]
            });
            self.append_chunk_metadata(descriptor_pk, field_id, chunk_meta, &mut puts_appends)?;

            // Shadow row_id chunk aligned to data chunk.
            let rid_fid = rowid_fid(field_id);
            let rid_chunk_pk = self.pager.alloc_many(1)?[0];
            let rid_ref: ArrayRef = Arc::new(rid_slice.clone());
            let rid_norm = zero_offset(&rid_ref);
            let rid_bytes = serialize_array(rid_norm.as_ref())?;
            puts_appends.push(BatchPut::Raw {
                key: rid_chunk_pk,
                bytes: rid_bytes.clone(),
            });

            let rid_meta = ChunkMetadata {
                chunk_pk: rid_chunk_pk,
                tombstone_pk: 0,
                value_order_perm_pk: 0,
                row_count: rows as u64,
                serialized_bytes: rid_bytes.len() as u64,
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

    /// LWW via in-place rewrite for a single field (no tombstones).
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

        // Identify chunks with any matching row_id.
        let mut hit_idxs: Vec<usize> = Vec::new();
        {
            let mut gets_rid = Vec::with_capacity(n);
            for i in 0..n {
                gets_rid.push(BatchGet::Raw {
                    key: metas_rid[i].chunk_pk,
                });
            }
            let rid_results = self.pager.batch_get(&gets_rid)?;
            let mut rid_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            for r in rid_results {
                match r {
                    GetResult::Raw { key, bytes } => {
                        rid_blobs.insert(key, bytes);
                    }
                    GetResult::Missing { .. } => return Err(Error::NotFound),
                }
            }
            for i in 0..n {
                let rid_blob = rid_blobs
                    .get(&metas_rid[i].chunk_pk)
                    .ok_or(Error::NotFound)?;
                let rid_arr = deserialize_array(rid_blob.clone())?;
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

        // Fetch data+rid for each hit chunk and rewrite with keep mask.
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
        let mut blob_map: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results {
            match r {
                GetResult::Raw { key, bytes } => {
                    blob_map.insert(key, bytes);
                }
                GetResult::Missing { .. } => return Err(Error::NotFound),
            }
        }

        for &i in &hit_idxs {
            let rid_blob = blob_map
                .get(&metas_rid[i].chunk_pk)
                .ok_or(Error::NotFound)?;
            let rid_arr_any = deserialize_array(rid_blob.clone())?;
            let rid_arr = rid_arr_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id chunk type mismatch".into()))?;
            let keep = BooleanArray::from_iter(
                (0..rid_arr.len()).map(|j| Some(!incoming_ids.contains(&rid_arr.value(j)))),
            );

            let data_blob = blob_map
                .get(&metas_data[i].chunk_pk)
                .ok_or(Error::NotFound)?;
            let data_arr = deserialize_array(data_blob.clone())?;
            let data_f = compute::filter(&data_arr, &keep)?;

            // Rewrite data chunk.
            let data_bytes = serialize_array(&data_f)?;
            puts.push(BatchPut::Raw {
                key: metas_data[i].chunk_pk,
                bytes: data_bytes.clone(),
            });
            metas_data[i].row_count = data_f.len() as u64;
            metas_data[i].serialized_bytes = data_bytes.len() as u64;
            metas_data[i].tombstone_pk = 0;

            // Rewrite row_id chunk.
            let rid_f = compute::filter(&rid_arr_any, &keep)?;
            let rid_bytes = serialize_array(&rid_f)?;
            puts.push(BatchPut::Raw {
                key: metas_rid[i].chunk_pk,
                bytes: rid_bytes.clone(),
            });
            metas_rid[i].row_count = rid_f.len() as u64;
            metas_rid[i].serialized_bytes = rid_bytes.len() as u64;

            // Refresh sort permutation if present.
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

        // Data metas
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            return Ok(());
        }

        // Optional shadow row_id metas
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

        let mut puts = Vec::new();

        // Single-pass walk over delete bitmap.
        let mut del_iter = rows_to_delete.iter();
        let mut cur_del = del_iter.next(); // Option<u32>

        let mut cum_rows = 0u64;

        for (i, meta) in metas.iter_mut().enumerate() {
            let start_u64 = cum_rows;
            let end_u64 = start_u64 + meta.row_count;

            // Fast-forward to first delete >= start.
            let start = start_u64 as u32;
            let end = end_u64 as u32;
            while let Some(d) = cur_del {
                if d < start {
                    cur_del = del_iter.next();
                } else {
                    break;
                }
            }

            // Collect deletes local to this chunk.
            let mut del_local: FxHashSet<u32> = FxHashSet::default();
            while let Some(d) = cur_del {
                if d >= end {
                    break;
                }
                del_local.insert(d - start);
                cur_del = del_iter.next();
            }

            if del_local.is_empty() {
                cum_rows = end_u64;
                continue;
            }

            // Build keep mask and rewrite data chunk.
            let keep = BooleanArray::from_iter(
                (0..meta.row_count as usize).map(|j| Some(!del_local.contains(&(j as u32)))),
            );

            let data_blob = self.pager.get_raw(meta.chunk_pk)?.ok_or(Error::NotFound)?;
            let data_arr = deserialize_array(data_blob.clone())?;
            let data_f = compute::filter(&data_arr, &keep)?;
            let data_bytes = serialize_array(&data_f)?;
            puts.push(BatchPut::Raw {
                key: meta.chunk_pk,
                bytes: data_bytes.clone(),
            });

            meta.row_count = data_f.len() as u64;
            meta.serialized_bytes = data_bytes.len() as u64;
            meta.tombstone_pk = 0;

            // Rewrite shadow row_id if present.
            if let (Some(_pk), Some(rm)) = (desc_pk_rid, metas_rid.get_mut(i)) {
                let rid_blob = self.pager.get_raw(rm.chunk_pk)?.ok_or(Error::NotFound)?;
                let rid_arr = deserialize_array(rid_blob.clone())?;
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

            // Refresh sort permutation if present.
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

            cum_rows = end_u64;
        }

        self.rewrite_descriptor_pages(descriptor_pk, &mut descriptor, &mut metas, &mut puts)?;
        if let (Some(pk), Some(mut d_rid)) = (desc_pk_rid, descriptor_rid) {
            self.rewrite_descriptor_pages(pk, &mut d_rid, &mut metas_rid, &mut puts)?;
        }

        if !puts.is_empty() {
            self.pager.batch_put(&puts)?;
        }
        Ok(())
    }

    /// Scan returns arrays in append order (no tombstones).
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

        let mut metas = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        // Fast path: single chunk
        if metas.len() == 1 {
            let blob = self
                .pager
                .get_raw(metas[0].chunk_pk)?
                .ok_or(Error::NotFound)?;
            let arr = deserialize_array(blob)?;
            return Ok(Box::new(std::iter::once(Ok(arr))));
        }

        // Multi-chunk: batch_get and map
        let mut gets = Vec::with_capacity(metas.len());
        for m in &metas {
            gets.push(BatchGet::Raw { key: m.chunk_pk });
        }
        let results = self.pager.batch_get(&gets)?;

        let mut blobs_map: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in results {
            match r {
                GetResult::Raw { key, bytes } => {
                    blobs_map.insert(key, bytes);
                }
                GetResult::Missing { .. } => return Err(Error::NotFound),
            }
        }

        let it = metas.into_iter().map(move |m| {
            let blob = blobs_map.get(&m.chunk_pk).ok_or(Error::NotFound)?;
            let array = deserialize_array(blob.clone())?;
            Ok(array)
        });

        Ok(Box::new(it))
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

        // Fetch all data chunks.
        let data_chunk_keys: Vec<PhysicalKey> =
            all_chunk_metadata.iter().map(|m| m.chunk_pk).collect();
        let gets: Vec<BatchGet> = data_chunk_keys
            .into_iter()
            .map(|k| BatchGet::Raw { key: k })
            .collect();
        let results = self.pager.batch_get(&gets)?;

        let mut chunks_map: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for result in results {
            match result {
                GetResult::Raw { key, bytes } => {
                    chunks_map.insert(key, bytes);
                }
                GetResult::Missing { .. } => return Err(Error::NotFound),
            }
        }

        let mut puts = Vec::new();

        for meta in all_chunk_metadata.iter_mut() {
            let chunk_blob = chunks_map.get(&meta.chunk_pk).ok_or(Error::NotFound)?;
            let chunk_array = deserialize_array(chunk_blob.clone())?;

            // Make row_count authoritative from the real array length.
            meta.row_count = chunk_array.len() as u64;

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

        // Rewrite descriptor pages to reflect updated metas.
        let mut current_page_pk = descriptor.head_page_pk;
        let mut page_start_chunk_idx = 0usize;
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

    pub fn scan_sorted(&self, field_id: LogicalFieldId) -> Result<MergeSortedIterator> {
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

        let mut blobs_map: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for r in get_results {
            match r {
                GetResult::Raw { key, bytes } => {
                    blobs_map.insert(key, bytes);
                }
                GetResult::Missing { .. } => return Err(Error::NotFound),
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

    /// Byte-based tail coalescing at append (fixed-width precise).
    fn coalesce_with_tail_bytes(
        &self,
        catalog: &mut ColumnCatalog,
        field_id: LogicalFieldId,
        incoming_data: &ArrayRef,
        incoming_row_ids: &UInt64Array,
        puts: &mut Vec<BatchPut>,
    ) -> Result<(ArrayRef, Arc<UInt64Array>)> {
        let bpv = match fixed_width_bpv(incoming_data.data_type()) {
            Some(b) => b,
            None => {
                return Ok((incoming_data.clone(), Arc::new(incoming_row_ids.clone())));
            }
        };

        // Need existing data and its shadow row_id descriptor.
        let desc_pk = match catalog.map.get(&field_id) {
            Some(&pk) => pk,
            None => {
                return Ok((incoming_data.clone(), Arc::new(incoming_row_ids.clone())));
            }
        };
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = match catalog.map.get(&rid_fid) {
            Some(&pk) => pk,
            None => {
                return Ok((incoming_data.clone(), Arc::new(incoming_row_ids.clone())));
            }
        };

        let desc_blob = self.pager.get_raw(desc_pk)?.ok_or(Error::NotFound)?;
        let mut desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        let desc_blob_rid = self.pager.get_raw(desc_pk_rid)?.ok_or(Error::NotFound)?;
        let mut desc_rid = ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref());

        let mut metas: Vec<ChunkMetadata> =
            DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk)
                .collect::<Result<_>>()?;
        let mut metas_rid: Vec<ChunkMetadata> =
            DescriptorIterator::new(self.pager.as_ref(), desc_rid.head_page_pk)
                .collect::<Result<_>>()?;

        if metas.is_empty() || incoming_data.len() == 0 {
            return Ok((incoming_data.clone(), Arc::new(incoming_row_ids.clone())));
        }

        // Touch tail only if small by bytes.
        let tail_idx = metas.len() - 1;
        let tail_bytes = metas[tail_idx].serialized_bytes as usize;
        if tail_bytes >= MIN_BYTES {
            return Ok((incoming_data.clone(), Arc::new(incoming_row_ids.clone())));
        }

        // Rows we can add to reach TARGET_BYTES.
        let budget = TARGET_BYTES.saturating_sub(tail_bytes);
        if budget == 0 {
            return Ok((incoming_data.clone(), Arc::new(incoming_row_ids.clone())));
        }

        let take_rows = (budget / bpv).min(incoming_data.len());
        if take_rows == 0 {
            return Ok((incoming_data.clone(), Arc::new(incoming_row_ids.clone())));
        }

        // Split head vs rest for data and row_ids.
        let head_data = incoming_data.slice(0, take_rows);
        let rest_data = incoming_data.slice(take_rows, incoming_data.len() - take_rows);
        let head_rid = Arc::new(
            incoming_row_ids
                .slice(0, take_rows)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id type mismatch".into()))?
                .clone(),
        );
        let rest_rid = Arc::new(
            incoming_row_ids
                .slice(take_rows, incoming_row_ids.len() - take_rows)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id type mismatch".into()))?
                .clone(),
        );

        // Fetch tail blobs and deserialize.
        let gets = &[
            BatchGet::Raw {
                key: metas[tail_idx].chunk_pk,
            },
            BatchGet::Raw {
                key: metas_rid[tail_idx].chunk_pk,
            },
        ];
        let got = self.pager.batch_get(gets)?;
        let mut by_pk = FxHashMap::default();
        for r in got {
            match r {
                GetResult::Raw { key, bytes } => {
                    by_pk.insert(key, bytes);
                }
                _ => return Err(Error::NotFound),
            }
        }

        let tail_data_blob = by_pk
            .remove(&metas[tail_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let tail_rid_blob = by_pk
            .remove(&metas_rid[tail_idx].chunk_pk)
            .ok_or(Error::NotFound)?;

        let tail_data_arr = deserialize_array(tail_data_blob)?;
        let tail_rid_any = deserialize_array(tail_rid_blob)?;
        let tail_rid_arr = tail_rid_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("row_id type mismatch".into()))?;

        // Merge data and row_ids; no tombstones.
        let merged_data = concat_arrays(&tail_data_arr, &head_data)?;
        let merged_rid =
            arrow::compute::concat(&[tail_rid_arr as &dyn Array, (&*head_rid) as &dyn Array])?;

        let merged_data_norm = zero_offset(&merged_data);
        let merged_rid_norm = zero_offset(&merged_rid);
        let merged_data_bytes = serialize_array(merged_data_norm.as_ref())?;
        let merged_rid_bytes = serialize_array(merged_rid_norm.as_ref())?;

        puts.push(BatchPut::Raw {
            key: metas[tail_idx].chunk_pk,
            bytes: merged_data_bytes.clone(),
        });
        puts.push(BatchPut::Raw {
            key: metas_rid[tail_idx].chunk_pk,
            bytes: merged_rid_bytes.clone(),
        });

        metas[tail_idx].row_count = merged_data.len() as u64;
        metas[tail_idx].serialized_bytes = merged_data_bytes.len() as u64;
        metas_rid[tail_idx].row_count = merged_rid.len() as u64;
        metas_rid[tail_idx].serialized_bytes = merged_rid_bytes.len() as u64;

        // If tail had a sort permutation, rebuild it.
        if metas[tail_idx].value_order_perm_pk != 0 {
            let sort_column = SortColumn {
                values: merged_data.clone(),
                options: None,
            };
            let indices = lexsort_to_indices(&[sort_column], None)?;
            let perm_bytes = serialize_array(&indices)?;
            puts.push(BatchPut::Raw {
                key: metas[tail_idx].value_order_perm_pk,
                bytes: perm_bytes,
            });
        }

        // Persist updated descriptors.
        self.rewrite_descriptor_pages(desc_pk, &mut desc, &mut metas, puts)?;
        self.rewrite_descriptor_pages(desc_pk_rid, &mut desc_rid, &mut metas_rid, puts)?;

        Ok((rest_data, rest_rid))
    }
}

// --- K-way merge implementation (EntryHandle-backed, zero-copy) ---

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

pub struct MergeSortedIterator {
    cursors: Vec<ChunkCursor>,
    heap: BinaryHeap<HeapItem>,
    _blobs: FxHashMap<PhysicalKey, EntryHandle>,
}

impl MergeSortedIterator {
    fn try_new(
        metadata: Vec<ChunkMetadata>,
        blobs: FxHashMap<PhysicalKey, EntryHandle>,
    ) -> Result<Self> {
        let mut cursors = Vec::with_capacity(metadata.len());
        for meta in &metadata {
            let data_blob = blobs.get(&meta.chunk_pk).ok_or(Error::NotFound)?.clone();
            let perm_blob = blobs
                .get(&meta.value_order_perm_pk)
                .ok_or(Error::NotFound)?
                .clone();

            let data_array = deserialize_array(data_blob)?;
            let perm_array = deserialize_array(perm_blob)?;
            let perm_indices = perm_array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Error::Internal("Invalid permutation index type".into()))?
                .clone();

            // Be tolerant: if storage ever drifted, guard here.
            if perm_indices.len() != data_array.len() {
                return Err(Error::Internal("perm/data length mismatch".into()));
            }

            cursors.push(ChunkCursor {
                perm_indices: Arc::new(perm_indices),
                data_array,
                pos: 0,
            });
        }

        let mut heap = BinaryHeap::new();
        for (i, cursor) in cursors.iter().enumerate() {
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

impl Iterator for MergeSortedIterator {
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
