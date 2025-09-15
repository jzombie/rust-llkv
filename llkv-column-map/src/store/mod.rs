//! The main ColumnStore API.

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
use std::cmp::{Ordering, Reverse};
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

/// Target byte size for descriptor pages. The pager may be mmapped, so page
/// alignment helps. Used to avoid overgrowing tail pages.
const TARGET_DESCRIPTOR_PAGE_BYTES: usize = 4096;

/// Target size for data chunks coalesced by the bounded compactor and used
/// on the ingest path when slicing big appends into multiple chunks.
const TARGET_CHUNK_BYTES: usize = 1 * 1024 * 1024; // ~1 MiB

/// Merge only chunks smaller than this threshold.
const MIN_CHUNK_BYTES: usize = TARGET_CHUNK_BYTES / 2; // ~512 KiB

/// Upper bound on bytes coalesced per run, to cap rewrite cost.
const MAX_MERGE_RUN_BYTES: usize = 16 * 1024 * 1024; // ~16 MiB

/// Normalize an array so its offset is zero without copying buffers.
#[inline]
fn zero_offset(arr: &ArrayRef) -> ArrayRef {
    if arr.offset() == 0 {
        return arr.clone();
    }
    let data = arr.to_data();
    let sliced = data.slice(arr.offset(), arr.len());
    make_array(sliced)
}

/// Concatenate many arrays; returns a zero-offset array.
#[inline]
fn concat_many(mut v: Vec<ArrayRef>) -> Result<ArrayRef> {
    if v.is_empty() {
        return Err(Error::Internal("concat_many: empty".into()));
    }
    if v.len() == 1 {
        return Ok(zero_offset(&v.pop().unwrap()));
    }
    let parts: Vec<&dyn Array> = v.iter().map(|a| a.as_ref()).collect();
    Ok(arrow::compute::concat(&parts)?)
}

/// Bytes-per-value for fixed-width primitives (Boolean approximated as 1).
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

/// Split array into ~target_bytes slices. Fixed-width uses bytes-based rows,
/// variable-width falls back to a conservative row count.
fn split_to_target_bytes(arr: &ArrayRef, target_bytes: usize) -> Vec<ArrayRef> {
    let n = arr.len();
    if n == 0 {
        return vec![];
    }
    if let Some(bpv) = fixed_width_bpv(arr.data_type()) {
        let rows_per = (target_bytes / bpv).max(1);
        let mut out = Vec::with_capacity((n + rows_per - 1) / rows_per);
        let mut off = 0usize;
        while off < n {
            let take = (n - off).min(rows_per);
            out.push(arr.slice(off, take));
            off += take;
        }
        return out;
    }
    // Variable-width fallback.
    let rows_per = 4096usize;
    let mut out = Vec::with_capacity((n + rows_per - 1) / rows_per);
    let mut off = 0usize;
    while off < n {
        let take = (n - off).min(rows_per);
        out.push(arr.slice(off, take));
        off += take;
    }
    out
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

    /// Append with automatic LWW (by row_id) using in-place rewrites.
    ///
    /// Requirements:
    /// - Batch must contain a non-nullable UInt64 column named "row_id".
    /// - For each data column (with "field_id" metadata), we:
    ///   1) rewrite existing chunks to remove rows whose per-field shadow
    ///      row_ids match incoming row_ids (LWW),
    ///   2) append the new data chunk(s) (ingest-time slicing to ~TARGET_CHUNK_BYTES),
    ///   3) append matching shadow row_id chunk(s) for the field.
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        let schema = batch.schema();
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
        let row_id_any: ArrayRef = Arc::clone(batch.column(row_id_idx));

        let mut incoming_ids = FxHashSet::default();
        incoming_ids.reserve(row_id_arr.len());
        for i in 0..row_id_arr.len() {
            incoming_ids.insert(row_id_arr.value(i));
        }

        let mut catalog_dirty = false;
        let mut catalog = self.catalog.write().unwrap();

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

        let mut puts_appends: Vec<BatchPut> = Vec::new();

        for (field_id, array) in data_cols.into_iter() {
            let descriptor_pk = *catalog.map.entry(field_id).or_insert_with(|| {
                catalog_dirty = true;
                self.pager.alloc_many(1).unwrap()[0]
            });
            let rid_fid = rowid_fid(field_id);
            let rid_descriptor_pk = *catalog.map.entry(rid_fid).or_insert_with(|| {
                catalog_dirty = true;
                self.pager.alloc_many(1).unwrap()[0]
            });

            // --- STATE MANAGEMENT FIX: Load state once before the loop ---
            let (mut data_descriptor, mut data_tail_page) =
                self.load_descriptor_state(descriptor_pk, field_id)?;
            let (mut rid_descriptor, mut rid_tail_page) =
                self.load_descriptor_state(rid_descriptor_pk, rid_fid)?;

            let slices = split_to_target_bytes(&array, TARGET_CHUNK_BYTES);
            let mut row_off = 0usize;

            for s in slices {
                let rows = s.len();

                // Data chunk
                let data_pk = self.pager.alloc_many(1)?[0];
                let s_norm = zero_offset(&s);
                let data_bytes = serialize_array(s_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: data_pk,
                    bytes: data_bytes.clone(),
                });
                let data_meta = ChunkMetadata {
                    chunk_pk: data_pk,
                    tombstone_pk: 0,
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: data_bytes.len() as u64,
                    min_val_u64: 0,
                    max_val_u64: u64::MAX,
                };
                self.append_meta_in_loop(
                    &mut data_descriptor,
                    &mut data_tail_page,
                    data_meta,
                    &mut puts_appends,
                )?;

                // Shadow row_id chunk
                let rid_slice: ArrayRef = row_id_any.slice(row_off, rows);
                let rid_norm = zero_offset(&rid_slice);
                let rid_pk = self.pager.alloc_many(1)?[0];
                let rid_bytes = serialize_array(rid_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: rid_pk,
                    bytes: rid_bytes.clone(),
                });
                let rid_meta = ChunkMetadata {
                    chunk_pk: rid_pk,
                    tombstone_pk: 0,
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: rid_bytes.len() as u64,
                    min_val_u64: 0,
                    max_val_u64: u64::MAX,
                };
                self.append_meta_in_loop(
                    &mut rid_descriptor,
                    &mut rid_tail_page,
                    rid_meta,
                    &mut puts_appends,
                )?;

                row_off += rows;
            }

            // --- STATE MANAGEMENT FIX: Write final state after the loop ---
            puts_appends.push(BatchPut::Raw {
                key: data_descriptor.tail_page_pk,
                bytes: data_tail_page,
            });
            puts_appends.push(BatchPut::Raw {
                key: descriptor_pk,
                bytes: data_descriptor.to_le_bytes(),
            });
            puts_appends.push(BatchPut::Raw {
                key: rid_descriptor.tail_page_pk,
                bytes: rid_tail_page,
            });
            puts_appends.push(BatchPut::Raw {
                key: rid_descriptor_pk,
                bytes: rid_descriptor.to_le_bytes(),
            });
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

        // Find hit chunks by scanning shadow row_ids.
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

        // Batch-get needed data + rid chunks for rewrites.
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

        // Rewrite each hit chunk in place.
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
            let data_bytes = serialize_array(&data_f)?;
            puts.push(BatchPut::Raw {
                key: metas_data[i].chunk_pk,
                bytes: data_bytes.clone(),
            });

            let rid_f = compute::filter(&rid_arr_any, &keep)?;
            let rid_bytes = serialize_array(&rid_f)?;
            puts.push(BatchPut::Raw {
                key: metas_rid[i].chunk_pk,
                bytes: rid_bytes.clone(),
            });

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
    /// Then bounded compaction of small adjacent chunks.
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

        // Gather metadata for data column.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            drop(catalog);
            // Still compact to handle empty-field cleanup.
            self.compact_field_bounded(field_id)?;
            return Ok(());
        }

        // Optional: corresponding shadow row_id descriptor + metas.
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

        // Single-pass walk over the delete bitmap.
        let mut del_iter = rows_to_delete.iter();
        let mut cur_del = del_iter.next();

        let mut cum_rows = 0u64;

        for (i, meta) in metas.iter_mut().enumerate() {
            let start_u64 = cum_rows;
            let end_u64 = start_u64 + meta.row_count;

            let start = start_u64 as u32;
            let end = end_u64 as u32;

            while let Some(d) = cur_del {
                if d < start {
                    cur_del = del_iter.next();
                } else {
                    break;
                }
            }

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

            let keep = BooleanArray::from_iter(
                (0..meta.row_count as usize).map(|j| Some(!del_local.contains(&(j as u32)))),
            );

            // Rewrite data chunk in place.
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

            // Rewrite matching shadow row_id chunk, if present.
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

            // If a sorted permutation exists for this chunk, refresh it.
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

        // Persist in-place rewrites before compaction.
        if !puts.is_empty() {
            self.pager.batch_put(&puts)?;
        }

        // Drop the read lock before compacting (compaction edits catalog).
        drop(catalog);

        // Bounded local compaction (no full-field rewrite).
        self.compact_field_bounded(field_id)?;

        Ok(())
    }

    /// Write a descriptor chain from a meta slice. Reuses first page when
    /// possible. Frees surplus pages via `frees`.
    fn write_descriptor_chain(
        &self,
        descriptor_pk: PhysicalKey,
        descriptor: &mut ColumnDescriptor,
        new_metas: &[ChunkMetadata],
        puts: &mut Vec<BatchPut>,
        frees: &mut Vec<PhysicalKey>,
    ) -> Result<()> {
        // Collect existing page chain.
        let mut old_pages = Vec::new();
        let mut pk = descriptor.head_page_pk;
        while pk != 0 {
            let page_blob = self.pager.get_raw(pk)?.ok_or(Error::NotFound)?;
            let header = DescriptorPageHeader::from_le_bytes(
                &page_blob.as_ref()[..DescriptorPageHeader::DISK_SIZE],
            );
            old_pages.push(pk);
            pk = header.next_page_pk;
        }

        // Required pages for new metas.
        let per = DESCRIPTOR_ENTRIES_PER_PAGE;
        let need_pages = if new_metas.is_empty() {
            0
        } else {
            (new_metas.len() + per - 1) / per
        };

        // If empty: free everything and clear counters.
        if need_pages == 0 {
            frees.extend(old_pages.iter().copied());
            descriptor.total_row_count = 0;
            descriptor.total_chunk_count = 0;
            puts.push(BatchPut::Raw {
                key: descriptor_pk,
                bytes: descriptor.to_le_bytes(),
            });
            return Ok(());
        }

        // Reuse first page; alloc more if needed.
        let mut pages = Vec::with_capacity(need_pages);
        if !old_pages.is_empty() {
            pages.push(old_pages[0]);
        } else {
            pages.push(self.pager.alloc_many(1)?[0]);
            descriptor.head_page_pk = pages[0];
        }
        if need_pages > pages.len() {
            let extra = self.pager.alloc_many(need_pages - pages.len())?;
            pages.extend(extra);
        }

        // Free surplus old pages.
        if old_pages.len() > need_pages {
            frees.extend(old_pages[need_pages..].iter().copied());
        }

        // Write page contents.
        let mut off = 0usize;
        for (i, page_pk) in pages.iter().copied().enumerate() {
            let remain = new_metas.len() - off;
            let count = remain.min(per);
            let next = if i + 1 < pages.len() { pages[i + 1] } else { 0 };
            let header = DescriptorPageHeader {
                next_page_pk: next,
                entry_count: count as u32,
                _padding: [0; 4],
            };
            let mut page_bytes = header.to_le_bytes().to_vec();
            for m in &new_metas[off..off + count] {
                page_bytes.extend_from_slice(&m.to_le_bytes());
            }
            puts.push(BatchPut::Raw {
                key: page_pk,
                bytes: page_bytes,
            });
            off += count;
        }

        descriptor.tail_page_pk = *pages.last().unwrap();
        descriptor.total_chunk_count = new_metas.len() as u64;
        descriptor.total_row_count = new_metas.iter().map(|m| m.row_count).sum();
        puts.push(BatchPut::Raw {
            key: descriptor_pk,
            bytes: descriptor.to_le_bytes(),
        });
        Ok(())
    }

    /// Bounded, local field compaction. Merges adjacent small chunks into
    /// ~TARGET_CHUNK_BYTES; leaves large chunks intact. Recomputes perms if
    /// any source chunk in a run had one. Frees obsolete chunks/pages.
    fn compact_field_bounded(&self, field_id: LogicalFieldId) -> Result<()> {
        // We may rewrite descriptors; take a write lock.
        let mut catalog = self.catalog.write().unwrap();

        let desc_pk = match catalog.map.get(&field_id) {
            Some(&pk) => pk,
            None => return Ok(()),
        };
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = match catalog.map.get(&rid_fid) {
            Some(&pk) => pk,
            None => return Ok(()),
        };

        let desc_blob = self.pager.get_raw(desc_pk)?.ok_or(Error::NotFound)?;
        let mut desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        let desc_blob_rid = self.pager.get_raw(desc_pk_rid)?.ok_or(Error::NotFound)?;
        let mut desc_rid = ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref());

        // Load metas.
        let mut metas = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), desc.head_page_pk) {
            metas.push(m?);
        }
        let mut metas_rid = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), desc_rid.head_page_pk) {
            metas_rid.push(m?);
        }
        if metas.is_empty() || metas_rid.is_empty() {
            return Ok(());
        }

        let mut puts: Vec<BatchPut> = Vec::new();
        let mut frees: Vec<PhysicalKey> = Vec::new();

        let mut new_metas: Vec<ChunkMetadata> = Vec::new();
        let mut new_rid_metas: Vec<ChunkMetadata> = Vec::new();

        let mut i = 0usize;
        while i < metas.len() {
            let sz = metas[i].serialized_bytes as usize;

            // Keep large chunks as-is.
            if sz >= MIN_CHUNK_BYTES {
                new_metas.push(metas[i]);
                new_rid_metas.push(metas_rid[i]);
                i += 1;
                continue;
            }

            // Build a small run [i, j) capped by MAX_MERGE_RUN_BYTES.
            let mut j = i;
            let mut run_bytes = 0usize;
            while j < metas.len() {
                let b = metas[j].serialized_bytes as usize;
                if b >= TARGET_CHUNK_BYTES {
                    break;
                }
                if run_bytes + b > MAX_MERGE_RUN_BYTES {
                    break;
                }
                run_bytes += b;
                j += 1;
            }
            if j == i + 1 && sz >= MIN_CHUNK_BYTES {
                new_metas.push(metas[i]);
                new_rid_metas.push(metas_rid[i]);
                i += 1;
                continue;
            }

            // Fetch and concatenate the run's data and row_ids.
            let mut gets = Vec::with_capacity((j - i) * 2);
            for k in i..j {
                gets.push(BatchGet::Raw {
                    key: metas[k].chunk_pk,
                });
                gets.push(BatchGet::Raw {
                    key: metas_rid[k].chunk_pk,
                });
            }
            let results = self.pager.batch_get(&gets)?;
            let mut by_pk: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            for r in results {
                match r {
                    GetResult::Raw { key, bytes } => {
                        by_pk.insert(key, bytes);
                    }
                    _ => return Err(Error::NotFound),
                }
            }
            let mut data_parts = Vec::with_capacity(j - i);
            let mut rid_parts = Vec::with_capacity(j - i);
            for k in i..j {
                let db = by_pk.get(&metas[k].chunk_pk).ok_or(Error::NotFound)?;
                data_parts.push(deserialize_array(db.clone())?);
                let rb = by_pk.get(&metas_rid[k].chunk_pk).ok_or(Error::NotFound)?;
                rid_parts.push(deserialize_array(rb.clone())?);
            }
            let merged_data = concat_many(data_parts)?;
            let merged_rid_any = concat_many(rid_parts)?;

            // Split merged run into ~target-sized chunks.
            let slices = split_to_target_bytes(&merged_data, TARGET_CHUNK_BYTES);
            let mut rid_off = 0usize;
            let mut need_perms = false;
            for k in i..j {
                if metas[k].value_order_perm_pk != 0 {
                    need_perms = true;
                    break;
                }
            }

            for s in slices {
                let rows = s.len();

                // Slice rid to match s (avoid double Arc).
                let rid_ref: ArrayRef = merged_rid_any.slice(rid_off, rows);
                let rid_norm = zero_offset(&rid_ref);
                let rid_pk = self.pager.alloc_many(1)?[0];
                let rid_bytes = serialize_array(rid_norm.as_ref())?;

                let data_pk = self.pager.alloc_many(1)?[0];
                let s_norm = zero_offset(&s);
                let data_bytes = serialize_array(s_norm.as_ref())?;

                puts.push(BatchPut::Raw {
                    key: data_pk,
                    bytes: data_bytes.clone(),
                });
                puts.push(BatchPut::Raw {
                    key: rid_pk,
                    bytes: rid_bytes.clone(),
                });

                let mut meta = ChunkMetadata {
                    chunk_pk: data_pk,
                    tombstone_pk: 0,
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: data_bytes.len() as u64,
                    min_val_u64: 0,
                    max_val_u64: u64::MAX,
                };
                // If any source chunk had a perm, recompute for this slice.
                if need_perms {
                    let sort_col = SortColumn {
                        values: s.clone(),
                        options: None,
                    };
                    let idx = lexsort_to_indices(&[sort_col], None)?;
                    let perm_bytes = serialize_array(&idx)?;
                    let perm_pk = self.pager.alloc_many(1)?[0];
                    puts.push(BatchPut::Raw {
                        key: perm_pk,
                        bytes: perm_bytes,
                    });
                    meta.value_order_perm_pk = perm_pk;
                }

                let rid_meta = ChunkMetadata {
                    chunk_pk: rid_pk,
                    tombstone_pk: 0,
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: rid_bytes.len() as u64,
                    min_val_u64: 0,
                    max_val_u64: u64::MAX,
                };

                new_metas.push(meta);
                new_rid_metas.push(rid_meta);
                rid_off += rows;
            }

            // Free all old data/rid/perms in the merged run.
            for k in i..j {
                frees.push(metas[k].chunk_pk);
                if metas[k].value_order_perm_pk != 0 {
                    frees.push(metas[k].value_order_perm_pk);
                }
                frees.push(metas_rid[k].chunk_pk);
            }

            i = j;
        }

        // If everything was deleted, drop the field entirely.
        if new_metas.is_empty() {
            // Drop descriptors and catalog mapping.
            self.write_descriptor_chain(desc_pk, &mut desc, &[], &mut puts, &mut frees)?;
            self.write_descriptor_chain(desc_pk_rid, &mut desc_rid, &[], &mut puts, &mut frees)?;
            catalog.map.remove(&field_id);
            catalog.map.remove(&rid_fid);
            puts.push(BatchPut::Raw {
                key: CATALOG_ROOT_PKEY,
                bytes: catalog.to_bytes(),
            });
            if !puts.is_empty() {
                self.pager.batch_put(&puts)?;
            }
            if !frees.is_empty() {
                self.pager.free_many(&frees)?;
            }
            return Ok(());
        }

        // Rewrite descriptor chains to match the new meta lists.
        self.write_descriptor_chain(desc_pk, &mut desc, &new_metas, &mut puts, &mut frees)?;
        self.write_descriptor_chain(
            desc_pk_rid,
            &mut desc_rid,
            &new_rid_metas,
            &mut puts,
            &mut frees,
        )?;

        // Persist new/updated blobs and free the old ones.
        if !puts.is_empty() {
            self.pager.batch_put(&puts)?;
        }
        if !frees.is_empty() {
            self.pager.free_many(&frees)?;
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
        for meta in DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk) {
            metas.push(meta?);
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

        // Rewrite descriptor pages with updated perm keys.
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
        for result in get_results {
            match result {
                GetResult::Raw { key, bytes } => {
                    blobs_map.insert(key, bytes);
                }
                GetResult::Missing { .. } => return Err(Error::NotFound),
            }
        }

        MergeSortedIterator::try_new(all_chunk_metadata, blobs_map)
    }

    /// (Internal) Loads the full state for a descriptor, creating it if it doesn't exist.
    fn load_descriptor_state(
        &self,
        descriptor_pk: PhysicalKey,
        field_id: LogicalFieldId,
    ) -> Result<(ColumnDescriptor, Vec<u8>)> {
        match self.pager.get_raw(descriptor_pk)? {
            Some(blob) => {
                let descriptor = ColumnDescriptor::from_le_bytes(blob.as_ref());
                let tail_page_bytes = self
                    .pager
                    .get_raw(descriptor.tail_page_pk)?
                    .ok_or(Error::NotFound)?
                    .as_ref()
                    .to_vec();
                Ok((descriptor, tail_page_bytes))
            }
            None => {
                let first_page_pk = self.pager.alloc_many(1)?[0];
                let descriptor = ColumnDescriptor {
                    field_id,
                    head_page_pk: first_page_pk,
                    tail_page_pk: first_page_pk,
                    total_row_count: 0,
                    total_chunk_count: 0,
                };
                let header = DescriptorPageHeader {
                    next_page_pk: 0,
                    entry_count: 0,
                    _padding: [0; 4],
                };
                let tail_page_bytes = header.to_le_bytes().to_vec();
                Ok((descriptor, tail_page_bytes))
            }
        }
    }

    /// (Internal) Helper for batch appends. Appends metadata to the current in-memory
    /// tail page, creating a new one if necessary.
    fn append_meta_in_loop(
        &self,
        descriptor: &mut ColumnDescriptor,
        tail_page_bytes: &mut Vec<u8>,
        meta: ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let mut header = DescriptorPageHeader::from_le_bytes(
            &tail_page_bytes[..DescriptorPageHeader::DISK_SIZE],
        );

        if tail_page_bytes.len() + ChunkMetadata::DISK_SIZE <= TARGET_DESCRIPTOR_PAGE_BYTES
            && (header.entry_count as usize) < DESCRIPTOR_ENTRIES_PER_PAGE
        {
            // Case 1: There's room in the current tail page.
            tail_page_bytes.extend_from_slice(&meta.to_le_bytes());
            header.entry_count += 1;
            tail_page_bytes[..DescriptorPageHeader::DISK_SIZE]
                .copy_from_slice(&header.to_le_bytes());
        } else {
            // Case 2: The tail page is full. Write it out and start a new one.
            let new_tail_pk = self.pager.alloc_many(1)?[0];
            header.next_page_pk = new_tail_pk;
            tail_page_bytes[..DescriptorPageHeader::DISK_SIZE]
                .copy_from_slice(&header.to_le_bytes());

            // --- PERFORMANCE FIX: Move the full page's bytes instead of cloning ---
            let full_page_to_write = std::mem::take(tail_page_bytes);
            puts.push(BatchPut::Raw {
                key: descriptor.tail_page_pk,
                bytes: full_page_to_write,
            });

            // Create the new tail page.
            let new_header = DescriptorPageHeader {
                next_page_pk: 0,
                entry_count: 1,
                _padding: [0; 4],
            };
            let mut new_page_bytes = new_header.to_le_bytes().to_vec();
            new_page_bytes.extend_from_slice(&meta.to_le_bytes());

            // Update our live state to point to the new tail.
            descriptor.tail_page_pk = new_tail_pk;
            *tail_page_bytes = new_page_bytes;
        }

        descriptor.total_row_count += meta.row_count;
        descriptor.total_chunk_count += 1;
        Ok(())
    }
}

// --- K-way merge implementation (EntryHandle-backed, zero-copy) ---

struct ChunkCursor {
    /// A fully sorted view of the chunk's data.
    data_sorted: ArrayRef,
    len: usize,
}

#[derive(Clone, Copy, Debug)]
struct HeapItem {
    cursor_idx: usize,
    /// The index into the `data_sorted` array for the current cursor.
    data_idx: usize,
    key_u128: u128,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key_u128 == other.key_u128 && self.cursor_idx == other.cursor_idx
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
        // Natural ascending order by key, with a stable tie-breaker (cursor_idx).
        // This is the correct logic for use with `Reverse` to create a min-heap.
        self.key_u128
            .cmp(&other.key_u128)
            .then_with(|| self.cursor_idx.cmp(&other.cursor_idx))
    }
}

#[derive(Clone, Copy)]
enum KeyType {
    U64,
    I32,
}

#[inline]
fn encode_key_u128(arr: &ArrayRef, idx: usize, kind: KeyType) -> u128 {
    match kind {
        KeyType::U64 => {
            let a = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
            a.value(idx) as u128
        }
        KeyType::I32 => {
            let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
            (a.value(idx) as i128 - i32::MIN as i128) as u128
        }
    }
}

pub struct MergeSortedIterator {
    cursors: Vec<ChunkCursor>,
    heap: BinaryHeap<Reverse<HeapItem>>, // min-heap by key
    key_kind: KeyType,
    _blobs: FxHashMap<PhysicalKey, EntryHandle>, // Keeps memory alive
}

impl MergeSortedIterator {
    pub fn try_new(
        metadata: Vec<ChunkMetadata>,
        blobs: FxHashMap<PhysicalKey, EntryHandle>,
    ) -> Result<Self> {
        // Determine key type from the first non-empty chunk.
        let key_kind = {
            let first_meta = metadata.iter().find(|m| m.row_count > 0);
            if first_meta.is_none() {
                // Handle case where all chunks are empty
                return Ok(Self {
                    cursors: Vec::new(),
                    heap: BinaryHeap::new(),
                    key_kind: KeyType::U64, // Default, won't be used
                    _blobs: blobs,
                });
            }
            let data_blob = blobs
                .get(&first_meta.unwrap().chunk_pk)
                .ok_or(Error::NotFound)?
                .clone();
            let data_array = deserialize_array(data_blob)?;
            match data_array.data_type() {
                &DataType::UInt64 => KeyType::U64,
                &DataType::Int32 => KeyType::I32,
                other => {
                    return Err(Error::Internal(format!(
                        "Unsupported sort type {:?}",
                        other
                    )));
                }
            }
        };

        let mut cursors = Vec::with_capacity(metadata.len());
        for meta in &metadata {
            if meta.row_count == 0 {
                continue;
            }
            let data_blob = blobs.get(&meta.chunk_pk).ok_or(Error::NotFound)?.clone();
            let data_arr = deserialize_array(data_blob)?;

            let perm_blob = blobs
                .get(&meta.value_order_perm_pk)
                .ok_or(Error::NotFound)?
                .clone();
            let perm_any = deserialize_array(perm_blob)?;
            let perm = perm_any
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or(Error::Internal("perm not u32".into()))?;

            // Materialize the sorted view of the chunk's data.
            let sorted = compute::take(&data_arr, perm, None)?;
            cursors.push(ChunkCursor {
                data_sorted: zero_offset(&sorted),
                len: sorted.len(),
            });
        }

        // Seed the heap with the first element from each chunk.
        let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        for (i, cursor) in cursors.iter().enumerate() {
            if !cursor.data_sorted.is_empty() {
                let key = encode_key_u128(&cursor.data_sorted, 0, key_kind);
                heap.push(Reverse(HeapItem {
                    cursor_idx: i,
                    data_idx: 0,
                    key_u128: key,
                }));
            }
        }

        Ok(Self {
            cursors,
            heap,
            key_kind,
            _blobs: blobs,
        })
    }
}

impl Iterator for MergeSortedIterator {
    type Item = Result<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        // Pop the item with the smallest key from the heap.
        match self.heap.pop() {
            Some(Reverse(item)) => {
                let cursor = &self.cursors[item.cursor_idx];

                // Create a 1-element slice to return.
                let output_slice = cursor.data_sorted.slice(item.data_idx, 1);

                // If there are more elements in this chunk, push the next one onto the heap.
                let next_data_idx = item.data_idx + 1;
                if next_data_idx < cursor.len {
                    let next_key =
                        encode_key_u128(&cursor.data_sorted, next_data_idx, self.key_kind);
                    self.heap.push(Reverse(HeapItem {
                        cursor_idx: item.cursor_idx,
                        data_idx: next_data_idx,
                        key_u128: next_key,
                    }));
                }

                Some(Ok(output_slice))
            }
            // If the heap is empty, the iteration is complete.
            None => None,
        }
    }
}
