//! The main ColumnStore API.

use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::types::{CATALOG_ROOT_PKEY, LogicalFieldId, Namespace, PhysicalKey};
use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, LargeBinaryArray, LargeStringArray, StringArray,
    UInt32Array, UInt64Array, make_array,
};
use arrow::compute::{self, SortColumn, lexsort_to_indices};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

pub mod catalog;
pub mod debug;
pub mod descriptor;

pub mod iter;
pub use iter::*;

const DESCRIPTOR_ENTRIES_PER_PAGE: usize = 256;

/// Sets the shadow row_id tag on a LogicalFieldId using the Namespace enum.
fn rowid_fid(fid: LogicalFieldId) -> LogicalFieldId {
    fid.with_namespace(Namespace::RowIdShadow)
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

/// Run-time configuration (no hidden constants).
#[derive(Debug, Clone)]
pub struct ColumnStoreConfig {
    /// Fallback rows-per-slice for *exotic* variable-width arrays that don't
    /// expose byte offsets we can use (e.g., List/Struct/Map). Used only as
    /// a last resort to avoid pathological slices.
    pub varwidth_fallback_rows_per_slice: usize,
}

impl Default for ColumnStoreConfig {
    fn default() -> Self {
        Self {
            varwidth_fallback_rows_per_slice: 4096,
        }
    }
}

/// Statistics for a single descriptor page.
#[derive(Debug, Clone)]
pub struct DescriptorPageStats {
    pub page_pk: PhysicalKey,
    pub entry_count: u32,
    pub page_size_bytes: usize,
}

/// Aggregated layout statistics for a single column.
#[derive(Debug, Clone)]
pub struct ColumnLayoutStats {
    pub field_id: LogicalFieldId,
    pub total_rows: u64,
    pub total_chunks: u64,
    pub pages: Vec<DescriptorPageStats>,
}

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
fn concat_many(v: Vec<&ArrayRef>) -> Result<ArrayRef> {
    if v.is_empty() {
        return Err(Error::Internal("concat_many: empty".into()));
    }
    if v.len() == 1 {
        return Ok(zero_offset(v[0]));
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

/// Split variable-width arrays (Utf8/Binary) using their offset buffers so each
/// slice is ~target_bytes of values payload. Offsets are 32-bit here.
#[inline]
fn split_varlen_offsets32(
    offsets: &[i32],
    len: usize,
    arr: &ArrayRef,
    target_bytes: usize,
) -> Vec<ArrayRef> {
    let mut out = Vec::new();
    if len == 0 {
        return out;
    }
    let base = offsets[0] as i64;
    let mut i = 0usize;
    // Ensure at least 1 row per slice; pack until ~target_bytes.
    while i < len {
        let start_off = (offsets[i] as i64 - base) as usize;
        let mut j = i + 1;
        while j <= len {
            let end_off = (offsets[j] as i64 - base) as usize;
            let bytes = end_off - start_off;
            if bytes >= target_bytes && j > i {
                break;
            }
            j += 1;
        }
        let take = (j - i).min(len - i);
        out.push(arr.slice(i, take));
        i += take;
    }
    out
}

/// Same as above but for LargeUtf8/LargeBinary (64-bit offsets).
#[inline]
fn split_varlen_offsets64(
    offsets: &[i64],
    len: usize,
    arr: &ArrayRef,
    target_bytes: usize,
) -> Vec<ArrayRef> {
    let mut out = Vec::new();
    if len == 0 {
        return out;
    }
    let base = offsets[0];
    let mut i = 0usize;
    while i < len {
        let start_off = (offsets[i] - base) as usize;
        let mut j = i + 1;
        while j <= len {
            let end_off = (offsets[j] - base) as usize;
            let bytes = end_off - start_off;
            if bytes >= target_bytes && j > i {
                break;
            }
            j += 1;
        }
        let take = (j - i).min(len - i);
        out.push(arr.slice(i, take));
        i += take;
    }
    out
}

/// Split array into ~target_bytes slices. Fixed-width uses bytes-based rows.
/// For var-width (Utf8/Binary), we use offsets to target value-bytes per slice.
/// For other exotic var-width types, we conservatively fall back to a row cap.
fn split_to_target_bytes(
    arr: &ArrayRef,
    target_bytes: usize,
    varwidth_fallback_rows_per_slice: usize,
) -> Vec<ArrayRef> {
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

    // Handle common var-width types using their offsets, after zeroing slice offset.
    let arr0 = zero_offset(arr);
    match arr0.data_type() {
        DataType::Utf8 => {
            let a = arr0.as_any().downcast_ref::<StringArray>().unwrap();
            split_varlen_offsets32(a.value_offsets(), n, &arr0, target_bytes)
        }
        DataType::Binary => {
            let a = arr0.as_any().downcast_ref::<BinaryArray>().unwrap();
            split_varlen_offsets32(a.value_offsets(), n, &arr0, target_bytes)
        }
        DataType::LargeUtf8 => {
            let a = arr0.as_any().downcast_ref::<LargeStringArray>().unwrap();
            split_varlen_offsets64(a.value_offsets(), n, &arr0, target_bytes)
        }
        DataType::LargeBinary => {
            let a = arr0.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            split_varlen_offsets64(a.value_offsets(), n, &arr0, target_bytes)
        }
        // Fallback for other var-width types (e.g., List/Struct/Map): conservative row cap.
        _ => {
            let rows_per = varwidth_fallback_rows_per_slice; // configurable
            let mut out = Vec::with_capacity((n + rows_per - 1) / rows_per);
            let mut off = 0usize;
            while off < n {
                let take = (n - off).min(rows_per);
                out.push(arr.slice(off, take));
                off += take;
            }
            out
        }
    }
}

pub struct ColumnStore<P: Pager> {
    pager: Arc<P>,
    catalog: RwLock<ColumnCatalog>,
    cfg: ColumnStoreConfig,
}

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle>,
{
    pub fn open(pager: Arc<P>) -> Result<Self> {
        Self::open_with_config(pager, ColumnStoreConfig::default())
    }

    pub fn open_with_config(pager: Arc<P>, cfg: ColumnStoreConfig) -> Result<Self> {
        let catalog = match pager
            .batch_get(&[BatchGet::Raw {
                key: CATALOG_ROOT_PKEY,
            }])?
            .pop()
        {
            Some(GetResult::Raw { bytes, .. }) => ColumnCatalog::from_bytes(bytes.as_ref())?,
            _ => ColumnCatalog::default(),
        };
        Ok(Self {
            pager,
            catalog: RwLock::new(catalog),
            cfg,
        })
    }

    // TODO: Convert all nulls to deletes (don't store them)
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        let schema = batch.schema();
        let row_id_idx = schema
            .index_of("row_id")
            .map_err(|_| Error::Internal("row_id column required".into()))?;
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
        let mut incoming_ids_map = FxHashMap::default();
        incoming_ids_map.reserve(row_id_arr.len());
        for i in 0..row_id_arr.len() {
            incoming_ids_map.insert(row_id_arr.value(i), i);
        }

        let mut catalog_dirty = false;
        let mut catalog = self.catalog.write().unwrap();
        let mut puts_rewrites: Vec<BatchPut> = Vec::new();
        let mut all_rewritten_ids = FxHashSet::default();
        // --- LWW Phase: Perform all rewrites first ---
        for i in 0..batch.num_columns() {
            if i == row_id_idx {
                continue;
            }
            let field = schema.field(i);
            if let Some(field_id_str) = field.metadata().get("field_id") {
                // The application is responsible for providing a u64 that can be
                // converted into the namespaced LogicalFieldId struct.
                let field_id = field_id_str
                    .parse::<u64>()
                    .map(LogicalFieldId::from)
                    .map_err(|e| Error::Internal(format!("Invalid field_id: {}", e)))?;

                let rewritten = self.lww_rewrite_for_field(
                    &mut catalog,
                    field_id,
                    &incoming_ids_map,
                    batch.column(i),
                    batch.column(row_id_idx),
                    &mut puts_rewrites,
                )?;
                all_rewritten_ids.extend(rewritten);
            }
        }

        if !puts_rewrites.is_empty() {
            self.pager.batch_put(&puts_rewrites)?;
        }

        // --- Append Phase: Filter the batch to only include NEW rows ---
        let batch_to_append = if !all_rewritten_ids.is_empty() {
            let keep_mask: Vec<bool> = (0..row_id_arr.len())
                .map(|i| !all_rewritten_ids.contains(&row_id_arr.value(i)))
                .collect();
            let keep_array = BooleanArray::from(keep_mask);
            compute::filter_record_batch(batch, &keep_array)?
        } else {
            batch.clone()
        };
        if batch_to_append.num_rows() == 0 {
            return Ok(());
        }

        // Now, proceed with appending the correctly filtered batch
        let append_schema = batch_to_append.schema();
        let append_row_id_idx = append_schema.index_of("row_id")?;
        let append_row_id_any: ArrayRef = Arc::clone(batch_to_append.column(append_row_id_idx));
        let mut puts_appends: Vec<BatchPut> = Vec::new();
        for (i, array) in batch_to_append.columns().iter().enumerate() {
            if i == append_row_id_idx {
                continue;
            }
            let field = append_schema.field(i);
            let field_id = field
                .metadata()
                .get("field_id")
                .ok_or_else(|| Error::Internal("Missing field_id".into()))?
                .parse::<u64>()
                .map(LogicalFieldId::from)
                .map_err(|e| Error::Internal(format!("Invalid field_id: {}", e)))?;

            let descriptor_pk = *catalog.map.entry(field_id).or_insert_with(|| {
                catalog_dirty = true;
                self.pager.alloc_many(1).unwrap()[0]
            });
            let rid_fid = rowid_fid(field_id);
            let rid_descriptor_pk = *catalog.map.entry(rid_fid).or_insert_with(|| {
                catalog_dirty = true;
                self.pager.alloc_many(1).unwrap()[0]
            });
            let (mut data_descriptor, mut data_tail_page) =
                self.load_descriptor_state(descriptor_pk, field_id)?;
            let (mut rid_descriptor, mut rid_tail_page) =
                self.load_descriptor_state(rid_descriptor_pk, rid_fid)?;
            let slices = split_to_target_bytes(
                array,
                TARGET_CHUNK_BYTES,
                self.cfg.varwidth_fallback_rows_per_slice,
            );
            let mut row_off = 0usize;

            for s in slices {
                let rows = s.len();
                let data_pk = self.pager.alloc_many(1)?[0];
                let s_norm = zero_offset(&s);
                let data_bytes = serialize_array(s_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: data_pk,
                    bytes: data_bytes,
                });
                let data_meta = ChunkMetadata {
                    chunk_pk: data_pk,
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: s_norm.get_array_memory_size() as u64,
                    min_val_u64: 0,
                    max_val_u64: u64::MAX,
                };
                self.append_meta_in_loop(
                    &mut data_descriptor,
                    &mut data_tail_page,
                    data_meta,
                    &mut puts_appends,
                )?;
                let rid_slice: ArrayRef = append_row_id_any.slice(row_off, rows);
                let rid_norm = zero_offset(&rid_slice);
                let rid_pk = self.pager.alloc_many(1)?[0];
                let rid_bytes = serialize_array(rid_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: rid_pk,
                    bytes: rid_bytes,
                });
                let rid_meta = ChunkMetadata {
                    chunk_pk: rid_pk,
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: rid_norm.get_array_memory_size() as u64,
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

    fn lww_rewrite_for_field(
        &self,
        catalog: &mut ColumnCatalog,
        field_id: LogicalFieldId,
        incoming_ids_map: &FxHashMap<u64, usize>,
        incoming_data: &ArrayRef,
        incoming_row_ids: &ArrayRef,
        puts: &mut Vec<BatchPut>,
    ) -> Result<FxHashSet<u64>> {
        let incoming_ids: FxHashSet<u64> = incoming_ids_map.keys().copied().collect();
        if incoming_ids.is_empty() {
            return Ok(FxHashSet::default());
        }

        let desc_pk_data = match catalog.map.get(&field_id) {
            Some(pk) => *pk,
            None => return Ok(FxHashSet::default()),
        };
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = match catalog.map.get(&rid_fid) {
            Some(pk) => *pk,
            None => return Ok(FxHashSet::default()),
        };

        // This is a TRUE BATCHING scenario.
        let gets = vec![
            BatchGet::Raw { key: desc_pk_data },
            BatchGet::Raw { key: desc_pk_rid },
        ];
        let results = self.pager.batch_get(&gets)?;
        let mut blobs_by_pk = FxHashMap::default();
        for res in results {
            if let GetResult::Raw { key, bytes } = res {
                blobs_by_pk.insert(key, bytes);
            }
        }

        let desc_blob_data = blobs_by_pk.remove(&desc_pk_data).ok_or(Error::NotFound)?;
        let mut descriptor_data = ColumnDescriptor::from_le_bytes(desc_blob_data.as_ref());

        let desc_blob_rid = blobs_by_pk.remove(&desc_pk_rid).ok_or(Error::NotFound)?;
        let mut descriptor_rid = ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref());

        let mut metas_data: Vec<ChunkMetadata> = Vec::new();
        let mut metas_rid: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor_data.head_page_pk) {
            metas_data.push(m?);
        }
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor_rid.head_page_pk) {
            metas_rid.push(m?);
        }

        let mut rewritten_ids = FxHashSet::default();
        let mut hit_chunks = FxHashMap::<usize, Vec<u64>>::default();
        let n = metas_data.len().min(metas_rid.len());
        if n > 0 {
            let mut gets_rid = Vec::with_capacity(n);
            for i in 0..n {
                gets_rid.push(BatchGet::Raw {
                    key: metas_rid[i].chunk_pk,
                });
            }
            let rid_results = self.pager.batch_get(&gets_rid)?;
            let mut rid_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            for r in rid_results {
                if let GetResult::Raw { key, bytes } = r {
                    rid_blobs.insert(key, bytes);
                }
            }

            for i in 0..n {
                if let Some(rid_blob) = rid_blobs.get(&metas_rid[i].chunk_pk) {
                    let rid_arr = deserialize_array(rid_blob.clone())?;
                    let rid_arr = rid_arr
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("rid type mismatch".into()))?;
                    for j in 0..rid_arr.len() {
                        let rid = rid_arr.value(j);
                        if incoming_ids.contains(&rid) {
                            hit_chunks.entry(i).or_default().push(rid);
                            rewritten_ids.insert(rid);
                        }
                    }
                }
            }
        }

        if hit_chunks.is_empty() {
            return Ok(rewritten_ids);
        }

        let hit_idxs: Vec<usize> = hit_chunks.keys().copied().collect();
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
            if let GetResult::Raw { key, bytes } = r {
                blob_map.insert(key, bytes);
            }
        }

        for (chunk_idx, rids_in_chunk) in hit_chunks {
            let rids_to_update: FxHashSet<u64> = rids_in_chunk.into_iter().collect();
            let old_data_arr = deserialize_array(
                blob_map
                    .get(&metas_data[chunk_idx].chunk_pk)
                    .unwrap()
                    .clone(),
            )?;
            let old_rid_arr_any = deserialize_array(
                blob_map
                    .get(&metas_rid[chunk_idx].chunk_pk)
                    .unwrap()
                    .clone(),
            )?;
            let old_rid_arr = old_rid_arr_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let keep_mask = BooleanArray::from_iter(
                (0..old_rid_arr.len())
                    .map(|j| Some(!rids_to_update.contains(&old_rid_arr.value(j)))),
            );
            let data_after_delete = compute::filter(&old_data_arr, &keep_mask)?;
            let rid_after_delete = compute::filter(&old_rid_arr_any, &keep_mask)?;

            // Arrow `take` requires UInt32 indices; build directly without an intermediate Vec
            let update_indices_arr =
                UInt32Array::from_iter_values(rids_to_update.iter().map(|rid| {
                    let idx = *incoming_ids_map.get(rid).expect("incoming id present");
                    u32::try_from(idx).expect("incoming batch index exceeds u32::MAX (arrow take)")
                }));

            let data_to_add = compute::take(incoming_data, &update_indices_arr, None)?;
            let rids_to_add = compute::take(incoming_row_ids, &update_indices_arr, None)?;
            let new_data_arr = concat_many(vec![&data_after_delete, &data_to_add])?;
            let new_rid_arr = concat_many(vec![&rid_after_delete, &rids_to_add])?;

            let new_data_bytes = serialize_array(&new_data_arr)?;
            let new_rid_bytes = serialize_array(&new_rid_arr)?;
            puts.push(BatchPut::Raw {
                key: metas_data[chunk_idx].chunk_pk,
                bytes: new_data_bytes,
            });
            puts.push(BatchPut::Raw {
                key: metas_rid[chunk_idx].chunk_pk,
                bytes: new_rid_bytes,
            });
            metas_data[chunk_idx].row_count = new_data_arr.len() as u64;
            metas_data[chunk_idx].serialized_bytes = new_data_arr.get_array_memory_size() as u64;
            metas_rid[chunk_idx].row_count = new_rid_arr.len() as u64;
            metas_rid[chunk_idx].serialized_bytes = new_rid_arr.get_array_memory_size() as u64;
        }

        self.rewrite_descriptor_pages(desc_pk_data, &mut descriptor_data, &mut metas_data, puts)?;
        self.rewrite_descriptor_pages(desc_pk_rid, &mut descriptor_rid, &mut metas_rid, puts)?;
        Ok(rewritten_ids)
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
            // This is a sequential walk, so a single-item batch is appropriate.
            let page_blob = self
                .pager
                .batch_get(&[BatchGet::Raw {
                    key: current_page_pk,
                }])?
                .pop()
                .and_then(|res| match res {
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

    /// Explicit delete by global row **positions** (0-based, ascending, unique) -> in-place chunk rewrite.
    /// Then bounded compaction of small adjacent chunks.
    ///
    /// Precondition: `rows_to_delete` yields strictly increasing `u64` positions.
    pub fn delete_rows<I>(&self, field_id: LogicalFieldId, rows_to_delete: I) -> Result<()>
    where
        I: IntoIterator<Item = u64>,
    {
        // Stream the iterator (no copies), while validating monotonicity.
        let mut del_iter = rows_to_delete.into_iter();
        let mut cur_del = del_iter.next();
        let mut last_seen: Option<u64> = None;
        if let Some(v) = cur_del {
            last_seen = Some(v);
        }

        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = catalog.map.get(&rid_fid).copied();

        // True batching: fetch both descriptors at once if they exist.
        let mut gets = vec![BatchGet::Raw { key: descriptor_pk }];
        if let Some(pk) = desc_pk_rid {
            gets.push(BatchGet::Raw { key: pk });
        }
        let results = self.pager.batch_get(&gets)?;
        let mut blobs_by_pk = FxHashMap::default();
        for res in results {
            if let GetResult::Raw { key, bytes } = res {
                blobs_by_pk.insert(key, bytes);
            }
        }

        let desc_blob = blobs_by_pk.remove(&descriptor_pk).ok_or(Error::NotFound)?;
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
        let mut metas_rid: Vec<ChunkMetadata> = Vec::new();
        let mut descriptor_rid: Option<ColumnDescriptor> = None;
        if let Some(pk) = desc_pk_rid {
            if let Some(desc_blob_rid) = blobs_by_pk.remove(&pk) {
                let d_rid = ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref());
                for m in DescriptorIterator::new(self.pager.as_ref(), d_rid.head_page_pk) {
                    metas_rid.push(m?);
                }
                descriptor_rid = Some(d_rid);
            }
        }

        let mut puts = Vec::new();
        let mut cum_rows = 0u64;

        for (i, meta) in metas.iter_mut().enumerate() {
            let start_u64 = cum_rows;
            let end_u64 = start_u64 + meta.row_count;

            // Advance deletes until they fall into [start, end).
            while let Some(d) = cur_del {
                if d < start_u64 {
                    // Enforce monotonicity (streaming check).
                    if let Some(prev) = last_seen {
                        if d < prev {
                            return Err(Error::Internal(
                                "rows_to_delete must be ascending/unique".into(),
                            ));
                        }
                    }
                    last_seen = Some(d);
                    cur_del = del_iter.next();
                } else {
                    break;
                }
            }

            // Local indices are per-chunk; track them in `usize` (native index type).
            let rows = meta.row_count as usize;

            let mut del_local: FxHashSet<usize> = FxHashSet::default();
            while let Some(d) = cur_del {
                if d >= end_u64 {
                    break;
                }
                // local index is (d - start); fits in usize because it < meta.row_count
                del_local.insert((d - start_u64) as usize);
                last_seen = Some(d);
                cur_del = del_iter.next();
            }

            if del_local.is_empty() {
                cum_rows = end_u64;
                continue;
            }

            let keep = BooleanArray::from_iter((0..rows).map(|j| Some(!del_local.contains(&j))));

            // Batch fetch the data and row_id chunks to be rewritten
            let mut chunk_gets = vec![BatchGet::Raw { key: meta.chunk_pk }];
            if let Some(rm) = metas_rid.get(i) {
                chunk_gets.push(BatchGet::Raw { key: rm.chunk_pk });
            }
            let chunk_results = self.pager.batch_get(&chunk_gets)?;
            let mut chunk_blobs = FxHashMap::default();
            for res in chunk_results {
                if let GetResult::Raw { key, bytes } = res {
                    chunk_blobs.insert(key, bytes);
                }
            }

            // Rewrite data chunk in place.
            let data_blob = chunk_blobs.remove(&meta.chunk_pk).ok_or(Error::NotFound)?;
            let data_arr = deserialize_array(data_blob)?;
            let data_f = compute::filter(&data_arr, &keep)?;
            let data_bytes = serialize_array(&data_f)?;
            puts.push(BatchPut::Raw {
                key: meta.chunk_pk,
                bytes: data_bytes,
            });
            meta.row_count = data_f.len() as u64;
            meta.serialized_bytes = data_f.get_array_memory_size() as u64;

            // Rewrite matching shadow row_id chunk, if present.
            if let (Some(_), Some(rm)) = (desc_pk_rid, metas_rid.get_mut(i)) {
                let rid_blob = chunk_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?;
                let rid_arr = deserialize_array(rid_blob)?;
                let rid_f = compute::filter(&rid_arr, &keep)?;
                let rid_bytes = serialize_array(&rid_f)?;
                puts.push(BatchPut::Raw {
                    key: rm.chunk_pk,
                    bytes: rid_bytes,
                });
                rm.row_count = rid_f.len() as u64;
                rm.serialized_bytes = rid_f.get_array_memory_size() as u64;
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
            let page_blob = self
                .pager
                .batch_get(&[BatchGet::Raw { key: pk }])?
                .pop()
                .and_then(|res| match res {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?;
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

        // True batching for the two descriptor reads.
        let gets = vec![
            BatchGet::Raw { key: desc_pk },
            BatchGet::Raw { key: desc_pk_rid },
        ];
        let results = self.pager.batch_get(&gets)?;
        let mut blobs_by_pk = FxHashMap::default();
        for res in results {
            if let GetResult::Raw { key, bytes } = res {
                blobs_by_pk.insert(key, bytes);
            }
        }
        let desc_blob = blobs_by_pk.remove(&desc_pk).ok_or(Error::NotFound)?;
        let mut desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        let desc_blob_rid = blobs_by_pk.remove(&desc_pk_rid).ok_or(Error::NotFound)?;
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
            let merged_data = concat_many(data_parts.iter().collect())?;
            let merged_rid_any = concat_many(rid_parts.iter().collect())?;

            // Split merged run into ~target-sized chunks.
            let slices = split_to_target_bytes(
                &merged_data,
                TARGET_CHUNK_BYTES,
                self.cfg.varwidth_fallback_rows_per_slice,
            );
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
                    bytes: data_bytes,
                });
                puts.push(BatchPut::Raw {
                    key: rid_pk,
                    bytes: rid_bytes,
                });
                let mut meta = ChunkMetadata {
                    chunk_pk: data_pk,
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: s_norm.get_array_memory_size() as u64,
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
                    value_order_perm_pk: 0,
                    row_count: rows as u64,
                    serialized_bytes: rid_norm.get_array_memory_size() as u64,
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
            &mut new_rid_metas,
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

    pub fn create_sort_index(&self, field_id: LogicalFieldId) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;

        let desc_blob = self
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
            let page_blob = self
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

            let header =
                DescriptorPageHeader::from_le_bytes(&page_blob[..DescriptorPageHeader::DISK_SIZE]);
            let chunks_on_this_page = header.entry_count as usize;
            let page_end_chunk_idx = page_start_chunk_idx + chunks_on_this_page;
            let updated_chunks = &all_chunk_metadata[page_start_chunk_idx..page_end_chunk_idx];
            let mut new_page_data = Vec::new();
            for chunk_meta in updated_chunks {
                new_page_data.extend_from_slice(&chunk_meta.to_le_bytes());
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
            page_start_chunk_idx = page_end_chunk_idx;
        }

        self.pager.batch_put(&puts)?;
        Ok(())
    }

    /// (Internal) Loads the full state for a descriptor, creating it if it doesn't exist.
    fn load_descriptor_state(
        &self,
        descriptor_pk: PhysicalKey,
        field_id: LogicalFieldId,
    ) -> Result<(ColumnDescriptor, Vec<u8>)> {
        match self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
        {
            Some(GetResult::Raw { bytes, .. }) => {
                let descriptor = ColumnDescriptor::from_le_bytes(bytes.as_ref());
                let tail_page_bytes = self
                    .pager
                    .batch_get(&[BatchGet::Raw {
                        key: descriptor.tail_page_pk,
                    }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?
                    .as_ref()
                    .to_vec();
                Ok((descriptor, tail_page_bytes))
            }
            _ => {
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

    /// Verifies the integrity of the column store's metadata.
    ///
    /// This check is useful for tests and debugging. It verifies:
    /// 1. The catalog can be read.
    /// 2. All descriptor chains are walkable.
    /// 3. The row and chunk counts in the descriptors match the sum of the chunk metadata.
    ///
    /// Returns `Ok(())` if consistent, otherwise returns an `Error`.
    pub fn verify_integrity(&self) -> Result<()> {
        let catalog = self.catalog.read().unwrap();
        for (&field_id, &descriptor_pk) in &catalog.map {
            let desc_blob = self
                .pager
                .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "Catalog points to missing descriptor pk={}",
                        descriptor_pk
                    ))
                })?;
            let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
            if descriptor.field_id != field_id {
                return Err(Error::Internal(format!(
                    "Descriptor at pk={} has wrong field_id: expected {:?}, got {:?}",
                    descriptor_pk, field_id, descriptor.field_id
                )));
            }

            let mut actual_rows = 0;
            let mut actual_chunks = 0;
            let mut current_page_pk = descriptor.head_page_pk;
            while current_page_pk != 0 {
                let page_blob = self
                    .pager
                    .batch_get(&[BatchGet::Raw {
                        key: current_page_pk,
                    }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Descriptor page chain broken at pk={}",
                            current_page_pk
                        ))
                    })?;
                let header = DescriptorPageHeader::from_le_bytes(
                    &page_blob.as_ref()[..DescriptorPageHeader::DISK_SIZE],
                );
                for i in 0..(header.entry_count as usize) {
                    let off = DescriptorPageHeader::DISK_SIZE + i * ChunkMetadata::DISK_SIZE;
                    let end = off + ChunkMetadata::DISK_SIZE;
                    let meta = ChunkMetadata::from_le_bytes(&page_blob.as_ref()[off..end]);
                    actual_rows += meta.row_count;
                    actual_chunks += 1;
                }
                current_page_pk = header.next_page_pk;
            }

            if descriptor.total_row_count != actual_rows {
                return Err(Error::Internal(format!(
                    "Row count mismatch for field {:?}: descriptor says {}, actual is {}",
                    field_id, descriptor.total_row_count, actual_rows
                )));
            }
            if descriptor.total_chunk_count != actual_chunks {
                return Err(Error::Internal(format!(
                    "Chunk count mismatch for field {:?}: descriptor says {}, actual is {}",
                    field_id, descriptor.total_chunk_count, actual_chunks
                )));
            }
        }
        Ok(())
    }

    /// Gathers detailed statistics about the storage layout.
    ///
    /// This method is designed for low-level analysis and debugging, allowing you to
    /// check for under- or over-utilization of descriptor pages.
    pub fn get_layout_stats(&self) -> Result<Vec<ColumnLayoutStats>> {
        let catalog = self.catalog.read().unwrap();
        let mut all_stats = Vec::new();

        for (&field_id, &descriptor_pk) in &catalog.map {
            let desc_blob = self
                .pager
                .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?;
            let descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

            let mut page_stats = Vec::new();
            let mut current_page_pk = descriptor.head_page_pk;
            while current_page_pk != 0 {
                let page_blob = self
                    .pager
                    .batch_get(&[BatchGet::Raw {
                        key: current_page_pk,
                    }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?;
                let header = DescriptorPageHeader::from_le_bytes(
                    &page_blob.as_ref()[..DescriptorPageHeader::DISK_SIZE],
                );
                page_stats.push(DescriptorPageStats {
                    page_pk: current_page_pk,
                    entry_count: header.entry_count,
                    page_size_bytes: page_blob.as_ref().len(),
                });
                current_page_pk = header.next_page_pk;
            }

            all_stats.push(ColumnLayoutStats {
                field_id,
                total_rows: descriptor.total_row_count,
                total_chunks: descriptor.total_chunk_count,
                pages: page_stats,
            });
        }
        Ok(all_stats)
    }
}
