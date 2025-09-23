use super::*;
use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::types::{CATALOG_ROOT_PKEY, LogicalFieldId, PhysicalKey};
use arrow::array::{Array, ArrayRef, BooleanArray, UInt32Array, UInt64Array};
use arrow::compute::{self, SortColumn, lexsort_to_indices};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

pub struct ColumnStore<P: Pager> {
    pub(crate) pager: Arc<P>,
    pub(crate) catalog: Arc<RwLock<ColumnCatalog>>,
    cfg: ColumnStoreConfig,
    dtype_cache: DTypeCache<P>,
    index_manager: IndexManager<P>,
}

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn open(pager: Arc<P>) -> Result<Self> {
        let cfg = ColumnStoreConfig::default();

        let catalog = match pager
            .batch_get(&[BatchGet::Raw {
                key: CATALOG_ROOT_PKEY,
            }])?
            .pop()
        {
            Some(GetResult::Raw { bytes, .. }) => ColumnCatalog::from_bytes(bytes.as_ref())?,
            _ => ColumnCatalog::default(),
        };

        let arc_catalog = Arc::new(RwLock::new(catalog));

        let index_manager = IndexManager::new(Arc::clone(&pager), Arc::clone(&arc_catalog)); // Add this line

        Ok(Self {
            pager: Arc::clone(&pager),
            catalog: Arc::clone(&arc_catalog),
            cfg,
            dtype_cache: DTypeCache::new(Arc::clone(&pager), Arc::clone(&arc_catalog)),
            index_manager,
        })
    }

    /// Registers an index for a given column, building it for existing data.
    pub fn register_index(&self, field_id: LogicalFieldId, kind: IndexKind) -> Result<()> {
        // First, build the physical index data for any data already in the column.
        self.index_manager.build_index(kind, field_id)?;

        // Then, register its name in the column descriptor so future operations know about it.
        self.index_manager.register_index(field_id, kind)
    }

    /// Unregisters a persisted index from a given column.
    pub fn unregister_index(&self, field_id: LogicalFieldId, kind: IndexKind) -> Result<()> {
        self.index_manager.unregister_index(field_id, kind)
    }

    /// Lists the names of all persisted indexes for a given column.
    pub fn list_persisted_indexes(&self, field_id: LogicalFieldId) -> Result<Vec<IndexKind>> {
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

        // Get the IndexKind enums and convert them to Strings for the caller.
        let kinds = descriptor.get_indexes()?;
        Ok(kinds)
    }

    /// Fast presence check using the presence index (row-id permutation) if available.
    /// Returns true if `row_id` exists in the column; false otherwise.
    pub fn has_row_id(&self, field_id: LogicalFieldId, row_id: u64) -> Result<bool> {
        let rid_fid = rowid_fid(field_id);
        let catalog = self.catalog.read().unwrap();
        let rid_desc_pk = *catalog.map.get(&rid_fid).ok_or(Error::NotFound)?;
        let rid_desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: rid_desc_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let rid_desc = ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());
        drop(catalog);

        // Walk metas; prune by min/max when available.
        for m in DescriptorIterator::new(self.pager.as_ref(), rid_desc.head_page_pk) {
            let meta = m?;
            if meta.row_count == 0 {
                continue;
            }
            if (meta.min_val_u64 != 0 || meta.max_val_u64 != 0) && row_id < meta.min_val_u64
                || row_id > meta.max_val_u64
            {
                continue;
            }
            // Fetch rid chunk and, if present, the presence perm
            let mut gets = vec![BatchGet::Raw { key: meta.chunk_pk }];
            if meta.value_order_perm_pk != 0 {
                gets.push(BatchGet::Raw {
                    key: meta.value_order_perm_pk,
                });
            }
            let results = self.pager.batch_get(&gets)?;
            let mut rid_blob: Option<EntryHandle> = None;
            let mut perm_blob: Option<EntryHandle> = None;
            for r in results {
                if let GetResult::Raw { key, bytes } = r {
                    if key == meta.chunk_pk {
                        rid_blob = Some(bytes);
                    } else if key == meta.value_order_perm_pk {
                        perm_blob = Some(bytes);
                    }
                }
            }
            // If the rid blob for this chunk is missing, treat as absent and continue
            let Some(rid_blob) = rid_blob else { continue };
            let rid_any = deserialize_array(rid_blob)?;
            let rids = rid_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("rid downcast".into()))?;
            if let Some(pblob) = perm_blob {
                let perm_any = deserialize_array(pblob)?;
                let perm = perm_any
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                // Binary search over sorted-by-perm view
                let mut lo: isize = 0;
                let mut hi: isize = (perm.len() as isize) - 1;
                while lo <= hi {
                    let mid = ((lo + hi) >> 1) as usize;
                    let rid = rids.value(perm.value(mid) as usize);
                    if rid == row_id {
                        return Ok(true);
                    } else if rid < row_id {
                        lo = mid as isize + 1;
                    } else {
                        hi = mid as isize - 1;
                    }
                }
            } else {
                // Assume rid chunk is sorted ascending (common for appends/compaction) and binary search
                let mut lo: isize = 0;
                let mut hi: isize = (rids.len() as isize) - 1;
                while lo <= hi {
                    let mid = ((lo + hi) >> 1) as usize;
                    let rid = rids.value(mid);
                    if rid == row_id {
                        return Ok(true);
                    } else if rid < row_id {
                        lo = mid as isize + 1;
                    } else {
                        hi = mid as isize - 1;
                    }
                }
            }
        }
        Ok(false)
    }

    #[allow(unused_variables, unused_assignments)] // TODO: Keep `presence_index_created`?
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        // Ensure we append rows in ascending row_id order to keep row_id chunks
        // naturally sorted and avoid building presence permutations later.
        // Do this up front on the incoming batch.
        let working_batch: RecordBatch;
        let batch = {
            let schema = batch.schema();
            let row_id_idx = schema
                .index_of("row_id")
                .map_err(|_| Error::Internal("row_id column required".into()))?;
            let row_id_any = batch.column(row_id_idx).clone();
            let row_id_arr = row_id_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id downcast failed".into()))?;
            // Quick ascending check
            let mut sorted = true;
            let mut prev = 0u64;
            for i in 0..row_id_arr.len() {
                let v = row_id_arr.value(i);
                if i > 0 && v < prev {
                    sorted = false;
                    break;
                }
                prev = v;
            }
            if sorted {
                batch.clone()
            } else {
                // Build sort indices by row_id and reorder all columns
                let sort_col = SortColumn {
                    values: row_id_any,
                    options: None,
                };
                let idx = lexsort_to_indices(&[sort_col], None)?;
                let perm = idx
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Internal("perm not u32".into()))?;
                let mut cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
                for i in 0..batch.num_columns() {
                    cols.push(compute::take(batch.column(i), perm, None)?);
                }
                working_batch = RecordBatch::try_new(schema.clone(), cols)
                    .map_err(|e| Error::Internal(format!("record batch rebuild: {e}")))?;
                working_batch
            }
        };

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
            compute::filter_record_batch(&batch, &keep_array)?
        } else {
            batch.clone()
        };
        if batch_to_append.num_rows() == 0 {
            return Ok(());
        }

        // Now, proceed with appending the correctly filtered batch.
        // Drop nulls per field (treat null as delete: do not store it).
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

            // Warm the dtype cache from schema; avoids a later chunk peek in this process.
            self.dtype_cache.insert(field_id, field.data_type().clone());

            // Filter out nulls in this column and keep matching row_ids only.
            let (array_clean, rids_clean) = if array.null_count() == 0 {
                (array.clone(), append_row_id_any.clone())
            } else {
                let keep =
                    BooleanArray::from_iter((0..array.len()).map(|j| Some(!array.is_null(j))));
                let a = compute::filter(array, &keep)?;
                let r = compute::filter(&append_row_id_any, &keep)?;
                (a, r)
            };
            if array_clean.is_empty() {
                continue;
            }

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
                ColumnDescriptor::load_or_create(Arc::clone(&self.pager), descriptor_pk, field_id)?;
            let (mut rid_descriptor, mut rid_tail_page) = ColumnDescriptor::load_or_create(
                Arc::clone(&self.pager),
                rid_descriptor_pk,
                rid_fid,
            )?;

            // TODO: Remove?
            // Persist/refresh dtype fingerprint for both data and row-id descriptors.
            // let fp_data = DTypeCache::<P>::dtype_fingerprint(field.data_type());
            // DTypeCache::<P>::set_desc_dtype_fingerprint(&mut data_descriptor, fp_data);
            // let fp_rid = DTypeCache::<P>::dtype_fingerprint(&DataType::UInt64);
            // DTypeCache::<P>::set_desc_dtype_fingerprint(&mut rid_descriptor, fp_rid);

            // Slice and append.
            let slices = split_to_target_bytes(
                &array_clean,
                TARGET_CHUNK_BYTES,
                self.cfg.varwidth_fallback_rows_per_slice,
            );
            let mut row_off = 0usize;

            let mut presence_index_created = false;

            for s in slices {
                let rows = s.len();

                // Data slice.
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

                // Row-id slice matched to this data slice.
                let rid_slice: ArrayRef = rids_clean.slice(row_off, rows);
                let rid_norm = zero_offset(&rid_slice);
                let rid_pk = self.pager.alloc_many(1)?[0];
                let rid_bytes = serialize_array(rid_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: rid_pk,
                    bytes: rid_bytes,
                });
                // Compute min/max and detect if row_ids are already sorted ascending
                let rid_any = rid_norm.clone();
                let rids = rid_any
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
                    presence_index_created = true;
                    // Build presence index (perm over row_ids)
                    let sort_col = SortColumn {
                        values: rid_any,
                        options: None,
                    };
                    let rid_idx = lexsort_to_indices(&[sort_col], None)?;
                    let perm_bytes = serialize_array(&rid_idx)?;
                    rid_perm_pk = self.pager.alloc_many(1)?[0];
                    puts_appends.push(BatchPut::Raw {
                        key: rid_perm_pk,
                        bytes: perm_bytes,
                    });
                }
                let rid_meta = ChunkMetadata {
                    chunk_pk: rid_pk,
                    value_order_perm_pk: rid_perm_pk,
                    row_count: rows as u64,
                    serialized_bytes: rid_norm.get_array_memory_size() as u64,
                    min_val_u64: if rows > 0 { min } else { 0 },
                    max_val_u64: if rows > 0 { max } else { 0 },
                };
                self.append_meta_in_loop(
                    &mut rid_descriptor,
                    &mut rid_tail_page,
                    rid_meta,
                    &mut puts_appends,
                )?;
                row_off += rows;
            }

            // TODO: Improve
            // A column with a row_id shadow always has a presence index.
            let mut indexes = data_descriptor.get_indexes()?;
            if !indexes.contains(&IndexKind::Presence) {
                indexes.push(IndexKind::Presence);
                data_descriptor.set_indexes(&indexes)?;
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
        use crate::store::descriptor::DescriptorIterator;
        use crate::store::ingest::ChunkEdit;

        // Fast exit if nothing to rewrite.
        if incoming_ids_map.is_empty() {
            return Ok(FxHashSet::default());
        }
        let incoming_ids: FxHashSet<u64> = incoming_ids_map.keys().copied().collect();

        // Resolve descriptors for data and row_id columns.
        let desc_pk_data = match catalog.map.get(&field_id) {
            Some(pk) => *pk,
            None => return Ok(FxHashSet::default()),
        };
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = match catalog.map.get(&rid_fid) {
            Some(pk) => *pk,
            None => return Ok(FxHashSet::default()),
        };

        // Batch fetch both descriptors.
        let gets = vec![
            BatchGet::Raw { key: desc_pk_data },
            BatchGet::Raw { key: desc_pk_rid },
        ];
        let results = self.pager.batch_get(&gets)?;
        let mut blobs_by_pk = FxHashMap::default();
        for r in results {
            if let GetResult::Raw { key, bytes } = r {
                blobs_by_pk.insert(key, bytes);
            }
        }

        let desc_blob_data = blobs_by_pk.remove(&desc_pk_data).ok_or(Error::NotFound)?;
        let mut descriptor_data = ColumnDescriptor::from_le_bytes(desc_blob_data.as_ref());

        let desc_blob_rid = blobs_by_pk.remove(&desc_pk_rid).ok_or(Error::NotFound)?;
        let mut descriptor_rid = ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref());

        // Collect chunk metadata.
        let mut metas_data: Vec<ChunkMetadata> = Vec::new();
        let mut metas_rid: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor_data.head_page_pk) {
            metas_data.push(m?);
        }
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor_rid.head_page_pk) {
            metas_rid.push(m?);
        }

        // Classify incoming rows: delete vs upsert.
        let rid_in = incoming_row_ids
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("row_id must be u64".into()))?;
        let mut ids_to_delete = FxHashSet::default();
        let mut ids_to_upsert = FxHashSet::default();
        for i in 0..rid_in.len() {
            let rid = rid_in.value(i);
            if incoming_data.is_null(i) {
                ids_to_delete.insert(rid);
            } else {
                ids_to_upsert.insert(rid);
            }
        }

        // Scan row_id chunks to find hits, bucketed by chunk index.
        let mut rewritten_ids = FxHashSet::default();
        let mut hit_up: FxHashMap<usize, Vec<u64>> = FxHashMap::default();
        let mut hit_del: FxHashMap<usize, Vec<u64>> = FxHashMap::default();

        let n = metas_data.len().min(metas_rid.len());
        if n > 0 {
            // Fetch only row_id chunks to locate matches.
            let mut gets_rid = Vec::with_capacity(n);
            for rm in metas_rid.iter().take(n) {
                gets_rid.push(BatchGet::Raw { key: rm.chunk_pk });
            }
            let rid_results = self.pager.batch_get(&gets_rid)?;
            let mut rid_blobs: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
            for r in rid_results {
                if let GetResult::Raw { key, bytes } = r {
                    rid_blobs.insert(key, bytes);
                }
            }

            for (i, meta_rid) in metas_rid.iter().enumerate().take(n) {
                if let Some(rid_blob) = rid_blobs.get(&meta_rid.chunk_pk) {
                    let rid_arr_any = deserialize_array(rid_blob.clone())?;
                    let rid_arr = rid_arr_any
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Internal("rid type mismatch".into()))?;
                    for j in 0..rid_arr.len() {
                        let rid = rid_arr.value(j);
                        if incoming_ids.contains(&rid) {
                            if ids_to_delete.contains(&rid) {
                                hit_del.entry(i).or_default().push(rid);
                            } else if ids_to_upsert.contains(&rid) {
                                hit_up.entry(i).or_default().push(rid);
                            }
                            rewritten_ids.insert(rid);
                        }
                    }
                }
            }
        }

        if hit_up.is_empty() && hit_del.is_empty() {
            return Ok(rewritten_ids);
        }

        // Batch fetch data+rid blobs for all chunks with hits.
        let mut hit_set = FxHashSet::default();
        hit_set.extend(hit_up.keys().copied());
        hit_set.extend(hit_del.keys().copied());
        let hit_idxs: Vec<usize> = hit_set.into_iter().collect();

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

        // Apply per-chunk edits and stage writebacks.
        for i in hit_idxs {
            let old_data_arr =
                deserialize_array(blob_map.get(&metas_data[i].chunk_pk).unwrap().clone())?;
            let old_rid_arr_any =
                deserialize_array(blob_map.get(&metas_rid[i].chunk_pk).unwrap().clone())?;
            let old_rid_arr = old_rid_arr_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();

            let up_vec = hit_up.remove(&i).unwrap_or_default();
            let del_vec = hit_del.remove(&i).unwrap_or_default();

            // Centralized LWW edit: builds keep mask and inject tails.
            let edit = ChunkEdit::from_lww_upsert(
                old_rid_arr,
                &up_vec,
                &del_vec,
                incoming_data,
                incoming_row_ids,
                incoming_ids_map,
            )?;

            let (new_data_arr, new_rid_arr) =
                ChunkEdit::apply_edit_to_arrays(&old_data_arr, Some(&old_rid_arr_any), &edit)?;

            // Stage data writeback.
            let data_bytes = serialize_array(&new_data_arr)?;
            puts.push(BatchPut::Raw {
                key: metas_data[i].chunk_pk,
                bytes: data_bytes,
            });
            metas_data[i].row_count = new_data_arr.len() as u64;
            metas_data[i].serialized_bytes = new_data_arr.get_array_memory_size() as u64;

            // Stage row_id writeback.
            if let Some(rarr) = new_rid_arr {
                let rid_bytes = serialize_array(&rarr)?;
                puts.push(BatchPut::Raw {
                    key: metas_rid[i].chunk_pk,
                    bytes: rid_bytes,
                });
                metas_rid[i].row_count = rarr.len() as u64;
                metas_rid[i].serialized_bytes = rarr.get_array_memory_size() as u64;
            }

            // Refresh permutation if present.
            if metas_data[i].value_order_perm_pk != 0 {
                let sort_col = SortColumn {
                    values: new_data_arr,
                    options: None,
                };
                let idx = lexsort_to_indices(&[sort_col], None)?;
                let perm_bytes = serialize_array(&idx)?;
                puts.push(BatchPut::Raw {
                    key: metas_data[i].value_order_perm_pk,
                    bytes: perm_bytes,
                });
            }
        }

        // Rewrite descriptor chains/totals for both columns.
        descriptor_data.rewrite_pages(
            Arc::clone(&self.pager),
            desc_pk_data,
            &mut metas_data,
            puts,
        )?;
        descriptor_rid.rewrite_pages(Arc::clone(&self.pager), desc_pk_rid, &mut metas_rid, puts)?;

        Ok(rewritten_ids)
    }

    // TODO: Remove pre-condition; accept vector of row ids instead
    /// Explicit delete by global row **positions** (0-based, ascending,
    /// unique) -> in-place chunk rewrite.
    ///
    /// Precondition: `rows_to_delete` yields strictly increasing positions.
    /// Explicit delete by global row **positions** (0-based, ascending,
    /// unique) -> in-place chunk rewrite.
    pub fn delete_rows<I>(&self, field_id: LogicalFieldId, rows_to_delete: I) -> Result<()>
    where
        I: IntoIterator<Item = u64>,
    {
        use crate::store::descriptor::DescriptorIterator;
        use crate::store::ingest::ChunkEdit;

        // Stream and validate ascending, unique positions.
        let mut del_iter = rows_to_delete.into_iter();
        let mut cur_del = del_iter.next();
        let mut last_seen: Option<u64> = None;
        if let Some(v) = cur_del {
            last_seen = Some(v);
        }

        // Lookup descriptors (data and optional row_id).
        let catalog = self.catalog.read().unwrap();
        let desc_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let rid_fid = rowid_fid(field_id);
        let desc_pk_rid = catalog.map.get(&rid_fid).copied();

        // Batch fetch descriptor blobs up front.
        let mut gets = vec![BatchGet::Raw { key: desc_pk }];
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

        let desc_blob = blobs_by_pk.remove(&desc_pk).ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // Build metas for data column.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            drop(catalog);
            self.compact_field_bounded(field_id)?;
            return Ok(());
        }

        // Optionally mirror metas for row_id column.
        let mut metas_rid: Vec<ChunkMetadata> = Vec::new();
        let mut descriptor_rid: Option<ColumnDescriptor> = None;
        if let Some(pk_rid) = desc_pk_rid
            && let Some(desc_blob_rid) = blobs_by_pk.remove(&pk_rid)
        {
            let d_rid = ColumnDescriptor::from_le_bytes(desc_blob_rid.as_ref());
            for m in DescriptorIterator::new(self.pager.as_ref(), d_rid.head_page_pk) {
                metas_rid.push(m?);
            }
            descriptor_rid = Some(d_rid);
        }

        let mut puts = Vec::new();
        let mut cum_rows = 0u64;

        for (i, meta) in metas.iter_mut().enumerate() {
            let start_u64 = cum_rows;
            let end_u64 = start_u64 + meta.row_count;

            // Advance deletes into this chunk window [start, end).
            while let Some(d) = cur_del {
                if d < start_u64
                    && let Some(prev) = last_seen
                {
                    if d < prev {
                        return Err(Error::Internal(
                            "rows_to_delete must be ascending/unique".into(),
                        ));
                    }

                    last_seen = Some(d);
                    cur_del = del_iter.next();
                } else {
                    break;
                }
            }

            // Collect local delete indices.
            let rows = meta.row_count as usize;
            let mut del_local: FxHashSet<usize> = FxHashSet::default();
            while let Some(d) = cur_del {
                if d >= end_u64 {
                    break;
                }
                del_local.insert((d - start_u64) as usize);
                last_seen = Some(d);
                cur_del = del_iter.next();
            }

            if del_local.is_empty() {
                cum_rows = end_u64;
                continue;
            }

            // Batch get chunk blobs (data and optional row_id).
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

            let data_blob = chunk_blobs.remove(&meta.chunk_pk).ok_or(Error::NotFound)?;
            let data_arr = deserialize_array(data_blob)?;

            let rid_arr_any = if let Some(rm) = metas_rid.get(i) {
                let rid_blob = chunk_blobs.remove(&rm.chunk_pk).ok_or(Error::NotFound)?;
                Some(deserialize_array(rid_blob)?)
            } else {
                None
            };

            // *** Wired: build edit via ingest helper.
            let edit = ChunkEdit::from_delete_indices(rows, &del_local);

            // Apply edit (pure array ops).
            let (new_data_arr, new_rid_arr) =
                ChunkEdit::apply_edit_to_arrays(&data_arr, rid_arr_any.as_ref(), &edit)?;

            // Write back data.
            let data_bytes = serialize_array(&new_data_arr)?;
            puts.push(BatchPut::Raw {
                key: meta.chunk_pk,
                bytes: data_bytes,
            });
            meta.row_count = new_data_arr.len() as u64;
            meta.serialized_bytes = new_data_arr.get_array_memory_size() as u64;

            // Write back row_ids if present.
            if let (Some(_), Some(rids)) = (metas_rid.get_mut(i), new_rid_arr) {
                let rm = metas_rid.get_mut(i).unwrap();
                let rid_bytes = serialize_array(&rids)?;
                puts.push(BatchPut::Raw {
                    key: rm.chunk_pk,
                    bytes: rid_bytes,
                });
                rm.row_count = rids.len() as u64;
                rm.serialized_bytes = rids.get_array_memory_size() as u64;
            }

            // Refresh permutation if this chunk has one.
            if meta.value_order_perm_pk != 0 {
                let sort_column = SortColumn {
                    values: new_data_arr,
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

        // Rewrite descriptor chains/totals and commit.
        descriptor.rewrite_pages(Arc::clone(&self.pager), desc_pk, &mut metas, &mut puts)?;
        if let (Some(rid_pk), Some(mut rid_desc)) = (desc_pk_rid, descriptor_rid) {
            rid_desc.rewrite_pages(Arc::clone(&self.pager), rid_pk, &mut metas_rid, &mut puts)?;
        }
        if !puts.is_empty() {
            self.pager.batch_put(&puts)?;
        }

        drop(catalog);
        self.compact_field_bounded(field_id)
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
            new_metas.len().div_ceil(per)
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
            let need_perms = metas[i..j].iter().any(|m| m.value_order_perm_pk != 0);

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

                // Build presence index for rids and min/max
                let rid_any = rid_norm.clone();
                let rids = rid_any
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("rid downcast".into()))?;
                let mut min = u64::MAX;
                let mut max = 0u64;
                let mut sorted_rids = true;
                let mut last_v = 0u64;
                for ii in 0..rids.len() {
                    let v = rids.value(ii);
                    if ii == 0 {
                        last_v = v;
                    } else if v < last_v {
                        sorted_rids = false;
                    } else {
                        last_v = v;
                    }
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                let mut rid_perm_pk = 0u64;
                if !sorted_rids {
                    let rid_sort_col = SortColumn {
                        values: rid_any,
                        options: None,
                    };
                    let rid_idx = lexsort_to_indices(&[rid_sort_col], None)?;
                    let rid_perm_bytes = serialize_array(&rid_idx)?;
                    rid_perm_pk = self.pager.alloc_many(1)?[0];
                    puts.push(BatchPut::Raw {
                        key: rid_perm_pk,
                        bytes: rid_perm_bytes,
                    });
                }
                let rid_meta = ChunkMetadata {
                    chunk_pk: rid_pk,
                    value_order_perm_pk: rid_perm_pk,
                    row_count: rows as u64,
                    serialized_bytes: rid_norm.get_array_memory_size() as u64,
                    min_val_u64: if rows > 0 { min } else { 0 },
                    max_val_u64: if rows > 0 { max } else { 0 },
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
                if metas_rid[k].value_order_perm_pk != 0 {
                    frees.push(metas_rid[k].value_order_perm_pk);
                }
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

    /// (Internal) Helper for batch appends. Appends metadata to the current
    /// in-memory tail page, creating a new one if necessary.
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
            // --- PERFORMANCE FIX: Move the full page's bytes instead of
            // cloning ---
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
    /// 3. The row and chunk counts in the descriptors match the sum of the
    ///    chunk metadata.
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
                    "Descriptor at pk={} has wrong field_id: expected {:?}, \
                     got {:?}",
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
                    "Row count mismatch for field {:?}: descriptor says {}, \
                     actual is {}",
                    field_id, descriptor.total_row_count, actual_rows
                )));
            }
            if descriptor.total_chunk_count != actual_chunks {
                return Err(Error::Internal(format!(
                    "Chunk count mismatch for field {:?}: descriptor says {}, \
                     actual is {}",
                    field_id, descriptor.total_chunk_count, actual_chunks
                )));
            }
        }
        Ok(())
    }

    /// Gathers detailed statistics about the storage layout.
    ///
    /// This method is designed for low-level analysis and debugging, allowing
    /// you to check for under- or over-utilization of descriptor pages.
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
