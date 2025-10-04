use super::*;
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::store::scan::filter::FilterDispatch;
use crate::store::scan::{FilterPrimitive, FilterResult};
use crate::types::{LogicalFieldId, Namespace, RowId, TableId};
use arrow::array::{Array, ArrayRef, BooleanArray, UInt32Array, UInt64Array};
use arrow::compute::{self, SortColumn, lexsort_to_indices};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use llkv_expr::typed_predicate::Predicate;
use llkv_result::{Error, Result};
use llkv_storage::{
    constants::CATALOG_ROOT_PKEY,
    pager::{BatchGet, BatchPut, GetResult, Pager},
    serialization::{deserialize_array, serialize_array},
    types::PhysicalKey,
};

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

/// Convenience alias for describing batched delete operations.
pub type DeleteRowsSpec = (LogicalFieldId, Vec<RowId>);

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

        let index_manager = IndexManager::new(Arc::clone(&pager));

        Ok(Self {
            pager: Arc::clone(&pager),
            catalog: Arc::clone(&arc_catalog),
            cfg,
            dtype_cache: DTypeCache::new(Arc::clone(&pager), Arc::clone(&arc_catalog)),
            index_manager,
        })
    }

    /// Registers an index for a given column, building it for existing data atomically and with low memory usage.
    pub fn register_index(&self, field_id: LogicalFieldId, kind: IndexKind) -> Result<()> {
        self.index_manager.register_index(self, field_id, kind)
    }

    /// Unregisters a persisted index from a given column atomically.
    pub fn unregister_index(&self, field_id: LogicalFieldId, kind: IndexKind) -> Result<()> {
        self.index_manager.unregister_index(self, field_id, kind)
    }

    /// Returns the Arrow data type of the given field, loading it if not cached.
    pub fn data_type(&self, field_id: LogicalFieldId) -> Result<DataType> {
        if let Some(dt) = self.dtype_cache.cached_data_type(field_id) {
            return Ok(dt);
        }
        self.dtype_cache.dtype_for_field(field_id)
    }

    /// Collects the row ids whose values satisfy the provided predicate.
    pub fn filter_row_ids<T>(
        &self,
        field_id: LogicalFieldId,
        predicate: &Predicate<T::Value>,
    ) -> Result<Vec<u64>>
    where
        T: FilterDispatch,
    {
        T::run_filter(self, field_id, predicate)
    }

    pub fn filter_matches<T, F>(
        &self,
        field_id: LogicalFieldId,
        predicate: F,
    ) -> Result<FilterResult>
    where
        T: FilterPrimitive,
        F: FnMut(T::Native) -> bool,
    {
        T::run_filter_with_result(self, field_id, predicate)
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

        let kinds = descriptor.get_indexes()?;
        Ok(kinds)
    }

    /// Return the total number of rows recorded for the given logical field.
    ///
    /// This reads the ColumnDescriptor.total_row_count stored in the catalog and
    /// descriptor pages and returns it as a u64. The value is updated by append
    /// / delete code paths and represents the persisted row count for that column.
    pub fn total_rows_for_field(&self, field_id: LogicalFieldId) -> Result<u64> {
        let catalog = self.catalog.read().unwrap();
        let desc_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        drop(catalog);

        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: desc_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;

        let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
        Ok(desc.total_row_count)
    }

    /// Return the total number of rows for a table by inspecting any persisted
    /// user-data column descriptor belonging to `table_id`.
    ///
    /// This method picks the first user-data column found in the in-memory
    /// catalog for the table and returns its descriptor.total_row_count. If the
    /// table has no persisted columns, `Ok(0)` is returned.
    pub fn total_rows_for_table(&self, table_id: crate::types::TableId) -> Result<u64> {
        use crate::types::Namespace;

        // Prefer the dedicated table-level row-id shadow column if present.
        let rid_fid = table_rowid_fid(table_id);
        {
            let catalog = self.catalog.read().unwrap();
            if let Some(desc_pk) = catalog.map.get(&rid_fid) {
                let desc_blob = self
                    .pager
                    .batch_get(&[BatchGet::Raw { key: *desc_pk }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?;
                let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
                return Ok(desc.total_row_count);
            }
        }

        // Fall back to scanning any user-data column descriptors.
        let catalog = self.catalog.read().unwrap();
        let candidates: Vec<LogicalFieldId> = catalog
            .map
            .keys()
            .filter(|fid| fid.namespace() == Namespace::UserData && fid.table_id() == table_id)
            .copied()
            .collect();
        drop(catalog);

        if candidates.is_empty() {
            return Ok(0);
        }

        let mut max_rows: u64 = 0;
        for field in candidates {
            let rows = self.total_rows_for_field(field)?;
            if rows > max_rows {
                max_rows = rows;
            }
        }
        Ok(max_rows)
    }

    /// Return the logical field identifiers for all user columns in `table_id`.
    pub fn user_field_ids_for_table(&self, table_id: crate::types::TableId) -> Vec<LogicalFieldId> {
        use crate::types::Namespace;

        let catalog = self.catalog.read().unwrap();
        catalog
            .map
            .keys()
            .filter(|fid| fid.namespace() == Namespace::UserData && fid.table_id() == table_id)
            .copied()
            .collect()
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

    // TODO: Set up a way to optionally auto-increment a row ID column if not provided.

    /// NOTE (logical table separation):
    ///
    /// The ColumnStore stores data per logical field (a LogicalFieldId) rather
    /// than per 'table' directly. A LogicalFieldId encodes a namespace and a
    /// table id together (see [crate::types::LogicalFieldId]). This design lets
    /// multiple tables share the same underlying storage layer without
    /// collisions because every persisted column descriptor, chunk chain and
    /// presence/permutation index is keyed by the full LogicalFieldId.
    ///
    /// Concretely:
    /// - `catalog.map` maps each LogicalFieldId -> descriptor physical key .
    /// - Each ColumnDescriptor contains chunk metadata and a persisted
    ///   `total_row_count` for that logical field only.
    /// - Row-id shadow columns (presence/permutation indices) are created per
    ///   logical field (they are derived from the LogicalFieldId) and are used
    ///   for fast existence checks and indexed gathers.
    ///
    /// The `append` implementation below expects incoming RecordBatches to be
    /// namespaced: each field's metadata contains a "field_id" that, when
    /// combined with the current table id, maps to the LogicalFieldId the
    /// ColumnStore will append into. This ensures that appends for different
    /// tables never overwrite each other's descriptors or chunks because the
    /// LogicalFieldId namespace keeps them separate.
    ///
    /// Constraints and recommended usage:
    /// - A single `RecordBatch` carries exactly one `ROW_ID` column which is
    ///   shared by all other columns in that batch. Because of this, a batch
    ///   is implicitly tied to a single table's row-id space and should not
    ///   mix columns from different tables.
    /// - To append data for multiple tables, create separate `RecordBatch`
    ///   instances (one per table) and call `append` for each. These calls
    ///   may be executed concurrently for throughput; the ColumnStore persists
    ///   data per-logical-field so physical writes won't collide.
    /// - If you need an atomic multi-table append, the current API does not
    ///   provide that; you'd need a higher-level coordinator that issues the
    ///   per-table appends within a single transaction boundary (or extend the
    ///   API to accept per-column row-id arrays). Such a change is more
    ///   invasive and requires reworking LWW/commit logic.
    #[allow(unused_variables, unused_assignments)] // TODO: Keep `presence_index_created`?
    pub fn append(&self, batch: &RecordBatch) -> Result<()> {
        // --- PHASE 1: PRE-PROCESSING THE INCOMING BATCH ---
        // The `append` logic relies on row IDs being processed in ascending order to handle
        // metadata updates efficiently and to ensure the shadow row_id chunks are naturally sorted.
        // This block checks if the incoming batch is already sorted by `row_id`. If not, it creates a
        // new, sorted `RecordBatch` to work with for the rest of the function.
        let working_batch: RecordBatch;
        let batch_ref = {
            let schema = batch.schema();
            let row_id_idx = schema
                .index_of(ROW_ID_COLUMN_NAME)
                .map_err(|_| Error::Internal("row_id column required".into()))?;
            let row_id_any = batch.column(row_id_idx).clone();
            let row_id_arr = row_id_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id downcast failed".into()))?;

            // Manually check if the row_id column is sorted.
            let mut is_sorted = true;
            if !row_id_arr.is_empty() {
                let mut last = row_id_arr.value(0);
                for i in 1..row_id_arr.len() {
                    let current = row_id_arr.value(i);
                    if current < last {
                        is_sorted = false;
                        break;
                    }
                    last = current;
                }
            }

            // If sorted, we can use the original batch directly.
            // Otherwise, we compute a permutation and reorder all columns.
            if is_sorted {
                batch
            } else {
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
                &working_batch
            }
        };

        // --- PHASE 2: LAST-WRITER-WINS (LWW) REWRITE ---
        // This phase handles updates. It identifies any rows in the incoming batch that
        // already exist in the store and rewrites them in-place. This is a separate
        // transaction that happens before the main append of new rows.
        let schema = batch_ref.schema();
        let row_id_idx = schema
            .index_of(ROW_ID_COLUMN_NAME)
            .map_err(|_| Error::Internal("row_id column required".into()))?;

        // Create a quick lookup map of incoming row IDs to their positions in the batch.
        let row_id_arr = batch_ref
            .column(row_id_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("row_id downcast failed".into()))?;
        let mut incoming_ids_map = FxHashMap::default();
        incoming_ids_map.reserve(row_id_arr.len());
        for i in 0..row_id_arr.len() {
            incoming_ids_map.insert(row_id_arr.value(i), i);
        }

        // These variables will track the state of the LWW transaction.
        let mut catalog_dirty = false;
        let mut puts_rewrites: Vec<BatchPut> = Vec::new();
        let mut all_rewritten_ids = FxHashSet::default();

        // Iterate through each column in the batch (except row_id) to perform rewrites.
        let mut catalog_lock = self.catalog.write().unwrap();
        for i in 0..batch_ref.num_columns() {
            if i == row_id_idx {
                continue;
            }
            let field = schema.field(i);
            if let Some(field_id_str) = field.metadata().get("field_id") {
                let field_id = field_id_str
                    .parse::<u64>()
                    .map(LogicalFieldId::from)
                    .map_err(|e| Error::Internal(format!("Invalid field_id: {}", e)))?;

                // `lww_rewrite_for_field` finds overlapping row IDs and rewrites the data chunks.
                // It returns the set of row IDs that were updated.
                let rewritten = self.lww_rewrite_for_field(
                    &mut catalog_lock,
                    field_id,
                    &incoming_ids_map,
                    batch_ref.column(i),
                    batch_ref.column(row_id_idx),
                    &mut puts_rewrites,
                )?;
                all_rewritten_ids.extend(rewritten);
            }
        }
        drop(catalog_lock);

        // Commit the LWW changes to the pager immediately.
        if !puts_rewrites.is_empty() {
            self.pager.batch_put(&puts_rewrites)?;
        }

        // --- PHASE 3: FILTERING FOR NEW ROWS ---
        // After handling updates, we filter the incoming batch to remove the rows that were
        // just rewritten. The remaining rows are guaranteed to be new additions to the store.
        let batch_to_append = if !all_rewritten_ids.is_empty() {
            let keep_mask: Vec<bool> = (0..row_id_arr.len())
                .map(|i| !all_rewritten_ids.contains(&row_id_arr.value(i)))
                .collect();
            let keep_array = BooleanArray::from(keep_mask);
            compute::filter_record_batch(batch_ref, &keep_array)?
        } else {
            batch_ref.clone()
        };

        // If no new rows are left, we are done.
        if batch_to_append.num_rows() == 0 {
            return Ok(());
        }

        // --- PHASE 4: APPENDING NEW DATA ---
        // This is the main append transaction. All writes generated in this phase will be
        // collected and committed atomically at the very end.
        let append_schema = batch_to_append.schema();
        let append_row_id_idx = append_schema.index_of(ROW_ID_COLUMN_NAME)?;
        let append_row_id_any: ArrayRef = Arc::clone(batch_to_append.column(append_row_id_idx));
        let mut puts_appends: Vec<BatchPut> = Vec::new();

        // Track the logical table id for the batch so we can update the
        // table-level row-id shadow column once per append.
        let mut table_id_for_batch: Option<TableId> = None;

        // Loop through each column of the filtered batch to append its data.
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

            match table_id_for_batch {
                Some(existing) => debug_assert_eq!(existing, field_id.table_id()),
                None => table_id_for_batch = Some(field_id.table_id()),
            }

            // Populate the data type cache for this column. This is a performance optimization
            // to avoid reading a chunk from storage later just to determine its type.
            self.dtype_cache.insert(field_id, field.data_type().clone());

            // Null values are treated as deletions, so we filter them out. The `rids_clean`
            // array contains the row IDs corresponding to the non-null values.
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

            // Get or create the physical keys for the column's data descriptor and its
            // shadow row_id descriptor from the catalog.
            let (descriptor_pk, rid_descriptor_pk, rid_fid) = {
                let mut catalog = self.catalog.write().unwrap();
                let pk1 = *catalog.map.entry(field_id).or_insert_with(|| {
                    catalog_dirty = true;
                    self.pager.alloc_many(1).unwrap()[0]
                });
                let r_fid = rowid_fid(field_id);
                let pk2 = *catalog.map.entry(r_fid).or_insert_with(|| {
                    catalog_dirty = true;
                    self.pager.alloc_many(1).unwrap()[0]
                });
                (pk1, pk2, r_fid)
            };

            // Load the descriptors and their tail metadata pages into memory.
            // If they don't exist, `load_or_create` will initialize new ones.
            let (mut data_descriptor, mut data_tail_page) =
                ColumnDescriptor::load_or_create(Arc::clone(&self.pager), descriptor_pk, field_id)?;
            let (mut rid_descriptor, mut rid_tail_page) = ColumnDescriptor::load_or_create(
                Arc::clone(&self.pager),
                rid_descriptor_pk,
                rid_fid,
            )?;

            // Logically register the Presence index on the main data descriptor. This ensures
            // that even if no physical index chunks are created (because data arrived sorted),
            // the system knows a Presence index is conceptually active for this column.
            self.index_manager
                .stage_index_registration(&mut data_descriptor, IndexKind::Presence)?;

            // Split the data to be appended into chunks of a target size.
            let slices = split_to_target_bytes(
                &array_clean,
                TARGET_CHUNK_BYTES,
                self.cfg.varwidth_fallback_rows_per_slice,
            );
            let mut row_off = 0usize;

            // Loop through each new slice to create and stage its chunks.
            for s in slices {
                let rows = s.len();
                // Create and stage the data chunk.
                let data_pk = self.pager.alloc_many(1)?[0];
                let s_norm = zero_offset(&s);
                let data_bytes = serialize_array(s_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: data_pk,
                    bytes: data_bytes,
                });

                // Create and stage the corresponding row_id chunk.
                let rid_slice: ArrayRef = rids_clean.slice(row_off, rows);
                let rid_norm = zero_offset(&rid_slice);
                let rid_pk = self.pager.alloc_many(1)?[0];
                let rid_bytes = serialize_array(rid_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: rid_pk,
                    bytes: rid_bytes,
                });

                // Compute min/max for the row_id chunk to enable pruning during scans.
                let rids_for_meta = rid_norm.as_any().downcast_ref::<UInt64Array>().unwrap();
                let (min, max) = if !rids_for_meta.is_empty() {
                    let mut min_val = rids_for_meta.value(0);
                    let mut max_val = rids_for_meta.value(0);
                    for i in 1..rids_for_meta.len() {
                        let v = rids_for_meta.value(i);
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    (min_val, max_val)
                } else {
                    (0, 0)
                };

                // Create the initial metadata for both chunks.
                // The `value_order_perm_pk` is initialized to 0 (None).
                let mut data_meta = ChunkMetadata {
                    chunk_pk: data_pk,
                    row_count: rows as u64,
                    serialized_bytes: s_norm.get_array_memory_size() as u64,
                    max_val_u64: u64::MAX,
                    ..Default::default()
                };
                let mut rid_meta = ChunkMetadata {
                    chunk_pk: rid_pk,
                    row_count: rows as u64,
                    serialized_bytes: rid_norm.get_array_memory_size() as u64,
                    min_val_u64: min,
                    max_val_u64: max,
                    ..Default::default()
                };

                // **GENERIC INDEX UPDATE DISPATCH**
                // This is the single, index-agnostic call. The IndexManager will look up all
                // active indexes for this column (e.g., Presence, Sort) and call their respective
                // `stage_update_for_new_chunk` methods. This is where the physical index data
                // (like permutation blobs) is created and staged.
                self.index_manager.stage_updates_for_new_chunk(
                    field_id,
                    &data_descriptor,
                    &s_norm,
                    &rid_norm,
                    &mut data_meta,
                    &mut rid_meta,
                    &mut puts_appends,
                )?;

                // Append the (potentially modified) metadata to their respective descriptor chains.
                self.append_meta_in_loop(
                    &mut data_descriptor,
                    &mut data_tail_page,
                    data_meta,
                    &mut puts_appends,
                )?;
                self.append_meta_in_loop(
                    &mut rid_descriptor,
                    &mut rid_tail_page,
                    rid_meta,
                    &mut puts_appends,
                )?;
                row_off += rows;
            }

            // After processing all slices, stage the final writes for the updated tail pages
            // and the root descriptor objects themselves.
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

        // After processing user-data columns, append the row ids to the
        // table-level shadow column so table-level metadata stays in sync.
        if let Some(table_id) = table_id_for_batch {
            let table_rid_fid = table_rowid_fid(table_id);
            let table_rid_descriptor_pk = {
                let mut catalog = self.catalog.write().unwrap();
                *catalog.map.entry(table_rid_fid).or_insert_with(|| {
                    catalog_dirty = true;
                    self.pager.alloc_many(1).unwrap()[0]
                })
            };

            let (mut table_rid_descriptor, mut table_rid_tail_page) =
                ColumnDescriptor::load_or_create(
                    Arc::clone(&self.pager),
                    table_rid_descriptor_pk,
                    table_rid_fid,
                )?;

            let rid_slices = split_to_target_bytes(
                &append_row_id_any,
                TARGET_CHUNK_BYTES,
                self.cfg.varwidth_fallback_rows_per_slice,
            );

            for rid_slice in rid_slices {
                let rows = rid_slice.len();
                if rows == 0 {
                    continue;
                }

                let rid_norm = zero_offset(&rid_slice);
                let rid_pk = self.pager.alloc_many(1)?[0];
                let rid_bytes = serialize_array(rid_norm.as_ref())?;
                puts_appends.push(BatchPut::Raw {
                    key: rid_pk,
                    bytes: rid_bytes,
                });

                let rids_for_meta = rid_norm
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| Error::Internal("row_id downcast failed".into()))?;

                let (min, max) = if !rids_for_meta.is_empty() {
                    let mut min_val = rids_for_meta.value(0);
                    let mut max_val = rids_for_meta.value(0);
                    for i in 1..rids_for_meta.len() {
                        let v = rids_for_meta.value(i);
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    (min_val, max_val)
                } else {
                    (0, 0)
                };

                let table_rid_meta = ChunkMetadata {
                    chunk_pk: rid_pk,
                    row_count: rows as u64,
                    serialized_bytes: rid_norm.get_array_memory_size() as u64,
                    min_val_u64: min,
                    max_val_u64: max,
                    ..Default::default()
                };

                // The table-level row-id column currently has no secondary
                // indexes, so skip stage_updates_for_new_chunk.
                self.append_meta_in_loop(
                    &mut table_rid_descriptor,
                    &mut table_rid_tail_page,
                    table_rid_meta,
                    &mut puts_appends,
                )?;
            }

            puts_appends.push(BatchPut::Raw {
                key: table_rid_descriptor.tail_page_pk,
                bytes: table_rid_tail_page,
            });
            puts_appends.push(BatchPut::Raw {
                key: table_rid_descriptor_pk,
                bytes: table_rid_descriptor.to_le_bytes(),
            });
        }

        // --- PHASE 5: FINAL ATOMIC COMMIT ---
        // If the catalog was modified (e.g., new columns were created), stage its write.
        if catalog_dirty {
            let catalog = self.catalog.read().unwrap();
            puts_appends.push(BatchPut::Raw {
                key: CATALOG_ROOT_PKEY,
                bytes: catalog.to_bytes(),
            });
        }

        // Commit all staged puts (new data chunks, new row_id chunks, new index permutations,
        // updated descriptor pages, updated root descriptors, and the updated catalog)
        // in a single atomic operation.
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

    fn stage_delete_rows_for_field(
        &self,
        field_id: LogicalFieldId,
        rows_to_delete: &[RowId],
        staged_puts: &mut Vec<BatchPut>,
    ) -> Result<bool> {
        use crate::store::descriptor::DescriptorIterator;
        use crate::store::ingest::ChunkEdit;

        if rows_to_delete.is_empty() {
            return Ok(false);
        }

        // Stream and validate ascending, unique positions.
        let mut del_iter = rows_to_delete.iter().copied();
        let mut cur_del = del_iter.next();
        let mut last_seen: Option<u64> = cur_del;

        // Lookup descriptors (data and optional row_id).
        let catalog = self.catalog.read().unwrap();
        let desc_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let is_table_rowid = field_id.namespace() == Namespace::RowIdShadow
            && field_id.field_id() == TABLE_ROW_ID_FIELD_ID;
        let desc_pk_rid = if is_table_rowid {
            None
        } else {
            let rid_fid = rowid_fid(field_id);
            catalog.map.get(&rid_fid).copied()
        };

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

        let desc_blob = blobs_by_pk.remove(&desc_pk).ok_or_else(|| {
            Error::Internal(format!(
                "descriptor pk={} missing during delete_rows for field {:?}",
                desc_pk, field_id
            ))
        })?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // Build metas for data column.
        let mut metas: Vec<ChunkMetadata> = Vec::new();
        for m in DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk) {
            metas.push(m?);
        }
        if metas.is_empty() {
            drop(catalog);
            return Ok(false);
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

        let mut cum_rows = 0u64;
        let mut any_changed = false;

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
            staged_puts.push(BatchPut::Raw {
                key: meta.chunk_pk,
                bytes: data_bytes,
            });
            meta.row_count = new_data_arr.len() as u64;
            meta.serialized_bytes = new_data_arr.get_array_memory_size() as u64;

            // Write back row_ids if present.
            if let (Some(_), Some(rids)) = (metas_rid.get_mut(i), new_rid_arr) {
                let rm = metas_rid.get_mut(i).unwrap();
                let rid_bytes = serialize_array(&rids)?;
                staged_puts.push(BatchPut::Raw {
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
                staged_puts.push(BatchPut::Raw {
                    key: meta.value_order_perm_pk,
                    bytes: perm_bytes,
                });
            }

            cum_rows = end_u64;
            any_changed = true;
        }

        // Rewrite descriptor chains/totals and commit.
        descriptor.rewrite_pages(Arc::clone(&self.pager), desc_pk, &mut metas, staged_puts)?;
        if let (Some(rid_pk), Some(mut rid_desc)) = (desc_pk_rid, descriptor_rid) {
            rid_desc.rewrite_pages(Arc::clone(&self.pager), rid_pk, &mut metas_rid, staged_puts)?;
        }
        drop(catalog);
        Ok(any_changed)
    }

    /// Delete row positions for one or more logical fields in a single atomic batch.
    ///
    /// Each entry contains a field id and the global row positions to delete
    /// for that field. All staged metadata and chunk updates are committed in a
    /// single pager batch.
    pub fn delete_rows(&self, deletes: &[DeleteRowsSpec]) -> Result<()> {
        if deletes.is_empty() {
            return Ok(());
        }

        let mut puts = Vec::new();
        let mut touched: FxHashSet<LogicalFieldId> = FxHashSet::default();
        let mut table_id: Option<TableId> = None;

        for (field_id, rows) in deletes {
            if rows.is_empty() {
                continue;
            }
            if let Some(expected) = table_id {
                if field_id.table_id() != expected {
                    return Err(Error::InvalidArgumentError(
                        "delete_rows requires fields from the same table".into(),
                    ));
                }
            } else {
                table_id = Some(field_id.table_id());
            }

            if self.stage_delete_rows_for_field(*field_id, rows, &mut puts)? {
                touched.insert(*field_id);
            }
        }

        if puts.is_empty() {
            return Ok(());
        }

        self.pager.batch_put(&puts)?;

        for field_id in touched {
            self.compact_field_bounded(field_id)?;
        }
        Ok(())
    }

    fn stage_table_row_id_deletes(
        &self,
        table_id: TableId,
        to_remove: &FxHashSet<RowId>,
        puts: &mut Vec<BatchPut>,
    ) -> Result<bool> {
        use crate::store::descriptor::DescriptorIterator;

        if to_remove.is_empty() {
            return Ok(false);
        }

        let field_id = table_rowid_fid(table_id);
        let catalog = self.catalog.read().unwrap();
        let Some(desc_pk) = catalog.map.get(&field_id).copied() else {
            return Ok(false);
        };
        drop(catalog);

        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: desc_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        let mut metas: Vec<ChunkMetadata> =
            DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk)
                .collect::<Result<_>>()?;
        if metas.is_empty() {
            return Ok(false);
        }

        let mut any_changed = false;

        for meta in metas.iter_mut() {
            if meta.row_count == 0 {
                continue;
            }

            let chunk_blob = self
                .pager
                .batch_get(&[BatchGet::Raw { key: meta.chunk_pk }])?
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?;
            let arr_any = deserialize_array(chunk_blob)?;
            let arr = arr_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id downcast failed".into()))?;

            if arr.is_empty() {
                continue;
            }

            let mut keep_mask = Vec::with_capacity(arr.len());
            let mut removed = false;
            for idx in 0..arr.len() {
                let rid = arr.value(idx);
                if to_remove.contains(&rid) {
                    keep_mask.push(false);
                    removed = true;
                } else {
                    keep_mask.push(true);
                }
            }

            if !removed {
                continue;
            }

            let keep_array = BooleanArray::from(keep_mask);
            let filtered_any = compute::filter(arr_any.as_ref(), &keep_array)?;
            let filtered = zero_offset(&filtered_any);
            let bytes = serialize_array(filtered.as_ref())?;
            puts.push(BatchPut::Raw {
                key: meta.chunk_pk,
                bytes,
            });

            let filtered_rids = filtered
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("row_id downcast failed".into()))?;
            meta.row_count = filtered_rids.len() as u64;
            meta.serialized_bytes = filtered.get_array_memory_size() as u64;
            if filtered_rids.is_empty() {
                meta.min_val_u64 = 0;
                meta.max_val_u64 = 0;
            } else {
                let mut min_val = filtered_rids.value(0);
                let mut max_val = filtered_rids.value(0);
                for i in 1..filtered_rids.len() {
                    let v = filtered_rids.value(i);
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
                meta.min_val_u64 = min_val;
                meta.max_val_u64 = max_val;
            }

            any_changed = true;
        }

        if !any_changed {
            return Ok(false);
        }

        descriptor.rewrite_pages(Arc::clone(&self.pager), desc_pk, &mut metas, puts)?;
        Ok(true)
    }

    /// Delete specific row ids from the table-level row-id shadow column.
    pub fn delete_table_row_ids<I>(&self, table_id: TableId, row_ids: I) -> Result<()>
    where
        I: IntoIterator<Item = u64>,
    {
        use rustc_hash::FxHashSet;

        let to_remove: FxHashSet<RowId> = row_ids.into_iter().collect();
        if to_remove.is_empty() {
            return Ok(());
        }

        let mut puts = Vec::new();
        if self.stage_table_row_id_deletes(table_id, &to_remove, &mut puts)? && !puts.is_empty() {
            self.pager.batch_put(&puts)?;
        }
        Ok(())
    }

    // TODO: Move to descriptor module?
    /// Write a descriptor chain from a meta slice. Reuses first page when
    /// possible. Frees surplus pages via `frees`.
    pub(crate) fn write_descriptor_chain(
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
                    row_count: rows as u64,
                    serialized_bytes: s_norm.get_array_memory_size() as u64,
                    max_val_u64: u64::MAX,
                    ..Default::default()
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
