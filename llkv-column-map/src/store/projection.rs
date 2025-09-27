use super::*;
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::types::LogicalFieldId;
use arrow::array::{Array, ArrayRef, BooleanArray, PrimitiveArray, UInt64Array, new_empty_array};
use arrow::compute;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use llkv_storage::{
    pager::{BatchGet, GetResult, Pager},
    serialization::deserialize_array,
    types::PhysicalKey,
};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherNullPolicy {
    /// Any missing row ID results in an error.
    ErrorOnMissing,
    /// Missing rows surface as nulls in the result arrays.
    IncludeNulls,
    /// Missing rows and rows where all projected columns are null are dropped from the output.
    DropNulls,
}

impl GatherNullPolicy {
    #[inline]
    fn allow_missing(self) -> bool {
        !matches!(self, Self::ErrorOnMissing)
    }
}

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Gathers values for the specified `row_ids`, returned in the same order as provided.
    /// When `include_nulls` is true, missing rows surface as nulls instead of an error.
    pub fn gather_rows(
        &self,
        field_id: LogicalFieldId,
        row_ids: &[u64],
        include_nulls: bool,
    ) -> Result<ArrayRef> {
        let policy = if include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::ErrorOnMissing
        };
        self.gather_rows_with_policy(field_id, row_ids, policy)
    }

    /// Gathers values using a configurable null-handling policy. The
    /// policy controls whether missing rows surfaces as nulls, produce
    /// errors, or are filtered alongside explicit nulls.
    pub fn gather_rows_with_policy(
        &self,
        field_id: LogicalFieldId,
        row_ids: &[u64],
        policy: GatherNullPolicy,
    ) -> Result<ArrayRef> {
        let batch = self.gather_rows_multi_with_policy(&[field_id], row_ids, policy)?;
        batch
            .columns()
            .first()
            .map(Arc::clone)
            .ok_or_else(|| Error::Internal("gather_rows_multi returned no columns".into()))
    }

    /// Prototype: gathers multiple primitive columns for the given `row_ids` in a single
    /// descriptor walk, reducing redundant pager fetches across columns. When `include_nulls`
    /// is true, missing row ids are surfaced as nulls (rather than producing an error), mirroring
    /// the semantics of `gather_rows_with_nulls` without requiring an explicit anchor.
    pub fn gather_rows_multi(
        &self,
        field_ids: &[LogicalFieldId],
        row_ids: &[u64],
        include_nulls: bool,
    ) -> Result<RecordBatch> {
        let policy = if include_nulls {
            GatherNullPolicy::IncludeNulls
        } else {
            GatherNullPolicy::ErrorOnMissing
        };
        self.gather_rows_multi_with_policy(field_ids, row_ids, policy)
    }

    /// Gathers multiple columns using a configurable null-handling policy.
    /// When [`GatherNullPolicy::DropNulls`] is selected, rows where all
    /// projected columns are null or missing are removed from the
    /// resulting batch.
    pub fn gather_rows_multi_with_policy(
        &self,
        field_ids: &[LogicalFieldId],
        row_ids: &[u64],
        policy: GatherNullPolicy,
    ) -> Result<RecordBatch> {
        if field_ids.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
        }

        let mut field_infos = Vec::with_capacity(field_ids.len());
        for &fid in field_ids {
            field_infos.push((fid, self.data_type(fid)?));
        }

        if row_ids.is_empty() {
            let mut arrays = Vec::with_capacity(field_infos.len());
            let mut fields = Vec::with_capacity(field_infos.len());
            for (fid, dtype) in &field_infos {
                arrays.push(new_empty_array(dtype));
                let field_name = format!("field_{}", u64::from(*fid));
                fields.push(Field::new(field_name, dtype.clone(), true));
            }
            let schema = Arc::new(Schema::new(fields));
            return RecordBatch::try_new(schema, arrays)
                .map_err(|e| Error::Internal(format!("gather_rows_multi empty batch: {e}")));
        }

        let mut row_index: FxHashMap<u64, usize> =
            FxHashMap::with_capacity_and_hasher(row_ids.len(), Default::default());
        for (idx, &row_id) in row_ids.iter().enumerate() {
            if row_index.insert(row_id, idx).is_some() {
                return Err(Error::Internal(
                    "duplicate row_id in gather_rows_multi".into(),
                ));
            }
        }

        let mut sorted_row_ids = row_ids.to_vec();
        sorted_row_ids.sort_unstable();

        struct FieldPlan {
            dtype: DataType,
            value_pk: PhysicalKey,
            row_pk: PhysicalKey,
            value_metas: Vec<ChunkMetadata>,
            row_metas: Vec<ChunkMetadata>,
            candidate_indices: Vec<usize>,
        }

        let mut plans = Vec::with_capacity(field_infos.len());
        {
            let catalog = self.catalog.read().unwrap();
            for (fid, dtype) in &field_infos {
                let value_pk = *catalog.map.get(fid).ok_or(Error::NotFound)?;
                let row_fid = rowid_fid(*fid);
                let row_pk = *catalog.map.get(&row_fid).ok_or(Error::NotFound)?;
                plans.push(FieldPlan {
                    dtype: dtype.clone(),
                    value_pk,
                    row_pk,
                    value_metas: Vec::new(),
                    row_metas: Vec::new(),
                    candidate_indices: Vec::new(),
                });
            }
        }

        let mut descriptor_requests = Vec::with_capacity(plans.len() * 2);
        for plan in &plans {
            descriptor_requests.push(BatchGet::Raw { key: plan.value_pk });
            descriptor_requests.push(BatchGet::Raw { key: plan.row_pk });
        }
        let descriptor_results = self.pager.batch_get(&descriptor_requests)?;
        let mut descriptor_map: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for result in descriptor_results {
            if let GetResult::Raw { key, bytes } = result {
                descriptor_map.insert(key, bytes);
            }
        }

        for plan in &mut plans {
            let value_desc_blob = descriptor_map
                .remove(&plan.value_pk)
                .ok_or(Error::NotFound)?;
            let value_desc = ColumnDescriptor::from_le_bytes(value_desc_blob.as_ref());
            plan.value_metas =
                Self::collect_non_empty_metas(self.pager.as_ref(), value_desc.head_page_pk)?;

            let row_desc_blob = descriptor_map.remove(&plan.row_pk).ok_or(Error::NotFound)?;
            let row_desc = ColumnDescriptor::from_le_bytes(row_desc_blob.as_ref());
            plan.row_metas =
                Self::collect_non_empty_metas(self.pager.as_ref(), row_desc.head_page_pk)?;

            if plan.value_metas.len() != plan.row_metas.len() {
                return Err(Error::Internal(
                    "gather_rows_multi: chunk count mismatch".into(),
                ));
            }

            plan.candidate_indices = plan
                .row_metas
                .iter()
                .enumerate()
                .filter_map(|(idx, meta)| {
                    if Self::chunk_intersects(&sorted_row_ids, meta) {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();
        }

        let mut chunk_keys: FxHashSet<PhysicalKey> = FxHashSet::default();
        for plan in &plans {
            for &idx in &plan.candidate_indices {
                chunk_keys.insert(plan.value_metas[idx].chunk_pk);
                chunk_keys.insert(plan.row_metas[idx].chunk_pk);
            }
        }

        let mut chunk_requests = Vec::with_capacity(chunk_keys.len());
        for &key in &chunk_keys {
            chunk_requests.push(BatchGet::Raw { key });
        }

        let mut chunk_map: FxHashMap<PhysicalKey, EntryHandle> =
            FxHashMap::with_capacity_and_hasher(chunk_requests.len(), Default::default());
        if !chunk_requests.is_empty() {
            let chunk_results = self.pager.batch_get(&chunk_requests)?;
            for result in chunk_results {
                if let GetResult::Raw { key, bytes } = result {
                    chunk_map.insert(key, bytes);
                }
            }
        }

        let allow_missing = policy.allow_missing();

        let mut outputs = Vec::with_capacity(plans.len());
        for plan in plans.into_iter() {
            let array = crate::with_integer_arrow_type!(
                plan.dtype.clone(),
                |ArrowTy| {
                    Self::gather_rows_from_chunks::<ArrowTy>(
                        &row_index,
                        row_ids.len(),
                        &plan.candidate_indices,
                        &plan.value_metas,
                        &plan.row_metas,
                        &mut chunk_map,
                        allow_missing,
                    )
                },
                Err(Error::Internal(format!(
                    "gather_rows_multi: unsupported dtype {:?}",
                    plan.dtype
                ))),
            )?;
            outputs.push(array);
        }

        let outputs = if matches!(policy, GatherNullPolicy::DropNulls) {
            Self::filter_rows_with_non_null(outputs)?
        } else {
            outputs
        };

        let mut fields = Vec::with_capacity(field_infos.len());
        for (idx, (fid, dtype)) in field_infos.iter().enumerate() {
            let array = &outputs[idx];
            let field_name = format!("field_{}", u64::from(*fid));
            let nullable = match policy {
                GatherNullPolicy::IncludeNulls => true,
                _ => array.null_count() > 0,
            };
            fields.push(Field::new(field_name, dtype.clone(), nullable));
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, outputs)
            .map_err(|e| Error::Internal(format!("gather_rows_multi batch: {e}")))
    }

    fn collect_non_empty_metas(pager: &P, head_page_pk: PhysicalKey) -> Result<Vec<ChunkMetadata>> {
        let mut metas = Vec::new();
        if head_page_pk == 0 {
            return Ok(metas);
        }
        for meta in DescriptorIterator::new(pager, head_page_pk) {
            let meta = meta?;
            if meta.row_count > 0 {
                metas.push(meta);
            }
        }
        Ok(metas)
    }

    #[inline]
    fn chunk_intersects(sorted_row_ids: &[u64], meta: &ChunkMetadata) -> bool {
        if sorted_row_ids.is_empty() || meta.row_count == 0 {
            return false;
        }
        let min = meta.min_val_u64;
        let max = meta.max_val_u64;
        if min == 0 && max == 0 && meta.row_count > 0 {
            return true;
        }
        if min > max {
            return true;
        }
        let min_req = sorted_row_ids[0];
        let max_req = *sorted_row_ids.last().unwrap();
        if max < min_req || min > max_req {
            return false;
        }
        let idx = sorted_row_ids.partition_point(|&rid| rid < min);
        idx < sorted_row_ids.len() && sorted_row_ids[idx] <= max
    }

    fn gather_rows_from_chunks<T>(
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        candidate_indices: &[usize],
        value_metas: &[ChunkMetadata],
        row_metas: &[ChunkMetadata],
        chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
        allow_missing: bool,
    ) -> Result<ArrayRef>
    where
        T: ArrowPrimitiveType,
    {
        let mut values: Vec<Option<T::Native>> = Vec::with_capacity(len);
        values.resize(len, None);
        let mut found: Vec<bool> = Vec::with_capacity(len);
        found.resize(len, false);

        for &idx in candidate_indices {
            let value_chunk = chunk_blobs
                .remove(&value_metas[idx].chunk_pk)
                .ok_or(Error::NotFound)?;
            let row_chunk = chunk_blobs
                .remove(&row_metas[idx].chunk_pk)
                .ok_or(Error::NotFound)?;

            let value_any = deserialize_array(value_chunk)?;
            let value_arr = value_any
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
            let row_any = deserialize_array(row_chunk)?;
            let row_arr = row_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

            let len_chunk = row_arr.len();
            for i in 0..len_chunk {
                let row_id = row_arr.value(i);
                if let Some(&out_idx) = row_index.get(&row_id) {
                    found[out_idx] = true;
                    let value = if value_arr.is_null(i) {
                        None
                    } else {
                        Some(value_arr.value(i))
                    };
                    values[out_idx] = value;
                }
            }
        }

        if !allow_missing && found.iter().any(|f| !*f) {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }

        if allow_missing {
            for (idx, was_found) in found.iter().enumerate() {
                if !*was_found {
                    values[idx] = None;
                }
            }
        }

        let array = PrimitiveArray::<T>::from_iter(values);
        Ok(Arc::new(array) as ArrayRef)
    }

    fn filter_rows_with_non_null(columns: Vec<ArrayRef>) -> Result<Vec<ArrayRef>> {
        if columns.is_empty() {
            return Ok(columns);
        }

        let len = columns[0].len();
        if len == 0 {
            return Ok(columns);
        }

        let mut keep = vec![false; len];
        for array in &columns {
            debug_assert_eq!(array.len(), len);
            if array.null_count() == 0 {
                keep.fill(true);
                break;
            }
            for i in 0..len {
                if array.is_valid(i) {
                    keep[i] = true;
                }
            }
            if keep.iter().all(|flag| *flag) {
                break;
            }
        }

        if keep.iter().all(|flag| *flag) {
            return Ok(columns);
        }

        let mask = BooleanArray::from(keep);

        let mut filtered = Vec::with_capacity(columns.len());
        for array in columns {
            let filtered_column = compute::filter(array.as_ref(), &mask)
                .map_err(|e| Error::Internal(format!("gather_rows_multi filter: {e}")))?;
            filtered.push(filtered_column);
        }
        Ok(filtered)
    }
}
