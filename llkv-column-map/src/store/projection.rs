use super::*;
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::types::LogicalFieldId;
use arrow::array::{new_empty_array, Array, ArrayRef, BooleanArray, PrimitiveArray, PrimitiveBuilder, UInt64Array};
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Projection {
    pub logical_field_id: LogicalFieldId,
    pub alias: Option<String>,
}

impl Projection {
    pub fn new(logical_field_id: LogicalFieldId) -> Self {
        Self {
            logical_field_id,
            alias: None,
        }
    }

    pub fn with_alias<S: Into<String>>(logical_field_id: LogicalFieldId, alias: S) -> Self {
        Self {
            logical_field_id,
            alias: Some(alias.into()),
        }
    }
}

impl From<LogicalFieldId> for Projection {
    fn from(logical_field_id: LogicalFieldId) -> Self {
        Projection::new(logical_field_id)
    }
}

impl<S: Into<String>> From<(LogicalFieldId, S)> for Projection {
    fn from(value: (LogicalFieldId, S)) -> Self {
        Projection::with_alias(value.0, value.1)
    }
}

impl GatherNullPolicy {
    #[inline]
    fn allow_missing(self) -> bool {
        !matches!(self, Self::ErrorOnMissing)
    }
}

pub struct MultiGatherContext {
    field_infos: Vec<(LogicalFieldId, DataType)>,
    plans: Vec<FieldPlan>,
    chunk_cache: FxHashMap<PhysicalKey, ArrayRef>,
    row_index: FxHashMap<u64, usize>,
    row_scratch: Vec<Option<(usize, usize)>>,
}

impl MultiGatherContext {
    fn new(
        field_infos: Vec<(LogicalFieldId, DataType)>,
        plans: Vec<FieldPlan>,
    ) -> Self {
        Self {
            chunk_cache: FxHashMap::default(),
            row_index: FxHashMap::default(),
            row_scratch: Vec::new(),
            field_infos,
            plans,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    #[inline]
    fn field_infos(&self) -> &[(LogicalFieldId, DataType)] {
        &self.field_infos
    }

    #[inline]
    fn plans(&self) -> &[FieldPlan] {
        &self.plans
    }

    #[inline]
    fn chunk_cache(&self) -> &FxHashMap<PhysicalKey, ArrayRef> {
        &self.chunk_cache
    }

    #[inline]
    fn chunk_cache_mut(&mut self) -> &mut FxHashMap<PhysicalKey, ArrayRef> {
        &mut self.chunk_cache
    }

    fn take_row_index(&mut self) -> FxHashMap<u64, usize> {
        std::mem::take(&mut self.row_index)
    }

    fn store_row_index(&mut self, row_index: FxHashMap<u64, usize>) {
        self.row_index = row_index;
    }

    fn take_row_scratch(&mut self) -> Vec<Option<(usize, usize)>> {
        std::mem::take(&mut self.row_scratch)
    }

    fn store_row_scratch(&mut self, scratch: Vec<Option<(usize, usize)>>) {
        self.row_scratch = scratch;
    }

    pub fn chunk_span_for_row(&self, row_id: u64) -> Option<(usize, u64, u64)> {
        let first_plan = self.plans.first()?;
        let mut chunk_idx = None;
        for (idx, meta) in first_plan.row_metas.iter().enumerate() {
            if row_id >= meta.min_val_u64 && row_id <= meta.max_val_u64 {
                chunk_idx = Some(idx);
                break;
            }
        }
        let idx = chunk_idx?;
        for plan in &self.plans {
            let meta = plan.row_metas.get(idx)?;
            if row_id < meta.min_val_u64 || row_id > meta.max_val_u64 {
                return None;
            }
        }
        let meta = &first_plan.row_metas[idx];
        Some((idx, meta.min_val_u64, meta.max_val_u64))
    }
}

#[derive(Clone, Debug)]
struct FieldPlan {
    dtype: DataType,
    value_metas: Vec<ChunkMetadata>,
    row_metas: Vec<ChunkMetadata>,
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
        let mut ctx = self.prepare_multi_gather_context(field_ids)?;
        self.gather_rows_multi_with_context(&mut ctx, row_ids, policy)
    }

    pub fn prepare_multi_gather_context(
        &self,
        field_ids: &[LogicalFieldId],
    ) -> Result<MultiGatherContext> {
        let mut field_infos = Vec::with_capacity(field_ids.len());
        for &fid in field_ids {
            field_infos.push((fid, self.data_type(fid)?));
        }

        if field_infos.is_empty() {
            return Ok(MultiGatherContext::new(Vec::new(), Vec::new()));
        }

        let catalog = self.catalog.read().unwrap();
        let mut key_pairs = Vec::with_capacity(field_infos.len());
        for (fid, _) in &field_infos {
            let value_pk = *catalog.map.get(fid).ok_or(Error::NotFound)?;
            let row_pk = *catalog.map.get(&rowid_fid(*fid)).ok_or(Error::NotFound)?;
            key_pairs.push((value_pk, row_pk));
        }
        drop(catalog);

        let mut descriptor_requests = Vec::with_capacity(key_pairs.len() * 2);
        for (value_pk, row_pk) in &key_pairs {
            descriptor_requests.push(BatchGet::Raw { key: *value_pk });
            descriptor_requests.push(BatchGet::Raw { key: *row_pk });
        }
        let descriptor_results = self.pager.batch_get(&descriptor_requests)?;
        let mut descriptor_map: FxHashMap<PhysicalKey, EntryHandle> = FxHashMap::default();
        for result in descriptor_results {
            if let GetResult::Raw { key, bytes } = result {
                descriptor_map.insert(key, bytes);
            }
        }

        let mut plans = Vec::with_capacity(field_infos.len());
        for ((_, dtype), (value_pk, row_pk)) in field_infos.iter().zip(key_pairs.iter()) {
            let value_desc_blob = descriptor_map
                .remove(value_pk)
                .ok_or(Error::NotFound)?;
            let value_desc = ColumnDescriptor::from_le_bytes(value_desc_blob.as_ref());
            let value_metas =
                Self::collect_non_empty_metas(self.pager.as_ref(), value_desc.head_page_pk)?;

            let row_desc_blob = descriptor_map
                .remove(row_pk)
                .ok_or(Error::NotFound)?;
            let row_desc = ColumnDescriptor::from_le_bytes(row_desc_blob.as_ref());
            let row_metas =
                Self::collect_non_empty_metas(self.pager.as_ref(), row_desc.head_page_pk)?;

            if value_metas.len() != row_metas.len() {
                return Err(Error::Internal(
                    "gather_rows_multi: chunk count mismatch".into(),
                ));
            }

            plans.push(FieldPlan {
                dtype: dtype.clone(),
                value_metas,
                row_metas,
            });
        }

        Ok(MultiGatherContext::new(field_infos, plans))
    }

    pub fn gather_rows_multi_with_context(
        &self,
        ctx: &mut MultiGatherContext,
        row_ids: &[u64],
        policy: GatherNullPolicy,
    ) -> Result<RecordBatch> {
        if ctx.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
        }

        if row_ids.is_empty() {
            let mut arrays = Vec::with_capacity(ctx.field_infos().len());
            let mut fields = Vec::with_capacity(ctx.field_infos().len());
            for (fid, dtype) in ctx.field_infos() {
                arrays.push(new_empty_array(dtype));
                let field_name = format!("field_{}", u64::from(*fid));
                fields.push(Field::new(field_name, dtype.clone(), true));
            }
            let schema = Arc::new(Schema::new(fields));
            return RecordBatch::try_new(schema, arrays)
                .map_err(|e| Error::Internal(format!("gather_rows_multi empty batch: {e}")));
        }

        let mut row_index = ctx.take_row_index();
        let mut row_scratch = ctx.take_row_scratch();

        let result: Result<RecordBatch> = (|| {
            row_index.clear();
            row_index.reserve(row_ids.len());
            for (idx, &row_id) in row_ids.iter().enumerate() {
                if row_index.insert(row_id, idx).is_some() {
                    return Err(Error::Internal(
                        "duplicate row_id in gather_rows_multi".into(),
                    ));
                }
            }

            if row_scratch.len() < row_ids.len() {
                row_scratch.resize(row_ids.len(), None);
            }

            let mut sorted_row_ids = row_ids.to_vec();
            sorted_row_ids.sort_unstable();

            let field_infos = ctx.field_infos().to_vec();
            let plans = ctx.plans().to_vec();

            let mut candidate_per_plan: Vec<Vec<usize>> = Vec::with_capacity(plans.len());
            let mut chunk_keys: FxHashSet<PhysicalKey> = FxHashSet::default();
            for plan in &plans {
                let mut candidates = Vec::new();
                for (idx, meta) in plan.row_metas.iter().enumerate() {
                    if Self::chunk_intersects(&sorted_row_ids, meta) {
                        candidates.push(idx);
                        chunk_keys.insert(plan.value_metas[idx].chunk_pk);
                        chunk_keys.insert(plan.row_metas[idx].chunk_pk);
                    }
                }
                candidate_per_plan.push(candidates);
            }

            let mut chunk_arrays: FxHashMap<PhysicalKey, ArrayRef> =
                FxHashMap::with_capacity_and_hasher(chunk_keys.len(), Default::default());
            {
                let cache = ctx.chunk_cache();
                for key in chunk_keys.iter() {
                    if let Some(arr) = cache.get(key) {
                        chunk_arrays.insert(*key, Arc::clone(arr));
                    }
                }
            }

            let missing: Vec<PhysicalKey> = chunk_keys
                .into_iter()
                .filter(|k| !chunk_arrays.contains_key(k))
                .collect();
            if !missing.is_empty() {
                let requests: Vec<BatchGet> = missing
                    .iter()
                    .map(|key| BatchGet::Raw { key: *key })
                    .collect();
                let chunk_results = self.pager.batch_get(&requests)?;
                for result in chunk_results {
                    if let GetResult::Raw { key, bytes } = result {
                        let array = deserialize_array(bytes)?;
                        ctx.chunk_cache_mut().insert(key, Arc::clone(&array));
                        chunk_arrays.insert(key, array);
                    }
                }
            }

            let allow_missing = policy.allow_missing();

            let mut outputs = Vec::with_capacity(plans.len());
            for (plan_idx, plan) in plans.iter().enumerate() {
                let candidates = &candidate_per_plan[plan_idx];
                let array = crate::with_integer_arrow_type!(
                    plan.dtype.clone(),
                    |ArrowTy| {
                        Self::gather_rows_from_chunks::<ArrowTy>(
                            row_ids,
                            &row_index,
                            row_ids.len(),
                            candidates,
                            plan,
                            &chunk_arrays,
                            &mut row_scratch,
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
        })();

        ctx.store_row_scratch(row_scratch);
        ctx.store_row_index(row_index);

        result
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
        row_ids: &[u64],
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        candidate_indices: &[usize],
        plan: &FieldPlan,
        chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
        row_scratch: &mut [Option<(usize, usize)>],
        allow_missing: bool,
    ) -> Result<ArrayRef>
    where
        T: ArrowPrimitiveType,
    {
        if len == 0 {
            return Ok(Arc::new(PrimitiveBuilder::<T>::new().finish()) as ArrayRef);
        }

        if candidate_indices.len() == 1 {
            let chunk_idx = candidate_indices[0];
            let value_any = chunk_arrays
                .get(&plan.value_metas[chunk_idx].chunk_pk)
                .ok_or(Error::NotFound)?;
            let row_any = chunk_arrays
                .get(&plan.row_metas[chunk_idx].chunk_pk)
                .ok_or(Error::NotFound)?;
            let _value_arr = value_any
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
            let row_arr = row_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

            if row_arr.null_count() == 0 && row_ids.windows(2).all(|w| w[0] <= w[1]) {
                let values = row_arr.values();
                if let Ok(start_idx) = values.binary_search(&row_ids[0]) {
                    if start_idx + len <= values.len()
                        && row_ids == &values[start_idx..start_idx + len]
                    {
                        return Ok(value_any.slice(start_idx, len));
                    }
                }
            }
        }

        for slot in row_scratch.iter_mut().take(len) {
            *slot = None;
        }

        let mut candidates: Vec<(usize, &PrimitiveArray<T>, &UInt64Array)> =
            Vec::with_capacity(candidate_indices.len());
        let mut chunk_lookup: FxHashMap<usize, usize> = FxHashMap::default();

        for (slot, &chunk_idx) in candidate_indices.iter().enumerate() {
            let value_any = chunk_arrays
                .get(&plan.value_metas[chunk_idx].chunk_pk)
                .ok_or(Error::NotFound)?;
            let value_arr = value_any
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
            let row_any = chunk_arrays
                .get(&plan.row_metas[chunk_idx].chunk_pk)
                .ok_or(Error::NotFound)?;
            let row_arr = row_any
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

            candidates.push((chunk_idx, value_arr, row_arr));
            chunk_lookup.insert(chunk_idx, slot);

            for i in 0..row_arr.len() {
                if !row_arr.is_valid(i) {
                    continue;
                }
                let row_id = row_arr.value(i);
                if let Some(&out_idx) = row_index.get(&row_id) {
                    row_scratch[out_idx] = Some((chunk_idx, i));
                }
            }
        }

        if !allow_missing {
            for slot in row_scratch.iter().take(len) {
                if slot.is_none() {
                    return Err(Error::Internal(
                        "gather_rows_multi: one or more requested row IDs were not found".into(),
                    ));
                }
            }
        }

        let mut builder = PrimitiveBuilder::<T>::with_capacity(len);
        for offset in 0..len {
            if let Some((chunk_idx, value_idx)) = row_scratch[offset] {
                if let Some(&slot) = chunk_lookup.get(&chunk_idx) {
                    let (idx, value_arr, _) = candidates[slot];
                    debug_assert_eq!(idx, chunk_idx);
                    if value_arr.is_null(value_idx) {
                        builder.append_null();
                    } else {
                        builder.append_value(value_arr.value(value_idx));
                    }
                } else {
                    builder.append_null();
                }
            } else {
                builder.append_null();
            }
        }

        Ok(Arc::new(builder.finish()) as ArrayRef)
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
            for (i, keep_item) in keep.iter_mut().enumerate().take(len) {
                if array.is_valid(i) {
                    *keep_item = true;
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
