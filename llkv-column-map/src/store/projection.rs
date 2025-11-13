//! Projection planners for building Arrow batches from column chunks.
//!
//! The projection subsystem wires descriptor metadata, chunk loading helpers,
//! and gather routines to materialize user-facing result sets. It centralizes
//! null-handling policies and keeps the hot paths monomorphized per Arrow type.

use super::*;
use crate::gather::{
    RowLocator, filter_rows_with_non_null as shared_filter_rows_with_non_null,
    gather_rows_from_chunks as shared_gather_rows_from_chunks,
    gather_rows_from_chunks_binary as shared_gather_rows_from_chunks_binary,
    gather_rows_from_chunks_bool as shared_gather_rows_from_chunks_bool,
    gather_rows_from_chunks_decimal128 as shared_gather_rows_from_chunks_decimal128,
    gather_rows_from_chunks_string as shared_gather_rows_from_chunks_string,
    gather_rows_from_chunks_struct as shared_gather_rows_from_chunks_struct,
    gather_rows_single_shot as shared_gather_rows_single_shot,
    gather_rows_single_shot_binary as shared_gather_rows_single_shot_binary,
    gather_rows_single_shot_bool as shared_gather_rows_single_shot_bool,
    gather_rows_single_shot_decimal128 as shared_gather_rows_single_shot_decimal128,
    gather_rows_single_shot_string as shared_gather_rows_single_shot_string,
    gather_rows_single_shot_struct as shared_gather_rows_single_shot_struct,
};
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::types::{LogicalFieldId, RowId};
use arrow::array::{ArrayRef, OffsetSizeTrait, new_empty_array};
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
use std::borrow::Cow;
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

/// Logical projection request describing a field and optional alias.
///
/// Used by planners to pull a column by [`LogicalFieldId`] and to rename the output slot when
/// needed.
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

/// Scratch structures shared across field plans during multi-column gathers.
///
/// Keeps cached chunk blobs and row indexes so repeated projection passes avoid redundant pager
/// reads.
pub struct MultiGatherContext {
    field_infos: Vec<(LogicalFieldId, DataType)>,
    fields: Vec<Field>,
    plans: Vec<FieldPlan>,
    chunk_cache: FxHashMap<PhysicalKey, ArrayRef>,
    row_index: FxHashMap<u64, usize>,
    row_scratch: Vec<Option<(usize, usize)>>,
    chunk_keys: Vec<PhysicalKey>,
}

impl MultiGatherContext {
    fn new(field_infos: Vec<(LogicalFieldId, DataType)>, plans: Vec<FieldPlan>) -> Self {
        // Build Field objects from field_infos with default nullable=true
        // This is for backward compatibility when no expected schema is provided
        let fields: Vec<Field> = field_infos
            .iter()
            .map(|(fid, dtype)| {
                let field_name = format!("field_{}", u64::from(*fid));
                Field::new(field_name, dtype.clone(), true)
            })
            .collect();

        Self {
            chunk_cache: FxHashMap::default(),
            row_index: FxHashMap::default(),
            row_scratch: Vec::new(),
            chunk_keys: Vec::new(),
            field_infos,
            fields,
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
    fn fields(&self) -> &[Field] {
        &self.fields
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

    #[inline]
    fn plans_mut(&mut self) -> &mut [FieldPlan] {
        &mut self.plans
    }

    fn take_chunk_keys(&mut self) -> Vec<PhysicalKey> {
        std::mem::take(&mut self.chunk_keys)
    }

    fn store_chunk_keys(&mut self, keys: Vec<PhysicalKey>) {
        self.chunk_keys = keys;
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

    pub fn chunk_span_for_row(&self, row_id: RowId) -> Option<(usize, RowId, RowId)> {
        let first_plan = self.plans.first()?;
        let mut chunk_idx = None;
        for (idx, meta) in first_plan.row_metas.iter().enumerate() {
            if row_id >= meta.min_val_u64 && row_id <= meta.max_val_u64 {
                chunk_idx = Some(idx);
                break;
            }
        }
        if chunk_idx.is_none() {
            let total_chunks = first_plan.row_metas.len();
            'outer: for idx in 0..total_chunks {
                for plan in &self.plans {
                    let meta = &plan.row_metas[idx];
                    if row_id >= meta.min_val_u64 && row_id <= meta.max_val_u64 {
                        chunk_idx = Some(idx);
                        break 'outer;
                    }
                }
            }
        }
        let idx = chunk_idx?;

        let mut span_min = u64::MAX;
        let mut span_max = 0u64;
        for plan in &self.plans {
            let meta = plan.row_metas.get(idx)?;
            span_min = span_min.min(meta.min_val_u64);
            span_max = span_max.max(meta.max_val_u64);
        }

        if span_min > span_max {
            return None;
        }

        Some((idx, span_min, span_max))
    }
}

#[derive(Clone, Debug)]
struct FieldPlan {
    dtype: DataType,
    value_metas: Vec<ChunkMetadata>,
    row_metas: Vec<ChunkMetadata>,
    candidate_indices: Vec<usize>,
}

impl<P> ColumnStore<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Gathers multiple columns using a configurable null-handling policy.
    /// When [`GatherNullPolicy::DropNulls`] is selected, rows where all
    /// projected columns are null or missing are removed from the
    /// resulting batch.
    pub fn gather_rows(
        &self,
        field_ids: &[LogicalFieldId],
        row_ids: &[u64],
        policy: GatherNullPolicy,
    ) -> Result<RecordBatch> {
        self.gather_rows_with_schema(field_ids, row_ids, policy, None)
    }

    /// Gather rows with an optional expected schema for empty result sets.
    ///
    /// When `expected_schema` is provided and `row_ids` is empty, the returned
    /// RecordBatch will use that schema instead of synthesizing one with all nullable fields.
    /// This ensures that non-nullable columns (e.g., PRIMARY KEYs) are correctly represented.
    pub fn gather_rows_with_schema(
        &self,
        field_ids: &[LogicalFieldId],
        row_ids: &[u64],
        policy: GatherNullPolicy,
        expected_schema: Option<Arc<Schema>>,
    ) -> Result<RecordBatch> {
        let mut ctx = self.prepare_gather_context(field_ids)?;
        self.execute_gather_single_pass_with_schema(&mut ctx, row_ids, policy, expected_schema)
    }

    /// Executes a one-off gather with optional schema for empty result sets.
    fn execute_gather_single_pass_with_schema(
        &self,
        ctx: &mut MultiGatherContext,
        row_ids: &[u64],
        policy: GatherNullPolicy,
        expected_schema: Option<Arc<Schema>>,
    ) -> Result<RecordBatch> {
        if ctx.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
        }

        let field_infos = ctx.field_infos().to_vec();

        if row_ids.is_empty() {
            // Use expected_schema if provided to preserve nullability
            let (schema, arrays) = if let Some(expected_schema) = expected_schema {
                let mut arrays = Vec::with_capacity(expected_schema.fields().len());
                for field in expected_schema.fields() {
                    arrays.push(new_empty_array(field.data_type()));
                }
                (expected_schema, arrays)
            } else {
                // Fallback: Use fields from context which have nullable=true by default
                // This is safe because nullable=true is always compatible
                let fields = ctx.fields();
                let mut arrays = Vec::with_capacity(fields.len());
                for field in fields {
                    arrays.push(new_empty_array(field.data_type()));
                }
                (Arc::new(Schema::new(fields.to_vec())), arrays)
            };
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

        let mut chunk_keys: FxHashSet<PhysicalKey> = FxHashSet::default();
        {
            let plans_mut = ctx.plans_mut();
            for plan in plans_mut.iter_mut() {
                plan.candidate_indices.clear();
                for (idx, meta) in plan.row_metas.iter().enumerate() {
                    if Self::chunk_intersects(&sorted_row_ids, meta) {
                        plan.candidate_indices.push(idx);
                        chunk_keys.insert(plan.value_metas[idx].chunk_pk);
                        chunk_keys.insert(plan.row_metas[idx].chunk_pk);
                    }
                }
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

        let mut outputs = Vec::with_capacity(ctx.plans().len());
        for plan in ctx.plans() {
            let array = match &plan.dtype {
                DataType::Utf8 => Self::gather_rows_single_shot_string::<i32>(
                    &row_index,
                    row_ids.len(),
                    plan,
                    &mut chunk_map,
                    allow_missing,
                ),
                DataType::LargeUtf8 => Self::gather_rows_single_shot_string::<i64>(
                    &row_index,
                    row_ids.len(),
                    plan,
                    &mut chunk_map,
                    allow_missing,
                ),
                DataType::Binary => Self::gather_rows_single_shot_binary::<i32>(
                    &row_index,
                    row_ids.len(),
                    plan,
                    &mut chunk_map,
                    allow_missing,
                ),
                DataType::LargeBinary => Self::gather_rows_single_shot_binary::<i64>(
                    &row_index,
                    row_ids.len(),
                    plan,
                    &mut chunk_map,
                    allow_missing,
                ),
                DataType::Boolean => Self::gather_rows_single_shot_bool(
                    &row_index,
                    row_ids.len(),
                    plan,
                    &mut chunk_map,
                    allow_missing,
                ),
                DataType::Struct(_) => Self::gather_rows_single_shot_struct(
                    &row_index,
                    row_ids.len(),
                    plan,
                    &mut chunk_map,
                    allow_missing,
                    &plan.dtype,
                ),
                DataType::Decimal128(_, _) => Self::gather_rows_single_shot_decimal128(
                    &row_index,
                    row_ids.len(),
                    plan,
                    &mut chunk_map,
                    allow_missing,
                    &plan.dtype,
                ),
                other => with_integer_arrow_type!(
                    other.clone(),
                    |ArrowTy| {
                        Self::gather_rows_single_shot::<ArrowTy>(
                            &row_index,
                            row_ids.len(),
                            plan,
                            &mut chunk_map,
                            allow_missing,
                        )
                    },
                    Err(Error::Internal(format!(
                        "gather_rows_multi: unsupported dtype {:?}",
                        other
                    ))),
                ),
            }?;
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

    pub fn prepare_gather_context(
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
            let value_desc_blob = descriptor_map.remove(value_pk).ok_or(Error::NotFound)?;
            let value_desc = ColumnDescriptor::from_le_bytes(value_desc_blob.as_ref());
            let value_metas =
                Self::collect_non_empty_metas(self.pager.as_ref(), value_desc.head_page_pk)?;

            let row_desc_blob = descriptor_map.remove(row_pk).ok_or(Error::NotFound)?;
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
                candidate_indices: Vec::new(),
            });
        }

        Ok(MultiGatherContext::new(field_infos, plans))
    }

    /// Gathers rows while reusing chunk caches and scratch buffers stored in the context.
    ///
    /// This path amortizes chunk fetch and decode costs across multiple calls by
    /// retaining Arrow arrays and scratch state inside the provided context.
    pub fn gather_rows_with_reusable_context(
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

        let field_infos = ctx.field_infos().to_vec();
        let mut chunk_keys = ctx.take_chunk_keys();

        let result: Result<RecordBatch> = (|| {
            let len = row_ids.len();
            if row_scratch.len() < len {
                row_scratch.resize(len, None);
            }

            let is_non_decreasing = len <= 1 || row_ids.windows(2).all(|w| w[0] <= w[1]);
            let sorted_row_ids_cow: Cow<'_, [u64]> = if is_non_decreasing {
                Cow::Borrowed(row_ids)
            } else {
                let mut buf = row_ids.to_vec();
                buf.sort_unstable();
                Cow::Owned(buf)
            };
            let sorted_row_ids: &[u64] = sorted_row_ids_cow.as_ref();

            let dense_base = if len == 0 {
                None
            } else if len == 1 || is_non_decreasing && row_ids.windows(2).all(|w| w[1] == w[0] + 1)
            {
                Some(row_ids[0])
            } else {
                None
            };

            if dense_base.is_none() {
                row_index.clear();
                row_index.reserve(len);
                for (idx, &row_id) in row_ids.iter().enumerate() {
                    if row_index.insert(row_id, idx).is_some() {
                        return Err(Error::Internal(
                            "duplicate row_id in gather_rows_multi".into(),
                        ));
                    }
                }
            } else {
                row_index.clear();
            }

            let row_locator = if let Some(base) = dense_base {
                RowLocator::Dense { base }
            } else {
                RowLocator::Sparse { index: &row_index }
            };

            chunk_keys.clear();

            {
                let plans_mut = ctx.plans_mut();
                for plan in plans_mut.iter_mut() {
                    plan.candidate_indices.clear();
                    for (idx, meta) in plan.row_metas.iter().enumerate() {
                        if Self::chunk_intersects(sorted_row_ids, meta) {
                            plan.candidate_indices.push(idx);
                            chunk_keys.push(plan.value_metas[idx].chunk_pk);
                            chunk_keys.push(plan.row_metas[idx].chunk_pk);
                        }
                    }
                }
            }

            chunk_keys.sort_unstable();
            chunk_keys.dedup();

            {
                let mut pending: Vec<BatchGet> = Vec::new();
                {
                    let cache = ctx.chunk_cache();
                    for &key in &chunk_keys {
                        if !cache.contains_key(&key) {
                            pending.push(BatchGet::Raw { key });
                        }
                    }
                }

                if !pending.is_empty() {
                    let chunk_results = self.pager.batch_get(&pending)?;
                    let cache = ctx.chunk_cache_mut();
                    for result in chunk_results {
                        if let GetResult::Raw { key, bytes } = result {
                            let array = deserialize_array(bytes)?;
                            cache.insert(key, Arc::clone(&array));
                        }
                    }
                }
            }

            let allow_missing = policy.allow_missing();

            let mut outputs = Vec::with_capacity(ctx.plans().len());
            for plan in ctx.plans() {
                let array = match &plan.dtype {
                    DataType::Utf8 => Self::gather_rows_from_chunks_string::<i32>(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        ctx.chunk_cache(),
                        &mut row_scratch,
                        allow_missing,
                    ),
                    DataType::LargeUtf8 => Self::gather_rows_from_chunks_string::<i64>(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        ctx.chunk_cache(),
                        &mut row_scratch,
                        allow_missing,
                    ),
                    DataType::Binary => Self::gather_rows_from_chunks_binary::<i32>(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        ctx.chunk_cache(),
                        &mut row_scratch,
                        allow_missing,
                    ),
                    DataType::LargeBinary => Self::gather_rows_from_chunks_binary::<i64>(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        ctx.chunk_cache(),
                        &mut row_scratch,
                        allow_missing,
                    ),
                    DataType::Boolean => Self::gather_rows_from_chunks_bool(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        ctx.chunk_cache(),
                        &mut row_scratch,
                        allow_missing,
                    ),
                    DataType::Struct(_) => Self::gather_rows_from_chunks_struct(
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        ctx.chunk_cache(),
                        &mut row_scratch,
                        allow_missing,
                        &plan.dtype,
                    ),
                    DataType::Decimal128(_, _) => Self::gather_rows_from_chunks_decimal128(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        ctx.chunk_cache(),
                        &mut row_scratch,
                        allow_missing,
                        &plan.dtype,
                    ),
                    other => with_integer_arrow_type!(
                        other.clone(),
                        |ArrowTy| {
                            Self::gather_rows_from_chunks::<ArrowTy>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                ctx.chunk_cache(),
                                &mut row_scratch,
                                allow_missing,
                            )
                        },
                        Err(Error::Internal(format!(
                            "gather_rows_multi: unsupported dtype {:?}",
                            other
                        ))),
                    ),
                }?;
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
        ctx.store_chunk_keys(chunk_keys);

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

    fn gather_rows_single_shot_string<O>(
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        plan: &FieldPlan,
        chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
        allow_missing: bool,
    ) -> Result<ArrayRef>
    where
        O: OffsetSizeTrait,
    {
        shared_gather_rows_single_shot_string::<O>(
            row_index,
            len,
            &plan.value_metas,
            &plan.row_metas,
            &plan.candidate_indices,
            chunk_blobs,
            allow_missing,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_rows_single_shot_bool(
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        plan: &FieldPlan,
        chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
        allow_missing: bool,
    ) -> Result<ArrayRef> {
        shared_gather_rows_single_shot_bool(
            row_index,
            len,
            &plan.value_metas,
            &plan.row_metas,
            &plan.candidate_indices,
            chunk_blobs,
            allow_missing,
        )
    }

    fn gather_rows_single_shot_struct(
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        plan: &FieldPlan,
        chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
        allow_missing: bool,
        dtype: &DataType,
    ) -> Result<ArrayRef> {
        shared_gather_rows_single_shot_struct(
            row_index,
            len,
            &plan.value_metas,
            &plan.row_metas,
            &plan.candidate_indices,
            chunk_blobs,
            allow_missing,
            dtype,
        )
    }

    fn gather_rows_single_shot_decimal128(
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        plan: &FieldPlan,
        chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
        allow_missing: bool,
        dtype: &DataType,
    ) -> Result<ArrayRef> {
        shared_gather_rows_single_shot_decimal128(
            row_index,
            len,
            &plan.value_metas,
            &plan.row_metas,
            &plan.candidate_indices,
            chunk_blobs,
            allow_missing,
            dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_rows_single_shot<T>(
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        plan: &FieldPlan,
        chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
        allow_missing: bool,
    ) -> Result<ArrayRef>
    where
        T: ArrowPrimitiveType,
    {
        shared_gather_rows_single_shot::<T>(
            row_index,
            len,
            &plan.value_metas,
            &plan.row_metas,
            &plan.candidate_indices,
            chunk_blobs,
            allow_missing,
        )
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Signature mirrors shared helper and avoids intermediate structs.
    fn gather_rows_from_chunks_string<O>(
        row_ids: &[u64],
        row_locator: RowLocator,
        len: usize,
        candidate_indices: &[usize],
        plan: &FieldPlan,
        chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
        row_scratch: &mut [Option<(usize, usize)>],
        allow_missing: bool,
    ) -> Result<ArrayRef>
    where
        O: OffsetSizeTrait,
    {
        shared_gather_rows_from_chunks_string::<O>(
            row_ids,
            row_locator,
            len,
            candidate_indices,
            &plan.value_metas,
            &plan.row_metas,
            chunk_arrays,
            row_scratch,
            allow_missing,
        )
    }

    fn gather_rows_single_shot_binary<O>(
        row_index: &FxHashMap<u64, usize>,
        len: usize,
        plan: &FieldPlan,
        chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
        allow_missing: bool,
    ) -> Result<ArrayRef>
    where
        O: OffsetSizeTrait,
    {
        shared_gather_rows_single_shot_binary::<O>(
            row_index,
            len,
            &plan.value_metas,
            &plan.row_metas,
            &plan.candidate_indices,
            chunk_blobs,
            allow_missing,
        )
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Signature mirrors shared helper and avoids intermediate structs.
    fn gather_rows_from_chunks_binary<O>(
        row_ids: &[u64],
        row_locator: RowLocator,
        len: usize,
        candidate_indices: &[usize],
        plan: &FieldPlan,
        chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
        row_scratch: &mut [Option<(usize, usize)>],
        allow_missing: bool,
    ) -> Result<ArrayRef>
    where
        O: OffsetSizeTrait,
    {
        shared_gather_rows_from_chunks_binary::<O>(
            row_ids,
            row_locator,
            len,
            candidate_indices,
            &plan.value_metas,
            &plan.row_metas,
            chunk_arrays,
            row_scratch,
            allow_missing,
        )
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Signature mirrors shared helper and avoids intermediate structs.
    fn gather_rows_from_chunks_bool(
        row_ids: &[u64],
        row_locator: RowLocator,
        len: usize,
        candidate_indices: &[usize],
        plan: &FieldPlan,
        chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
        row_scratch: &mut [Option<(usize, usize)>],
        allow_missing: bool,
    ) -> Result<ArrayRef> {
        shared_gather_rows_from_chunks_bool(
            row_ids,
            row_locator,
            len,
            candidate_indices,
            &plan.value_metas,
            &plan.row_metas,
            chunk_arrays,
            row_scratch,
            allow_missing,
        )
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Signature mirrors shared helper and avoids intermediate structs.
    fn gather_rows_from_chunks_struct(
        row_locator: RowLocator,
        len: usize,
        candidate_indices: &[usize],
        plan: &FieldPlan,
        chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
        row_scratch: &mut [Option<(usize, usize)>],
        allow_missing: bool,
        dtype: &DataType,
    ) -> Result<ArrayRef> {
        shared_gather_rows_from_chunks_struct(
            row_locator,
            len,
            candidate_indices,
            &plan.value_metas,
            &plan.row_metas,
            chunk_arrays,
            row_scratch,
            allow_missing,
            dtype,
        )
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Signature mirrors shared helper and avoids intermediate structs.
    fn gather_rows_from_chunks_decimal128(
        row_ids: &[u64],
        row_locator: RowLocator,
        len: usize,
        candidate_indices: &[usize],
        plan: &FieldPlan,
        chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
        row_scratch: &mut [Option<(usize, usize)>],
        allow_missing: bool,
        dtype: &DataType,
    ) -> Result<ArrayRef> {
        shared_gather_rows_from_chunks_decimal128(
            row_ids,
            row_locator,
            len,
            candidate_indices,
            &plan.value_metas,
            &plan.row_metas,
            chunk_arrays,
            row_scratch,
            allow_missing,
            dtype,
        )
    }

    #[allow(clippy::too_many_arguments)] // NOTE: Signature mirrors shared helper and keeps type monomorphization straightforward.
    fn gather_rows_from_chunks<T>(
        row_ids: &[u64],
        row_locator: RowLocator,
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
        shared_gather_rows_from_chunks::<T>(
            row_ids,
            row_locator,
            len,
            candidate_indices,
            &plan.value_metas,
            &plan.row_metas,
            chunk_arrays,
            row_scratch,
            allow_missing,
        )
    }

    fn filter_rows_with_non_null(columns: Vec<ArrayRef>) -> Result<Vec<ArrayRef>> {
        shared_filter_rows_with_non_null(columns)
    }
}
