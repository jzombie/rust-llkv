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
};
use crate::serialization::deserialize_array;
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use arrow::array::{
    ArrayRef, BooleanBuilder, Decimal128Builder, GenericBinaryBuilder, GenericStringBuilder,
    OffsetSizeTrait, PrimitiveBuilder, new_empty_array,
};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Date32Type, Date64Type, Field, Float32Type, Float64Type,
    Int16Type, Int32Type, Int64Type, Int8Type, Schema, UInt16Type, UInt32Type, UInt64Type,
    UInt8Type,
};
use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use llkv_storage::{
    pager::{BatchGet, GetResult, Pager},
    types::PhysicalKey,
};
use llkv_types::ids::{LogicalFieldId, RowId};
use rustc_hash::FxHashMap;
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
enum ColumnOutputBuilder {
    Utf8(GenericStringBuilder<i32>),
    LargeUtf8(GenericStringBuilder<i64>),
    Binary(GenericBinaryBuilder<i32>),
    LargeBinary(GenericBinaryBuilder<i64>),
    Boolean(BooleanBuilder),
    Decimal128(Decimal128Builder),
    Primitive(PrimitiveBuilderKind),
    Passthrough,
}

impl ColumnOutputBuilder {
    fn from_dtype(dtype: &DataType) -> Result<Self> {
        use DataType::*;
        let builder = match dtype {
            Utf8 => ColumnOutputBuilder::Utf8(GenericStringBuilder::<i32>::new()),
            LargeUtf8 => ColumnOutputBuilder::LargeUtf8(GenericStringBuilder::<i64>::new()),
            Binary => ColumnOutputBuilder::Binary(GenericBinaryBuilder::<i32>::new()),
            LargeBinary => ColumnOutputBuilder::LargeBinary(GenericBinaryBuilder::<i64>::new()),
            Boolean => ColumnOutputBuilder::Boolean(BooleanBuilder::new()),
            Decimal128(precision, scale) => ColumnOutputBuilder::Decimal128(
                Decimal128Builder::new()
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| Error::Internal(format!(
                        "invalid Decimal128 precision/scale: {e}"
                    )))?,
            ),
            Struct(_) => ColumnOutputBuilder::Passthrough,
            other => {
                if let Some(kind) = PrimitiveBuilderKind::for_type(other) {
                    ColumnOutputBuilder::Primitive(kind)
                } else {
                    return Err(Error::Internal(format!(
                        "unsupported gather datatype {other:?}"
                    )));
                }
            }
        };
        Ok(builder)
    }
}

enum PrimitiveBuilderKind {
    UInt64(PrimitiveBuilder<UInt64Type>),
    UInt32(PrimitiveBuilder<UInt32Type>),
    UInt16(PrimitiveBuilder<UInt16Type>),
    UInt8(PrimitiveBuilder<UInt8Type>),
    Int64(PrimitiveBuilder<Int64Type>),
    Int32(PrimitiveBuilder<Int32Type>),
    Int16(PrimitiveBuilder<Int16Type>),
    Int8(PrimitiveBuilder<Int8Type>),
    Float64(PrimitiveBuilder<Float64Type>),
    Float32(PrimitiveBuilder<Float32Type>),
    Date64(PrimitiveBuilder<Date64Type>),
    Date32(PrimitiveBuilder<Date32Type>),
}

impl PrimitiveBuilderKind {
    fn for_type(dtype: &DataType) -> Option<Self> {
        use DataType::*;
        Some(match dtype {
            UInt64 => PrimitiveBuilderKind::UInt64(PrimitiveBuilder::<UInt64Type>::new()),
            UInt32 => PrimitiveBuilderKind::UInt32(PrimitiveBuilder::<UInt32Type>::new()),
            UInt16 => PrimitiveBuilderKind::UInt16(PrimitiveBuilder::<UInt16Type>::new()),
            UInt8 => PrimitiveBuilderKind::UInt8(PrimitiveBuilder::<UInt8Type>::new()),
            Int64 => PrimitiveBuilderKind::Int64(PrimitiveBuilder::<Int64Type>::new()),
            Int32 => PrimitiveBuilderKind::Int32(PrimitiveBuilder::<Int32Type>::new()),
            Int16 => PrimitiveBuilderKind::Int16(PrimitiveBuilder::<Int16Type>::new()),
            Int8 => PrimitiveBuilderKind::Int8(PrimitiveBuilder::<Int8Type>::new()),
            Float64 => PrimitiveBuilderKind::Float64(PrimitiveBuilder::<Float64Type>::new()),
            Float32 => PrimitiveBuilderKind::Float32(PrimitiveBuilder::<Float32Type>::new()),
            Date64 => PrimitiveBuilderKind::Date64(PrimitiveBuilder::<Date64Type>::new()),
            Date32 => PrimitiveBuilderKind::Date32(PrimitiveBuilder::<Date32Type>::new()),
            _ => return None,
        })
    }

}

pub struct MultiGatherContext {
    field_infos: Vec<(LogicalFieldId, DataType)>,
    plans: Vec<FieldPlan>,
    chunk_cache: FxHashMap<PhysicalKey, ArrayRef>,
    row_index: FxHashMap<u64, usize>,
    row_scratch: Vec<Option<(usize, usize)>>,
    chunk_keys: Vec<PhysicalKey>,
    builders: Vec<ColumnOutputBuilder>,
    cached_schema: Option<(Vec<bool>, Arc<Schema>)>,
}

impl MultiGatherContext {
    fn new(field_infos: Vec<(LogicalFieldId, DataType)>, plans: Vec<FieldPlan>) -> Result<Self> {
        let mut builders = Vec::with_capacity(field_infos.len());
        for (_, dtype) in &field_infos {
            builders.push(ColumnOutputBuilder::from_dtype(dtype)?);
        }

        Ok(Self {
            chunk_cache: FxHashMap::default(),
            row_index: FxHashMap::default(),
            row_scratch: Vec::new(),
            chunk_keys: Vec::new(),
            field_infos,
            plans,
            builders,
            cached_schema: None,
        })
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

    #[inline]
    fn plans_mut(&mut self) -> &mut [FieldPlan] {
        &mut self.plans
    }

    fn take_builders(&mut self) -> Vec<ColumnOutputBuilder> {
        std::mem::take(&mut self.builders)
    }

    fn restore_builders(&mut self, builders: Vec<ColumnOutputBuilder>) {
        self.builders = builders;
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

    fn schema_for_nullability(&mut self, nullability: &[bool]) -> Arc<Schema> {
        if let Some((cached_flags, schema)) = &self.cached_schema {
            if cached_flags == nullability {
                return Arc::clone(schema);
            }
        }

        let mut fields = Vec::with_capacity(self.field_infos.len());
        for ((fid, dtype), nullable) in self.field_infos.iter().zip(nullability.iter()) {
            let field_name = format!("field_{}", u64::from(*fid));
            fields.push(Field::new(field_name, dtype.clone(), *nullable));
        }

        let schema = Arc::new(Schema::new(fields));
        self.cached_schema = Some((nullability.to_vec(), Arc::clone(&schema)));
        schema
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
        self.gather_rows_with_reusable_context_impl(&mut ctx, row_ids, policy, expected_schema)
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
            return MultiGatherContext::new(Vec::new(), Vec::new());
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

        MultiGatherContext::new(field_infos, plans)
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
        self.gather_rows_with_reusable_context_impl(ctx, row_ids, policy, None)
    }

    fn gather_rows_with_reusable_context_impl(
        &self,
        ctx: &mut MultiGatherContext,
        row_ids: &[u64],
        policy: GatherNullPolicy,
        expected_schema: Option<Arc<Schema>>,
    ) -> Result<RecordBatch> {
        if ctx.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
        }

        if row_ids.is_empty() {
            let mut arrays = Vec::new();
            let schema = if let Some(schema) = expected_schema {
                arrays.reserve(schema.fields().len());
                for field in schema.fields() {
                    arrays.push(new_empty_array(field.data_type()));
                }
                schema
            } else {
                arrays.reserve(ctx.field_infos().len());
                let mut fields = Vec::with_capacity(ctx.field_infos().len());
                for (fid, dtype) in ctx.field_infos() {
                    arrays.push(new_empty_array(dtype));
                    let field_name = format!("field_{}", u64::from(*fid));
                    fields.push(Field::new(field_name, dtype.clone(), true));
                }
                Arc::new(Schema::new(fields))
            };
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
            let mut builders = ctx.take_builders();

            let outputs = {
                let chunk_arrays = ctx.chunk_cache();
                let mut column_outputs = Vec::with_capacity(ctx.plans().len());
                for (plan, builder) in ctx.plans().iter().zip(builders.iter_mut()) {
                    let array = match builder {
                    ColumnOutputBuilder::Utf8(builder) => Self::gather_rows_from_chunks_string::<i32>(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        chunk_arrays,
                        &mut row_scratch,
                        allow_missing,
                        builder,
                    ),
                    ColumnOutputBuilder::LargeUtf8(builder) => {
                        Self::gather_rows_from_chunks_string::<i64>(
                            row_ids,
                            row_locator,
                            len,
                            &plan.candidate_indices,
                            plan,
                            chunk_arrays,
                            &mut row_scratch,
                            allow_missing,
                            builder,
                        )
                    }
                    ColumnOutputBuilder::Binary(builder) => Self::gather_rows_from_chunks_binary::<i32>(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        chunk_arrays,
                        &mut row_scratch,
                        allow_missing,
                        builder,
                    ),
                    ColumnOutputBuilder::LargeBinary(builder) => {
                        Self::gather_rows_from_chunks_binary::<i64>(
                            row_ids,
                            row_locator,
                            len,
                            &plan.candidate_indices,
                            plan,
                            chunk_arrays,
                            &mut row_scratch,
                            allow_missing,
                            builder,
                        )
                    }
                    ColumnOutputBuilder::Boolean(builder) => Self::gather_rows_from_chunks_bool(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        chunk_arrays,
                        &mut row_scratch,
                        allow_missing,
                        builder,
                    ),
                    ColumnOutputBuilder::Decimal128(builder) => Self::gather_rows_from_chunks_decimal128(
                        row_ids,
                        row_locator,
                        len,
                        &plan.candidate_indices,
                        plan,
                        chunk_arrays,
                        &mut row_scratch,
                        allow_missing,
                        builder,
                    ),
                    ColumnOutputBuilder::Primitive(kind) => match kind {
                        PrimitiveBuilderKind::UInt64(builder) => {
                            Self::gather_rows_from_chunks::<UInt64Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::UInt32(builder) => {
                            Self::gather_rows_from_chunks::<UInt32Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::UInt16(builder) => {
                            Self::gather_rows_from_chunks::<UInt16Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::UInt8(builder) => {
                            Self::gather_rows_from_chunks::<UInt8Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Int64(builder) => {
                            Self::gather_rows_from_chunks::<Int64Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Int32(builder) => {
                            Self::gather_rows_from_chunks::<Int32Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Int16(builder) => {
                            Self::gather_rows_from_chunks::<Int16Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Int8(builder) => {
                            Self::gather_rows_from_chunks::<Int8Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Float64(builder) => {
                            Self::gather_rows_from_chunks::<Float64Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Float32(builder) => {
                            Self::gather_rows_from_chunks::<Float32Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Date64(builder) => {
                            Self::gather_rows_from_chunks::<Date64Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                        PrimitiveBuilderKind::Date32(builder) => {
                            Self::gather_rows_from_chunks::<Date32Type>(
                                row_ids,
                                row_locator,
                                len,
                                &plan.candidate_indices,
                                plan,
                                chunk_arrays,
                                &mut row_scratch,
                                allow_missing,
                                builder,
                            )
                        }
                    },
                    ColumnOutputBuilder::Passthrough => match &plan.dtype {
                        DataType::Struct(_) => Self::gather_rows_from_chunks_struct(
                            row_locator,
                            len,
                            &plan.candidate_indices,
                            plan,
                            chunk_arrays,
                            &mut row_scratch,
                            allow_missing,
                            &plan.dtype,
                        ),
                        other => Err(Error::Internal(format!(
                            "gather_rows_multi: unsupported dtype {other:?}"
                        ))),
                    },
                    }?;
                    column_outputs.push(array);
                }
                column_outputs
            };

            ctx.restore_builders(builders);

            let outputs = if matches!(policy, GatherNullPolicy::DropNulls) {
                Self::filter_rows_with_non_null(outputs)?
            } else {
                outputs
            };

            let mut nullability = Vec::with_capacity(field_infos.len());
            for (idx, _) in field_infos.iter().enumerate() {
                let array = &outputs[idx];
                let nullable = match policy {
                    GatherNullPolicy::IncludeNulls => true,
                    _ => array.null_count() > 0,
                };
                nullability.push(nullable);
            }

            let schema = ctx.schema_for_nullability(&nullability);
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

    /// Gather a row window into a `RecordBatch`, reusing or creating a gather context.
    pub fn gather_row_window_with_context(
        &self,
        logical_fields: &[LogicalFieldId],
        row_ids: &[u64],
        null_policy: GatherNullPolicy,
        ctx: Option<&mut MultiGatherContext>,
    ) -> Result<RecordBatch> {
        if logical_fields.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
        }

        let mut local_ctx;
        let ctx_ref = match ctx {
            Some(existing) => existing,
            None => {
                local_ctx = self.prepare_gather_context(logical_fields)?;
                &mut local_ctx
            }
        };

        self.gather_rows_with_reusable_context(ctx_ref, row_ids, null_policy)
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
        builder: &mut GenericStringBuilder<O>,
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
            builder,
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
        builder: &mut GenericBinaryBuilder<O>,
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
            builder,
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
        builder: &mut BooleanBuilder,
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
            builder,
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
        builder: &mut Decimal128Builder,
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
            builder,
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
        builder: &mut PrimitiveBuilder<T>,
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
            builder,
        )
    }

    fn filter_rows_with_non_null(columns: Vec<ArrayRef>) -> Result<Vec<ArrayRef>> {
        shared_filter_rows_with_non_null(columns)
    }
}
