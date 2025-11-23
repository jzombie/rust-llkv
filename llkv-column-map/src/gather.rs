//! Row gathering helpers for assembling Arrow arrays across chunks.
//!
//! These utilities provide shared implementations used by projections, joins,
//! and multi-column scans. They focus on minimizing temporary allocations while
//! preserving row order guarantees.

use std::mem;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, Decimal128Builder, GenericBinaryArray,
    GenericBinaryBuilder, GenericStringArray, GenericStringBuilder, OffsetSizeTrait,
    PrimitiveArray, PrimitiveBuilder, RecordBatch, StringViewArray, StringViewBuilder, StructArray,
    UInt32Array, UInt64Array, new_empty_array,
};
use arrow::compute::{self, take};
use arrow::datatypes::{ArrowPrimitiveType, DataType};

use crate::serialization::deserialize_array;
use crate::store::descriptor::ChunkMetadata;
use crate::types::RowId;
use crate::{Error, Result};
use llkv_result::Result as LlkvResult;
use llkv_storage::types::PhysicalKey;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;

/// Gather rows from a single [`RecordBatch`] according to the provided indices.
///
/// The function batches the `take` operation by reusing a shared Arrow index
/// array, avoiding the per-row allocations that arise from repeatedly slicing
/// the input column. The order of `indices` is preserved in the returned
/// columns. An empty `indices` slice yields an empty vector.
///
/// # Examples
///
/// ```
/// use arrow::array::Int32Array;
/// use arrow::datatypes::{DataType, Field, Schema};
/// use arrow::record_batch::RecordBatch;
/// use llkv_column_map::gather::gather_indices;
/// use std::sync::Arc;
///
/// let schema = Schema::new(vec![Field::new("values", DataType::Int32, false)]);
/// let batch = RecordBatch::try_new(
///     Arc::new(schema),
///     vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
/// )
/// .unwrap();
///
/// let columns = gather_indices(&batch, &[2, 0]).unwrap();
/// let values = columns[0].as_any().downcast_ref::<Int32Array>().unwrap();
/// let collected: Vec<_> = values.values().iter().copied().collect();
/// assert_eq!(collected, vec![30, 10]);
/// ```
pub fn gather_indices(batch: &RecordBatch, indices: &[usize]) -> LlkvResult<Vec<ArrayRef>> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }

    let mut indices_vec: Vec<u32> = Vec::with_capacity(indices.len());
    indices_vec.extend(indices.iter().map(|&idx| idx as u32));
    let indices_array = UInt32Array::from(indices_vec);

    let mut result = Vec::with_capacity(batch.num_columns());
    for column in batch.columns() {
        let gathered = take(column.as_ref(), &indices_array, None)?;
        result.push(gathered);
    }

    Ok(result)
}

/// Gather rows from multiple [`RecordBatch`]es using `(batch_idx, row_idx)` pairs.
///
/// Indices are expected to be ordered by probe evaluation, but not necessarily
/// grouped by batch. The implementation groups contiguous references to the
/// same batch so each underlying column is scanned once per run, minimizing
/// Arrow allocations. The returned column slices are concatenated in the input
/// order. Supplying an empty `indices` slice returns an empty vector.
///
/// # Examples
///
/// ```
/// use arrow::array::Int32Array;
/// use arrow::datatypes::{DataType, Field, Schema};
/// use arrow::record_batch::RecordBatch;
/// use llkv_column_map::gather::gather_indices_from_batches;
/// use std::sync::Arc;
///
/// let schema = Arc::new(Schema::new(vec![Field::new("values", DataType::Int32, false)]));
/// let batch_a = RecordBatch::try_new(
///     schema.clone(),
///     vec![Arc::new(Int32Array::from(vec![1, 2]))],
/// )
/// .unwrap();
/// let batch_b = RecordBatch::try_new(
///     schema,
///     vec![Arc::new(Int32Array::from(vec![3, 4]))],
/// )
/// .unwrap();
///
/// let batches = vec![batch_a.clone(), batch_b.clone()];
/// let columns = gather_indices_from_batches(&batches, &[(0, 1), (1, 0)]).unwrap();
/// let values = columns[0].as_any().downcast_ref::<Int32Array>().unwrap();
/// let collected: Vec<_> = values.values().iter().copied().collect();
/// assert_eq!(collected, vec![2, 3]);
/// ```
pub fn gather_indices_from_batches(
    batches: &[RecordBatch],
    indices: &[(usize, usize)],
) -> LlkvResult<Vec<ArrayRef>> {
    if batches.is_empty() || indices.is_empty() {
        return Ok(Vec::new());
    }

    let mut grouped: Vec<(usize, UInt32Array)> = Vec::new();
    let mut current_batch = indices[0].0;
    let mut current_rows: Vec<u32> = Vec::new();

    for &(batch_idx, row_idx) in indices {
        if batch_idx == current_batch {
            current_rows.push(row_idx as u32);
        } else {
            if !current_rows.is_empty() {
                let array = UInt32Array::from(mem::take(&mut current_rows));
                grouped.push((current_batch, array));
            }
            current_batch = batch_idx;
            current_rows.push(row_idx as u32);
        }
    }

    if !current_rows.is_empty() {
        let array = UInt32Array::from(mem::take(&mut current_rows));
        grouped.push((current_batch, array));
    }

    let num_columns = batches[0].num_columns();
    let mut result = Vec::with_capacity(num_columns);

    for col_idx in 0..num_columns {
        let mut segments: Vec<ArrayRef> = Vec::with_capacity(grouped.len());
        for (batch_idx, rows) in &grouped {
            let column = batches[*batch_idx].column(col_idx);
            let segment = take(column.as_ref(), rows, None)?;
            segments.push(segment);
        }

        let concatenated = if segments.len() == 1 {
            segments.pop().unwrap()
        } else {
            let inputs: Vec<&dyn Array> = segments.iter().map(|a| a.as_ref()).collect();
            arrow::compute::concat(&inputs)?
        };
        result.push(concatenated);
    }

    Ok(result)
}

/// Gather rows from batches with optional matches, producing NULL runs when a
/// `None` entry is encountered.
///
/// This helper is tailored for left/outer joins where probe rows may lack a
/// match on the build side. It collapses consecutive `None` entries into a
/// single NULL array, drastically reducing temporary allocations compared to
/// emitting a one-row NULL array per miss.
pub fn gather_optional_indices_from_batches(
    batches: &[RecordBatch],
    indices: &[Option<(usize, usize)>],
) -> LlkvResult<Vec<ArrayRef>> {
    if batches.is_empty() || indices.is_empty() {
        return Ok(Vec::new());
    }

    enum Segment {
        Gather { batch_idx: usize, rows: UInt32Array },
        Null { len: usize },
    }

    let mut segments: Vec<Segment> = Vec::new();
    let mut current_batch: Option<usize> = None;
    let mut current_rows: Vec<u32> = Vec::new();
    let mut pending_nulls: usize = 0;

    let flush_batch = |segments: &mut Vec<Segment>,
                       current_batch: &mut Option<usize>,
                       current_rows: &mut Vec<u32>| {
        if let Some(batch_idx) = current_batch.take()
            && !current_rows.is_empty()
        {
            let rows = UInt32Array::from(mem::take(current_rows));
            segments.push(Segment::Gather { batch_idx, rows });
        }
    };

    let flush_nulls = |segments: &mut Vec<Segment>, pending_nulls: &mut usize| {
        if *pending_nulls > 0 {
            segments.push(Segment::Null {
                len: *pending_nulls,
            });
            *pending_nulls = 0;
        }
    };

    for opt in indices {
        match opt {
            Some((batch_idx, row_idx)) => {
                flush_nulls(&mut segments, &mut pending_nulls);
                if current_batch == Some(*batch_idx) {
                    current_rows.push(*row_idx as u32);
                } else {
                    flush_batch(&mut segments, &mut current_batch, &mut current_rows);
                    current_batch = Some(*batch_idx);
                    current_rows.push(*row_idx as u32);
                }
            }
            None => {
                flush_batch(&mut segments, &mut current_batch, &mut current_rows);
                pending_nulls += 1;
            }
        }
    }

    flush_batch(&mut segments, &mut current_batch, &mut current_rows);
    flush_nulls(&mut segments, &mut pending_nulls);

    let num_columns = batches[0].num_columns();
    let mut result = Vec::with_capacity(num_columns);

    for col_idx in 0..num_columns {
        let mut column_segments: Vec<ArrayRef> = Vec::with_capacity(segments.len());
        for segment in &segments {
            match segment {
                Segment::Gather { batch_idx, rows } => {
                    let column = batches[*batch_idx].column(col_idx);
                    let gathered = take(column.as_ref(), rows, None)?;
                    column_segments.push(gathered);
                }
                Segment::Null { len } => {
                    let template = batches[0].column(col_idx);
                    let null_array = arrow::array::new_null_array(template.data_type(), *len);
                    column_segments.push(null_array);
                }
            }
        }

        let concatenated = if column_segments.len() == 1 {
            column_segments.pop().unwrap()
        } else {
            let inputs: Vec<&dyn Array> = column_segments.iter().map(|a| a.as_ref()).collect();
            arrow::compute::concat(&inputs)?
        };
        result.push(concatenated);
    }

    Ok(result)
}

#[derive(Clone, Copy)]
pub(crate) enum RowLocator<'a> {
    Dense { base: RowId },
    Sparse { index: &'a FxHashMap<RowId, usize> },
}

impl<'a> RowLocator<'a> {
    #[inline]
    pub(crate) fn lookup(&self, row_id: RowId, len: usize) -> Option<usize> {
        match self {
            RowLocator::Dense { base } => {
                let offset = row_id.checked_sub(*base)?;
                if offset < len as u64 {
                    Some(offset as usize)
                } else {
                    None
                }
            }
            RowLocator::Sparse { index } => index.get(&row_id).copied(),
        }
    }
}

pub(crate) fn gather_rows_single_shot_string<O>(
    row_index: &FxHashMap<u64, usize>,
    len: usize,
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    candidate_indices: &[usize],
    chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
    allow_missing: bool,
) -> Result<ArrayRef>
where
    O: OffsetSizeTrait,
{
    if len == 0 {
        let mut builder = GenericStringBuilder::<O>::new();
        return Ok(Arc::new(builder.finish()) as ArrayRef);
    }

    let mut values: Vec<Option<String>> = vec![None; len];
    let mut found: Vec<bool> = vec![false; len];

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
            .downcast_ref::<GenericStringArray<O>>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = deserialize_array(row_chunk)?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        for i in 0..row_arr.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            if let Some(&out_idx) = row_index.get(&row_id) {
                found[out_idx] = true;
                if value_arr.is_null(i) {
                    values[out_idx] = None;
                } else {
                    values[out_idx] = Some(value_arr.value(i).to_owned());
                }
            }
        }
    }

    if !allow_missing {
        if found.iter().any(|f| !*f) {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    } else {
        for (idx, was_found) in found.iter().enumerate() {
            if !*was_found {
                values[idx] = None;
            }
        }
    }

    let total_bytes: usize = values
        .iter()
        .filter_map(|v| v.as_ref().map(|s| s.len()))
        .sum();

    let mut builder = GenericStringBuilder::<O>::with_capacity(len, total_bytes);
    for value in values {
        match value {
            Some(s) => builder.append_value(&s),
            None => builder.append_null(),
        }
    }

    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn gather_rows_single_shot_string_view(
    row_index: &FxHashMap<u64, usize>,
    len: usize,
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    candidate_indices: &[usize],
    chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
    allow_missing: bool,
) -> Result<ArrayRef> {
    if len == 0 {
        let mut builder = StringViewBuilder::new();
        return Ok(Arc::new(builder.finish()) as ArrayRef);
    }

    let mut values: Vec<Option<String>> = vec![None; len];
    let mut found: Vec<bool> = vec![false; len];

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
            .downcast_ref::<StringViewArray>()
            .ok_or_else(|| {
                Error::Internal(
                    "gather_rows_multi: dtype mismatch (expected StringViewArray)".into(),
                )
            })?;
        let row_any = deserialize_array(row_chunk)?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        for i in 0..row_arr.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            if let Some(&out_idx) = row_index.get(&row_id) {
                found[out_idx] = true;
                if value_arr.is_null(i) {
                    values[out_idx] = None;
                } else {
                    values[out_idx] = Some(value_arr.value(i).to_string());
                }
            }
        }
    }

    if !allow_missing {
        if found.iter().any(|f| !*f) {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    } else {
        for (idx, was_found) in found.iter().enumerate() {
            if !*was_found {
                values[idx] = None;
            }
        }
    }

    let mut builder = StringViewBuilder::with_capacity(len);
    for v in values {
        if let Some(s) = v {
            builder.append_value(s);
        } else {
            builder.append_null();
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn gather_rows_single_shot_bool(
    row_index: &FxHashMap<u64, usize>,
    len: usize,
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    candidate_indices: &[usize],
    chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
    allow_missing: bool,
) -> Result<ArrayRef> {
    if len == 0 {
        let empty = BooleanArray::from(Vec::<bool>::new());
        return Ok(Arc::new(empty) as ArrayRef);
    }

    let mut values: Vec<Option<bool>> = vec![None; len];
    let mut found: Vec<bool> = vec![false; len];

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
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = deserialize_array(row_chunk)?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        for i in 0..row_arr.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            if let Some(&out_idx) = row_index.get(&row_id) {
                found[out_idx] = true;
                if value_arr.is_null(i) {
                    values[out_idx] = None;
                } else {
                    values[out_idx] = Some(value_arr.value(i));
                }
            }
        }
    }

    if !allow_missing && found.iter().any(|f| !*f) {
        return Err(Error::Internal(
            "gather_rows_multi: one or more requested row IDs were not found".into(),
        ));
    }

    let array = BooleanArray::from(values);
    Ok(Arc::new(array) as ArrayRef)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_single_shot_struct(
    row_index: &FxHashMap<u64, usize>,
    len: usize,
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    candidate_indices: &[usize],
    chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
    allow_missing: bool,
    dtype: &DataType,
) -> Result<ArrayRef> {
    if len == 0 {
        if let DataType::Struct(fields) = dtype {
            let empty_columns: Vec<ArrayRef> = fields
                .iter()
                .map(|f| new_empty_array(f.data_type()))
                .collect();
            let empty = StructArray::try_new(fields.clone(), empty_columns, None)
                .map_err(|e| Error::Internal(format!("failed to create empty struct: {e}")))?;
            return Ok(Arc::new(empty) as ArrayRef);
        }
        return Err(Error::Internal("expected Struct dtype".into()));
    }

    let mut all_structs = Vec::new();
    let mut all_row_ids = Vec::new();

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
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::Internal("gather_rows_struct: dtype mismatch".into()))?;

        let row_any = deserialize_array(row_chunk)?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_struct: row_id downcast".into()))?;

        all_structs.push(value_arr.clone());
        all_row_ids.push(row_arr.clone());
    }

    let mut row_to_chunk_pos: FxHashMap<u64, (usize, usize)> = FxHashMap::default();
    for (chunk_idx, row_arr) in all_row_ids.iter().enumerate() {
        for i in 0..row_arr.len() {
            if row_arr.is_valid(i) {
                let row_id = row_arr.value(i);
                row_to_chunk_pos.insert(row_id, (chunk_idx, i));
            }
        }
    }

    let mut take_indices = vec![None; len];
    let mut found = vec![false; len];

    for (row_id, &out_idx) in row_index {
        if let Some(&(chunk_idx, pos)) = row_to_chunk_pos.get(row_id) {
            found[out_idx] = true;
            take_indices[out_idx] = Some((chunk_idx, pos));
        }
    }

    if !allow_missing && found.iter().any(|f| !*f) {
        return Err(Error::Internal(
            "gather_rows_struct: one or more requested row IDs were not found".into(),
        ));
    }

    let concat_refs: Vec<&dyn Array> = all_structs.iter().map(|a| a as &dyn Array).collect();
    let concatenated = compute::concat(&concat_refs)?;
    let concat_struct = concatenated
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| Error::Internal("concat result not a struct".into()))?;

    let mut cumulative_offsets = vec![0];
    for arr in &all_structs {
        cumulative_offsets.push(cumulative_offsets.last().unwrap() + arr.len());
    }

    let mut global_indices = Vec::with_capacity(len);
    for opt_chunk_pos in take_indices {
        if let Some((chunk_idx, pos)) = opt_chunk_pos {
            let global_idx = cumulative_offsets[chunk_idx] + pos;
            global_indices.push(Some(global_idx as u64));
        } else {
            global_indices.push(None);
        }
    }

    let indices = UInt64Array::from(global_indices);
    let result = compute::take(concat_struct, &indices, None)?;

    Ok(result)
}

pub(crate) fn gather_rows_single_shot<T>(
    row_index: &FxHashMap<u64, usize>,
    len: usize,
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    candidate_indices: &[usize],
    chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
    allow_missing: bool,
) -> Result<ArrayRef>
where
    T: ArrowPrimitiveType,
{
    if len == 0 {
        let empty = PrimitiveBuilder::<T>::new().finish();
        return Ok(Arc::new(empty) as ArrayRef);
    }

    let mut values: Vec<Option<T::Native>> = vec![None; len];
    let mut found: Vec<bool> = vec![false; len];

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

        for i in 0..row_arr.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            if let Some(&out_idx) = row_index.get(&row_id) {
                found[out_idx] = true;
                if value_arr.is_null(i) {
                    values[out_idx] = None;
                } else {
                    values[out_idx] = Some(value_arr.value(i));
                }
            }
        }
    }

    if !allow_missing {
        if found.iter().any(|f| !*f) {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    } else {
        for (idx, was_found) in found.iter().enumerate() {
            if !*was_found {
                values[idx] = None;
            }
        }
    }

    let array = PrimitiveArray::<T>::from_iter(values);
    Ok(Arc::new(array) as ArrayRef)
}

/// Gather rows for Decimal128, preserving precision and scale from the schema.
///
/// This is a specialized version of `gather_rows_single_shot` that handles
/// Decimal128 types. Unlike the generic primitive gather, we need to preserve
/// the exact precision and scale metadata from the schema's DataType.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_single_shot_decimal128(
    row_index: &FxHashMap<u64, usize>,
    len: usize,
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    candidate_indices: &[usize],
    chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
    allow_missing: bool,
    dtype: &DataType,
) -> Result<ArrayRef> {
    let (precision, scale) = match dtype {
        DataType::Decimal128(p, s) => (*p, *s),
        _ => {
            return Err(Error::Internal(
                "gather_rows_single_shot_decimal128: expected Decimal128 dtype".into(),
            ));
        }
    };

    if len == 0 {
        let empty = Decimal128Builder::new()
            .with_precision_and_scale(precision, scale)
            .map_err(|e| Error::Internal(format!("invalid Decimal128 precision/scale: {}", e)))?
            .finish();
        return Ok(Arc::new(empty) as ArrayRef);
    }

    let mut values: Vec<Option<i128>> = vec![None; len];
    let mut found: Vec<bool> = vec![false; len];

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
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = deserialize_array(row_chunk)?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        for i in 0..row_arr.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            if let Some(&out_idx) = row_index.get(&row_id) {
                found[out_idx] = true;
                if value_arr.is_null(i) {
                    values[out_idx] = None;
                } else {
                    values[out_idx] = Some(value_arr.value(i));
                }
            }
        }
    }

    if !allow_missing {
        if found.iter().any(|f| !*f) {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    } else {
        for (idx, was_found) in found.iter().enumerate() {
            if !*was_found {
                values[idx] = None;
            }
        }
    }

    let mut builder = Decimal128Builder::new()
        .with_precision_and_scale(precision, scale)
        .map_err(|e| Error::Internal(format!("invalid Decimal128 precision/scale: {}", e)))?;

    for value in values {
        match value {
            Some(v) => builder.append_value(v),
            None => builder.append_null(),
        }
    }

    let array = builder.finish();
    Ok(Arc::new(array) as ArrayRef)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_from_chunks_string<O>(
    row_ids: &[u64],
    row_locator: RowLocator,
    len: usize,
    candidate_indices: &[usize],
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
    row_scratch: &mut [Option<(usize, usize)>],
    allow_missing: bool,
) -> Result<ArrayRef>
where
    O: OffsetSizeTrait,
{
    if len == 0 {
        let mut builder = GenericStringBuilder::<O>::new();
        return Ok(Arc::new(builder.finish()) as ArrayRef);
    }

    if candidate_indices.len() == 1 {
        let chunk_idx = candidate_indices[0];
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let _value_arr = value_any
            .as_any()
            .downcast_ref::<GenericStringArray<O>>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        if row_arr.null_count() == 0 && row_ids.windows(2).all(|w| w[0] <= w[1]) {
            let values = row_arr.values();
            if let Ok(start_idx) = values.binary_search(&row_ids[0])
                && start_idx + len <= values.len()
                && row_ids == &values[start_idx..start_idx + len]
            {
                return Ok(value_any.slice(start_idx, len));
            }
        }
    }

    for slot in row_scratch.iter_mut().take(len) {
        *slot = None;
    }

    let mut candidates: Vec<(usize, &GenericStringArray<O>, &UInt64Array)> =
        Vec::with_capacity(candidate_indices.len());
    let mut chunk_lookup: FxHashMap<usize, usize> = FxHashMap::default();

    for (slot, &chunk_idx) in candidate_indices.iter().enumerate() {
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let value_arr = value_any
            .as_any()
            .downcast_ref::<GenericStringArray<O>>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
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
            if let Some(out_idx) = row_locator.lookup(row_id, len) {
                row_scratch[out_idx] = Some((chunk_idx, i));
            }
        }
    }

    let mut total_bytes = 0usize;
    for row_scratch_item in row_scratch.iter().take(len) {
        if let Some((chunk_idx, value_idx)) = row_scratch_item {
            let slot = *chunk_lookup
                .get(chunk_idx)
                .ok_or_else(|| Error::Internal("gather_rows_multi: chunk lookup missing".into()))?;
            let (_, value_arr, _) = candidates[slot];
            if !value_arr.is_null(*value_idx) {
                total_bytes += value_arr.value(*value_idx).len();
            }
        } else if !allow_missing {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    }

    let mut builder = GenericStringBuilder::<O>::with_capacity(len, total_bytes);
    for row_scratch_item in row_scratch.iter().take(len) {
        match row_scratch_item {
            Some((chunk_idx, value_idx)) => {
                let slot = *chunk_lookup.get(chunk_idx).ok_or_else(|| {
                    Error::Internal("gather_rows_multi: chunk lookup missing".into())
                })?;
                let (_, value_arr, _) = candidates[slot];
                if value_arr.is_null(*value_idx) {
                    builder.append_null();
                } else {
                    builder.append_value(value_arr.value(*value_idx));
                }
            }
            None => {
                if allow_missing {
                    builder.append_null();
                } else {
                    return Err(Error::Internal(
                        "gather_rows_multi: one or more requested row IDs were not found".into(),
                    ));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn gather_rows_single_shot_binary<O>(
    row_index: &FxHashMap<u64, usize>,
    len: usize,
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    candidate_indices: &[usize],
    chunk_blobs: &mut FxHashMap<PhysicalKey, EntryHandle>,
    allow_missing: bool,
) -> Result<ArrayRef>
where
    O: OffsetSizeTrait,
{
    if len == 0 {
        let mut builder = GenericBinaryBuilder::<O>::new();
        return Ok(Arc::new(builder.finish()) as ArrayRef);
    }

    let mut values: Vec<Option<Vec<u8>>> = vec![None; len];
    let mut found: Vec<bool> = vec![false; len];

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
            .downcast_ref::<GenericBinaryArray<O>>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = deserialize_array(row_chunk)?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        for i in 0..row_arr.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            if let Some(&out_idx) = row_index.get(&row_id) {
                found[out_idx] = true;
                if value_arr.is_null(i) {
                    values[out_idx] = None;
                } else {
                    values[out_idx] = Some(value_arr.value(i).to_vec());
                }
            }
        }
    }

    if !allow_missing {
        if found.iter().any(|f| !*f) {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    } else {
        for (idx, was_found) in found.iter().enumerate() {
            if !*was_found {
                values[idx] = None;
            }
        }
    }

    let total_bytes: usize = values
        .iter()
        .filter_map(|v| v.as_ref().map(|b| b.len()))
        .sum();

    let mut builder = GenericBinaryBuilder::<O>::with_capacity(len, total_bytes);
    for value in values {
        match value {
            Some(bytes) => builder.append_value(&bytes),
            None => builder.append_null(),
        }
    }

    Ok(Arc::new(builder.finish()) as ArrayRef)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_from_chunks_binary<O>(
    row_ids: &[u64],
    row_locator: RowLocator,
    len: usize,
    candidate_indices: &[usize],
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
    row_scratch: &mut [Option<(usize, usize)>],
    allow_missing: bool,
) -> Result<ArrayRef>
where
    O: OffsetSizeTrait,
{
    if len == 0 {
        let mut builder = GenericBinaryBuilder::<O>::new();
        return Ok(Arc::new(builder.finish()) as ArrayRef);
    }

    if candidate_indices.len() == 1 {
        let chunk_idx = candidate_indices[0];
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let _value_arr = value_any
            .as_any()
            .downcast_ref::<GenericBinaryArray<O>>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        if row_arr.null_count() == 0 && row_ids.windows(2).all(|w| w[0] <= w[1]) {
            let values_slice = row_arr.values();
            if let Ok(start_idx) = values_slice.binary_search(&row_ids[0])
                && start_idx + len <= values_slice.len()
                && row_ids == &values_slice[start_idx..start_idx + len]
            {
                return Ok(value_any.slice(start_idx, len));
            }
        }
    }

    for slot in row_scratch.iter_mut().take(len) {
        *slot = None;
    }

    let mut candidates: Vec<(usize, &GenericBinaryArray<O>, &UInt64Array)> =
        Vec::with_capacity(candidate_indices.len());
    let mut chunk_lookup: FxHashMap<usize, usize> = FxHashMap::default();

    for (slot, &chunk_idx) in candidate_indices.iter().enumerate() {
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let value_arr = value_any
            .as_any()
            .downcast_ref::<GenericBinaryArray<O>>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
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
            if let Some(out_idx) = row_locator.lookup(row_id, len) {
                row_scratch[out_idx] = Some((chunk_idx, i));
            }
        }
    }

    let mut total_bytes = 0usize;
    for row_scratch_item in row_scratch.iter().take(len) {
        if let Some((chunk_idx, value_idx)) = row_scratch_item {
            let slot = *chunk_lookup
                .get(chunk_idx)
                .ok_or_else(|| Error::Internal("gather_rows_multi: chunk lookup missing".into()))?;
            let (_, value_arr, _) = candidates[slot];
            if !value_arr.is_null(*value_idx) {
                total_bytes += value_arr.value(*value_idx).len();
            }
        } else if !allow_missing {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    }

    let mut builder = GenericBinaryBuilder::<O>::with_capacity(len, total_bytes);
    for row_scratch_item in row_scratch.iter().take(len) {
        match row_scratch_item {
            Some((chunk_idx, value_idx)) => {
                let slot = *chunk_lookup.get(chunk_idx).ok_or_else(|| {
                    Error::Internal("gather_rows_multi: chunk lookup missing".into())
                })?;
                let (_, value_arr, _) = candidates[slot];
                if value_arr.is_null(*value_idx) {
                    builder.append_null();
                } else {
                    builder.append_value(value_arr.value(*value_idx));
                }
            }
            None => {
                if allow_missing {
                    builder.append_null();
                } else {
                    return Err(Error::Internal(
                        "gather_rows_multi: one or more requested row IDs were not found".into(),
                    ));
                }
            }
        }
    }

    Ok(Arc::new(builder.finish()) as ArrayRef)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_from_chunks_bool(
    row_ids: &[u64],
    row_locator: RowLocator,
    len: usize,
    candidate_indices: &[usize],
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
    row_scratch: &mut [Option<(usize, usize)>],
    allow_missing: bool,
) -> Result<ArrayRef> {
    if len == 0 {
        let empty = BooleanArray::from(Vec::<bool>::new());
        return Ok(Arc::new(empty) as ArrayRef);
    }

    if candidate_indices.len() == 1 {
        let chunk_idx = candidate_indices[0];
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let _value_arr = value_any
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        if row_arr.null_count() == 0 && row_ids.windows(2).all(|w| w[0] <= w[1]) {
            let values = row_arr.values();
            if let Ok(start_idx) = values.binary_search(&row_ids[0])
                && start_idx + len <= values.len()
                && row_ids == &values[start_idx..start_idx + len]
            {
                return Ok(value_any.slice(start_idx, len));
            }
        }
    }

    for slot in row_scratch.iter_mut().take(len) {
        *slot = None;
    }

    let mut candidates: Vec<(usize, &BooleanArray, &UInt64Array)> =
        Vec::with_capacity(candidate_indices.len());
    let mut chunk_lookup: FxHashMap<usize, usize> = FxHashMap::default();

    for (slot, &chunk_idx) in candidate_indices.iter().enumerate() {
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let value_arr = value_any
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
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
            if let Some(out_idx) = row_locator.lookup(row_id, len) {
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

    let mut values: Vec<Option<bool>> = vec![None; len];
    for (out_idx, row_scratch_item) in row_scratch.iter().take(len).enumerate() {
        if let Some((chunk_idx, value_idx)) = row_scratch_item
            && let Some(&slot) = chunk_lookup.get(chunk_idx)
        {
            let (_idx, value_arr, _) = &candidates[slot];
            if value_arr.is_null(*value_idx) {
                values[out_idx] = None;
            } else {
                values[out_idx] = Some(value_arr.value(*value_idx));
            }
        }
    }

    let array = BooleanArray::from(values);
    Ok(Arc::new(array) as ArrayRef)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_from_chunks_struct(
    row_locator: RowLocator,
    len: usize,
    candidate_indices: &[usize],
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
    row_scratch: &mut [Option<(usize, usize)>],
    allow_missing: bool,
    dtype: &DataType,
) -> Result<ArrayRef> {
    if len == 0 {
        if let DataType::Struct(fields) = dtype {
            let empty_columns: Vec<ArrayRef> = fields
                .iter()
                .map(|f| new_empty_array(f.data_type()))
                .collect();
            let empty = StructArray::try_new(fields.clone(), empty_columns, None)
                .map_err(|e| Error::Internal(format!("failed to create empty struct: {e}")))?;
            return Ok(Arc::new(empty) as ArrayRef);
        }
        return Err(Error::Internal("expected Struct dtype".into()));
    }

    for slot in row_scratch.iter_mut().take(len) {
        *slot = None;
    }

    let mut candidates: Vec<(usize, &StructArray, &UInt64Array)> =
        Vec::with_capacity(candidate_indices.len());
    let mut chunk_lookup: FxHashMap<usize, usize> = FxHashMap::default();

    for (slot, &chunk_idx) in candidate_indices.iter().enumerate() {
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let value_arr = value_any
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::Internal("gather_rows_struct: dtype mismatch".into()))?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_struct: row_id downcast".into()))?;

        candidates.push((chunk_idx, value_arr, row_arr));
        chunk_lookup.insert(chunk_idx, slot);

        for i in 0..row_arr.len() {
            if !row_arr.is_valid(i) {
                continue;
            }
            let row_id = row_arr.value(i);
            if let Some(out_idx) = row_locator.lookup(row_id, len) {
                row_scratch[out_idx] = Some((chunk_idx, i));
            }
        }
    }

    if !allow_missing {
        for slot in row_scratch.iter().take(len) {
            if slot.is_none() {
                return Err(Error::Internal(
                    "gather_rows_struct: one or more requested row IDs were not found".into(),
                ));
            }
        }
    }

    let mut chunk_takes: FxHashMap<usize, Vec<Option<u64>>> = FxHashMap::default();
    for (chunk_idx, _, _) in &candidates {
        chunk_takes.insert(*chunk_idx, Vec::new());
    }

    for row_scratch_item in row_scratch.iter().take(len) {
        if let Some((chunk_idx, value_idx)) = row_scratch_item {
            for (cand_chunk_idx, _, _) in &candidates {
                if *cand_chunk_idx == *chunk_idx {
                    chunk_takes
                        .get_mut(cand_chunk_idx)
                        .unwrap()
                        .push(Some(*value_idx as u64));
                } else {
                    chunk_takes.get_mut(cand_chunk_idx).unwrap().push(None);
                }
            }
        } else {
            for (cand_chunk_idx, _, _) in &candidates {
                chunk_takes.get_mut(cand_chunk_idx).unwrap().push(None);
            }
        }
    }

    let mut results = Vec::new();
    for (chunk_idx, value_arr, _) in &candidates {
        let indices_vec = chunk_takes.get(chunk_idx).unwrap();
        if indices_vec.iter().any(|x| x.is_some()) {
            let indices = UInt64Array::from(indices_vec.clone());
            let taken = compute::take(*value_arr as &dyn Array, &indices, None)?;
            results.push(taken);
        }
    }

    if results.is_empty() {
        if let DataType::Struct(fields) = dtype {
            let empty_columns: Vec<ArrayRef> = fields
                .iter()
                .map(|f| new_empty_array(f.data_type()))
                .collect();
            let empty = StructArray::try_new(fields.clone(), empty_columns, None)
                .map_err(|e| Error::Internal(format!("failed to create empty struct: {e}")))?;
            return Ok(Arc::new(empty) as ArrayRef);
        }
        return Err(Error::Internal("no results for struct gather".into()));
    }

    Ok(results.into_iter().next().unwrap())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_from_chunks<T>(
    row_ids: &[u64],
    row_locator: RowLocator,
    len: usize,
    candidate_indices: &[usize],
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
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
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
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
            if let Ok(start_idx) = values.binary_search(&row_ids[0])
                && start_idx + len <= values.len()
                && row_ids == &values[start_idx..start_idx + len]
            {
                return Ok(value_any.slice(start_idx, len));
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
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let value_arr = value_any
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
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
            if let Some(out_idx) = row_locator.lookup(row_id, len) {
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
    for row_scratch_item in row_scratch.iter().take(len) {
        if let Some((chunk_idx, value_idx)) = *row_scratch_item {
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

/// Gather rows from chunks for Decimal128, preserving precision and scale from the schema.
///
/// This is a specialized version of `gather_rows_from_chunks` that handles Decimal128 types.
/// We need to preserve the exact precision and scale metadata from the schema's DataType.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows_from_chunks_decimal128(
    row_ids: &[u64],
    row_locator: RowLocator,
    len: usize,
    candidate_indices: &[usize],
    value_metas: &[ChunkMetadata],
    row_metas: &[ChunkMetadata],
    chunk_arrays: &FxHashMap<PhysicalKey, ArrayRef>,
    row_scratch: &mut [Option<(usize, usize)>],
    allow_missing: bool,
    dtype: &DataType,
) -> Result<ArrayRef> {
    let (precision, scale) = match dtype {
        DataType::Decimal128(p, s) => (*p, *s),
        _ => {
            return Err(Error::Internal(
                "gather_rows_from_chunks_decimal128: expected Decimal128 dtype".into(),
            ));
        }
    };

    if len == 0 {
        let empty = Decimal128Builder::new()
            .with_precision_and_scale(precision, scale)
            .map_err(|e| Error::Internal(format!("invalid Decimal128 precision/scale: {}", e)))?
            .finish();
        return Ok(Arc::new(empty) as ArrayRef);
    }

    if candidate_indices.len() == 1 {
        let chunk_idx = candidate_indices[0];
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let _value_arr = value_any
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_arr = row_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: row_id downcast".into()))?;

        if row_arr.null_count() == 0 && row_ids.windows(2).all(|w| w[0] <= w[1]) {
            let values = row_arr.values();
            if let Ok(start_idx) = values.binary_search(&row_ids[0])
                && start_idx + len <= values.len()
                && row_ids == &values[start_idx..start_idx + len]
            {
                return Ok(value_any.slice(start_idx, len));
            }
        }
    }

    for slot in row_scratch.iter_mut().take(len) {
        *slot = None;
    }

    let mut candidates: Vec<(usize, &Decimal128Array, &UInt64Array)> =
        Vec::with_capacity(candidate_indices.len());
    let mut chunk_lookup: FxHashMap<usize, usize> = FxHashMap::default();

    for (slot, &chunk_idx) in candidate_indices.iter().enumerate() {
        let value_any = chunk_arrays
            .get(&value_metas[chunk_idx].chunk_pk)
            .ok_or(Error::NotFound)?;
        let value_arr = value_any
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| Error::Internal("gather_rows_multi: dtype mismatch".into()))?;
        let row_any = chunk_arrays
            .get(&row_metas[chunk_idx].chunk_pk)
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
            if let Some(out_idx) = row_locator.lookup(row_id, len) {
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

    let mut builder = Decimal128Builder::new()
        .with_precision_and_scale(precision, scale)
        .map_err(|e| Error::Internal(format!("invalid Decimal128 precision/scale: {}", e)))?;

    for row_scratch_item in row_scratch.iter().take(len) {
        if let Some((chunk_idx, value_idx)) = *row_scratch_item {
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

pub(crate) fn filter_rows_with_non_null(columns: Vec<ArrayRef>) -> Result<Vec<ArrayRef>> {
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
