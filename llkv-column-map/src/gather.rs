//! Row gathering helpers for assembling Arrow arrays across chunks.
//!
//! These utilities provide shared implementations used by projections, joins,
//! and multi-column scans. They focus on minimizing temporary allocations while
//! preserving row order guarantees.

use std::mem;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Decimal128Array, Decimal128Builder,
    GenericBinaryArray, GenericBinaryBuilder, GenericStringArray, GenericStringBuilder,
    OffsetSizeTrait, PrimitiveArray, PrimitiveBuilder, RecordBatch, StructArray, UInt32Array,
    UInt64Array, new_empty_array,
};
use arrow::compute::{self, take};
use arrow::datatypes::{ArrowPrimitiveType, DataType};

use crate::store::descriptor::ChunkMetadata;
use crate::{Error, Result};
use llkv_result::Result as LlkvResult;
use llkv_storage::types::PhysicalKey;
use llkv_types::ids::RowId;
use rustc_hash::FxHashMap;

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

    // Zero-copy fast path: contiguous, increasing row window.
    let start = indices[0];
    let is_contiguous = indices.iter().enumerate().all(|(i, &idx)| idx == start + i);

    if is_contiguous {
        let len = indices.len();
        let mut result = Vec::with_capacity(batch.num_columns());
        for column in batch.columns() {
            result.push(column.slice(start, len));
        }
        return Ok(result);
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

    // Zero-copy fast path: all rows come from a single batch in a contiguous window.
    let single_batch = indices.iter().all(|(b, _)| *b == indices[0].0);
    if single_batch {
        let start = indices[0].1;
        let contiguous = indices
            .iter()
            .enumerate()
            .all(|(i, &(_, row))| row == start + i);
        if contiguous {
            let len = indices.len();
            let batch = &batches[indices[0].0];
            let mut result = Vec::with_capacity(batch.num_columns());
            for column in batch.columns() {
                result.push(column.slice(start, len));
            }
            return Ok(result);
        }
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

/// Gather a projected subset of columns from batches using `(batch_idx, row_idx)` pairs.
///
/// This mirrors [`gather_indices_from_batches`] but restricts the output to the selected
/// `projection` columns, preserving the input row order. Contiguous windows from a single
/// batch are returned as zero-copy slices; scattered references fall back to batched `take`
/// operations grouped by batch.
pub fn gather_projected_indices_from_batches(
    batches: &[RecordBatch],
    indices: &[(usize, usize)],
    projection: &[usize],
) -> LlkvResult<Vec<ArrayRef>> {
    if batches.is_empty() || indices.is_empty() || projection.is_empty() {
        return Ok(Vec::new());
    }

    let num_columns = batches[0].num_columns();
    for &col_idx in projection {
        if col_idx >= num_columns {
            return Err(Error::InvalidArgumentError(format!(
                "projection index {} out of bounds ({} columns)",
                col_idx, num_columns
            )));
        }
    }

    let single_batch = indices.iter().all(|(b, _)| *b == indices[0].0);
    if single_batch {
        let start = indices[0].1;
        let contiguous = indices
            .iter()
            .enumerate()
            .all(|(i, &(_, row))| row == start + i);
        if contiguous {
            let len = indices.len();
            let batch = &batches[indices[0].0];
            let mut result = Vec::with_capacity(projection.len());
            for &col_idx in projection {
                result.push(batch.column(col_idx).slice(start, len));
            }
            return Ok(result);
        }
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

    let mut result = Vec::with_capacity(projection.len());

    for &col_idx in projection {
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

    // Zero-copy fast path: all indices are Some, from one batch, contiguous.
    if indices.iter().all(|opt| opt.is_some()) {
        let first = indices[0].expect("checked is_some above");
        let single_batch = indices.iter().all(|opt| opt.unwrap().0 == first.0);
        if single_batch {
            let start = first.1;
            let contiguous = indices
                .iter()
                .enumerate()
                .all(|(i, opt)| opt.unwrap().1 == start + i);
            if contiguous {
                let len = indices.len();
                let batch = &batches[first.0];
                let mut result = Vec::with_capacity(batch.num_columns());
                for column in batch.columns() {
                    result.push(column.slice(start, len));
                }
                return Ok(result);
            }
        }
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

/// Gather a projected subset of columns from batches with optional matches, producing NULL
/// runs when a `None` entry is encountered. Contiguous windows from a single batch are
/// returned as zero-copy slices when all entries are `Some`.
pub fn gather_optional_projected_indices_from_batches(
    batches: &[RecordBatch],
    indices: &[Option<(usize, usize)>],
    projection: &[usize],
) -> LlkvResult<Vec<ArrayRef>> {
    if batches.is_empty() || indices.is_empty() || projection.is_empty() {
        return Ok(Vec::new());
    }

    let num_columns = batches[0].num_columns();
    for &col_idx in projection {
        if col_idx >= num_columns {
            return Err(Error::InvalidArgumentError(format!(
                "projection index {} out of bounds ({} columns)",
                col_idx, num_columns
            )));
        }
    }

    if indices.iter().all(|opt| opt.is_some()) {
        let first = indices[0].expect("checked is_some above");
        let single_batch = indices.iter().all(|opt| opt.unwrap().0 == first.0);
        if single_batch {
            let start = first.1;
            let contiguous = indices
                .iter()
                .enumerate()
                .all(|(i, opt)| opt.unwrap().1 == start + i);
            if contiguous {
                let len = indices.len();
                let batch = &batches[first.0];
                let mut result = Vec::with_capacity(projection.len());
                for &col_idx in projection {
                    result.push(batch.column(col_idx).slice(start, len));
                }
                return Ok(result);
            }
        }
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

    let mut result = Vec::with_capacity(projection.len());

    for &col_idx in projection {
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
    builder: &mut GenericStringBuilder<O>,
) -> Result<ArrayRef>
where
    O: OffsetSizeTrait,
{
    if len == 0 {
        let array = builder.finish();
        return Ok(Arc::new(array) as ArrayRef);
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

    for row_scratch_item in row_scratch.iter().take(len) {
        if row_scratch_item.is_none() && !allow_missing {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    }

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
    builder: &mut GenericBinaryBuilder<O>,
) -> Result<ArrayRef>
where
    O: OffsetSizeTrait,
{
    if len == 0 {
        let array = builder.finish();
        return Ok(Arc::new(array) as ArrayRef);
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

    for row_scratch_item in row_scratch.iter().take(len) {
        if row_scratch_item.is_none() && !allow_missing {
            return Err(Error::Internal(
                "gather_rows_multi: one or more requested row IDs were not found".into(),
            ));
        }
    }

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
    builder: &mut BooleanBuilder,
) -> Result<ArrayRef> {
    if len == 0 {
        let array = builder.finish();
        return Ok(Arc::new(array) as ArrayRef);
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

    for row_scratch_item in row_scratch.iter().take(len) {
        match row_scratch_item {
            Some((chunk_idx, value_idx)) => {
                let slot = *chunk_lookup.get(chunk_idx).ok_or_else(|| {
                    Error::Internal("gather_rows_multi: chunk lookup missing".into())
                })?;
                let (_, value_arr, _) = &candidates[slot];
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
    builder: &mut PrimitiveBuilder<T>,
) -> Result<ArrayRef>
where
    T: ArrowPrimitiveType,
{
    if len == 0 {
        let array = builder.finish();
        return Ok(Arc::new(array) as ArrayRef);
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
    builder: &mut Decimal128Builder,
) -> Result<ArrayRef> {
    if len == 0 {
        let array = builder.finish();
        return Ok(Arc::new(array) as ArrayRef);
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

#[cfg(test)]
mod tests {
    use super::{
        gather_optional_projected_indices_from_batches, gather_projected_indices_from_batches,
    };
    use arrow::array::{Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use llkv_result::Result as LlkvResult;
    use std::sync::Arc;

    #[test]
    fn projected_indices_zero_copy_contiguous() -> LlkvResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("label", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            ],
        )?;

        let out = gather_projected_indices_from_batches(&[batch.clone()], &[(0, 1), (0, 2)], &[1])?;

        let values = out[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        assert_eq!(values.value(0), "b");
        assert_eq!(values.value(1), "c");

        // Contiguous rows from a single batch should share backing buffers.
        let original = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        assert!(std::ptr::eq(
            values.values().as_ptr(),
            original.values().as_ptr()
        ));

        Ok(())
    }

    #[test]
    fn projected_indices_scattered_batches() -> LlkvResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let batch_a =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2]))])?;
        let batch_b = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![3, 4]))])?;

        let out = gather_projected_indices_from_batches(
            &[batch_a, batch_b],
            &[(0, 1), (1, 0), (1, 1)],
            &[0],
        )?;

        let values = out[0]
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int column");
        assert_eq!(values.values(), &[2, 3, 4]);

        Ok(())
    }

    #[test]
    fn projected_optional_indices_emit_nulls() -> LlkvResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![Some(5), None, Some(7)]))],
        )?;

        let out = gather_optional_projected_indices_from_batches(
            &[batch],
            &[Some((0, 0)), None, Some((0, 2))],
            &[0],
        )?;

        let values = out[0]
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int column");
        assert_eq!(values.len(), 3);
        assert!(!values.is_null(0));
        assert!(values.is_null(1));
        assert_eq!(values.value(2), 7);

        Ok(())
    }
}
