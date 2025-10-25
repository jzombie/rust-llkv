//! Cartesian product helpers used by the executor and join engine.
//!
//! Cross joins conceptually replicate every row from the left input for each
//! row on the right. The helpers in this module centralize the expansion logic
//! so higher layers do not need to duplicate Arrow array plumbing.

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::Schema;
use llkv_result::Error;
use llkv_result::Result as LlkvResult;
use std::sync::Arc;

type SchemaRef = Arc<Schema>;

/// Build a cross join batch for a single pair of record batches.
///
/// The caller must supply an `output_schema` whose fields are the left schema
/// followed by the right schema. This keeps column ordering deterministic while
/// allowing higher layers to apply their own naming conventions.
pub fn cross_join_pair(
    left: &RecordBatch,
    right: &RecordBatch,
    output_schema: &SchemaRef,
) -> LlkvResult<RecordBatch> {
    let left_rows = left.num_rows();
    let right_rows = right.num_rows();

    if left_rows == 0 || right_rows == 0 {
        return Ok(RecordBatch::new_empty(Arc::clone(output_schema)));
    }

    let expected_columns = left.num_columns() + right.num_columns();
    if output_schema.fields().len() != expected_columns {
        return Err(Error::Internal(format!(
            "cross join schema mismatch: left columns {} + right columns {} != {}",
            left.num_columns(),
            right.num_columns(),
            output_schema.fields().len()
        )));
    }

    let total_rows = left_rows.checked_mul(right_rows).ok_or_else(|| {
        Error::InvalidArgumentError(format!(
            "cross join would produce {} x {} rows, exceeding limits",
            left_rows, right_rows
        ))
    })?;

    if left_rows > u32::MAX as usize {
        return Err(Error::InvalidArgumentError(
            "cross join left side exceeds supported row index range".into(),
        ));
    }
    if right_rows > u32::MAX as usize {
        return Err(Error::InvalidArgumentError(
            "cross join right side exceeds supported row index range".into(),
        ));
    }

    let mut left_indices = Vec::with_capacity(total_rows);
    let mut right_indices = Vec::with_capacity(total_rows);

    for left_idx in 0..left_rows {
        let left_as_u32 = left_idx as u32;
        for right_idx in 0..right_rows {
            left_indices.push(left_as_u32);
            right_indices.push(right_idx as u32);
        }
    }

    let left_index_array = UInt32Array::from(left_indices);
    let right_index_array = UInt32Array::from(right_indices);

    let mut columns = Vec::with_capacity(expected_columns);

    for (col_idx, column) in left.columns().iter().enumerate() {
        let expanded = take(column, &left_index_array, None).map_err(|err| {
            Error::InvalidArgumentError(format!(
                "failed to expand left column {col_idx} during cross join: {err}"
            ))
        })?;
        columns.push(expanded);
    }

    for (col_idx, column) in right.columns().iter().enumerate() {
        let expanded = take(column, &right_index_array, None).map_err(|err| {
            Error::InvalidArgumentError(format!(
                "failed to expand right column {col_idx} during cross join: {err}"
            ))
        })?;
        columns.push(expanded);
    }

    RecordBatch::try_new(Arc::clone(output_schema), columns).map_err(|err| {
        Error::Internal(format!(
            "failed to build cross join batch with {} rows: {}",
            total_rows, err
        ))
    })
}
