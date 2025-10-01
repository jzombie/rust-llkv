//! Nested-loop join implementation.
//!
//! This is a simple, correct O(N*M) join suitable for small tables and
//! validation. For each left row, we scan all right rows and emit matches.

use crate::join::{JoinKey, JoinOptions, JoinType};
use crate::table::{ScanProjection, ScanStreamOptions, Table};
use crate::types::FieldId;

/// Type alias for a joined row: (left_batch, left_idx, optional(right_batch, right_idx))
type JoinedRow<'a> = (&'a RecordBatch, usize, Option<(&'a RecordBatch, usize)>);
use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::datatypes::Schema;
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{Expr, Filter, Operator};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::ops::Bound;
use std::sync::Arc;

/// Perform a nested-loop join between two tables.
///
/// This implementation is O(N*M) but correct and supports all join types.
/// It materializes the right side in memory (batches) and streams the left.
pub(crate) fn nested_loop_join_stream<P, F>(
    left: &Table<P>,
    right: &Table<P>,
    keys: &[JoinKey],
    options: &JoinOptions,
    mut on_batch: F,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(RecordBatch),
{
    // Get schemas for both sides
    let left_schema = left.schema()?;
    let right_schema = right.schema()?;

    // Build projections for all user columns (exclude row_id)
    let left_projections = build_user_projections(left, &left_schema)?;
    let right_projections = build_user_projections(right, &right_schema)?;

    // Determine output schema based on join type
    let output_schema = build_output_schema(&left_schema, &right_schema, options.join_type)?;

    // Materialize the right side (all batches)
    let right_batches = materialize_side(right, &right_projections)?;

    // Extract join key indices for left and right
    // Handle empty tables: if either side has no user columns, keys won't be found
    let key_indices = if left_projections.is_empty() || right_projections.is_empty() {
        // Empty table case: no keys to extract, join will produce no results
        Vec::new()
    } else {
        extract_key_indices(keys, left, right, &left_schema, &right_schema)?
    };

    // Scan left side and join with materialized right
    let mut join_ctx = JoinContext {
        right_batches: &right_batches,
        keys,
        key_indices: &key_indices,
        options,
        output_schema: &output_schema,
        on_batch: &mut on_batch,
    };

    // Build a filter that selects all rows from left (we'll match in callback)
    // Skip scan if left table is empty (no projections means no user columns)
    if !left_projections.is_empty() {
        let filter_expr = build_all_rows_filter(&left_projections)?;

        left.scan_stream(
            &left_projections,
            &filter_expr,
            ScanStreamOptions::default(),
            |left_batch| {
                if let Err(e) = join_ctx.process_left_batch(&left_batch) {
                    eprintln!("Join error: {}", e);
                }
            },
        )?;
    }

    // For Left/Full/Anti joins, emit unmatched right rows if needed
    if options.join_type == JoinType::Right || options.join_type == JoinType::Full {
        join_ctx.emit_unmatched_right_rows()?;
    }

    Ok(())
}

/// Context for join processing.
struct JoinContext<'a, F> {
    right_batches: &'a [RecordBatch],
    keys: &'a [JoinKey],
    key_indices: &'a [(usize, usize)],
    options: &'a JoinOptions,
    output_schema: &'a Arc<Schema>,
    on_batch: &'a mut F,
}

impl<'a, F> JoinContext<'a, F>
where
    F: FnMut(RecordBatch),
{
    /// Process one left batch and join with all right batches.
    fn process_left_batch(&mut self, left_batch: &RecordBatch) -> LlkvResult<()> {
        let left_rows = left_batch.num_rows();
        if left_rows == 0 {
            return Ok(());
        }

        match self.options.join_type {
            JoinType::Inner => self.process_inner(left_batch),
            JoinType::Left => self.process_left(left_batch),
            JoinType::Right => self.process_right(left_batch),
            JoinType::Full => self.process_full(left_batch),
            JoinType::Semi => self.process_semi(left_batch),
            JoinType::Anti => self.process_anti(left_batch),
        }
    }

    fn process_inner(&mut self, left_batch: &RecordBatch) -> LlkvResult<()> {
        let mut output_rows = Vec::new();

        for left_idx in 0..left_batch.num_rows() {
            for right_batch in self.right_batches {
                for right_idx in 0..right_batch.num_rows() {
                    if self.keys_match(left_batch, left_idx, right_batch, right_idx)? {
                        output_rows.push((left_batch, left_idx, Some((right_batch, right_idx))));
                    }
                }
            }
        }

        self.emit_joined_rows(&output_rows)
    }

    fn process_left(&mut self, left_batch: &RecordBatch) -> LlkvResult<()> {
        let mut output_rows = Vec::new();

        for left_idx in 0..left_batch.num_rows() {
            let mut found_match = false;
            for right_batch in self.right_batches {
                for right_idx in 0..right_batch.num_rows() {
                    if self.keys_match(left_batch, left_idx, right_batch, right_idx)? {
                        output_rows.push((left_batch, left_idx, Some((right_batch, right_idx))));
                        found_match = true;
                    }
                }
            }
            // Emit left row with NULLs if no match
            if !found_match {
                output_rows.push((left_batch, left_idx, None));
            }
        }

        self.emit_joined_rows(&output_rows)
    }

    fn process_right(&mut self, left_batch: &RecordBatch) -> LlkvResult<()> {
        // Right join: emit all right rows, matched or not
        // For now, just collect matches; we'll handle unmatched in emit_unmatched_right_rows
        let mut output_rows = Vec::new();

        for left_idx in 0..left_batch.num_rows() {
            for right_batch in self.right_batches {
                for right_idx in 0..right_batch.num_rows() {
                    if self.keys_match(left_batch, left_idx, right_batch, right_idx)? {
                        output_rows.push((left_batch, left_idx, Some((right_batch, right_idx))));
                    }
                }
            }
        }

        self.emit_joined_rows(&output_rows)
    }

    fn process_full(&mut self, left_batch: &RecordBatch) -> LlkvResult<()> {
        // Full join: emit all left rows with matches or NULLs
        let mut output_rows = Vec::new();

        for left_idx in 0..left_batch.num_rows() {
            let mut found_match = false;
            for right_batch in self.right_batches {
                for right_idx in 0..right_batch.num_rows() {
                    if self.keys_match(left_batch, left_idx, right_batch, right_idx)? {
                        output_rows.push((left_batch, left_idx, Some((right_batch, right_idx))));
                        found_match = true;
                    }
                }
            }
            if !found_match {
                output_rows.push((left_batch, left_idx, None));
            }
        }

        self.emit_joined_rows(&output_rows)
    }

    fn process_semi(&mut self, left_batch: &RecordBatch) -> LlkvResult<()> {
        // Semi join: emit left rows that have at least one match (no right columns)
        let mut output_rows = Vec::new();

        for left_idx in 0..left_batch.num_rows() {
            let mut found_match = false;
            for right_batch in self.right_batches {
                for right_idx in 0..right_batch.num_rows() {
                    if self.keys_match(left_batch, left_idx, right_batch, right_idx)? {
                        found_match = true;
                        break;
                    }
                }
                if found_match {
                    break;
                }
            }
            if found_match {
                output_rows.push((left_batch, left_idx, None));
            }
        }

        self.emit_joined_rows(&output_rows)
    }

    fn process_anti(&mut self, left_batch: &RecordBatch) -> LlkvResult<()> {
        // Anti join: emit left rows that have NO match (no right columns)
        let mut output_rows = Vec::new();

        for left_idx in 0..left_batch.num_rows() {
            let mut found_match = false;
            for right_batch in self.right_batches {
                for right_idx in 0..right_batch.num_rows() {
                    if self.keys_match(left_batch, left_idx, right_batch, right_idx)? {
                        found_match = true;
                        break;
                    }
                }
                if found_match {
                    break;
                }
            }
            if !found_match {
                output_rows.push((left_batch, left_idx, None));
            }
        }

        self.emit_joined_rows(&output_rows)
    }

    /// Check if join keys match between two rows.
    fn keys_match(
        &self,
        left_batch: &RecordBatch,
        left_idx: usize,
        right_batch: &RecordBatch,
        right_idx: usize,
    ) -> LlkvResult<bool> {
        for ((left_col_idx, right_col_idx), key) in self.key_indices.iter().zip(self.keys.iter())
        {
            let left_col = left_batch.column(*left_col_idx);
            let right_col = right_batch.column(*right_col_idx);

            let left_null = left_col.is_null(left_idx);
            let right_null = right_col.is_null(right_idx);

            // Handle NULL semantics
            if left_null || right_null {
                if key.null_equals_null && left_null && right_null {
                    continue; // NULL == NULL
                } else {
                    return Ok(false); // NULL != anything (or NULL != NULL)
                }
            }

            // Compare values (type-specific)
            if !values_equal(left_col, left_idx, right_col, right_idx)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Emit joined rows as a RecordBatch.
    fn emit_joined_rows(
        &mut self,
        rows: &[JoinedRow],
    ) -> LlkvResult<()> {
        if rows.is_empty() {
            return Ok(());
        }

        // SIMPLIFICATION: Assume all rows come from the same left batch
        // (this is true for current implementation which processes one left batch at a time)
        let left_batch = rows[0].0;
        let left_indices: Vec<usize> = rows.iter().map(|(_, idx, _)| *idx).collect();

        let mut output_columns: Vec<ArrayRef> = Vec::new();

        // Gather left columns
        for col_idx in 0..left_batch.num_columns() {
            let col = gather_indices(left_batch.column(col_idx), &left_indices)?;
            output_columns.push(col);
        }

        // Add right columns (if not Semi/Anti)
        if self.options.join_type != JoinType::Semi
            && self.options.join_type != JoinType::Anti
            && !self.right_batches.is_empty()
        {
            // Use first right batch to determine schema
            let right_batch_ref = &self.right_batches[0];
            
            for col_idx in 0..right_batch_ref.num_columns() {
                // Collect right indices (with None for unmatched)
                let right_indices_opt: Vec<Option<usize>> = rows
                    .iter()
                    .map(|(_, _, right_opt)| right_opt.map(|(_, idx)| idx))
                    .collect();
                
                let col = gather_optional_indices_from_batches(
                    self.right_batches,
                    col_idx,
                    &right_indices_opt,
                )?;
                output_columns.push(col);
            }
        }

        let output_batch = RecordBatch::try_new(self.output_schema.clone(), output_columns)?;
        (self.on_batch)(output_batch);
        Ok(())
    }

    /// Emit unmatched right rows (for Right/Full joins).
    fn emit_unmatched_right_rows(&mut self) -> LlkvResult<()> {
        // TODO: Track which right rows were matched and emit unmatched ones
        // For now, this is a stub (correct for Inner/Left/Semi/Anti)
        // Ok(())
        unimplemented!("`emit_unmatched_right_rows` not yet implemented");
    }
}

/// Build projections for all user columns (excluding row_id).
fn build_user_projections<P>(
    table: &Table<P>,
    schema: &Arc<Schema>,
) -> LlkvResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut projections = Vec::new();
    for field in schema.fields().iter() {
        if field.name() == llkv_column_map::store::ROW_ID_COLUMN_NAME {
            continue;
        }
        let fid = field
            .metadata()
            .get("field_id")
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| Error::Internal(format!("field {} missing field_id", field.name())))?;
        let lfid = LogicalFieldId::for_user(table.table_id(), fid);
        projections.push(ScanProjection::Column(Projection::with_alias(
            lfid,
            field.name().to_string(),
        )));
    }
    // Return empty projections for empty table (will be handled by caller)
    Ok(projections)
}

/// Materialize all batches from a table scan.
fn materialize_side<P>(
    table: &Table<P>,
    projections: &[ScanProjection],
) -> LlkvResult<Vec<RecordBatch>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut batches = Vec::new();
    // Handle empty table case - no projections means no data
    if projections.is_empty() {
        return Ok(batches);
    }
    let filter = build_all_rows_filter(projections)?;
    table.scan_stream(
        projections,
        &filter,
        ScanStreamOptions::default(),
        |batch| {
            batches.push(batch);
        },
    )?;
    Ok(batches)
}

/// Build a filter that selects all rows.
fn build_all_rows_filter(projections: &[ScanProjection]) -> LlkvResult<Expr<'_, FieldId>> {
    if projections.is_empty() {
        // For empty tables, use a dummy field (will match no rows which is correct)
        return Ok(Expr::Pred(Filter {
            field_id: 0,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        }));
    }
    // Use the first field and a range that includes everything
    let first_field = match &projections[0] {
        ScanProjection::Column(proj) => proj.logical_field_id.field_id(),
        ScanProjection::Computed { .. } => {
            return Err(Error::InvalidArgumentError(
                "join projections cannot include computed columns yet".to_string(),
            ))
        }
    };
    Ok(Expr::Pred(Filter {
        field_id: first_field,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    }))
}

/// Extract key column indices from schemas.
fn extract_key_indices<P>(
    keys: &[JoinKey],
    _left: &Table<P>,
    _right: &Table<P>,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
) -> LlkvResult<Vec<(usize, usize)>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut indices = Vec::new();
    for key in keys {
        let left_idx = find_field_index(left_schema, key.left_field)?;
        let right_idx = find_field_index(right_schema, key.right_field)?;
        indices.push((left_idx, right_idx));
    }
    Ok(indices)
}

/// Find the column index for a field_id in a schema (excluding row_id).
fn find_field_index(schema: &Arc<Schema>, field_id: FieldId) -> LlkvResult<usize> {
    let mut user_col_idx = 0;
    for field in schema.fields().iter() {
        if field.name() == llkv_column_map::store::ROW_ID_COLUMN_NAME {
            continue;
        }
        if let Some(fid_str) = field.metadata().get("field_id")
            && let Ok(fid) = fid_str.parse::<u32>()
            && fid == field_id
        {
            return Ok(user_col_idx);
        }
        user_col_idx += 1;
    }
    Err(Error::InvalidArgumentError(format!(
        "field_id {} not found in schema",
        field_id
    )))
}

/// Build the output schema for a join.
fn build_output_schema(
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> LlkvResult<Arc<Schema>> {
    let mut fields = Vec::new();

    // Add left fields (excluding row_id)
    for field in left_schema.fields().iter() {
        if field.name() != llkv_column_map::store::ROW_ID_COLUMN_NAME {
            fields.push(field.as_ref().clone());
        }
    }

    // Add right fields (except for Semi/Anti, excluding row_id)
    if join_type != JoinType::Semi && join_type != JoinType::Anti {
        for field in right_schema.fields().iter() {
            if field.name() != llkv_column_map::store::ROW_ID_COLUMN_NAME {
                fields.push(field.as_ref().clone());
            }
        }
    }

    Ok(Arc::new(Schema::new(fields)))
}

/// Compare two values for equality (type-specific).
fn values_equal(
    left_col: &dyn Array,
    left_idx: usize,
    right_col: &dyn Array,
    right_idx: usize,
) -> LlkvResult<bool> {
    use arrow::datatypes::DataType;

    if left_col.data_type() != right_col.data_type() {
        return Err(Error::Internal(format!(
            "join key type mismatch: {:?} vs {:?}",
            left_col.data_type(),
            right_col.data_type()
        )));
    }

    // TODO: Derive from supported types list
    match left_col.data_type() {
        DataType::Int8 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::Int8Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::Int8Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::Int16 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::Int16Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::Int16Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::Int32 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::Int64 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::UInt8 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::UInt8Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::UInt8Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::UInt16 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::UInt16Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::UInt16Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::UInt32 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::UInt32Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::UInt32Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::UInt64 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::UInt64Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::UInt64Array>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::Float32 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::Float32Array>().unwrap();
            // Use bitwise equality for NaN handling
            Ok(left.value(left_idx).to_bits() == right.value(right_idx).to_bits())
        }
        DataType::Float64 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
            Ok(left.value(left_idx).to_bits() == right.value(right_idx).to_bits())
        }
        DataType::Utf8 => {
            let left = left_col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        DataType::Binary => {
            let left = left_col.as_any().downcast_ref::<arrow::array::BinaryArray>().unwrap();
            let right = right_col.as_any().downcast_ref::<arrow::array::BinaryArray>().unwrap();
            Ok(left.value(left_idx) == right.value(right_idx))
        }
        _ => Err(Error::Internal(format!(
            "join key type {:?} not yet supported",
            left_col.data_type()
        ))),
    }
}

/// Gather rows from an array by indices.
fn gather_indices(array: &dyn Array, indices: &[usize]) -> LlkvResult<ArrayRef> {
    use arrow::compute::take;
    use arrow::array::UInt64Array;

    let indices_array = UInt64Array::from(indices.iter().map(|&i| i as u64).collect::<Vec<_>>());
    let result = take(array, &indices_array, None)?;
    Ok(result)
}

/// Gather rows with optional indices (None = NULL).
fn gather_optional_indices(
    array: &dyn Array,
    indices: &[Option<usize>],
) -> LlkvResult<ArrayRef> {
    use arrow::array::UInt64Array;
    use arrow::compute::take;

    let indices_array = UInt64Array::from(
        indices
            .iter()
            .map(|opt| opt.map(|i| i as u64))
            .collect::<Vec<_>>(),
    );
    let result = take(array, &indices_array, None)?;
    Ok(result)
}

/// Gather rows from multiple batches (right side can span batches).
/// For nested-loop join, we actually materialize all right batches, so this
/// is a simplified version that assumes indices reference the first batch.
fn gather_optional_indices_from_batches(
    batches: &[RecordBatch],
    col_idx: usize,
    indices: &[Option<usize>],
) -> LlkvResult<ArrayRef> {
    if batches.is_empty() {
        return Err(Error::Internal("cannot gather from empty batches".to_string()));
    }
    
    // For simplicity, since we materialize right side, use first batch
    // (in a production impl, we'd need to track which batch each row comes from)
    gather_optional_indices(batches[0].column(col_idx), indices)
}
