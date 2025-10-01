//! Hash join implementation.
//!
//! Hash join is O(N+M) compared to nested-loop's O(N×M), making it suitable
//! for production workloads with large datasets.
//!
//! Algorithm:
//! 1. Build phase: Scan the smaller table (build side) and insert rows into a hash map
//!    keyed by join columns. This creates an index for fast lookup.
//! 2. Probe phase: Scan the larger table (probe side) and for each row, look up
//!    matching rows in the hash map and emit joined results.
//!
//! For a 1M × 1M join:
//! - Nested-loop: 1 trillion comparisons (~hours)
//! - Hash join: 2 million rows processed (~seconds)

use crate::join::{JoinKey, JoinOptions, JoinType};
use crate::table::{ScanProjection, ScanStreamOptions, Table};
use crate::types::FieldId;
use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::take;
use arrow::datatypes::{DataType, Schema};
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{Expr, Filter, Operator};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::hash::{Hash, Hasher};
use std::ops::Bound;
use std::sync::Arc;

/// A hash key representing join column values for a single row.
#[derive(Debug, Clone, Eq)]
struct HashKey {
    values: Vec<KeyValue>,
}

/// A single join column value, with NULL handling.
#[derive(Debug, Clone)]
enum KeyValue {
    Null,
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(u32), // Store as bits for hashing
    Float64(u64), // Store as bits for hashing
    Utf8(String),
    Binary(Vec<u8>),
}

impl PartialEq for KeyValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (KeyValue::Null, KeyValue::Null) => false, // NULL != NULL by default
            (KeyValue::Int8(a), KeyValue::Int8(b)) => a == b,
            (KeyValue::Int16(a), KeyValue::Int16(b)) => a == b,
            (KeyValue::Int32(a), KeyValue::Int32(b)) => a == b,
            (KeyValue::Int64(a), KeyValue::Int64(b)) => a == b,
            (KeyValue::UInt8(a), KeyValue::UInt8(b)) => a == b,
            (KeyValue::UInt16(a), KeyValue::UInt16(b)) => a == b,
            (KeyValue::UInt32(a), KeyValue::UInt32(b)) => a == b,
            (KeyValue::UInt64(a), KeyValue::UInt64(b)) => a == b,
            (KeyValue::Float32(a), KeyValue::Float32(b)) => a == b,
            (KeyValue::Float64(a), KeyValue::Float64(b)) => a == b,
            (KeyValue::Utf8(a), KeyValue::Utf8(b)) => a == b,
            (KeyValue::Binary(a), KeyValue::Binary(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for KeyValue {}

impl Hash for KeyValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            KeyValue::Null => 0u8.hash(state),
            KeyValue::Int8(v) => v.hash(state),
            KeyValue::Int16(v) => v.hash(state),
            KeyValue::Int32(v) => v.hash(state),
            KeyValue::Int64(v) => v.hash(state),
            KeyValue::UInt8(v) => v.hash(state),
            KeyValue::UInt16(v) => v.hash(state),
            KeyValue::UInt32(v) => v.hash(state),
            KeyValue::UInt64(v) => v.hash(state),
            KeyValue::Float32(v) => v.hash(state),
            KeyValue::Float64(v) => v.hash(state),
            KeyValue::Utf8(v) => v.hash(state),
            KeyValue::Binary(v) => v.hash(state),
        }
    }
}

impl PartialEq for HashKey {
    fn eq(&self, other: &Self) -> bool {
        if self.values.len() != other.values.len() {
            return false;
        }
        self.values.iter().zip(&other.values).all(|(a, b)| a == b)
    }
}

impl Hash for HashKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in &self.values {
            value.hash(state);
        }
    }
}

/// A reference to a row in a batch: (batch_index, row_index)
type RowRef = (usize, usize);

/// Hash table mapping join keys to lists of matching rows.
type HashTable = FxHashMap<HashKey, Vec<RowRef>>;

/// Entry point for hash join algorithm.
pub(crate) fn hash_join_stream<P, F>(
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
    // Get schemas
    let left_schema = left.schema()?;
    let right_schema = right.schema()?;

    // Build projections for all user columns
    let left_projections = build_user_projections(left, &left_schema)?;
    let right_projections = build_user_projections(right, &right_schema)?;

    // Determine output schema based on join type
    let output_schema = build_output_schema(&left_schema, &right_schema, options.join_type)?;

    // For now, always use right as build side (future: choose smaller table)
    // Build phase: create hash table from right side
    let (hash_table, build_batches) = if right_projections.is_empty() {
        (HashTable::default(), Vec::new())
    } else {
        build_hash_table(right, &right_projections, keys, &right_schema)?
    };

    // Get key indices for probe side (left)
    let probe_key_indices = if left_projections.is_empty() || right_projections.is_empty() {
        Vec::new()
    } else {
        extract_left_key_indices(keys, &left_schema)?
    };

    // Probe phase: scan left side and emit matches
    let batch_size = options.batch_size;

    if !left_projections.is_empty() {
        let filter_expr = build_all_rows_filter(&left_projections)?;

        left.scan_stream(
            &left_projections,
            &filter_expr,
            ScanStreamOptions::default(),
            |probe_batch| {
                let result = match options.join_type {
                    JoinType::Inner => process_inner_probe(
                        &probe_batch,
                        &probe_key_indices,
                        &hash_table,
                        &build_batches,
                        &output_schema,
                        keys,
                        batch_size,
                        &mut on_batch,
                    ),
                    JoinType::Left => process_left_probe(
                        &probe_batch,
                        &probe_key_indices,
                        &hash_table,
                        &build_batches,
                        &output_schema,
                        keys,
                        batch_size,
                        &mut on_batch,
                    ),
                    JoinType::Semi => process_semi_probe(
                        &probe_batch,
                        &probe_key_indices,
                        &hash_table,
                        &output_schema,
                        keys,
                        batch_size,
                        &mut on_batch,
                    ),
                    JoinType::Anti => process_anti_probe(
                        &probe_batch,
                        &probe_key_indices,
                        &hash_table,
                        &output_schema,
                        keys,
                        batch_size,
                        &mut on_batch,
                    ),
                    _ => {
                        eprintln!(
                            "Hash join does not yet support {:?}, falling back would cause issues",
                            options.join_type
                        );
                        Ok(())
                    }
                };

                if let Err(e) = result {
                    eprintln!("Join error: {}", e);
                }
            },
        )?;
    }

    // For Right/Full joins, also emit unmatched build side rows
    if matches!(options.join_type, JoinType::Right | JoinType::Full) {
        return Err(Error::Internal(
            "Right and Full outer joins not yet implemented for hash join".to_string(),
        ));
    }

    Ok(())
}

/// Build hash table from the build side table.
fn build_hash_table<P>(
    table: &Table<P>,
    projections: &[ScanProjection],
    join_keys: &[JoinKey],
    schema: &Arc<Schema>,
) -> LlkvResult<(HashTable, Vec<RecordBatch>)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut hash_table = HashTable::default();
    let mut batches = Vec::new();
    let key_indices = extract_right_key_indices(join_keys, schema)?;
    let filter_expr = build_all_rows_filter(projections)?;

    table.scan_stream(
        projections,
        &filter_expr,
        ScanStreamOptions::default(),
        |batch| {
            let batch_idx = batches.len();

            // Extract keys for all rows in this batch
            for row_idx in 0..batch.num_rows() {
                if let Ok(key) = extract_hash_key(&batch, &key_indices, row_idx, join_keys) {
                    hash_table
                        .entry(key)
                        .or_default()
                        .push((batch_idx, row_idx));
                }
            }

            batches.push(batch.clone());
        },
    )?;

    Ok((hash_table, batches))
}

/// Extract hash key from a row.
fn extract_hash_key(
    batch: &RecordBatch,
    key_indices: &[usize],
    row_idx: usize,
    join_keys: &[JoinKey],
) -> LlkvResult<HashKey> {
    let mut values = Vec::with_capacity(key_indices.len());

    for (&col_idx, join_key) in key_indices.iter().zip(join_keys) {
        let column = batch.column(col_idx);

        // Handle NULL
        if column.is_null(row_idx) {
            if join_key.null_equals_null {
                values.push(KeyValue::Utf8("<NULL>".to_string())); // Treat NULLs as equal
            } else {
                values.push(KeyValue::Null);
            }
            continue;
        }

        let value = extract_key_value(column, row_idx)?;
        values.push(value);
    }

    Ok(HashKey { values })
}

/// Extract a single key value from an array.
fn extract_key_value(column: &ArrayRef, row_idx: usize) -> LlkvResult<KeyValue> {
    use arrow::array::*;

    let value = match column.data_type() {
        DataType::Int8 => KeyValue::Int8(
            column
                .as_any()
                .downcast_ref::<Int8Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::Int16 => KeyValue::Int16(
            column
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::Int32 => KeyValue::Int32(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::Int64 => KeyValue::Int64(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::UInt8 => KeyValue::UInt8(
            column
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::UInt16 => KeyValue::UInt16(
            column
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::UInt32 => KeyValue::UInt32(
            column
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::UInt64 => KeyValue::UInt64(
            column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row_idx),
        ),
        DataType::Float32 => {
            let val = column
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row_idx);
            KeyValue::Float32(val.to_bits())
        }
        DataType::Float64 => {
            let val = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row_idx);
            KeyValue::Float64(val.to_bits())
        }
        DataType::Utf8 => KeyValue::Utf8(
            column
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row_idx)
                .to_string(),
        ),
        DataType::Binary => KeyValue::Binary(
            column
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(row_idx)
                .to_vec(),
        ),
        dt => {
            return Err(Error::Internal(format!(
                "Unsupported join key type: {:?}",
                dt
            )));
        }
    };

    Ok(value)
}

/// Process inner join probe phase.
#[allow(clippy::too_many_arguments)]
fn process_inner_probe<F>(
    probe_batch: &RecordBatch,
    probe_key_indices: &[usize],
    hash_table: &HashTable,
    build_batches: &[RecordBatch],
    output_schema: &Arc<Schema>,
    join_keys: &[JoinKey],
    batch_size: usize,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    let mut probe_indices = Vec::new();
    let mut build_indices = Vec::new();

    for probe_row_idx in 0..probe_batch.num_rows() {
        if let Ok(key) = extract_hash_key(probe_batch, probe_key_indices, probe_row_idx, join_keys)
            && let Some(build_rows) = hash_table.get(&key)
        {
            for &(batch_idx, row_idx) in build_rows {
                probe_indices.push(probe_row_idx);
                build_indices.push((batch_idx, row_idx));
            }
        }

        // Emit batch if we've accumulated enough rows
        if probe_indices.len() >= batch_size {
            emit_joined_batch(
                probe_batch,
                &probe_indices,
                build_batches,
                &build_indices,
                output_schema,
                on_batch,
            )?;
            probe_indices.clear();
            build_indices.clear();
        }
    }

    // Emit remaining rows
    if !probe_indices.is_empty() {
        emit_joined_batch(
            probe_batch,
            &probe_indices,
            build_batches,
            &build_indices,
            output_schema,
            on_batch,
        )?;
    }

    Ok(())
}

/// Process left join probe phase.
#[allow(clippy::too_many_arguments)]
fn process_left_probe<F>(
    probe_batch: &RecordBatch,
    probe_key_indices: &[usize],
    hash_table: &HashTable,
    build_batches: &[RecordBatch],
    output_schema: &Arc<Schema>,
    join_keys: &[JoinKey],
    batch_size: usize,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    let mut probe_indices = Vec::new();
    let mut build_indices = Vec::new();

    for probe_row_idx in 0..probe_batch.num_rows() {
        let mut found_match = false;

        if let Ok(key) = extract_hash_key(probe_batch, probe_key_indices, probe_row_idx, join_keys)
            && let Some(build_rows) = hash_table.get(&key)
        {
            for &(batch_idx, row_idx) in build_rows {
                probe_indices.push(probe_row_idx);
                build_indices.push(Some((batch_idx, row_idx)));
                found_match = true;
            }
        }

        if !found_match {
            // No match - emit probe row with NULLs for build side
            probe_indices.push(probe_row_idx);
            build_indices.push(None);
        }

        // Emit batch if we've accumulated enough rows
        if probe_indices.len() >= batch_size {
            emit_left_joined_batch(
                probe_batch,
                &probe_indices,
                build_batches,
                &build_indices,
                output_schema,
                on_batch,
            )?;
            probe_indices.clear();
            build_indices.clear();
        }
    }

    // Emit remaining rows
    if !probe_indices.is_empty() {
        emit_left_joined_batch(
            probe_batch,
            &probe_indices,
            build_batches,
            &build_indices,
            output_schema,
            on_batch,
        )?;
    }

    Ok(())
}

/// Process semi join probe phase (only emit probe side if match exists).
#[allow(clippy::too_many_arguments)]
fn process_semi_probe<F>(
    probe_batch: &RecordBatch,
    probe_key_indices: &[usize],
    hash_table: &HashTable,
    output_schema: &Arc<Schema>,
    join_keys: &[JoinKey],
    batch_size: usize,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    let mut probe_indices = Vec::new();

    for probe_row_idx in 0..probe_batch.num_rows() {
        if let Ok(key) = extract_hash_key(probe_batch, probe_key_indices, probe_row_idx, join_keys)
            && hash_table.contains_key(&key)
        {
            probe_indices.push(probe_row_idx);
        }

        // Emit batch if we've accumulated enough rows
        if probe_indices.len() >= batch_size {
            emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
            probe_indices.clear();
        }
    }

    // Emit remaining rows
    if !probe_indices.is_empty() {
        emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
    }

    Ok(())
}

/// Process anti join probe phase (only emit probe side if no match).
#[allow(clippy::too_many_arguments)]
fn process_anti_probe<F>(
    probe_batch: &RecordBatch,
    probe_key_indices: &[usize],
    hash_table: &HashTable,
    output_schema: &Arc<Schema>,
    join_keys: &[JoinKey],
    batch_size: usize,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    let mut probe_indices = Vec::new();

    for probe_row_idx in 0..probe_batch.num_rows() {
        let mut found = false;
        if let Ok(key) = extract_hash_key(probe_batch, probe_key_indices, probe_row_idx, join_keys)
        {
            found = hash_table.contains_key(&key);
        }

        if !found {
            probe_indices.push(probe_row_idx);
        }

        // Emit batch if we've accumulated enough rows
        if probe_indices.len() >= batch_size {
            emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
            probe_indices.clear();
        }
    }

    // Emit remaining rows
    if !probe_indices.is_empty() {
        emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
    }

    Ok(())
}

/// Emit a joined batch for inner join.
fn emit_joined_batch<F>(
    probe_batch: &RecordBatch,
    probe_indices: &[usize],
    build_batches: &[RecordBatch],
    build_indices: &[(usize, usize)],
    output_schema: &Arc<Schema>,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    let probe_arrays = gather_indices(probe_batch, probe_indices)?;
    let build_arrays = gather_indices_from_batches(build_batches, build_indices)?;

    let output_arrays: Vec<ArrayRef> = probe_arrays.into_iter().chain(build_arrays).collect();

    let output_batch = RecordBatch::try_new(output_schema.clone(), output_arrays)?;
    on_batch(output_batch);
    Ok(())
}

/// Emit a joined batch for left join.
fn emit_left_joined_batch<F>(
    probe_batch: &RecordBatch,
    probe_indices: &[usize],
    build_batches: &[RecordBatch],
    build_indices: &[Option<(usize, usize)>],
    output_schema: &Arc<Schema>,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    let probe_arrays = gather_indices(probe_batch, probe_indices)?;
    let build_arrays = gather_optional_indices_from_batches(build_batches, build_indices)?;

    let output_arrays: Vec<ArrayRef> = probe_arrays.into_iter().chain(build_arrays).collect();

    let output_batch = RecordBatch::try_new(output_schema.clone(), output_arrays)?;
    on_batch(output_batch);
    Ok(())
}

/// Emit a batch for semi/anti join (probe side only).
fn emit_semi_batch<F>(
    probe_batch: &RecordBatch,
    probe_indices: &[usize],
    output_schema: &Arc<Schema>,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    F: FnMut(RecordBatch),
{
    let probe_arrays = gather_indices(probe_batch, probe_indices)?;
    let output_batch = RecordBatch::try_new(output_schema.clone(), probe_arrays)?;
    on_batch(output_batch);
    Ok(())
}

/// Helper functions (adapted from nested_loop.rs)
fn build_user_projections<P>(
    table: &Table<P>,
    schema: &Arc<Schema>,
) -> LlkvResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut projections = Vec::new();

    for field in schema.fields() {
        if field.name() == "row_id" {
            continue;
        }

        if let Some(field_id_str) = field.metadata().get("field_id") {
            let field_id: u32 = field_id_str.parse().map_err(|_| {
                Error::Internal(format!("Invalid field_id in schema: {}", field_id_str))
            })?;
            let lfid = LogicalFieldId::for_user(table.table_id(), field_id);
            projections.push(ScanProjection::Column(Projection::with_alias(
                lfid,
                field.name().to_string(),
            )));
        }
    }

    Ok(projections)
}

fn build_all_rows_filter(projections: &[ScanProjection]) -> LlkvResult<Expr<'static, FieldId>> {
    if projections.is_empty() {
        return Ok(Expr::Pred(Filter {
            field_id: 0,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        }));
    }

    let first_field = match &projections[0] {
        ScanProjection::Column(proj) => proj.logical_field_id.field_id(),
        ScanProjection::Computed { .. } => {
            return Err(Error::InvalidArgumentError(
                "join projections cannot include computed columns yet".to_string(),
            ));
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

fn extract_left_key_indices(keys: &[JoinKey], schema: &Arc<Schema>) -> LlkvResult<Vec<usize>> {
    keys.iter()
        .map(|key| find_field_index(schema, key.left_field))
        .collect()
}

fn extract_right_key_indices(keys: &[JoinKey], schema: &Arc<Schema>) -> LlkvResult<Vec<usize>> {
    keys.iter()
        .map(|key| find_field_index(schema, key.right_field))
        .collect()
}

fn find_field_index(schema: &Schema, target_field_id: FieldId) -> LlkvResult<usize> {
    let mut user_col_idx = 0;

    for field in schema.fields() {
        if field.name() == "row_id" {
            continue;
        }

        if let Some(field_id_str) = field.metadata().get("field_id") {
            let field_id: u32 = field_id_str.parse().map_err(|_| {
                Error::Internal(format!("Invalid field_id in schema: {}", field_id_str))
            })?;

            if field_id == target_field_id {
                return Ok(user_col_idx);
            }
        }

        user_col_idx += 1;
    }

    Err(Error::Internal(format!(
        "field_id {} not found in schema",
        target_field_id
    )))
}

fn build_output_schema(
    left_schema: &Schema,
    right_schema: &Schema,
    join_type: JoinType,
) -> LlkvResult<Arc<Schema>> {
    let mut fields = Vec::new();

    // For semi/anti joins, only include left side
    if matches!(join_type, JoinType::Semi | JoinType::Anti) {
        for field in left_schema.fields() {
            if field.name() != "row_id" {
                fields.push(field.clone());
            }
        }
        return Ok(Arc::new(Schema::new(fields)));
    }

    // For other joins, include both sides
    for field in left_schema.fields() {
        if field.name() != "row_id" {
            fields.push(field.clone());
        }
    }

    for field in right_schema.fields() {
        if field.name() != "row_id" {
            fields.push(field.clone());
        }
    }

    Ok(Arc::new(Schema::new(fields)))
}

fn gather_indices(batch: &RecordBatch, indices: &[usize]) -> LlkvResult<Vec<ArrayRef>> {
    let indices_array =
        arrow::array::UInt32Array::from(indices.iter().map(|&i| i as u32).collect::<Vec<_>>());

    let mut result = Vec::new();
    for column in batch.columns() {
        let gathered = take(column.as_ref(), &indices_array, None)?;
        result.push(gathered);
    }

    Ok(result)
}

fn gather_indices_from_batches(
    batches: &[RecordBatch],
    indices: &[(usize, usize)],
) -> LlkvResult<Vec<ArrayRef>> {
    if batches.is_empty() || indices.is_empty() {
        return Ok(Vec::new());
    }

    let num_columns = batches[0].num_columns();
    let mut result = Vec::with_capacity(num_columns);

    for col_idx in 0..num_columns {
        let mut column_data: Vec<ArrayRef> = Vec::new();

        for &(batch_idx, row_idx) in indices {
            let batch = &batches[batch_idx];
            let column = batch.column(col_idx);
            let single_row = take(
                column.as_ref(),
                &arrow::array::UInt32Array::from(vec![row_idx as u32]),
                None,
            )?;
            column_data.push(single_row);
        }

        let concatenated =
            arrow::compute::concat(&column_data.iter().map(|a| a.as_ref()).collect::<Vec<_>>())?;
        result.push(concatenated);
    }

    Ok(result)
}

fn gather_optional_indices_from_batches(
    batches: &[RecordBatch],
    indices: &[Option<(usize, usize)>],
) -> LlkvResult<Vec<ArrayRef>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let num_columns = batches[0].num_columns();
    let mut result = Vec::with_capacity(num_columns);

    for col_idx in 0..num_columns {
        let mut column_data: Vec<ArrayRef> = Vec::new();

        for opt_idx in indices {
            if let Some((batch_idx, row_idx)) = opt_idx {
                let batch = &batches[*batch_idx];
                let column = batch.column(col_idx);
                let single_row = take(
                    column.as_ref(),
                    &arrow::array::UInt32Array::from(vec![*row_idx as u32]),
                    None,
                )?;
                column_data.push(single_row);
            } else {
                // NULL value for unmatched row
                let column = batches[0].column(col_idx);
                let null_array = arrow::array::new_null_array(column.data_type(), 1);
                column_data.push(null_array);
            }
        }

        let concatenated =
            arrow::compute::concat(&column_data.iter().map(|a| a.as_ref()).collect::<Vec<_>>())?;
        result.push(concatenated);
    }

    Ok(result)
}
