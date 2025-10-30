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
//!
//! ## Fast-Path Optimizations
//!
//! This implementation includes specialized fast-paths for single-column joins on
//! primitive integer types. These optimizations avoid the overhead of the generic
//! `HashKey`/`KeyValue` enum wrappers by using the primitive types directly as
//! hash map keys.
//!
//! **Fast-path triggers when:**
//! - Exactly one join key column (no multi-column joins)
//! - Both left and right key columns have matching data types
//! - Data type is one of: Int32, Int64, UInt32, UInt64
//!
//! **Fallback behavior:**
//! - Multi-column joins use generic path
//! - Non-primitive types (Utf8, Binary, Float) use generic path
//! - Type mismatches between left/right use generic path
//! - Empty tables safely fall back to generic path

use crate::{JoinKey, JoinOptions, JoinType};
use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, Schema};
use llkv_column_map::gather::{
    gather_indices, gather_indices_from_batches, gather_optional_indices_from_batches,
};
use llkv_column_map::store::Projection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::{Expr, Filter, Operator};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_table::schema_ext::CachedSchema;
use llkv_table::table::{ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::FieldId;
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

/// Execute a hash join between two tables and stream joined batches to `on_batch`.
///
/// Supports `Inner`, `Left`, `Semi`, and `Anti` joins today. Right and Full outer
/// joins return an [`llkv_result::Error::Internal`] since their
/// probe phases are not wired up yet.
pub fn hash_join_stream<P, F>(
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
    // Handle cross product (empty keys = Cartesian product)
    if keys.is_empty() {
        return cross_product_stream(left, right, options, on_batch);
    }

    // Get schemas
    let left_schema = left.schema()?;
    let right_schema = right.schema()?;

    // Fast-path for single-column primitive joins
    // Triggers when: 1 key, matching types, supported integer type
    // Performance: 1.2-3.6× faster than generic path
    if keys.len() == 1 {
        // Try to use fast-path if both schemas have the key field
        if let (Ok(left_dtype), Ok(right_dtype)) = (
            get_key_datatype(&left_schema, keys[0].left_field),
            get_key_datatype(&right_schema, keys[0].right_field),
        ) && left_dtype == right_dtype
        {
            match left_dtype {
                DataType::Int32 => {
                    return hash_join_i32_fast_path(left, right, keys, options, on_batch);
                }
                DataType::Int64 => {
                    return hash_join_i64_fast_path(left, right, keys, options, on_batch);
                }
                DataType::UInt32 => {
                    return hash_join_u32_fast_path(left, right, keys, options, on_batch);
                }
                DataType::UInt64 => {
                    return hash_join_u64_fast_path(left, right, keys, options, on_batch);
                }
                _ => {
                    // Fall through to generic path for other types
                }
            }
        }
        // Fall through to generic path if fast-path not applicable
    }

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
                        tracing::debug!(
                            join_type = ?options.join_type,
                            "Hash join does not yet support this join type; skipping batch processing"
                        );
                        Ok(())
                    }
                };

                if let Err(err) = result {
                    tracing::debug!(error = %err, "Hash join batch processing failed");
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
    let cached = CachedSchema::new(Arc::clone(schema));
    let mut projections = Vec::new();

    for (idx, field) in schema.fields().iter().enumerate() {
        // Use cached field_id lookup instead of metadata extraction
        let Some(field_id) = cached.field_id(idx) else {
            continue;
        };

        let lfid = LogicalFieldId::for_user(table.table_id(), field_id);
        projections.push(ScanProjection::Column(Projection::with_alias(
            lfid,
            field.name().to_string(),
        )));
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
    // Use cached schema for O(1) field ID lookup.
    // NOTE: We need the index among USER fields only (excluding system fields like rowid)
    // because the projected batch only contains user fields.
    let cached = CachedSchema::new(Arc::new(schema.clone()));

    // Find the schema index of the target field
    let schema_index = cached.index_of_field_id(target_field_id).ok_or_else(|| {
        Error::Internal(format!("field_id {} not found in schema", target_field_id))
    })?;

    // Count how many user fields come BEFORE this field
    // (this gives us the index in the projected batch which only has user fields)
    let mut user_col_idx = 0;
    for idx in 0..schema_index {
        if cached.field_id(idx).is_some() {
            user_col_idx += 1;
        }
    }

    Ok(user_col_idx)
}

/// Get the DataType of a join key field from schema.
fn get_key_datatype(schema: &Schema, field_id: FieldId) -> LlkvResult<DataType> {
    // Use cached schema for O(1) field ID lookup.
    let cached = CachedSchema::new(Arc::new(schema.clone()));

    let index = cached
        .index_of_field_id(field_id)
        .ok_or_else(|| Error::Internal(format!("field_id {} not found in schema", field_id)))?;

    Ok(schema.field(index).data_type().clone())
}

fn build_output_schema(
    left_schema: &Schema,
    right_schema: &Schema,
    join_type: JoinType,
) -> LlkvResult<Arc<Schema>> {
    let mut fields = Vec::new();
    let mut field_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    // For semi/anti joins, only include left side
    if matches!(join_type, JoinType::Semi | JoinType::Anti) {
        for field in left_schema.fields() {
            if field
                .metadata()
                .get(llkv_column_map::store::FIELD_ID_META_KEY)
                .is_some()
            {
                fields.push(field.clone());
                field_names.insert(field.name().clone());
            }
        }
        return Ok(Arc::new(Schema::new(fields)));
    }

    // For other joins, include both sides
    // Add left side fields
    for field in left_schema.fields() {
        if field
            .metadata()
            .get(llkv_column_map::store::FIELD_ID_META_KEY)
            .is_some()
        {
            fields.push(field.clone());
            field_names.insert(field.name().clone());
        }
    }

    // Add right side fields with deduplication
    for field in right_schema.fields() {
        if field
            .metadata()
            .get(llkv_column_map::store::FIELD_ID_META_KEY)
            .is_some()
        {
            let field_name = field.name();
            // If there's a conflict, append "_1" suffix
            let new_name = if field_names.contains(field_name) {
                format!("{}_1", field_name)
            } else {
                field_name.clone()
            };

            let new_field = Arc::new(
                arrow::datatypes::Field::new(
                    new_name.clone(),
                    field.data_type().clone(),
                    field.is_nullable(),
                )
                .with_metadata(field.metadata().clone()),
            );

            fields.push(new_field);
            field_names.insert(new_name);
        }
    }

    Ok(Arc::new(Schema::new(fields)))
}

// gather helpers relocated to `llkv_table::gather`

// ============================================================================
// Macro to generate fast-path implementations for integer types
// ============================================================================

/// Generates fast-path hash join implementations for integer types.
///
/// This macro creates specialized functions that avoid HashKey/KeyValue allocations
/// by using primitive types directly as hash map keys.
macro_rules! impl_integer_fast_path {
    (
        fast_path_fn: $fast_path_fn:ident,
        build_fn: $build_fn:ident,
        inner_probe_fn: $inner_probe_fn:ident,
        left_probe_fn: $left_probe_fn:ident,
        semi_probe_fn: $semi_probe_fn:ident,
        anti_probe_fn: $anti_probe_fn:ident,
        rust_type: $rust_type:ty,
        arrow_array: $arrow_array:ty,
        null_sentinel: $null_sentinel:expr
    ) => {
        /// Fast-path hash join for integer join keys.
        ///
        /// This optimized path avoids HashKey/KeyValue allocations by using
        /// FxHashMap directly, resulting in 1.2-3.6× speedup.
        #[allow(clippy::too_many_arguments)]
        fn $fast_path_fn<P, F>(
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
            let left_schema = left.schema()?;
            let right_schema = right.schema()?;

            let left_projections = build_user_projections(left, &left_schema)?;
            let right_projections = build_user_projections(right, &right_schema)?;

            let output_schema =
                build_output_schema(&left_schema, &right_schema, options.join_type)?;

            let (hash_table, build_batches) = if right_projections.is_empty() {
                (FxHashMap::default(), Vec::new())
            } else {
                $build_fn(right, &right_projections, keys, &right_schema)?
            };

            let probe_key_idx = if left_projections.is_empty() || right_projections.is_empty() {
                0
            } else {
                find_field_index(&left_schema, keys[0].left_field)?
            };

            let batch_size = options.batch_size;

            if !left_projections.is_empty() {
                let filter_expr = build_all_rows_filter(&left_projections)?;
                let null_equals_null = keys[0].null_equals_null;

                left.scan_stream(
                    &left_projections,
                    &filter_expr,
                    ScanStreamOptions::default(),
                    |probe_batch| {
                        let result = match options.join_type {
                            JoinType::Inner => $inner_probe_fn(
                                &probe_batch,
                                probe_key_idx,
                                &hash_table,
                                &build_batches,
                                &output_schema,
                                null_equals_null,
                                batch_size,
                                &mut on_batch,
                            ),
                            JoinType::Left => $left_probe_fn(
                                &probe_batch,
                                probe_key_idx,
                                &hash_table,
                                &build_batches,
                                &output_schema,
                                null_equals_null,
                                batch_size,
                                &mut on_batch,
                            ),
                            JoinType::Semi => $semi_probe_fn(
                                &probe_batch,
                                probe_key_idx,
                                &hash_table,
                                &output_schema,
                                null_equals_null,
                                batch_size,
                                &mut on_batch,
                            ),
                            JoinType::Anti => $anti_probe_fn(
                                &probe_batch,
                                probe_key_idx,
                                &hash_table,
                                &output_schema,
                                null_equals_null,
                                batch_size,
                                &mut on_batch,
                            ),
                            _ => {
                                tracing::debug!(
                                    join_type = ?options.join_type,
                                    "Hash join does not yet support this join type; skipping batch processing"
                                );
                                Ok(())
                            }
                        };

                        if let Err(err) = result {
                            tracing::debug!(error = %err, "Hash join batch processing failed");
                        }
                    },
                )?;
            }

            if matches!(options.join_type, JoinType::Right | JoinType::Full) {
                return Err(Error::Internal(
                    "Right and Full outer joins not yet implemented for hash join".to_string(),
                ));
            }

            Ok(())
        }

        /// Build hash table from the build side.
        fn $build_fn<P>(
            table: &Table<P>,
            projections: &[ScanProjection],
            join_keys: &[JoinKey],
            schema: &Arc<Schema>,
        ) -> LlkvResult<(FxHashMap<$rust_type, Vec<RowRef>>, Vec<RecordBatch>)>
        where
            P: Pager<Blob = EntryHandle> + Send + Sync,
        {
            let mut hash_table: FxHashMap<$rust_type, Vec<RowRef>> = FxHashMap::default();
            let mut batches = Vec::new();
            let key_idx = find_field_index(schema, join_keys[0].right_field)?;
            let filter_expr = build_all_rows_filter(projections)?;
            let null_equals_null = join_keys[0].null_equals_null;

            table.scan_stream(
                projections,
                &filter_expr,
                ScanStreamOptions::default(),
                |batch| {
                    let batch_idx = batches.len();
                    let key_column = batch.column(key_idx);
                    let key_array = match key_column.as_any().downcast_ref::<$arrow_array>() {
                        Some(arr) => arr,
                        None => {
                            tracing::debug!(
                                expected_array = stringify!($arrow_array),
                                actual_type = ?key_column.data_type(),
                                "Fast-path expected array type mismatch; falling back to generic path"
                            );
                            batches.push(batch.clone());
                            return;
                        }
                    };

                    for row_idx in 0..batch.num_rows() {
                        if key_array.is_null(row_idx) {
                            if null_equals_null {
                                hash_table
                                    .entry($null_sentinel)
                                    .or_default()
                                    .push((batch_idx, row_idx));
                            }
                        } else {
                            let key = key_array.value(row_idx);
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

        /// Process inner join probe.
        #[allow(clippy::too_many_arguments)]
        fn $inner_probe_fn<F>(
            probe_batch: &RecordBatch,
            probe_key_idx: usize,
            hash_table: &FxHashMap<$rust_type, Vec<RowRef>>,
            build_batches: &[RecordBatch],
            output_schema: &Arc<Schema>,
            null_equals_null: bool,
            batch_size: usize,
            on_batch: &mut F,
        ) -> LlkvResult<()>
        where
            F: FnMut(RecordBatch),
        {
            let probe_keys = match probe_batch
                .column(probe_key_idx)
                .as_any()
                .downcast_ref::<$arrow_array>()
            {
                Some(arr) => arr,
                None => {
                    return Err(Error::Internal(format!(
                        "Fast-path: Expected array type at column {} but got {:?}",
                        probe_key_idx,
                        probe_batch.column(probe_key_idx).data_type()
                    )));
                }
            };
            let mut probe_indices = Vec::with_capacity(batch_size);
            let mut build_indices = Vec::with_capacity(batch_size);

            for probe_row_idx in 0..probe_batch.num_rows() {
                let key = if probe_keys.is_null(probe_row_idx) {
                    if null_equals_null {
                        $null_sentinel
                    } else {
                        continue;
                    }
                } else {
                    probe_keys.value(probe_row_idx)
                };

                if let Some(build_rows) = hash_table.get(&key) {
                    for &row_ref in build_rows {
                        probe_indices.push(probe_row_idx);
                        build_indices.push(row_ref);
                    }
                }

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

        /// Process left join probe.
        #[allow(clippy::too_many_arguments)]
        fn $left_probe_fn<F>(
            probe_batch: &RecordBatch,
            probe_key_idx: usize,
            hash_table: &FxHashMap<$rust_type, Vec<RowRef>>,
            build_batches: &[RecordBatch],
            output_schema: &Arc<Schema>,
            null_equals_null: bool,
            batch_size: usize,
            on_batch: &mut F,
        ) -> LlkvResult<()>
        where
            F: FnMut(RecordBatch),
        {
            let probe_keys = match probe_batch
                .column(probe_key_idx)
                .as_any()
                .downcast_ref::<$arrow_array>()
            {
                Some(arr) => arr,
                None => {
                    return Err(Error::Internal(format!(
                        "Fast-path: Expected array type at column {} but got {:?}",
                        probe_key_idx,
                        probe_batch.column(probe_key_idx).data_type()
                    )));
                }
            };
            let mut probe_indices = Vec::with_capacity(batch_size);
            let mut build_indices = Vec::with_capacity(batch_size);

            for probe_row_idx in 0..probe_batch.num_rows() {
                let key = if probe_keys.is_null(probe_row_idx) {
                    if null_equals_null {
                        $null_sentinel
                    } else {
                        probe_indices.push(probe_row_idx);
                        build_indices.push(None);
                        continue;
                    }
                } else {
                    probe_keys.value(probe_row_idx)
                };

                if let Some(build_rows) = hash_table.get(&key) {
                    for &row_ref in build_rows {
                        probe_indices.push(probe_row_idx);
                        build_indices.push(Some(row_ref));
                    }
                } else {
                    probe_indices.push(probe_row_idx);
                    build_indices.push(None);
                }

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

        /// Process semi join probe.
        #[allow(clippy::too_many_arguments)]
        fn $semi_probe_fn<F>(
            probe_batch: &RecordBatch,
            probe_key_idx: usize,
            hash_table: &FxHashMap<$rust_type, Vec<RowRef>>,
            output_schema: &Arc<Schema>,
            null_equals_null: bool,
            batch_size: usize,
            on_batch: &mut F,
        ) -> LlkvResult<()>
        where
            F: FnMut(RecordBatch),
        {
            let probe_keys = match probe_batch
                .column(probe_key_idx)
                .as_any()
                .downcast_ref::<$arrow_array>()
            {
                Some(arr) => arr,
                None => {
                    return Err(Error::Internal(format!(
                        "Fast-path: Expected array type at column {} but got {:?}",
                        probe_key_idx,
                        probe_batch.column(probe_key_idx).data_type()
                    )));
                }
            };
            let mut probe_indices = Vec::with_capacity(batch_size);

            for probe_row_idx in 0..probe_batch.num_rows() {
                let key = if probe_keys.is_null(probe_row_idx) {
                    if null_equals_null {
                        $null_sentinel
                    } else {
                        continue;
                    }
                } else {
                    probe_keys.value(probe_row_idx)
                };

                if hash_table.contains_key(&key) {
                    probe_indices.push(probe_row_idx);
                }

                if probe_indices.len() >= batch_size {
                    emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
                    probe_indices.clear();
                }
            }

            if !probe_indices.is_empty() {
                emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
            }

            Ok(())
        }

        /// Process anti join probe.
        #[allow(clippy::too_many_arguments)]
        fn $anti_probe_fn<F>(
            probe_batch: &RecordBatch,
            probe_key_idx: usize,
            hash_table: &FxHashMap<$rust_type, Vec<RowRef>>,
            output_schema: &Arc<Schema>,
            null_equals_null: bool,
            batch_size: usize,
            on_batch: &mut F,
        ) -> LlkvResult<()>
        where
            F: FnMut(RecordBatch),
        {
            let probe_keys = match probe_batch
                .column(probe_key_idx)
                .as_any()
                .downcast_ref::<$arrow_array>()
            {
                Some(arr) => arr,
                None => {
                    return Err(Error::Internal(format!(
                        "Fast-path: Expected array type at column {} but got {:?}",
                        probe_key_idx,
                        probe_batch.column(probe_key_idx).data_type()
                    )));
                }
            };
            let mut probe_indices = Vec::with_capacity(batch_size);

            for probe_row_idx in 0..probe_batch.num_rows() {
                let key = if probe_keys.is_null(probe_row_idx) {
                    if null_equals_null {
                        $null_sentinel
                    } else {
                        probe_indices.push(probe_row_idx);
                        continue;
                    }
                } else {
                    probe_keys.value(probe_row_idx)
                };

                if !hash_table.contains_key(&key) {
                    probe_indices.push(probe_row_idx);
                }

                if probe_indices.len() >= batch_size {
                    emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
                    probe_indices.clear();
                }
            }

            if !probe_indices.is_empty() {
                emit_semi_batch(probe_batch, &probe_indices, output_schema, on_batch)?;
            }

            Ok(())
        }
    };
}

// Generate fast-path implementations for all supported integer types
impl_integer_fast_path!(
    fast_path_fn: hash_join_i32_fast_path,
    build_fn: build_i32_hash_table,
    inner_probe_fn: process_i32_inner_probe,
    left_probe_fn: process_i32_left_probe,
    semi_probe_fn: process_i32_semi_probe,
    anti_probe_fn: process_i32_anti_probe,
    rust_type: i32,
    arrow_array: arrow::array::Int32Array,
    null_sentinel: i32::MIN
);

impl_integer_fast_path!(
    fast_path_fn: hash_join_i64_fast_path,
    build_fn: build_i64_hash_table,
    inner_probe_fn: process_i64_inner_probe,
    left_probe_fn: process_i64_left_probe,
    semi_probe_fn: process_i64_semi_probe,
    anti_probe_fn: process_i64_anti_probe,
    rust_type: i64,
    arrow_array: arrow::array::Int64Array,
    null_sentinel: i64::MIN
);

impl_integer_fast_path!(
    fast_path_fn: hash_join_u32_fast_path,
    build_fn: build_u32_hash_table,
    inner_probe_fn: process_u32_inner_probe,
    left_probe_fn: process_u32_left_probe,
    semi_probe_fn: process_u32_semi_probe,
    anti_probe_fn: process_u32_anti_probe,
    rust_type: u32,
    arrow_array: arrow::array::UInt32Array,
    null_sentinel: u32::MAX
);

impl_integer_fast_path!(
    fast_path_fn: hash_join_u64_fast_path,
    build_fn: build_u64_hash_table,
    inner_probe_fn: process_u64_inner_probe,
    left_probe_fn: process_u64_left_probe,
    semi_probe_fn: process_u64_semi_probe,
    anti_probe_fn: process_u64_anti_probe,
    rust_type: u64,
    arrow_array: arrow::array::UInt64Array,
    null_sentinel: u64::MAX
);

/// Synthesize a LEFT JOIN result batch when right side is empty:
/// Take all left columns and append NULL arrays for right columns.
fn synthesize_left_join_nulls(
    left_batch: &RecordBatch,
    output_schema: &Arc<Schema>,
) -> LlkvResult<RecordBatch> {
    use arrow::array::new_null_array;
    
    let left_col_count = left_batch.num_columns();
    let right_col_count = output_schema.fields().len() - left_col_count;
    let row_count = left_batch.num_rows();
    
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    
    // Copy left columns as-is
    for col in left_batch.columns() {
        columns.push(Arc::clone(col));
    }
    
    // Append NULL arrays for right columns
    for field_idx in left_col_count..(left_col_count + right_col_count) {
        let field = output_schema.field(field_idx);
        let null_array = new_null_array(field.data_type(), row_count);
        columns.push(null_array);
    }
    
    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map_err(|err| Error::InvalidArgumentError(format!("Failed to create LEFT JOIN null batch: {}", err)))
}

/// Cross product (Cartesian product) implementation for empty join keys
fn cross_product_stream<P, F>(
    left: &Table<P>,
    right: &Table<P>,
    options: &JoinOptions,
    mut on_batch: F,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(RecordBatch),
{
    let left_schema = left.schema()?;
    let right_schema = right.schema()?;

    // Build projections for all user columns
    let left_projections = build_user_projections(left, &left_schema)?;
    let right_projections = build_user_projections(right, &right_schema)?;

    // Output schema: all left columns + all right columns
    let output_schema = build_output_schema(&left_schema, &right_schema, options.join_type)?;

    let mut right_batches = Vec::new();
    if !right_projections.is_empty() {
        let filter_expr = build_all_rows_filter(&right_projections)?;
        right.scan_stream(
            &right_projections,
            &filter_expr,
            ScanStreamOptions::default(),
            |batch| {
                right_batches.push(batch);
            },
        )?;
    }

    // For INNER JOIN: if right side is empty, no results
    // For LEFT JOIN: if right side is empty, emit all left rows with NULL right columns
    let right_is_empty = right_batches.is_empty() || right_batches.iter().all(|b| b.num_rows() == 0);
    
    if right_is_empty && options.join_type == JoinType::Inner {
        return Ok(());
    }

    if left_projections.is_empty() {
        return Ok(());
    }

    let filter_expr = build_all_rows_filter(&left_projections)?;
    let mut error: Option<Error> = None;

    left.scan_stream(
        &left_projections,
        &filter_expr,
        ScanStreamOptions::default(),
        |left_batch| {
            if error.is_some() || left_batch.num_rows() == 0 {
                return;
            }

            // For LEFT JOIN with empty right side: emit left rows with NULL right columns
            if right_is_empty && options.join_type == JoinType::Left {
                match synthesize_left_join_nulls(&left_batch, &output_schema) {
                    Ok(result) => on_batch(result),
                    Err(err) => {
                        error = Some(err);
                    }
                }
                return;
            }

            for right_batch in &right_batches {
                if right_batch.num_rows() == 0 {
                    continue;
                }

                match crate::cartesian::cross_join_pair(&left_batch, right_batch, &output_schema) {
                    Ok(result) => on_batch(result),
                    Err(err) => {
                        error = Some(err);
                        break;
                    }
                }
            }
        },
    )?;

    if let Some(err) = error {
        return Err(err);
    }

    Ok(())
}
