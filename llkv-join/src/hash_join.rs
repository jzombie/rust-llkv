//! Hash join implementation optimized for row-id streaming (zero-copy friendly).
//! Materialization is intentionally avoided; consumers can project rows lazily using
//! the recorded row references.

use crate::{JoinIndexBatch, JoinKey, JoinOptions, JoinRowRef, JoinType};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::Projection;
use llkv_expr::{Expr, Filter, Operator};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_table::schema_ext::CachedSchema;
use llkv_table::table::{RowIdFilter, ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::FieldId;
use llkv_types::LogicalFieldId;
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

/// A single join column value with explicit NULL handling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyValue {
    NullEqual,
    NullSide { is_left: bool, id: usize },
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(u32),
    Float64(u64),
    Utf8(String),
    Binary(Vec<u8>),
}

impl KeyValue {
    fn null_as_key(is_left: bool, row_idx: usize, null_equals_null: bool) -> Self {
        if null_equals_null {
            KeyValue::NullEqual
        } else {
            KeyValue::NullSide {
                is_left,
                id: row_idx,
            }
        }
    }
}

impl PartialEq for HashKey {
    fn eq(&self, other: &Self) -> bool {
        self.values.len() == other.values.len()
            && self.values.iter().zip(&other.values).all(|(a, b)| a == b)
    }
}

impl Hash for HashKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in &self.values {
            value.hash(state);
        }
    }
}

/// A reference to a row in a batch: (batch_index, row_index).
type RowRef = (usize, usize);

/// Hash table mapping join keys to lists of matching rows.
type HashTable = FxHashMap<HashKey, Vec<RowRef>>;

/// Execute a hash join but emit row-id pairs for zero-copy late materialization.
pub fn hash_join_rowid_stream<P, F>(
    left: &Table<P>,
    right: &Table<P>,
    keys: &[JoinKey],
    options: &JoinOptions,
    on_batch: F,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(JoinIndexBatch<'_>),
{
    hash_join_rowid_stream_with_filters(left, right, keys, options, None, None, on_batch)
}

pub fn hash_join_rowid_stream_with_filters<P, F>(
    left: &Table<P>,
    right: &Table<P>,
    keys: &[JoinKey],
    options: &JoinOptions,
    left_filter: Option<Arc<dyn RowIdFilter<P>>>,
    right_filter: Option<Arc<dyn RowIdFilter<P>>>,
    mut on_batch: F,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(JoinIndexBatch<'_>),
{
    if matches!(options.join_type, JoinType::Right | JoinType::Full) {
        return Err(Error::Internal(
            "Right and Full outer joins not yet implemented for hash join".to_string(),
        ));
    }

    if keys.is_empty() {
        return cross_product_rowid_stream(
            left,
            right,
            options,
            left_filter,
            right_filter,
            on_batch,
        );
    }

    let left_schema = left.schema()?;
    let right_schema = right.schema()?;

    let left_projections = build_user_projections(left, &left_schema)?;
    let right_projections = build_user_projections(right, &right_schema)?;

    let (hash_table, build_batches) = build_hash_table(
        right,
        &right_projections,
        keys,
        &right_schema,
        right_filter.clone(),
    )?;
    let probe_key_indices = extract_left_key_indices(keys, &left_schema)?;

    let batch_size = options.batch_size.max(1);
    let probe_filter = build_all_rows_filter(&left_projections)?;

    fn flush_batch<'a, FCb>(
        left_batch: &'a RecordBatch,
        left_rows: &mut Vec<usize>,
        right_rows: &mut Vec<Option<JoinRowRef>>,
        right_batches: &'a [RecordBatch],
        cb: &mut FCb,
    ) where
        FCb: FnMut(JoinIndexBatch<'a>),
    {
        if left_rows.is_empty() {
            return;
        }
        let batch = JoinIndexBatch {
            left_batch,
            left_rows: std::mem::take(left_rows),
            right_rows: std::mem::take(right_rows),
            right_batches,
        };
        cb(batch);
    }

    left.scan_stream(
        &left_projections,
        &probe_filter,
        ScanStreamOptions {
            row_id_filter: left_filter,
            ..ScanStreamOptions::default()
        },
        |probe_batch| {
            let mut left_rows: Vec<usize> = Vec::with_capacity(batch_size);
            let mut right_rows: Vec<Option<JoinRowRef>> = Vec::with_capacity(batch_size);

            for probe_row_idx in 0..probe_batch.num_rows() {
                let key = match extract_hash_key(
                    &probe_batch,
                    &probe_key_indices,
                    probe_row_idx,
                    keys,
                    true,
                ) {
                    Ok(key) => key,
                    Err(err) => panic!("{err}"),
                };
                let matches = hash_table.get(&key);

                match options.join_type {
                    JoinType::Inner => {
                        if let Some(build_rows) = matches {
                            for &(build_batch_idx, build_row_idx) in build_rows {
                                left_rows.push(probe_row_idx);
                                right_rows.push(Some(JoinRowRef {
                                    batch: build_batch_idx,
                                    row: build_row_idx,
                                }));

                                if left_rows.len() >= batch_size {
                                    flush_batch(
                                        &probe_batch,
                                        &mut left_rows,
                                        &mut right_rows,
                                        &build_batches,
                                        &mut on_batch,
                                    );
                                }
                            }
                        }
                    }
                    JoinType::Left => {
                        if let Some(build_rows) = matches {
                            for &(build_batch_idx, build_row_idx) in build_rows {
                                left_rows.push(probe_row_idx);
                                right_rows.push(Some(JoinRowRef {
                                    batch: build_batch_idx,
                                    row: build_row_idx,
                                }));

                                if left_rows.len() >= batch_size {
                                    flush_batch(
                                        &probe_batch,
                                        &mut left_rows,
                                        &mut right_rows,
                                        &build_batches,
                                        &mut on_batch,
                                    );
                                }
                            }
                        } else {
                            left_rows.push(probe_row_idx);
                            right_rows.push(None);

                            if left_rows.len() >= batch_size {
                                flush_batch(
                                    &probe_batch,
                                    &mut left_rows,
                                    &mut right_rows,
                                    &build_batches,
                                    &mut on_batch,
                                );
                            }
                        }
                    }
                    JoinType::Semi => {
                        if matches.map(|m| !m.is_empty()).unwrap_or(false) {
                            left_rows.push(probe_row_idx);
                            right_rows.push(None);

                            if left_rows.len() >= batch_size {
                                flush_batch(
                                    &probe_batch,
                                    &mut left_rows,
                                    &mut right_rows,
                                    &build_batches,
                                    &mut on_batch,
                                );
                            }
                        }
                    }
                    JoinType::Anti => {
                        if matches.is_none() || matches.unwrap().is_empty() {
                            left_rows.push(probe_row_idx);
                            right_rows.push(None);

                            if left_rows.len() >= batch_size {
                                flush_batch(
                                    &probe_batch,
                                    &mut left_rows,
                                    &mut right_rows,
                                    &build_batches,
                                    &mut on_batch,
                                );
                            }
                        }
                    }
                    JoinType::Right | JoinType::Full => {
                        panic!("Right and Full outer joins not yet implemented for hash join");
                    }
                }
            }

            flush_batch(
                &probe_batch,
                &mut left_rows,
                &mut right_rows,
                &build_batches,
                &mut on_batch,
            );
        },
    )?;

    Ok(())
}

fn cross_product_rowid_stream<P, F>(
    left: &Table<P>,
    right: &Table<P>,
    options: &JoinOptions,
    left_filter: Option<Arc<dyn RowIdFilter<P>>>,
    right_filter: Option<Arc<dyn RowIdFilter<P>>>,
    mut on_batch: F,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(JoinIndexBatch<'_>),
{
    let left_schema = left.schema()?;
    let right_schema = right.schema()?;
    let left_projections = build_user_projections(left, &left_schema)?;
    let right_projections = build_user_projections(right, &right_schema)?;

    let mut right_batches = Vec::new();
    let filter_left = build_all_rows_filter(&left_projections)?;
    let filter_right = build_all_rows_filter(&right_projections)?;

    right.scan_stream(
        &right_projections,
        &filter_right,
        ScanStreamOptions {
            row_id_filter: right_filter,
            ..ScanStreamOptions::default()
        },
        |batch| right_batches.push(batch.clone()),
    )?;

    fn flush_cross<'a, FCb>(
        left_batch: &'a RecordBatch,
        left_rows: &mut Vec<usize>,
        right_rows: &mut Vec<Option<JoinRowRef>>,
        right_batches: &'a [RecordBatch],
        cb: &mut FCb,
    ) where
        FCb: FnMut(JoinIndexBatch<'a>),
    {
        if left_rows.is_empty() {
            return;
        }

        let batch = JoinIndexBatch {
            left_batch,
            left_rows: std::mem::take(left_rows),
            right_rows: std::mem::take(right_rows),
            right_batches,
        };
        cb(batch);
    }

    let batch_size = options.batch_size.max(1);
    left.scan_stream(
        &left_projections,
        &filter_left,
        ScanStreamOptions {
            row_id_filter: left_filter,
            ..ScanStreamOptions::default()
        },
        |left_batch| {
            let mut left_rows: Vec<usize> = Vec::with_capacity(batch_size);
            let mut right_rows: Vec<Option<JoinRowRef>> = Vec::with_capacity(batch_size);

            for l_row in 0..left_batch.num_rows() {
                for (r_idx, r_batch) in right_batches.iter().enumerate() {
                    for r_row in 0..r_batch.num_rows() {
                        left_rows.push(l_row);
                        right_rows.push(Some(JoinRowRef {
                            batch: r_idx,
                            row: r_row,
                        }));
                        if left_rows.len() >= batch_size {
                            flush_cross(
                                &left_batch,
                                &mut left_rows,
                                &mut right_rows,
                                &right_batches,
                                &mut on_batch,
                            );
                        }
                    }
                }
            }

            flush_cross(
                &left_batch,
                &mut left_rows,
                &mut right_rows,
                &right_batches,
                &mut on_batch,
            );
        },
    )?;

    Ok(())
}

fn build_hash_table<P>(
    table: &Table<P>,
    projections: &[ScanProjection],
    join_keys: &[JoinKey],
    schema: &Arc<Schema>,
    row_filter: Option<Arc<dyn RowIdFilter<P>>>,
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
        ScanStreamOptions {
            row_id_filter: row_filter,
            ..ScanStreamOptions::default()
        },
        |batch| {
            let batch_idx = batches.len();

            for row_idx in 0..batch.num_rows() {
                if let Ok(key) = extract_hash_key(&batch, &key_indices, row_idx, join_keys, false) {
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

fn extract_hash_key(
    batch: &RecordBatch,
    key_indices: &[usize],
    row_idx: usize,
    join_keys: &[JoinKey],
    is_left: bool,
) -> LlkvResult<HashKey> {
    let mut values = Vec::with_capacity(key_indices.len());

    for (&col_idx, join_key) in key_indices.iter().zip(join_keys) {
        let column = batch.column(col_idx);

        if column.is_null(row_idx) {
            values.push(KeyValue::null_as_key(
                is_left,
                row_idx,
                join_key.null_equals_null,
            ));
            continue;
        }

        let value = extract_key_value(column, row_idx)?;
        values.push(value);
    }

    Ok(HashKey { values })
}

fn extract_key_value(column: &arrow::array::ArrayRef, row_idx: usize) -> LlkvResult<KeyValue> {
    use arrow::array::*;

    let value = match column.data_type() {
        DataType::Int8 => KeyValue::Int8(
            column
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| Error::Internal("Expected Int8Array".to_string()))?
                .value(row_idx),
        ),
        DataType::Int16 => KeyValue::Int16(
            column
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| Error::Internal("Expected Int16Array".to_string()))?
                .value(row_idx),
        ),
        DataType::Int32 => KeyValue::Int32(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::Internal("Expected Int32Array".to_string()))?
                .value(row_idx),
        ),
        DataType::Int64 => KeyValue::Int64(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::Internal("Expected Int64Array".to_string()))?
                .value(row_idx),
        ),
        DataType::UInt8 => KeyValue::UInt8(
            column
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| Error::Internal("Expected UInt8Array".to_string()))?
                .value(row_idx),
        ),
        DataType::UInt16 => KeyValue::UInt16(
            column
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| Error::Internal("Expected UInt16Array".to_string()))?
                .value(row_idx),
        ),
        DataType::UInt32 => KeyValue::UInt32(
            column
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Error::Internal("Expected UInt32Array".to_string()))?
                .value(row_idx),
        ),
        DataType::UInt64 => KeyValue::UInt64(
            column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("Expected UInt64Array".to_string()))?
                .value(row_idx),
        ),
        DataType::Float32 => {
            let val = column
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::Internal("Expected Float32Array".to_string()))?
                .value(row_idx);
            KeyValue::Float32(val.to_bits())
        }
        DataType::Float64 => {
            let val = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::Internal("Expected Float64Array".to_string()))?
                .value(row_idx);
            KeyValue::Float64(val.to_bits())
        }
        DataType::Utf8 => KeyValue::Utf8(
            column
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::Internal("Expected StringArray".to_string()))?
                .value(row_idx)
                .to_string(),
        ),
        DataType::Binary => KeyValue::Binary(
            column
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| Error::Internal("Expected BinaryArray".to_string()))?
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
    let cached = CachedSchema::new(Arc::new(schema.clone()));

    let schema_index = cached.index_of_field_id(target_field_id).ok_or_else(|| {
        Error::Internal(format!("field_id {} not found in schema", target_field_id))
    })?;

    let mut user_col_idx = 0;
    for idx in 0..schema_index {
        if cached.field_id(idx).is_some() {
            user_col_idx += 1;
        }
    }

    Ok(user_col_idx)
}
