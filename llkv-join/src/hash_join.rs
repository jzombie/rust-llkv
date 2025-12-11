//! Hash join implementation optimized for row-id streaming (zero-copy friendly).
//! Materialization is intentionally avoided; consumers can project rows lazily using
//! the recorded row references.
//!
//! This implementation uses `arrow::row::RowConverter` to vectorize the hashing of
//! composite keys, avoiding per-row allocations and scalar value wrapping.

use crate::{JoinIndexBatch, JoinKey, JoinOptions, JoinRowRef, JoinSide, JoinType};
use arrow::array::{Array, PrimitiveArray, RecordBatch};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Int32Type, Int64Type, Schema, UInt32Type, UInt64Type,
};
use arrow::row::{RowConverter, Rows, SortField};
use llkv_column_map::store::Projection;
use llkv_expr::{Expr, Filter, Operator};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_table::schema_ext::CachedSchema;
use llkv_table::table::{RowIdFilter, ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::FieldId;
use llkv_types::LogicalFieldId;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use simd_r_drive_entry_handle::EntryHandle;
use std::hash::{Hash, Hasher};
use std::ops::Bound;
use std::sync::Arc;

/// A reference to a row in a batch: (batch_index, row_index).
type RowRef = (usize, usize);

/// Hash table keyed by hashed row bytes to lists of matching rows.
/// Row buffers stay shared to avoid cloning the underlying bytes.
type HashTable = FxHashMap<u64, Vec<RowRef>>;

/// Key wrapper that encodes NULLs without materializing byte buffers.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ScalarKey<T: Copy + Eq + std::hash::Hash> {
    Null,
    Value(T),
}

#[inline]
fn hash_row(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    bytes.hash(&mut hasher);
    hasher.finish()
}

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

    // Fast path for single-key primitive joins to avoid per-row byte clones.
    if keys.len() == 1 {
        let left_dtype = key_datatype(&left_schema, keys[0].left_field)?;
        let right_dtype = key_datatype(&right_schema, keys[0].right_field)?;

        if left_dtype == right_dtype {
            match left_dtype {
                DataType::Int32 => {
                    return hash_join_rowid_single_key::<Int32Type, _, _>(
                        left,
                        right,
                        keys,
                        options,
                        left_filter,
                        right_filter,
                        on_batch,
                    );
                }
                DataType::Int64 => {
                    return hash_join_rowid_single_key::<Int64Type, _, _>(
                        left,
                        right,
                        keys,
                        options,
                        left_filter,
                        right_filter,
                        on_batch,
                    );
                }
                DataType::UInt32 => {
                    return hash_join_rowid_single_key::<UInt32Type, _, _>(
                        left,
                        right,
                        keys,
                        options,
                        left_filter,
                        right_filter,
                        on_batch,
                    );
                }
                DataType::UInt64 => {
                    return hash_join_rowid_single_key::<UInt64Type, _, _>(
                        left,
                        right,
                        keys,
                        options,
                        left_filter,
                        right_filter,
                        on_batch,
                    );
                }
                _ => {}
            }
        }
    }

    let left_projections = build_user_projections(left, &left_schema)?;
    let right_projections = build_user_projections(right, &right_schema)?;

    // 1. Build Phase (Right Side)
    let (hash_table, build_batches, build_rows) = build_hash_table(
        right,
        &right_projections,
        keys,
        &right_schema,
        right_filter.clone(),
    )?;

    // 2. Probe Phase (Left Side)
    let prepared_left = prepare_keys(&left_schema, keys, JoinSide::Left)?;
    let batch_size = options.batch_size.max(1);
    let probe_filter = build_all_rows_filter(&left_projections)?;

    // Create converter for probe side
    let probe_converter =
        RowConverter::new(prepared_left.sort_fields).map_err(|e| Error::Internal(e.to_string()))?;

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

            let mut flush = |left_rows: &mut Vec<usize>,
                             right_rows: &mut Vec<Option<JoinRowRef>>,
                             force: bool| {
                if left_rows.is_empty() {
                    return;
                }
                if !force && left_rows.len() < batch_size {
                    return;
                }

                let batch = JoinIndexBatch {
                    left_batch: &probe_batch,
                    left_rows: std::mem::take(left_rows),
                    right_rows: std::mem::take(right_rows),
                    right_batches: &build_batches,
                };
                on_batch(batch);
            };

            // Extract probe keys
            let probe_columns: Vec<_> = prepared_left
                .batch_indices
                .iter()
                .map(|&idx| probe_batch.column(idx).clone())
                .collect();

            // Convert to rows (vectorized)
            let probe_rows = match probe_converter.convert_columns(&probe_columns) {
                Ok(rows) => rows,
                Err(e) => panic!("Failed to convert probe rows: {}", e),
            };

            // Iterate and lookup in parallel chunks to reduce contention and keep order.
            let total_rows = probe_rows.num_rows();
            let chunk_size = batch_size;
            let chunk_count = (total_rows + chunk_size - 1) / chunk_size;

            #[derive(Default)]
            struct ChunkBuffers {
                left_rows: Vec<usize>,
                right_rows: Vec<Option<JoinRowRef>>,
            }

            let mut chunk_buffers: Vec<ChunkBuffers> =
                (0..chunk_count).map(|_| ChunkBuffers::default()).collect();

            chunk_buffers
                .par_iter_mut()
                .enumerate()
                .for_each(|(chunk_idx, buffers)| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(total_rows);

                    for row_idx in start..end {
                        let has_nulls = keys
                            .iter()
                            .zip(&probe_columns)
                            .any(|(k, col)| !k.null_equals_null && col.is_null(row_idx));

                        if has_nulls {
                            if matches!(options.join_type, JoinType::Left) {
                                buffers.left_rows.push(row_idx);
                                buffers.right_rows.push(None);
                            }
                            continue;
                        }

                        let row = probe_rows.row(row_idx);
                        let key_hash = hash_row(row.data());
                        let matches = hash_table.get(&key_hash);

                        match options.join_type {
                            JoinType::Inner | JoinType::Left => {
                                let mut matched = false;
                                if let Some(build_refs) = matches {
                                    for &(build_batch_idx, build_row_idx) in build_refs {
                                        let build_row =
                                            build_rows[build_batch_idx].row(build_row_idx);
                                        if row == build_row {
                                            buffers.left_rows.push(row_idx);
                                            buffers.right_rows.push(Some(JoinRowRef {
                                                batch: build_batch_idx,
                                                row: build_row_idx,
                                            }));
                                            matched = true;
                                        }
                                    }
                                }

                                if options.join_type == JoinType::Left && !matched {
                                    buffers.left_rows.push(row_idx);
                                    buffers.right_rows.push(None);
                                }
                            }
                            JoinType::Semi => {
                                if let Some(build_refs) = matches {
                                    if build_refs.iter().any(|&(b_idx, r_idx)| {
                                        probe_rows.row(row_idx) == build_rows[b_idx].row(r_idx)
                                    }) {
                                        buffers.left_rows.push(row_idx);
                                        buffers.right_rows.push(None);
                                    }
                                }
                            }
                            JoinType::Anti => {
                                let matched = matches
                                    .map(|build_refs| {
                                        build_refs.iter().any(|&(b_idx, r_idx)| {
                                            probe_rows.row(row_idx) == build_rows[b_idx].row(r_idx)
                                        })
                                    })
                                    .unwrap_or(false);

                                if !matched {
                                    buffers.left_rows.push(row_idx);
                                    buffers.right_rows.push(None);
                                }
                            }
                            _ => panic!("Unsupported join type"),
                        }
                    }
                });

            for chunk in chunk_buffers {
                for (l, r) in chunk
                    .left_rows
                    .into_iter()
                    .zip(chunk.right_rows.into_iter())
                {
                    left_rows.push(l);
                    right_rows.push(r);
                    if left_rows.len() >= batch_size {
                        flush(&mut left_rows, &mut right_rows, true);
                    }
                }
            }

            // Force flush at the end of the batch
            flush(&mut left_rows, &mut right_rows, true);
        },
    )?;

    Ok(())
}

fn hash_join_rowid_single_key<A, P, F>(
    left: &Table<P>,
    right: &Table<P>,
    keys: &[JoinKey],
    options: &JoinOptions,
    left_filter: Option<Arc<dyn RowIdFilter<P>>>,
    right_filter: Option<Arc<dyn RowIdFilter<P>>>,
    mut on_batch: F,
) -> LlkvResult<()>
where
    A: ArrowPrimitiveType,
    A::Native: Send + Sync + Eq + std::hash::Hash + Copy,
    P: Pager<Blob = EntryHandle> + Send + Sync,
    F: FnMut(JoinIndexBatch<'_>),
{
    let join_key = &keys[0];
    let left_schema = left.schema()?;
    let right_schema = right.schema()?;

    let left_projections = build_user_projections(left, &left_schema)?;
    let right_projections = build_user_projections(right, &right_schema)?;

    let (hash_table, build_batches) = build_hash_table_single_key::<A, P>(
        right,
        &right_projections,
        join_key,
        &right_schema,
        right_filter,
    )?;

    let prepared_left = prepare_keys(&left_schema, keys, JoinSide::Left)?;
    let key_idx = prepared_left.batch_indices[0];
    let batch_size = options.batch_size.max(1);
    let probe_filter = build_all_rows_filter(&left_projections)?;

    // Parallelize probe processing per batch while keeping output ordered.
    struct ChunkBuffers {
        left_rows: Vec<usize>,
        right_rows: Vec<Option<JoinRowRef>>,
    }
    impl Default for ChunkBuffers {
        fn default() -> Self {
            Self {
                left_rows: Vec::new(),
                right_rows: Vec::new(),
            }
        }
    }

    left.scan_stream(
        &left_projections,
        &probe_filter,
        ScanStreamOptions {
            row_id_filter: left_filter,
            ..ScanStreamOptions::default()
        },
        |probe_batch| {
            let key_array = probe_batch
                .column(key_idx)
                .as_any()
                .downcast_ref::<PrimitiveArray<A>>()
                .unwrap_or_else(|| {
                    panic!(
                        "Expected primitive array at column {} but found {:?}",
                        key_idx,
                        probe_batch.column(key_idx).data_type()
                    )
                });

            let total_rows = key_array.len();
            if total_rows == 0 {
                return;
            }

            let chunk_size = batch_size;
            let chunk_count = (total_rows + chunk_size - 1) / chunk_size;
            let mut chunk_buffers: Vec<ChunkBuffers> =
                (0..chunk_count).map(|_| ChunkBuffers::default()).collect();

            chunk_buffers
                .par_iter_mut()
                .enumerate()
                .for_each(|(chunk_idx, buffers)| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(total_rows);

                    for row_idx in start..end {
                        let key = if key_array.is_null(row_idx) {
                            if !join_key.null_equals_null {
                                continue;
                            }
                            ScalarKey::Null
                        } else {
                            ScalarKey::Value(key_array.value(row_idx))
                        };

                        let matches = hash_table.get(&key);
                        match options.join_type {
                            JoinType::Inner => {
                                if let Some(build_rows) = matches {
                                    for &(build_batch_idx, build_row_idx) in build_rows {
                                        buffers.left_rows.push(row_idx);
                                        buffers.right_rows.push(Some(JoinRowRef {
                                            batch: build_batch_idx,
                                            row: build_row_idx,
                                        }));
                                    }
                                }
                            }
                            JoinType::Left => {
                                if let Some(build_rows) = matches {
                                    for &(build_batch_idx, build_row_idx) in build_rows {
                                        buffers.left_rows.push(row_idx);
                                        buffers.right_rows.push(Some(JoinRowRef {
                                            batch: build_batch_idx,
                                            row: build_row_idx,
                                        }));
                                    }
                                } else {
                                    buffers.left_rows.push(row_idx);
                                    buffers.right_rows.push(None);
                                }
                            }
                            JoinType::Semi => {
                                if matches.map(|m| !m.is_empty()).unwrap_or(false) {
                                    buffers.left_rows.push(row_idx);
                                    buffers.right_rows.push(None);
                                }
                            }
                            JoinType::Anti => {
                                if matches.is_none() || matches.unwrap().is_empty() {
                                    buffers.left_rows.push(row_idx);
                                    buffers.right_rows.push(None);
                                }
                            }
                            _ => {
                                panic!("Unsupported join type for fast-path hash join");
                            }
                        }
                    }
                });

            let mut left_rows: Vec<usize> = Vec::with_capacity(batch_size);
            let mut right_rows: Vec<Option<JoinRowRef>> = Vec::with_capacity(batch_size);

            let mut flush = |left_rows: &mut Vec<usize>,
                             right_rows: &mut Vec<Option<JoinRowRef>>,
                             force: bool| {
                if left_rows.is_empty() {
                    return;
                }
                if !force && left_rows.len() < batch_size {
                    return;
                }

                let batch = JoinIndexBatch {
                    left_batch: &probe_batch,
                    left_rows: std::mem::take(left_rows),
                    right_rows: std::mem::take(right_rows),
                    right_batches: &build_batches,
                };
                on_batch(batch);
            };

            for chunk in chunk_buffers {
                for (l, r) in chunk
                    .left_rows
                    .into_iter()
                    .zip(chunk.right_rows.into_iter())
                {
                    left_rows.push(l);
                    right_rows.push(r);
                    if left_rows.len() >= batch_size {
                        flush(&mut left_rows, &mut right_rows, true);
                    }
                }
            }

            flush(&mut left_rows, &mut right_rows, true);
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
) -> LlkvResult<(HashTable, Vec<RecordBatch>, Vec<Arc<Rows>>)>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut hash_table = HashTable::default();
    let mut batches = Vec::new();
    let mut build_rows: Vec<Arc<Rows>> = Vec::new();
    let filter_expr = build_all_rows_filter(projections)?;

    // Prepare keys and converter
    let prepared = prepare_keys(schema, join_keys, JoinSide::Right)?;
    let converter =
        RowConverter::new(prepared.sort_fields).map_err(|e| Error::Internal(e.to_string()))?;

    table.scan_stream(
        projections,
        &filter_expr,
        ScanStreamOptions {
            row_id_filter: row_filter,
            ..ScanStreamOptions::default()
        },
        |batch| {
            let batch_idx = batches.len();

            // Extract key columns
            let key_columns: Vec<_> = prepared
                .batch_indices
                .iter()
                .map(|&idx| batch.column(idx).clone())
                .collect();

            // Convert to rows
            let rows = match converter.convert_columns(&key_columns) {
                Ok(rows) => rows,
                Err(e) => panic!("Failed to convert build rows: {}", e),
            };
            let rows = Arc::new(rows);

            // Insert into hash table
            for row_idx in 0..rows.num_rows() {
                // Handle NULLs: if any key part is null and !null_equals_null, skip.
                let has_nulls = join_keys
                    .iter()
                    .zip(&key_columns)
                    .any(|(k, col)| !k.null_equals_null && col.is_null(row_idx));

                if !has_nulls {
                    let row_bytes = rows.row(row_idx);
                    let key_hash = hash_row(row_bytes.data());
                    hash_table
                        .entry(key_hash)
                        .or_default()
                        .push((batch_idx, row_idx));
                }
            }

            batches.push(batch.clone());
            build_rows.push(rows);
        },
    )?;

    Ok((hash_table, batches, build_rows))
}

fn build_hash_table_single_key<A, P>(
    table: &Table<P>,
    projections: &[ScanProjection],
    join_key: &JoinKey,
    schema: &Arc<Schema>,
    row_filter: Option<Arc<dyn RowIdFilter<P>>>,
) -> LlkvResult<(
    FxHashMap<ScalarKey<A::Native>, Vec<RowRef>>,
    Vec<RecordBatch>,
)>
where
    A: ArrowPrimitiveType,
    A::Native: Copy + Eq + std::hash::Hash,
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut hash_table: FxHashMap<ScalarKey<A::Native>, Vec<RowRef>> = FxHashMap::default();
    let mut batches = Vec::new();
    let filter_expr = build_all_rows_filter(projections)?;

    let prepared = prepare_keys(schema, std::slice::from_ref(join_key), JoinSide::Right)?;
    let key_idx = prepared.batch_indices[0];

    table.scan_stream(
        projections,
        &filter_expr,
        ScanStreamOptions {
            row_id_filter: row_filter,
            ..ScanStreamOptions::default()
        },
        |batch| {
            let batch_idx = batches.len();
            let key_array = batch
                .column(key_idx)
                .as_any()
                .downcast_ref::<PrimitiveArray<A>>()
                .unwrap_or_else(|| {
                    panic!(
                        "Expected primitive array at column {} but found {:?}",
                        key_idx,
                        batch.column(key_idx).data_type()
                    )
                });

            for row_idx in 0..key_array.len() {
                let key = if key_array.is_null(row_idx) {
                    if !join_key.null_equals_null {
                        continue;
                    }
                    ScalarKey::Null
                } else {
                    ScalarKey::Value(key_array.value(row_idx))
                };

                hash_table
                    .entry(key)
                    .or_default()
                    .push((batch_idx, row_idx));
            }

            batches.push(batch.clone());
        },
    )?;

    Ok((hash_table, batches))
}

// --- Helper functions ---

struct PreparedKeys {
    /// Indices into the projected RecordBatch
    batch_indices: Vec<usize>,
    /// SortFields for RowConverter (must match projected columns)
    sort_fields: Vec<SortField>,
}

fn prepare_keys(schema: &Schema, keys: &[JoinKey], side: JoinSide) -> LlkvResult<PreparedKeys> {
    let cached = CachedSchema::new(Arc::new(schema.clone()));

    // Map FieldId -> Batch Index
    // This MUST match build_user_projections logic exactly.
    let mut field_id_to_batch_idx = FxHashMap::default();
    let mut current_batch_idx = 0;
    for (i, _field) in schema.fields().iter().enumerate() {
        if let Some(fid) = cached.field_id(i) {
            field_id_to_batch_idx.insert(fid, current_batch_idx);
            current_batch_idx += 1;
        }
    }

    let mut batch_indices = Vec::with_capacity(keys.len());
    let mut sort_fields = Vec::with_capacity(keys.len());

    for k in keys {
        let fid = if side == JoinSide::Left {
            k.left_field
        } else {
            k.right_field
        };

        let batch_idx = *field_id_to_batch_idx.get(&fid).ok_or_else(|| {
            Error::Internal(format!(
                "Join key field_id {} not found in projected columns",
                fid
            ))
        })?;

        // For SortField, we need the DataType of the projected column.
        let schema_idx = cached.index_of_field_id(fid).ok_or_else(|| {
            Error::Internal(format!("Join key field_id {} not found in schema", fid))
        })?;
        let field = schema.field(schema_idx);

        batch_indices.push(batch_idx);
        sort_fields.push(SortField::new(field.data_type().clone()));
    }

    Ok(PreparedKeys {
        batch_indices,
        sort_fields,
    })
}

fn key_datatype(schema: &Schema, field_id: FieldId) -> LlkvResult<DataType> {
    let cached = CachedSchema::new(Arc::new(schema.clone()));
    let schema_idx = cached.index_of_field_id(field_id).ok_or_else(|| {
        Error::Internal(format!(
            "Join key field_id {} not found in schema",
            field_id
        ))
    })?;
    Ok(schema.field(schema_idx).data_type().clone())
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

/// Streaming hash join implementation for the executor.
pub struct HashJoinStream {
    schema: Arc<Schema>,
    left_stream: Box<dyn Iterator<Item = LlkvResult<RecordBatch>> + Send>,
    right_batch: RecordBatch,
    hash_table: HashTable,
    build_rows: Option<Rows>,
    join_type: JoinType,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
    probe_converter: RowConverter,
    filter: Option<crate::JoinFilter>,

    // State for batch splitting
    active_left_batch: Option<RecordBatch>,
    active_probe_rows: Option<Rows>,
    next_left_row_idx: usize,
    target_batch_size: usize,
}

impl HashJoinStream {
    pub fn try_new(
        schema: Arc<Schema>,
        left_stream: Box<dyn Iterator<Item = LlkvResult<RecordBatch>> + Send>,
        right_batch: RecordBatch,
        join_type: JoinType,
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
        filter: Option<crate::JoinFilter>,
    ) -> LlkvResult<Self> {
        // 1. Build Hash Table from right_batch
        let mut hash_table = HashTable::default();

        // Prepare build converter
        let build_fields: Vec<_> = right_indices
            .iter()
            .map(|&i| SortField::new(right_batch.schema().field(i).data_type().clone()))
            .collect();

        let build_converter =
            RowConverter::new(build_fields.clone()).map_err(|e| Error::Internal(e.to_string()))?;

        let build_columns: Vec<_> = right_indices
            .iter()
            .map(|&i| right_batch.column(i).clone())
            .collect();

        let build_rows = if build_columns.is_empty() {
            for row_idx in 0..right_batch.num_rows() {
                hash_table.entry(0).or_default().push((0, row_idx));
            }
            None
        } else {
            let build_rows = build_converter
                .convert_columns(&build_columns)
                .map_err(|e| Error::Internal(e.to_string()))?;

            for row_idx in 0..build_rows.num_rows() {
                // Handle NULLs: if any key part is null (and we assume standard SQL equality), skip.
                let has_nulls = build_columns.iter().any(|col| col.is_null(row_idx));

                if !has_nulls {
                    let key_hash = hash_row(build_rows.row(row_idx).data());
                    hash_table.entry(key_hash).or_default().push((0, row_idx));
                }
            }
            Some(build_rows)
        };

        // 2. Prepare probe converter
        // Left fields must match right fields types for hashing to work
        let probe_fields = build_fields; // Same types
        let probe_converter =
            RowConverter::new(probe_fields).map_err(|e| Error::Internal(e.to_string()))?;

        if left_indices.is_empty() {
            tracing::warn!(
                "HashJoinStream::try_new: WARNING: Cross Join detected (no join keys). Right batch rows: {}",
                right_batch.num_rows()
            );
        } else {
            // tracing::trace!("HashJoinStream::try_new: Inner/Left Join. Keys: {}, Right rows: {}", left_indices.len(), right_batch.num_rows());
        }

        Ok(Self {
            schema,
            left_stream,
            right_batch,
            hash_table,
            build_rows,
            join_type,
            left_indices,
            right_indices,
            probe_converter,
            filter,
            active_left_batch: None,
            active_probe_rows: None,
            next_left_row_idx: 0,
            target_batch_size: 4096,
        })
    }
}

impl Iterator for HashJoinStream {
    type Item = LlkvResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        use arrow::array::UInt64Builder;
        use arrow::compute::take;

        loop {
            // 1. Ensure we have an active left batch
            if self.active_left_batch.is_none() {
                match self.left_stream.next() {
                    Some(Ok(batch)) => {
                        // Prepare probe rows for the new batch
                        let probe_columns = match self
                            .left_indices
                            .iter()
                            .zip(self.right_indices.iter())
                            .map(|(&left_idx, &right_idx)| {
                                let col = batch.column(left_idx);
                                let target_type = self
                                    .right_batch
                                    .schema()
                                    .field(right_idx)
                                    .data_type()
                                    .clone();
                                if col.data_type() != &target_type {
                                    arrow::compute::cast(col, &target_type)
                                        .map_err(|e| Error::Internal(e.to_string()))
                                } else {
                                    Ok(col.clone())
                                }
                            })
                            .collect::<LlkvResult<Vec<_>>>()
                        {
                            Ok(cols) => cols,
                            Err(e) => return Some(Err(e)),
                        };

                        if !probe_columns.is_empty() {
                            match self.probe_converter.convert_columns(&probe_columns) {
                                Ok(rows) => self.active_probe_rows = Some(rows),
                                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
                            }
                        } else {
                            self.active_probe_rows = None;
                        }

                        self.active_left_batch = Some(batch);
                        self.next_left_row_idx = 0;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None, // End of stream
                }
            }

            let left_batch = self.active_left_batch.as_ref().unwrap();
            let num_rows = left_batch.num_rows();

            // 2. Process a chunk of the active batch
            let mut left_builder = UInt64Builder::with_capacity(self.target_batch_size);
            let mut right_builder = UInt64Builder::with_capacity(self.target_batch_size);
            let mut rows_produced = 0;

            let start_idx = self.next_left_row_idx;
            let mut current_idx = start_idx;

            let empty_hash = 0_u64; // For cross join case
            let build_rows = self.build_rows.as_ref();

            while current_idx < num_rows && rows_produced < self.target_batch_size {
                let (matches, probe_row) = if let Some(rows) = &self.active_probe_rows {
                    let row = rows.row(current_idx);
                    let key_hash = hash_row(row.data());
                    (self.hash_table.get(&key_hash), Some(row))
                } else {
                    // No join keys -> Cross Join behavior (if hash table has empty key)
                    (self.hash_table.get(&empty_hash), None)
                };

                match self.join_type {
                    JoinType::Inner => {
                        if let Some(build_rows_indices) = matches {
                            for &(_, build_row_idx) in build_rows_indices {
                                if let (Some(build_rows), Some(probe_row)) = (build_rows, probe_row)
                                {
                                    if probe_row != build_rows.row(build_row_idx) {
                                        continue;
                                    }
                                }
                                left_builder.append_value(current_idx as u64);
                                right_builder.append_value(build_row_idx as u64);
                                rows_produced += 1;
                            }
                        }
                    }
                    JoinType::Left => {
                        if let Some(build_rows_indices) = matches {
                            for &(_, build_row_idx) in build_rows_indices {
                                if let (Some(build_rows), Some(probe_row)) = (build_rows, probe_row)
                                {
                                    if probe_row != build_rows.row(build_row_idx) {
                                        continue;
                                    }
                                }
                                left_builder.append_value(current_idx as u64);
                                right_builder.append_value(build_row_idx as u64);
                                rows_produced += 1;
                            }
                        } else {
                            left_builder.append_value(current_idx as u64);
                            right_builder.append_null();
                            rows_produced += 1;
                        }
                    }
                    _ => {
                        return Some(Err(Error::Internal(
                            "Unsupported join type in stream".to_string(),
                        )));
                    }
                }

                current_idx += 1;
            }

            self.next_left_row_idx = current_idx;

            // If we finished the batch, clear it so we fetch next one in next iteration
            let finished_batch = self.next_left_row_idx >= num_rows;

            // 3. If we produced rows, materialize and return
            let mut left_take_indices = left_builder.finish();
            let mut right_take_indices = right_builder.finish();

            if left_take_indices.is_empty() {
                if finished_batch {
                    self.active_left_batch = None;
                    self.active_probe_rows = None;
                }
                // If we didn't produce any rows (e.g. all filtered out in Inner join),
                // loop again to fetch next batch or continue processing
                continue;
            }

            let right_batch = &self.right_batch;
            let schema = self.schema.clone();

            let materialize_batch = |l_indices: &arrow::array::UInt64Array,
                                     r_indices: &arrow::array::UInt64Array|
             -> LlkvResult<RecordBatch> {
                let mut output_columns =
                    Vec::with_capacity(left_batch.num_columns() + right_batch.num_columns());
                for col in left_batch.columns() {
                    output_columns.push(
                        take(col, l_indices, None).map_err(|e| Error::Internal(e.to_string()))?,
                    );
                }
                for col in right_batch.columns() {
                    output_columns.push(
                        take(col, r_indices, None).map_err(|e| Error::Internal(e.to_string()))?,
                    );
                }
                RecordBatch::try_new(schema.clone(), output_columns)
                    .map_err(|e| Error::Internal(e.to_string()))
            };

            let mut batch = match materialize_batch(&left_take_indices, &right_take_indices) {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            if let Some(filter) = &self.filter {
                let mask = match filter(&batch) {
                    Ok(m) => m,
                    Err(e) => return Some(Err(e)),
                };

                if self.join_type == JoinType::Left {
                    let mut new_left = UInt64Builder::with_capacity(left_take_indices.len());
                    let mut new_right = UInt64Builder::with_capacity(right_take_indices.len());

                    let mut i = 0;
                    let num_rows = left_take_indices.len();

                    while i < num_rows {
                        let l_idx = left_take_indices.value(i);
                        let start = i;

                        // Find range of rows with same l_idx
                        while i < num_rows && left_take_indices.value(i) == l_idx {
                            i += 1;
                        }
                        let end = i;

                        let mut any_kept = false;
                        for j in start..end {
                            let keep = mask.is_valid(j) && mask.value(j);
                            if right_take_indices.is_null(j) {
                                new_left.append_value(l_idx);
                                new_right.append_null();
                                any_kept = true;
                            } else if keep {
                                new_left.append_value(l_idx);
                                new_right.append_value(right_take_indices.value(j));
                                any_kept = true;
                            }
                        }

                        if !any_kept {
                            new_left.append_value(l_idx);
                            new_right.append_null();
                        }
                    }

                    left_take_indices = new_left.finish();
                    right_take_indices = new_right.finish();

                    batch = match materialize_batch(&left_take_indices, &right_take_indices) {
                        Ok(b) => b,
                        Err(e) => return Some(Err(e)),
                    };
                } else {
                    use arrow::compute::filter_record_batch;
                    match filter_record_batch(&batch, &mask) {
                        Ok(filtered) => return Some(Ok(filtered)),
                        Err(e) => return Some(Err(Error::Internal(e.to_string()))),
                    }
                }
            }

            if finished_batch {
                self.active_left_batch = None;
                self.active_probe_rows = None;
            }

            return Some(Ok(batch));
        }
    }
}
