use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use croaring::Treemap;
use llkv_compute::analysis::computed_expr_requires_numeric;
use llkv_compute::eval::ScalarEvaluator;
use llkv_compute::projection::{
    ProjectionLiteral, emit_synthetic_null_batch, infer_computed_dtype,
};
use llkv_expr::Expr;
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::{FieldId, LogicalFieldId, RowId};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::row_stream::{
    ColumnProjectionInfo, ComputedProjectionInfo, NumericArrayMap, ProjectionEval, ProjectionPlan,
    RowIdSource, RowStream, RowStreamBuilder, materialize_row_window,
};
use crate::{ScanProjection, ScanStorage, ScanStreamOptions, ordering::sort_row_ids_with_order};
use llkv_column_map::store::GatherNullPolicy;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

/// Streaming chunk size for row materialization.
///
/// The benches spend most of their time in compute once allocations are pooled,
/// so larger chunks help amortize planner and gather overhead. 16K keeps the
/// batches cache-friendly while letting arithmetic kernels stay vectorized.
const ROW_STREAM_CHUNK_SIZE: usize = 16_384;

struct FullTableStreamPlan<'a> {
    projection_evals: &'a [ProjectionEval],
    unique_index: &'a FxHashMap<LogicalFieldId, usize>,
    unique_lfids: &'a [LogicalFieldId],
    numeric_fields: &'a FxHashSet<FieldId>,
    requires_numeric: bool,
    null_policy: GatherNullPolicy,
    out_schema: Arc<Schema>,
    projection_plan: Vec<crate::row_stream::ProjectionPlan>,
}

/// Execute a table scan using the shared scan machinery.
pub fn execute_scan<'expr, P, S, F>(
    storage: &S,
    table_id: llkv_types::TableId,
    projections: &[ScanProjection],
    filter_expr: &Expr<'expr, FieldId>,
    options: ScanStreamOptions<P>,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
    F: FnMut(RecordBatch),
{
    if projections.is_empty() {
        return Err(Error::InvalidArgumentError(
            "scan requires at least one projection".into(),
        ));
    }

    // Determine projection evaluation plan and output schema.
    let mut projection_evals = Vec::with_capacity(projections.len());
    let mut unique_index: FxHashMap<LogicalFieldId, usize> = FxHashMap::default();
    let mut unique_lfids: Vec<LogicalFieldId> = Vec::new();
    let mut numeric_fields: FxHashSet<FieldId> = FxHashSet::default();
    let mut lfid_dtypes: FxHashMap<LogicalFieldId, DataType> = FxHashMap::default();

    for proj in projections {
        match proj {
            ScanProjection::Column(p) => {
                let lfid = p.logical_field_id;
                let dtype = storage.field_data_type(lfid)?;
                if let std::collections::hash_map::Entry::Vacant(entry) = unique_index.entry(lfid) {
                    entry.insert(unique_lfids.len());
                    unique_lfids.push(lfid);
                }
                let fallback = lfid.field_id().to_string();
                let output_name = p.alias.clone().unwrap_or(fallback);
                projection_evals.push(ProjectionEval::Column(ColumnProjectionInfo {
                    logical_field_id: lfid,
                    data_type: dtype,
                    output_name,
                }));
            }
            ScanProjection::Computed { expr, alias } => {
                let simplified = ScalarEvaluator::simplify(expr);
                let mut fields_set: FxHashSet<FieldId> = FxHashSet::default();
                ScalarEvaluator::collect_fields(&simplified, &mut fields_set);
                for fid in fields_set.iter().copied() {
                    numeric_fields.insert(fid);
                    let lfid = LogicalFieldId::for_user(table_id, fid);
                    let dtype = storage.field_data_type(lfid)?;
                    lfid_dtypes.entry(lfid).or_insert_with(|| dtype.clone());
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        unique_index.entry(lfid)
                    {
                        entry.insert(unique_lfids.len());
                        unique_lfids.push(lfid);
                    }
                }
                projection_evals.push(ProjectionEval::Computed(ComputedProjectionInfo {
                    expr: simplified,
                    alias: alias.clone(),
                }));
            }
        }
    }

    let passthrough_fields: Vec<Option<FieldId>> = projection_evals
        .iter()
        .map(|eval| match eval {
            ProjectionEval::Computed(info) => ScalarEvaluator::passthrough_column(&info.expr),
            _ => None,
        })
        .collect();

    let mut projection_plan = Vec::with_capacity(projection_evals.len());
    for (idx, eval) in projection_evals.iter().enumerate() {
        match eval {
            ProjectionEval::Column(info) => {
                let arr_idx = *unique_index
                    .get(&info.logical_field_id)
                    .expect("logical field id missing from index");
                projection_plan.push(ProjectionPlan::Column {
                    source_idx: arr_idx,
                });
            }
            ProjectionEval::Computed(_) => {
                let passthrough_idx = passthrough_fields[idx].as_ref().map(|fid| {
                    let lfid = LogicalFieldId::for_user(table_id, *fid);
                    *unique_index
                        .get(&lfid)
                        .expect("passthrough field missing from index")
                });
                projection_plan.push(ProjectionPlan::Computed {
                    eval_idx: idx,
                    passthrough_idx,
                });
            }
        }
    }

    let null_policy = if options.include_nulls {
        GatherNullPolicy::IncludeNulls
    } else {
        GatherNullPolicy::DropNulls
    };

    let requires_numeric = projection_evals.iter().enumerate().any(|(idx, eval)| {
        matches!(
            eval,
            ProjectionEval::Computed(info)
            if passthrough_fields[idx].is_none()
                && computed_expr_requires_numeric(&info.expr)
        )
    });

    let mut schema_fields: Vec<Field> = Vec::with_capacity(projection_evals.len());
    for (idx, eval) in projection_evals.iter().enumerate() {
        match eval {
            ProjectionEval::Column(info) => schema_fields.push(Field::new(
                info.output_name.clone(),
                info.data_type.clone(),
                true,
            )),
            ProjectionEval::Computed(info) => {
                if let Some(fid) = passthrough_fields[idx] {
                    let lfid = LogicalFieldId::for_user(table_id, fid);
                    let dtype = lfid_dtypes
                        .get(&lfid)
                        .cloned()
                        .ok_or_else(|| Error::Internal("missing dtype for passthrough".into()))?;
                    schema_fields.push(Field::new(info.alias.clone(), dtype, true));
                } else {
                    let dtype = infer_computed_dtype(&info.expr, table_id, &lfid_dtypes)?;
                    schema_fields.push(Field::new(info.alias.clone(), dtype, true));
                }
            }
        }
    }
    let out_schema = Arc::new(Schema::new(schema_fields));

    let ScanStreamOptions {
        include_nulls: _,
        order,
        row_id_filter,
        include_row_ids,
    } = options;

    let can_stream_full_table = !include_row_ids
        && row_id_filter.is_none()
        && order.is_none()
        && filter_expr.is_trivially_true();

    if can_stream_full_table {
        let plan = FullTableStreamPlan {
            projection_evals: &projection_evals,
            unique_index: &unique_index,
            unique_lfids: &unique_lfids,
            numeric_fields: &numeric_fields,
            requires_numeric,
            null_policy,
            out_schema: Arc::clone(&out_schema),
            projection_plan: projection_plan.clone(),
        };
        stream_full_table_scan(storage, table_id, plan, on_batch)?;
        return Ok(());
    }

    let mut row_source = RowIdSource::Bitmap(storage.filter_row_ids(filter_expr)?);

    if row_id_filter.is_some() || order.is_some() {
        let mut bitmap = match row_source {
            RowIdSource::Bitmap(b) => b,
            RowIdSource::Vector(v) => Treemap::from_iter(v),
        };

        if let Some(filter) = row_id_filter.as_ref() {
            bitmap = filter.filter(table_id, storage, bitmap)?;
        }

        row_source = if let Some(order_spec) = order {
            let sorted = sort_row_ids_with_order(storage, &bitmap, order_spec)?;
            RowIdSource::Vector(sorted)
        } else {
            RowIdSource::Bitmap(bitmap)
        };
    }

    let is_empty = match &row_source {
        RowIdSource::Bitmap(b) => b.is_empty(),
        RowIdSource::Vector(v) => v.is_empty(),
    };

    if is_empty {
        if row_id_filter.is_none() && filter_expr.is_trivially_true() {
            let total_rows = storage.total_rows()?;
            let row_count = usize::try_from(total_rows).map_err(|_| {
                Error::InvalidArgumentError(
                    "table row count exceeds supported range for synthetic batch".into(),
                )
            })?;
            let projection_literals = build_projection_literals(&projection_evals, &out_schema);
            if let Some(batch) =
                emit_synthetic_null_batch(&projection_literals, &out_schema, row_count)?
            {
                on_batch(batch);
            }
        }
        return Ok(());
    }

    let projection_evals = Arc::new(projection_evals);
    let passthrough_fields = Arc::new(passthrough_fields);
    let unique_index = Arc::new(unique_index);
    let unique_lfids = Arc::new(unique_lfids);
    let numeric_fields = Arc::new(numeric_fields);

    let mut row_stream = RowStreamBuilder::new(
        storage,
        table_id,
        Arc::clone(&out_schema),
        Arc::clone(&unique_lfids),
        Arc::clone(&projection_evals),
        Arc::clone(&passthrough_fields),
        Arc::clone(&unique_index),
        Arc::clone(&numeric_fields),
        requires_numeric,
        null_policy,
        row_source,
        ROW_STREAM_CHUNK_SIZE,
        include_row_ids,
    )
    .build()?;

    let expected_columns = row_stream.schema().fields().len();
    while let Some(chunk) = row_stream.next_chunk()? {
        let batch = chunk.to_record_batch();
        debug_assert_eq!(batch.num_columns(), expected_columns);
        if batch.num_rows() > 0 {
            on_batch(batch);
        }
    }

    Ok(())
}

fn stream_full_table_scan<P, S, F>(
    storage: &S,
    table_id: llkv_types::TableId,
    plan: FullTableStreamPlan<'_>,
    on_batch: &mut F,
) -> LlkvResult<()>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
    S: ScanStorage<P>,
    F: FnMut(RecordBatch),
{
    let mut gather_ctx = if plan.unique_lfids.is_empty() {
        None
    } else {
        Some(storage.prepare_gather_context(plan.unique_lfids)?)
    };

    let mut columns_buf = Vec::with_capacity(plan.projection_evals.len());
    let mut numeric_cache = if plan.requires_numeric {
        Some(NumericArrayMap::default())
    } else {
        None
    };

    let mut emitted_rows = false;
    let mut process_chunk = |chunk: &[RowId]| -> LlkvResult<()> {
        if chunk.is_empty() {
            return Ok(());
        }

        if let Some(batch) = materialize_row_window(
            storage,
            table_id,
            plan.unique_lfids,
            plan.projection_evals,
            &plan.projection_plan,
            plan.unique_index,
            plan.numeric_fields,
            plan.requires_numeric,
            plan.null_policy,
            &plan.out_schema,
            chunk,
            gather_ctx.as_mut(),
            numeric_cache.as_mut().map(|m| m),
            &mut columns_buf,
        )?
        .filter(|batch| batch.num_rows() > 0)
        {
            emitted_rows = true;
            on_batch(batch);
        }

        Ok(())
    };

    storage.stream_row_ids(ROW_STREAM_CHUNK_SIZE, &mut process_chunk)?;

    if !emitted_rows {
        let total_rows = storage.total_rows()?;
        let row_count = usize::try_from(total_rows).map_err(|_| {
            Error::InvalidArgumentError(
                "table row count exceeds supported range for synthetic batch".into(),
            )
        })?;
        let projection_literals =
            build_projection_literals(plan.projection_evals, &plan.out_schema);
        if let Some(batch) =
            emit_synthetic_null_batch(&projection_literals, &plan.out_schema, row_count)?
        {
            on_batch(batch);
        }
    }

    Ok(())
}

fn build_projection_literals(
    projection_evals: &[ProjectionEval],
    out_schema: &Arc<Schema>,
) -> Vec<ProjectionLiteral<FieldId>> {
    projection_evals
        .iter()
        .enumerate()
        .map(|(idx, eval)| match eval {
            ProjectionEval::Column(_) => ProjectionLiteral::Column {
                data_type: out_schema.field(idx).data_type().clone(),
            },
            ProjectionEval::Computed(info) => ProjectionLiteral::Computed {
                info: info.clone(),
                data_type: out_schema.field(idx).data_type().clone(),
            },
        })
        .collect()
}
