use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use llkv_compute::analysis::computed_expr_requires_numeric;
use llkv_compute::eval::ScalarEvaluator;
use llkv_compute::projection::infer_computed_dtype;
use llkv_expr::Expr;
use llkv_result::{Error, Result as LlkvResult};
use llkv_types::{FieldId, LogicalFieldId};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::row_stream::{ColumnProjectionInfo, ComputedProjectionInfo, ProjectionEval, RowIdSource, RowStream, RowStreamBuilder};
use crate::{ScanProjection, ScanStorage, ScanStreamOptions};
use llkv_column_map::store::GatherNullPolicy;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

/// Execute a table scan using the shared scan machinery. Supports basic scans
/// without ordering or custom row-id filters; callers should handle those
/// features separately for now.
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
    if options.order.is_some() || options.row_id_filter.is_some() {
        return Err(Error::InvalidArgumentError(
            "execute_scan does not yet support order or row_id_filter".into(),
        ));
    }

    if projections.is_empty() {
        return Err(Error::InvalidArgumentError(
            "scan requires at least one projection".into(),
        ));
    }

    // Collect row ids via storage filter.
    let row_ids = storage.filter_row_ids(filter_expr)?;
    let row_source = if row_ids.is_empty() {
        RowIdSource::Bitmap(row_ids)
    } else {
        RowIdSource::Bitmap(row_ids)
    };

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
                    if let std::collections::hash_map::Entry::Vacant(entry) = unique_index.entry(lfid) {
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

    let mut row_stream = RowStreamBuilder::new(
        storage,
        table_id,
        Arc::clone(&out_schema),
        Arc::new(unique_lfids),
        Arc::new(projection_evals),
        Arc::new(passthrough_fields),
        Arc::new(unique_index),
        Arc::new(numeric_fields),
        requires_numeric,
        null_policy,
        row_source,
        1024,
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
