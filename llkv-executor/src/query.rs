use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::datatypes::{Field as ArrowField, Schema, SchemaRef};
use arrow::row::{OwnedRow, RowConverter, SortField};
use arrow::record_batch::RecordBatch;
use arrow::compute::cast;
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};
use llkv_expr::expr::{CompareOp, Expr as LlkvExpr, ScalarExpr};
use llkv_expr::AggregateCall;
use llkv_column_map::gather::gather_optional_projected_indices_from_batches;
use llkv_compute::eval::{ScalarEvaluator, ScalarExprTypeExt};
use llkv_join::{JoinKey, JoinOptions, JoinType, TableJoinRowIdExt};
use llkv_plan::logical_planner::{LogicalPlan, LogicalPlanner, ResolvedJoin};
use llkv_plan::physical::table::{ExecutionTable, TableProvider};
use llkv_plan::physical::PhysicalPlan;
use llkv_plan::planner::PhysicalPlanner;
use llkv_plan::plans::SelectPlan;
use llkv_plan::plans::JoinPlan;
use llkv_plan::schema::PlanSchema;
use llkv_result::Error;
use llkv_scan::RowIdFilter;
use llkv_storage::pager::Pager;
use llkv_types::FieldId;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use crate::types::{ExecutorTable, ExecutorTableProvider};
use crate::ExecutorResult;

/// Plan-driven SELECT executor bridging planner output to storage.
pub struct QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    logical_planner: LogicalPlanner<P>,
    physical_planner: PhysicalPlanner<P>,
}

impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn ExecutorTableProvider<P>>) -> Self {
        let planner_provider = Arc::new(PlannerTableProvider { inner: provider });
        Self {
            logical_planner: LogicalPlanner::new(planner_provider),
            physical_planner: PhysicalPlanner::new(),
        }
    }

    pub fn execute_select(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        self.execute_select_with_filter(plan, None)
    }

    pub fn execute_select_with_filter(
        &self,
        plan: SelectPlan,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let logical_plan = self.logical_planner.create_logical_plan(&plan)?;

        match logical_plan {
            LogicalPlan::Single(single) => {
                let table_name = single.table_name.clone();
                let physical_plan = self
                    .physical_planner
                    .create_physical_plan(&single, row_filter)
                    .map_err(Error::Internal)?;
                let schema = physical_plan.schema();
                let batches = collect_batches(&*physical_plan)?;

                if !plan.aggregates.is_empty() {
                    if !plan.group_by.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "GROUP BY aggregates are not supported yet".into(),
                        ));
                    }

                    let (agg_schema, agg_batches) = aggregate_select(
                        &plan,
                        &single.schema,
                        &schema,
                        batches,
                    )?;
                    let trimmed = apply_offset_limit(agg_batches, plan.offset, plan.limit);
                    Ok(SelectExecution::new(table_name, agg_schema, trimmed))
                } else {
                    let batches = if plan.distinct {
                        distinct_record_batches(&schema, batches)?
                    } else {
                        batches
                    };
                    let batches = apply_offset_limit(batches, plan.offset, plan.limit);
                    Ok(SelectExecution::new(table_name, schema, batches))
                }
            }
            LogicalPlan::Multi(multi) => {
                if !multi.aggregates.is_empty()
                    || !multi.group_by.is_empty()
                    || multi.distinct
                    || multi.compound.is_some()
                    || multi.having.is_some()
                {
                    return Err(Error::InvalidArgumentError(
                        "multi-table aggregates, DISTINCT, HAVING, or compound queries are not supported yet"
                            .into(),
                    ));
                }

                if !multi.order_by.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "ORDER BY for multi-table queries is not supported yet".into(),
                    ));
                }

                if multi.tables.len() < 2 {
                    return Err(Error::InvalidArgumentError(
                        "multi-table SELECT requires at least two tables".into(),
                    ));
                }

                let table_count = multi.tables.len();
                let mut table_index_by_fid: Vec<FxHashMap<FieldId, usize>> = Vec::with_capacity(table_count);
                let mut exec_tables: Vec<Arc<ExecutorTable<P>>> = Vec::with_capacity(table_count);
                let mut field_id_by_table_index: Vec<Vec<FieldId>> = Vec::with_capacity(table_count);

                for table in &multi.tables {
                    let adapter = table
                        .table
                        .as_any()
                        .downcast_ref::<ExecutionTableAdapter<P>>()
                        .ok_or_else(|| Error::Internal("unexpected table adapter type".into()))?;
                    exec_tables.push(adapter.executor_table());
                    field_id_by_table_index.push(table.schema.columns.iter().map(|c| c.field_id).collect());
                    table_index_by_fid.push(index_fields_by_id(&table.schema));
                }

                let mut output_fields = Vec::with_capacity(multi.projections.len());
                let mut computed_projections: Vec<ComputedProjectionPlan> = Vec::new();
                let mut gather_plan_per_table: Vec<Vec<usize>> = vec![Vec::new(); table_count];
                let mut projection_out_map: Vec<Vec<(usize, usize)>> = vec![Vec::new(); table_count];
                let mut gather_index_cache: Vec<FxHashMap<usize, usize>> =
                    vec![FxHashMap::default(); table_count];

                for proj in &multi.projections {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column {
                            table_index,
                            logical_field_id,
                            alias,
                        } => {
                            if *table_index >= table_count {
                                return Err(Error::InvalidArgumentError(format!(
                                    "projection references missing table {}",
                                    table_index
                                )));
                            }

                            let schema = &multi.tables[*table_index].schema;
                            let index_by_fid = &table_index_by_fid[*table_index];
                            let field_id = logical_field_id.field_id();
                            let Some(col_idx) = index_by_fid.get(&field_id).copied() else {
                                return Err(Error::InvalidArgumentError(
                                    format!("field_id {field_id} not found in table {table_index}"),
                                ));
                            };

                            let col = schema
                                .columns
                                .get(col_idx)
                                .ok_or_else(|| Error::Internal("projection index out of bounds".into()))?;

                            let field = ArrowField::new(
                                alias.clone().unwrap_or_else(|| col.name.clone()),
                                col.data_type.clone(),
                                col.is_nullable,
                            );
                            output_fields.push(field);
                            let gather_idx = if let Some(existing) = gather_index_cache[*table_index].get(&col_idx) {
                                *existing
                            } else {
                                let next = gather_plan_per_table[*table_index].len();
                                gather_plan_per_table[*table_index].push(col_idx);
                                gather_index_cache[*table_index].insert(col_idx, next);
                                next
                            };
                            projection_out_map[*table_index].push((output_fields.len() - 1, gather_idx));
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { expr, alias } => {
                            let remapped_expr = remap_scalar_expr(expr);

                            let mut resolver = |key: (usize, FieldId)| {
                                table_index_by_fid.get(key.0).and_then(|idx_map| {
                                    idx_map
                                        .get(&key.1)
                                        .and_then(|col_idx| multi.tables[key.0]
                                            .schema
                                            .columns
                                            .get(*col_idx)
                                            .map(|c| c.data_type.clone()))
                                })
                            };

                            let data_type = remapped_expr
                                .infer_result_type(&mut resolver)
                                .ok_or_else(|| {
                                    Error::InvalidArgumentError(
                                        "unable to infer type for computed projection".into(),
                                    )
                                })?;

                            output_fields.push(ArrowField::new(alias.clone(), data_type.clone(), true));
                            let out_index = output_fields.len() - 1;

                            let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                            ScalarEvaluator::collect_fields(&remapped_expr, &mut fields);
                            for (tbl_idx, fid) in fields {
                                let col_idx = table_index_by_fid
                                    .get(tbl_idx)
                                    .and_then(|map| map.get(&fid).copied())
                                    .ok_or_else(|| {
                                        Error::InvalidArgumentError(format!(
                                            "field_id {fid} not found in table {tbl_idx}"
                                        ))
                                    })?;

                                if let std::collections::hash_map::Entry::Vacant(entry) =
                                    gather_index_cache[tbl_idx].entry(col_idx)
                                {
                                    let next = gather_plan_per_table[tbl_idx].len();
                                    gather_plan_per_table[tbl_idx].push(col_idx);
                                    entry.insert(next);
                                }
                            }

                            computed_projections.push(ComputedProjectionPlan {
                                expr: remapped_expr,
                                data_type,
                                out_index,
                            });
                        }
                    }
                }

                let mut join_plan_by_left: Vec<Option<&ResolvedJoin>> = vec![None; table_count - 1];
                for join in &multi.joins {
                    if join.left_table_index + 1 >= table_count {
                        return Err(Error::InvalidArgumentError(
                            "join references table outside the FROM clause".into(),
                        ));
                    }
                    if join_plan_by_left[join.left_table_index].is_some() {
                        return Err(Error::InvalidArgumentError(
                            "only one join per adjacent table pair is supported".into(),
                        ));
                    }
                    join_plan_by_left[join.left_table_index] = Some(join);
                }

                let mut table_batches: Vec<Option<Vec<RecordBatch>>> = vec![None; table_count];
                let mut adjacencies = Vec::with_capacity(table_count - 1);

                let output_schema = Arc::new(Schema::new(output_fields));
                let join_batch_size = JoinOptions::default().batch_size.max(1);

                for left_index in 0..table_count - 1 {
                    let join_meta = join_plan_by_left[left_index];
                    let join_type = match join_meta {
                        Some(meta) => map_join_type(meta.join_type)?,
                        None => JoinType::Inner,
                    };
                    let join_keys = match join_meta {
                        Some(meta) => extract_join_keys(meta)?,
                        None => Vec::new(),
                    };

                    let options = JoinOptions {
                        join_type,
                        ..Default::default()
                    };

                    let right_index = left_index + 1;
                    let left_exec = &exec_tables[left_index];
                    let right_exec = &exec_tables[right_index];
                    let mut matches: FxHashMap<(usize, usize), Vec<Option<llkv_join::JoinRowRef>>> =
                        FxHashMap::default();

                    left_exec.table.join_rowid_stream_with_filter(
                        &right_exec.table,
                        &join_keys,
                        &options,
                        row_filter.clone(),
                        row_filter.clone(),
                        |index_batch| {
                            if table_batches[left_index].is_none() {
                                table_batches[left_index] =
                                    Some(index_batch.left_batches.iter().cloned().collect());
                            }
                            if table_batches[right_index].is_none() {
                                table_batches[right_index] =
                                    Some(index_batch.right_batches.iter().cloned().collect());
                            }

                            for (left_row, right_row) in index_batch
                                .left_rows
                                .iter()
                                .zip(index_batch.right_rows.iter())
                            {
                                matches
                                    .entry((left_row.batch, left_row.row))
                                    .or_default()
                                    .push(*right_row);
                            }
                        },
                    )?;

                    adjacencies.push(JoinAdjacency {
                        left_index,
                        _right_index: right_index,
                        join_type,
                        matches,
                    });
                }

                for (idx, batches) in table_batches.iter().enumerate() {
                    if batches.is_none() {
                        return Err(Error::Internal(format!("missing batches for table {idx}")));
                    }
                }

                let mut buffered_rows: Vec<Vec<Option<llkv_join::JoinRowRef>>> = Vec::new();
                let mut result_batches = Vec::new();

                let table_zero_batches = table_batches[0].as_ref().expect("validated above");
                let mut emit_path = |
                    path: &[Option<llkv_join::JoinRowRef>],
                    buffered_rows: &mut Vec<Vec<Option<llkv_join::JoinRowRef>>>,
                    result_batches: &mut Vec<RecordBatch>,
                | -> ExecutorResult<()> {
                    buffered_rows.push(path.to_vec());
                    if buffered_rows.len() >= join_batch_size {
                        flush_join_rows(
                            buffered_rows,
                            &gather_plan_per_table,
                            &projection_out_map,
                            &table_batches,
                            &output_schema,
                            &computed_projections,
                            &field_id_by_table_index,
                            result_batches,
                        )?;
                    }
                    Ok(())
                };

                let mut path: Vec<Option<llkv_join::JoinRowRef>> = Vec::with_capacity(table_count);

                for (batch_idx, batch) in table_zero_batches.iter().enumerate() {
                    for row_idx in 0..batch.num_rows() {
                        path.clear();
                        path.push(Some(llkv_join::JoinRowRef {
                            batch: batch_idx,
                            row: row_idx,
                        }));
                        walk_join_paths(
                            0,
                            &adjacencies,
                            &mut path,
                            &mut emit_path,
                            &mut buffered_rows,
                            &mut result_batches,
                        )?;
                    }
                }

                if !buffered_rows.is_empty() {
                    flush_join_rows(
                        &mut buffered_rows,
                        &gather_plan_per_table,
                        &projection_out_map,
                        &table_batches,
                        &output_schema,
                        &computed_projections,
                        &field_id_by_table_index,
                        &mut result_batches,
                    )?;
                }

                let schema = output_schema;
                let batches = apply_offset_limit(result_batches, plan.offset, plan.limit);
                let table_name = multi
                    .table_order
                    .first()
                    .map(|t| t.qualified_name())
                    .unwrap_or_default();
                Ok(SelectExecution::new(table_name, schema, batches))
            }
        }
    }
}

fn flush_join_rows(
    rows: &mut Vec<Vec<Option<llkv_join::JoinRowRef>>>,
    gather_plan_per_table: &[Vec<usize>],
    projection_out_map: &[Vec<(usize, usize)>],
    table_batches: &[Option<Vec<RecordBatch>>],
    output_schema: &SchemaRef,
    computed_projections: &[ComputedProjectionPlan],
    field_id_by_table_index: &[Vec<FieldId>],
    result_batches: &mut Vec<RecordBatch>,
) -> ExecutorResult<()> {
    if rows.is_empty() {
        return Ok(());
    }

    let table_count = gather_plan_per_table.len();
    let mut per_table_rows: Vec<Vec<Option<(usize, usize)>>> = vec![Vec::new(); table_count];

    for row in rows.iter() {
        for (tbl_idx, cell) in row.iter().enumerate() {
            if let Some(rr) = cell {
                per_table_rows[tbl_idx].push(Some((rr.batch, rr.row)));
            } else {
                per_table_rows[tbl_idx].push(None);
            }
        }
    }

    let mut table_arrays: Vec<Vec<arrow::array::ArrayRef>> = vec![Vec::new(); table_count];

    for tbl_idx in 0..table_count {
        if gather_plan_per_table[tbl_idx].is_empty() {
            continue;
        }

        let batches = table_batches[tbl_idx]
            .as_ref()
            .ok_or_else(|| Error::Internal("validated earlier for all tables".into()))?;
        let arrays = gather_optional_projected_indices_from_batches(
            batches,
            &per_table_rows[tbl_idx],
            &gather_plan_per_table[tbl_idx],
        )
        .map_err(|e| Error::Internal(e.to_string()))?;
        table_arrays[tbl_idx] = arrays;
    }

    let total_columns: usize = output_schema.fields().len();
    let mut columns: Vec<Option<arrow::array::ArrayRef>> = vec![None; total_columns];

    for (tbl_idx, out_map) in projection_out_map.iter().enumerate() {
        for (out_idx, proj_idx) in out_map {
            let arr = table_arrays
                .get(tbl_idx)
                .and_then(|arrays| arrays.get(*proj_idx))
                .ok_or_else(|| Error::Internal("projection index missing".into()))?
                .clone();
            columns[*out_idx] = Some(arr);
        }
    }

    if !computed_projections.is_empty() {
        let mut field_arrays: FxHashMap<(usize, FieldId), arrow::array::ArrayRef> = FxHashMap::default();
        for (tbl_idx, gather_plan) in gather_plan_per_table.iter().enumerate() {
            if gather_plan.is_empty() {
                continue;
            }

            let arrays = table_arrays
                .get(tbl_idx)
                .ok_or_else(|| Error::Internal("projection arrays missing".into()))?;
            let field_ids = field_id_by_table_index
                .get(tbl_idx)
                .ok_or_else(|| Error::Internal("field ids missing".into()))?;

            for (arr_idx, col_idx) in gather_plan.iter().enumerate() {
                let fid = *field_ids
                    .get(*col_idx)
                    .ok_or_else(|| Error::Internal("field id missing for column".into()))?;
                let array = arrays
                    .get(arr_idx)
                    .ok_or_else(|| Error::Internal("array missing for gather index".into()))?
                    .clone();
                field_arrays.insert((tbl_idx, fid), array);
            }
        }

        let row_count = rows.len();
        let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, row_count);

        for computed in computed_projections {
            let evaluated = ScalarEvaluator::evaluate_batch_simplified(
                &computed.expr,
                row_count,
                &numeric_arrays,
            )
            .map_err(|e| Error::Internal(e.to_string()))?;

            let value = if evaluated.data_type() != &computed.data_type {
                cast(&evaluated, &computed.data_type)
                    .map_err(|e| Error::Internal(e.to_string()))?
            } else {
                evaluated
            };

            columns[computed.out_index] = Some(value);
        }
    }

    let columns: Vec<_> = columns
        .into_iter()
        .map(|c| c.ok_or_else(|| Error::Internal("projection column missing".into())))
        .collect::<Result<_, _>>()?;

    let batch = RecordBatch::try_new(output_schema.clone(), columns)
        .map_err(|e| Error::Internal(e.to_string()))?;
    result_batches.push(batch);
    rows.clear();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn walk_join_paths<F>(
    depth: usize,
    adjacencies: &[JoinAdjacency],
    path: &mut Vec<Option<llkv_join::JoinRowRef>>,
    emit_path: &mut F,
    buffered_rows: &mut Vec<Vec<Option<llkv_join::JoinRowRef>>>,
    result_batches: &mut Vec<RecordBatch>,
) -> ExecutorResult<()>
where
    F: FnMut(
        &[Option<llkv_join::JoinRowRef>],
        &mut Vec<Vec<Option<llkv_join::JoinRowRef>>>,
        &mut Vec<RecordBatch>,
    ) -> ExecutorResult<()>,
{
    if depth >= adjacencies.len() {
        emit_path(path, buffered_rows, result_batches)?;
        return Ok(());
    }

    let adj = &adjacencies[depth];
    let left_cell = path.get(adj.left_index).and_then(|c| *c);

    let candidates = left_cell
        .and_then(|lr| adj.matches.get(&(lr.batch, lr.row)));

    match candidates {
        Some(list) if !list.is_empty() => {
            for candidate in list {
                path.push(*candidate);
                walk_join_paths(
                    depth + 1,
                    adjacencies,
                    path,
                    emit_path,
                    buffered_rows,
                    result_batches,
                )?;
                path.pop();
            }
        }
        _ if matches!(adj.join_type, JoinType::Left) => {
            path.push(None);
            walk_join_paths(
                depth + 1,
                adjacencies,
                path,
                emit_path,
                buffered_rows,
                result_batches,
            )?;
            path.pop();
        }
        _ => {}
    }

    Ok(())
}

struct JoinAdjacency {
    left_index: usize,
    _right_index: usize,
    join_type: JoinType,
    matches: FxHashMap<(usize, usize), Vec<Option<llkv_join::JoinRowRef>>>,
}

fn collect_batches(plan: &dyn PhysicalPlan) -> ExecutorResult<Vec<RecordBatch>> {
    let iter = plan.execute().map_err(Error::Internal)?;
    let mut out = Vec::new();
    for batch in iter {
        out.push(batch.map_err(Error::Internal)?);
    }
    Ok(out)
}

fn apply_offset_limit(
    batches: Vec<RecordBatch>,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Vec<RecordBatch> {
    let mut remaining_offset = offset.unwrap_or(0);
    let mut remaining_limit = limit;
    let mut trimmed = Vec::new();

    for batch in batches {
        let mut start = 0usize;
        let mut len = batch.num_rows();

        if remaining_offset > 0 {
            if remaining_offset >= len {
                remaining_offset -= len;
                continue;
            }
            start = remaining_offset;
            len -= remaining_offset;
            remaining_offset = 0;
        }

        if let Some(limit) = remaining_limit {
            if limit == 0 {
                break;
            }
            if len > limit {
                len = limit;
            }
            remaining_limit = Some(limit - len);
        }

        trimmed.push(batch.slice(start, len));
        if let Some(limit) = remaining_limit {
            if limit == 0 {
                break;
            }
        }
    }

    trimmed
}

fn aggregate_select(
    plan: &SelectPlan,
    logical_schema: &llkv_plan::schema::PlanSchema,
    physical_schema: &SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<(SchemaRef, Vec<RecordBatch>), Error> {
    let mut field_index_by_id: FxHashMap<FieldId, usize> = FxHashMap::default();
    for (idx, field) in physical_schema.fields().iter().enumerate() {
        if let Some(fid_str) = field.metadata().get("field_id") {
            if let Ok(fid) = fid_str.parse::<FieldId>() {
                field_index_by_id.insert(fid, idx);
            }
        }
    }

    let mut specs = Vec::with_capacity(plan.aggregates.len());
    for agg in &plan.aggregates {
        match agg {
            llkv_plan::AggregateExpr::CountStar { alias, distinct } => {
                specs.push(AggregateSpec {
                    alias: alias.clone(),
                    kind: AggregateKind::Count {
                        field_id: None,
                        distinct: *distinct,
                    },
                });
            }
            llkv_plan::AggregateExpr::Column {
                column,
                alias,
                function,
                distinct,
            } => {
                let col = logical_schema.column_by_name(column).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column '{}' in aggregate",
                        column
                    ))
                })?;

                let kind = match function {
                    llkv_plan::AggregateFunction::Count => AggregateKind::Count {
                        field_id: Some(col.field_id),
                        distinct: *distinct,
                    },
                    llkv_plan::AggregateFunction::SumInt64 => AggregateKind::Sum {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                        distinct: *distinct,
                    },
                    llkv_plan::AggregateFunction::TotalInt64 => AggregateKind::Total {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                        distinct: *distinct,
                    },
                    llkv_plan::AggregateFunction::MinInt64 => AggregateKind::Min {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                    },
                    llkv_plan::AggregateFunction::MaxInt64 => AggregateKind::Max {
                        field_id: col.field_id,
                        data_type: col.data_type.clone(),
                    },
                    llkv_plan::AggregateFunction::CountNulls => AggregateKind::CountNulls {
                        field_id: col.field_id,
                    },
                    llkv_plan::AggregateFunction::GroupConcat => {
                        return Err(Error::InvalidArgumentError(
                            "GROUP_CONCAT aggregate is not supported yet".into(),
                        ));
                    }
                };

                specs.push(AggregateSpec {
                    alias: alias.clone(),
                    kind,
                });
            }
        }
    }

    let mut states = Vec::with_capacity(specs.len());
    for spec in specs {
        let projection_idx = spec
            .kind
            .field_id()
            .and_then(|fid| field_index_by_id.get(&fid).copied());

        if spec.kind.field_id().is_some() && projection_idx.is_none() {
            return Err(Error::InvalidArgumentError(
                "aggregate input column missing from scan output".into(),
            ));
        }

        let accumulator = AggregateAccumulator::new_with_projection_index(&spec, projection_idx, None)?;
        states.push(AggregateState::new(spec.alias.clone(), accumulator, None));
    }

    for batch in &batches {
        for state in states.iter_mut() {
            state.update(batch)?;
        }
    }

    let mut fields = Vec::with_capacity(states.len());
    let mut arrays = Vec::with_capacity(states.len());
    for state in states.into_iter() {
        let (field, array) = state.finalize()?;
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
        .map_err(|e| Error::Internal(e.to_string()))?;
    Ok((schema, vec![batch]))
}

/// Materialised SELECT result batches.
#[derive(Clone, Debug)]
pub struct SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table_name: String,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    _marker: PhantomData<P>,
}

impl<P> SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(table_name: String, schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self {
            table_name,
            schema,
            batches,
            _marker: PhantomData,
        }
    }

    pub fn new_single_batch(
        table_name: String,
        schema: SchemaRef,
        batch: RecordBatch,
    ) -> Self {
        Self::new(table_name, schema, vec![batch])
    }

    pub fn from_batch(table_name: String, schema: SchemaRef, batch: RecordBatch) -> Self {
        Self::new_single_batch(table_name, schema, batch)
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn collect(&self) -> ExecutorResult<Vec<RecordBatch>> {
        Ok(self.batches.clone())
    }

    pub fn stream<F>(&self, mut on_batch: F) -> ExecutorResult<()>
    where
        F: FnMut(RecordBatch) -> ExecutorResult<()>,
    {
        for batch in &self.batches {
            on_batch(batch.clone())?;
        }
        Ok(())
    }
}

/// Adapter to expose executor tables to the planner as `ExecutionTable`s.
struct PlannerTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    inner: Arc<dyn ExecutorTableProvider<P>>,
}

impl<P> TableProvider<P> for PlannerTableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, name: &str) -> llkv_result::Result<Arc<dyn ExecutionTable<P>>> {
        let table = self.inner.get_table(name)?;
        Ok(Arc::new(ExecutionTableAdapter::new(table)))
    }
}

struct ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: Arc<ExecutorTable<P>>,
    plan_schema: Arc<llkv_plan::schema::PlanSchema>,
}

struct ComputedProjectionPlan {
    expr: ScalarExpr<(usize, FieldId)>,
    data_type: arrow::datatypes::DataType,
    out_index: usize,
}

impl<P> fmt::Debug for ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionTableAdapter")
            .field("table_id", &self.table.table_id())
            .finish()
    }
}

impl<P> ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new(table: Arc<ExecutorTable<P>>) -> Self {
        let mut name_to_index = FxHashMap::default();
        let columns: Vec<llkv_plan::schema::PlanColumn> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                name_to_index.insert(c.name.to_ascii_lowercase(), idx);
                llkv_plan::schema::PlanColumn {
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    field_id: c.field_id,
                    is_nullable: c.is_nullable,
                    is_primary_key: c.is_primary_key,
                    is_unique: c.is_unique,
                    default_value: c.default_value.clone(),
                    check_expr: c.check_expr.clone(),
                }
            })
            .collect();
        let plan_schema = Arc::new(llkv_plan::schema::PlanSchema { columns, name_to_index });
        Self { table, plan_schema }
    }

    fn executor_table(&self) -> Arc<ExecutorTable<P>> {
        Arc::clone(&self.table)
    }
}

fn index_fields_by_id(schema: &PlanSchema) -> FxHashMap<FieldId, usize> {
    let mut map = FxHashMap::default();
    for (idx, col) in schema.columns.iter().enumerate() {
        map.insert(col.field_id, idx);
    }
    map
}

fn distinct_record_batches(
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
) -> ExecutorResult<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let sort_fields: Vec<SortField> = schema
        .fields()
        .iter()
        .map(|f| SortField::new(f.data_type().clone()))
        .collect();
    let converter = RowConverter::new(sort_fields).map_err(|e| Error::Internal(e.to_string()))?;

    let mut seen: FxHashSet<OwnedRow> = FxHashSet::default();
    let mut unique_rows: Vec<Option<(usize, usize)>> = Vec::new();

    for (batch_idx, batch) in batches.iter().enumerate() {
        let rows = converter
            .convert_columns(batch.columns())
            .map_err(|e| Error::Internal(e.to_string()))?;

        for row_idx in 0..batch.num_rows() {
            let owned = rows.row(row_idx).owned();
            if seen.insert(owned) {
                unique_rows.push(Some((batch_idx, row_idx)));
            }
        }
    }

    if unique_rows.is_empty() {
        return Ok(Vec::new());
    }

    let projection: Vec<usize> = (0..schema.fields().len()).collect();
    let arrays = gather_optional_projected_indices_from_batches(
        &batches,
        &unique_rows,
        &projection,
    )
    .map_err(|e| Error::Internal(e.to_string()))?;

    let batch = RecordBatch::try_new(Arc::clone(schema), arrays)
        .map_err(|e| Error::Internal(e.to_string()))?;
    Ok(vec![batch])
}

fn remap_scalar_expr(
    expr: &ScalarExpr<llkv_plan::logical_planner::ResolvedFieldRef>,
) -> ScalarExpr<(usize, FieldId)> {
    match expr {
        ScalarExpr::Column(resolved) => {
            ScalarExpr::Column((resolved.table_index, resolved.logical_field_id.field_id()))
        }
        ScalarExpr::Literal(lit) => ScalarExpr::Literal(lit.clone()),
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(remap_scalar_expr(left)),
            op: *op,
            right: Box::new(remap_scalar_expr(right)),
        },
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(remap_scalar_expr(left)),
            op: *op,
            right: Box::new(remap_scalar_expr(right)),
        },
        ScalarExpr::Not(inner) => ScalarExpr::Not(Box::new(remap_scalar_expr(inner))),
        ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
            expr: Box::new(remap_scalar_expr(expr)),
            negated: *negated,
        },
        ScalarExpr::Aggregate(call) => ScalarExpr::Aggregate(match call {
            AggregateCall::CountStar => AggregateCall::CountStar,
            AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
            },
            AggregateCall::Min(expr) => AggregateCall::Min(Box::new(remap_scalar_expr(expr))),
            AggregateCall::Max(expr) => AggregateCall::Max(Box::new(remap_scalar_expr(expr))),
            AggregateCall::CountNulls(expr) => {
                AggregateCall::CountNulls(Box::new(remap_scalar_expr(expr)))
            }
            AggregateCall::GroupConcat {
                expr,
                distinct,
                separator,
            } => AggregateCall::GroupConcat {
                expr: Box::new(remap_scalar_expr(expr)),
                distinct: *distinct,
                separator: separator.clone(),
            },
        }),
        ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
            base: Box::new(remap_scalar_expr(base)),
            field_name: field_name.clone(),
        },
        ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
            expr: Box::new(remap_scalar_expr(expr)),
            data_type: data_type.clone(),
        },
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => ScalarExpr::Case {
            operand: operand
                .as_deref()
                .map(remap_scalar_expr)
                .map(Box::new),
            branches: branches
                .iter()
                .map(|(when_expr, then_expr)| {
                    (
                        remap_scalar_expr(when_expr),
                        remap_scalar_expr(then_expr),
                    )
                })
                .collect(),
            else_expr: else_expr
                .as_deref()
                .map(remap_scalar_expr)
                .map(Box::new),
        },
        ScalarExpr::Coalesce(items) => {
            ScalarExpr::Coalesce(items.iter().map(remap_scalar_expr).collect())
        }
        ScalarExpr::Random => ScalarExpr::Random,
        ScalarExpr::ScalarSubquery(subquery) => ScalarExpr::ScalarSubquery(subquery.clone()),
    }
}

fn map_join_type(join_plan: JoinPlan) -> ExecutorResult<JoinType> {
    match join_plan {
        JoinPlan::Inner => Ok(JoinType::Inner),
        JoinPlan::Left => Ok(JoinType::Left),
        JoinPlan::Right => Err(Error::InvalidArgumentError(
            "RIGHT JOIN is not supported yet".into(),
        )),
        JoinPlan::Full => Err(Error::InvalidArgumentError(
            "FULL JOIN is not supported yet".into(),
        )),
    }
}

fn extract_join_keys(join: &ResolvedJoin) -> ExecutorResult<Vec<JoinKey>> {
    let Some(on) = &join.on else {
        // Treat missing ON as cross product
        return Ok(Vec::new());
    };

    let mut keys = Vec::new();
    collect_join_keys(on, join.left_table_index, &mut keys)?;

    if keys.is_empty() {
        return Err(Error::InvalidArgumentError(
            "join ON clause must include at least one equality between tables".into(),
        ));
    }

    Ok(keys)
}

fn collect_join_keys(
    expr: &LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
    left_table_index: usize,
    out: &mut Vec<JoinKey>,
) -> ExecutorResult<()> {
    match expr {
        LlkvExpr::And(list) => {
            for e in list {
                collect_join_keys(e, left_table_index, out)?;
            }
            Ok(())
        }
        LlkvExpr::Compare { left, op, right } => {
            if !matches!(op, CompareOp::Eq) {
                return Err(Error::InvalidArgumentError(
                    "only equality predicates are supported in JOIN ON".into(),
                ));
            }

            match (left, right) {
                (ScalarExpr::Column(l), ScalarExpr::Column(r))
                    if (l.table_index == left_table_index
                        && r.table_index == left_table_index + 1)
                        || (r.table_index == left_table_index
                            && l.table_index == left_table_index + 1) =>
                {
                    let (left_col, right_col) = if l.table_index == left_table_index {
                        (l, r)
                    } else {
                        (r, l)
                    };

                    out.push(JoinKey {
                        left_field: left_col.logical_field_id.field_id(),
                        right_field: right_col.logical_field_id.field_id(),
                        null_equals_null: false,
                    });
                    Ok(())
                }
                _ => Err(Error::InvalidArgumentError(
                    "only equality predicates between consecutive tables are supported in JOIN ON"
                        .into(),
                )),
            }
        }
        _ => Err(Error::InvalidArgumentError(
            "unsupported predicate in JOIN ON clause".into(),
        )),
    }
}

impl<P> ExecutionTable<P> for ExecutionTableAdapter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn schema(&self) -> Arc<llkv_plan::schema::PlanSchema> {
        Arc::clone(&self.plan_schema)
    }

    fn table_id(&self) -> llkv_types::ids::TableId {
        self.table.table_id()
    }

    fn scan_stream(
        &self,
        projections: &[llkv_scan::ScanProjection],
        predicate: &llkv_expr::Expr<'static, llkv_types::FieldId>,
        options: llkv_scan::ScanStreamOptions<P>,
        callback: &mut dyn FnMut(RecordBatch),
    ) -> llkv_result::Result<()> {
        self.table
            .storage()
            .scan_stream(projections, predicate, options, callback)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

