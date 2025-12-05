use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::datatypes::{Field as ArrowField, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};
use llkv_expr::expr::{CompareOp, Expr as LlkvExpr, ScalarExpr};
use llkv_join::{JoinKey, JoinOptions, JoinSide, JoinType, TableJoinRowIdExt, project_join_columns};
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
use rustc_hash::FxHashMap;
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
                    let batches = apply_offset_limit(batches, plan.offset, plan.limit);
                    Ok(SelectExecution::new(table_name, schema, batches))
                }
            }
            LogicalPlan::Multi(multi) => {
                if multi.tables.len() != 2 {
                    return Err(Error::InvalidArgumentError(
                        "multi-table SELECT currently supports exactly two tables".into(),
                    ));
                }

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

                let join = multi
                    .joins
                    .first()
                    .ok_or_else(|| Error::InvalidArgumentError("missing join metadata".into()))?;

                if join.left_table_index != 0 {
                    return Err(Error::InvalidArgumentError(
                        "only left-deep joins starting from the first table are supported".into(),
                    ));
                }

                let left = multi.tables.get(0).ok_or_else(|| {
                    Error::InvalidArgumentError("missing left table in multi-table plan".into())
                })?;
                let right = multi.tables.get(1).ok_or_else(|| {
                    Error::InvalidArgumentError("missing right table in multi-table plan".into())
                })?;

                let left_exec = left
                    .table
                    .as_any()
                    .downcast_ref::<ExecutionTableAdapter<P>>()
                    .ok_or_else(|| Error::Internal("unexpected table adapter type".into()))?
                    .executor_table();
                let right_exec = right
                    .table
                    .as_any()
                    .downcast_ref::<ExecutionTableAdapter<P>>()
                    .ok_or_else(|| Error::Internal("unexpected table adapter type".into()))?
                    .executor_table();

                let join_type = map_join_type(join.join_type)?;
                let join_keys = extract_join_keys(join)?;

                let left_index_by_fid = index_fields_by_id(&left.schema);
                let right_index_by_fid = index_fields_by_id(&right.schema);

                let mut output_fields = Vec::with_capacity(multi.projections.len());
                let mut projection_plan = Vec::with_capacity(multi.projections.len());

                for proj in &multi.projections {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column {
                            table_index,
                            logical_field_id,
                            alias,
                        } => {
                            let (schema, index_by_fid) = if *table_index == 0 {
                                (&left.schema, &left_index_by_fid)
                            } else {
                                (&right.schema, &right_index_by_fid)
                            };

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
                            projection_plan.push((*table_index, col_idx));
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { .. } => {
                            return Err(Error::InvalidArgumentError(
                                "computed projections are not supported yet for multi-table SELECT".into(),
                            ))
                        }
                    }
                }

                let mut left_projection = Vec::new();
                let mut right_projection = Vec::new();
                let mut left_out_map = Vec::new();
                let mut right_out_map = Vec::new();

                for (out_idx, (table_idx, col_idx)) in projection_plan.iter().enumerate() {
                    if *table_idx == 0 {
                        let proj_idx = left_projection.len();
                        left_projection.push(*col_idx);
                        left_out_map.push((out_idx, proj_idx));
                    } else {
                        let proj_idx = right_projection.len();
                        right_projection.push(*col_idx);
                        right_out_map.push((out_idx, proj_idx));
                    }
                }

                let mut result_batches = Vec::new();
                let options = JoinOptions {
                    join_type,
                    ..Default::default()
                };

                let output_schema = Arc::new(Schema::new(output_fields.clone()));

                left_exec.table.join_rowid_stream_with_filter(
                    &right_exec.table,
                    &join_keys,
                    &options,
                    row_filter.clone(),
                    row_filter.clone(),
                    |index_batch| {
                        let filtered_batch = index_batch;

                        if filtered_batch.left_rows.is_empty() {
                            return;
                        }

                        let left_arrays = if left_projection.is_empty() {
                            Vec::new()
                        } else {
                            project_join_columns(
                                &filtered_batch,
                                JoinSide::Left,
                                &left_projection,
                            )
                            .expect("left projection gather failed")
                        };

                        let right_arrays = if right_projection.is_empty() {
                            Vec::new()
                        } else {
                            project_join_columns(
                                &filtered_batch,
                                JoinSide::Right,
                                &right_projection,
                            )
                            .expect("right projection gather failed")
                        };

                        let mut columns = vec![None; projection_plan.len()];
                        for (out_idx, proj_idx) in &left_out_map {
                            columns[*out_idx] = Some(left_arrays[*proj_idx].clone());
                        }
                        for (out_idx, proj_idx) in &right_out_map {
                            columns[*out_idx] = Some(right_arrays[*proj_idx].clone());
                        }

                        let columns: Vec<_> = columns
                            .into_iter()
                            .map(|c| c.expect("projection column missing"))
                            .collect();

                        let batch = RecordBatch::try_new(output_schema.clone(), columns)
                            .expect("join projection record batch creation failed");
                        result_batches.push(batch);
                    },
                )?;

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

