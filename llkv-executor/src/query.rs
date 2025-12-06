use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, mpsc};

use arrow::compute::concat_batches;
use arrow::array::ArrayRef;
use arrow::datatypes::{Field as ArrowField, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{OwnedRow, RowConverter, SortField};
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};
use llkv_column_map::gather::gather_optional_projected_indices_from_batches;
use llkv_column_map::store::Projection;

use llkv_compute::eval::ScalarEvaluator;
use llkv_expr::AggregateCall;
use llkv_expr::expr::{CompareOp, Expr as LlkvExpr, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_expr::{Expr, Filter, Operator};
use llkv_join::{JoinKey, JoinType};
use llkv_plan::logical_planner::{LogicalPlan, LogicalPlanner, ResolvedJoin};
use llkv_plan::physical::table::{ExecutionTable, TableProvider};
use llkv_plan::planner::PhysicalPlanner;
use llkv_plan::plans::JoinPlan;
use llkv_plan::plans::SelectPlan;
use llkv_result::Error;
use llkv_scan::{RowIdFilter, ScanProjection};
use llkv_storage::pager::Pager;
use llkv_types::FieldId;
use llkv_types::LogicalFieldId;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;

use crate::ExecutorResult;
use crate::types::{ExecutorTable, ExecutorTableProvider};
use llkv_join::vectorized::VectorizedHashJoinStream;

pub type BatchIter = Box<dyn Iterator<Item = ExecutorResult<RecordBatch>> + Send>;
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

                let base_iter: BatchIter = Box::new(
                    physical_plan
                        .execute()
                        .map_err(Error::Internal)?
                        .map(|b| b.map_err(Error::Internal)),
                );

                if !plan.aggregates.is_empty() {
                    if !plan.group_by.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "GROUP BY aggregates are not supported yet".into(),
                        ));
                    }

                    let agg_iter = AggregateStream::new(base_iter, &plan, &single.schema, &schema)?;
                    let trimmed =
                        apply_offset_limit_stream(Box::new(agg_iter), plan.offset, plan.limit);
                    return Ok(SelectExecution::from_stream(table_name, schema, trimmed));
                }

                let mut iter: BatchIter = base_iter;
                if plan.distinct {
                    iter = Box::new(DistinctStream::new(schema.clone(), iter)?);
                }

                let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);
                Ok(SelectExecution::from_stream(table_name, schema, trimmed))
            }
            LogicalPlan::Multi(multi) => {
                let table_count = multi.tables.len();
                
                // 1. Analyze required columns
                let mut required_fields: Vec<FxHashSet<FieldId>> = vec![FxHashSet::default(); table_count];
                
                for proj in &multi.projections {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, .. } => {
                            required_fields[*table_index].insert(logical_field_id.field_id());
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { expr, .. } => {
                             let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                             ScalarEvaluator::collect_fields(&remap_scalar_expr(expr), &mut fields);
                             for (tbl, fid) in fields {
                                 required_fields[tbl].insert(fid);
                             }
                        }
                    }
                }
                
                for join in &multi.joins {
                    let (keys, filters) = extract_join_keys_and_filters(join)?;
                    for key in keys {
                        required_fields[join.left_table_index].insert(key.left_field);
                        required_fields[join.left_table_index + 1].insert(key.right_field);
                    }
                    for filter in filters {
                         let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                         ScalarEvaluator::collect_fields(&remap_filter_expr(&filter)?, &mut fields);
                         for (tbl, fid) in fields {
                             required_fields[tbl].insert(fid);
                         }
                    }
                }
                
                if let Some(filter) = &multi.filter {
                     let mut fields: FxHashSet<(usize, FieldId)> = FxHashSet::default();
                     ScalarEvaluator::collect_fields(&remap_filter_expr(filter)?, &mut fields);
                     for (tbl, fid) in fields {
                         required_fields[tbl].insert(fid);
                     }
                }

                // 2. Create Streams
                let mut streams: Vec<BatchIter> = Vec::with_capacity(table_count);
                let mut schemas: Vec<SchemaRef> = Vec::with_capacity(table_count);
                let mut table_field_map: Vec<Vec<FieldId>> = Vec::with_capacity(table_count);

                for (i, table) in multi.tables.iter().enumerate() {
                    let adapter = table.table.as_any().downcast_ref::<ExecutionTableAdapter<P>>()
                        .ok_or_else(|| Error::Internal("unexpected table adapter type".into()))?;
                    
                    let mut fields: Vec<FieldId> = required_fields[i].iter().copied().collect();
                    if fields.is_empty() {
                        if let Some(col) = table.schema.columns.first() {
                            fields.push(col.field_id);
                        }
                    }
                    fields.sort_unstable();
                    table_field_map.push(fields.clone());
                    
                    let projections: Vec<ScanProjection> = fields.iter().map(|&fid| {
                        let col = table.schema.columns.iter().find(|c| c.field_id == fid).unwrap();
                        let lfid = LogicalFieldId::for_user(adapter.executor_table().table_id(), fid);
                        ScanProjection::Column(Projection::with_alias(lfid, col.name.clone()))
                    }).collect();
                    
                    let arrow_fields: Vec<ArrowField> = fields.iter().map(|&fid| {
                        let col = table.schema.columns.iter().find(|c| c.field_id == fid).unwrap();
                        ArrowField::new(col.name.clone(), col.data_type.clone(), col.is_nullable)
                    }).collect();
                    schemas.push(Arc::new(Schema::new(arrow_fields)));

                    let (tx, rx) = mpsc::sync_channel(16);
                    let storage = adapter.executor_table().storage().clone();
                    let row_filter = row_filter.clone();
                    
                    std::thread::spawn(move || {
                        let res = storage.scan_stream(
                            &projections,
                            &Expr::Pred(Filter { field_id: 0, op: Operator::Range { lower: std::ops::Bound::Unbounded, upper: std::ops::Bound::Unbounded } }),
                            llkv_scan::ScanStreamOptions { row_id_filter: row_filter.clone(), ..Default::default() },
                            &mut |batch| {
                                tx.send(Ok(batch)).ok();
                            },
                        );
                        if let Err(e) = res {
                            tx.send(Err(e)).ok();
                        }
                    });
                    
                    streams.push(Box::new(rx.into_iter()));
                }

                // 3. Build Pipeline
                let mut current_stream = streams.remove(0);
                let mut current_schema = schemas[0].clone();
                let mut col_mapping: FxHashMap<(usize, FieldId), usize> = FxHashMap::default();
                
                for (idx, fid) in table_field_map[0].iter().enumerate() {
                    col_mapping.insert((0, *fid), idx);
                }

                for i in 0..table_count - 1 {
                    let right_stream = streams.remove(0);
                    let right_schema = schemas[i+1].clone();
                    
                    let right_batches: Vec<RecordBatch> = right_stream.collect::<Result<Vec<_>, _>>()?;
                    let right_batch = if right_batches.is_empty() {
                        RecordBatch::new_empty(right_schema.clone())
                    } else {
                        concat_batches(&right_schema, &right_batches)?
                    };
                    
                    let join_opt = multi.joins.iter().find(|j| j.left_table_index == i);
                    
                    let (join_type, left_indices, right_indices) = if let Some(join) = join_opt {
                        let (keys, _) = extract_join_keys_and_filters(join)?;
                        let mut left_indices = Vec::new();
                        let mut right_indices = Vec::new();
                        
                        for key in keys {
                            let left_idx = *col_mapping.get(&(join.left_table_index, key.left_field))
                                .ok_or_else(|| Error::Internal("left join key missing".into()))?;
                            left_indices.push(left_idx);
                            
                            let right_idx = table_field_map[i+1].iter().position(|&f| f == key.right_field)
                                .ok_or_else(|| Error::Internal("right join key missing".into()))?;
                            right_indices.push(right_idx);
                        }
                        (map_join_type(join.join_type)?, left_indices, right_indices)
                    } else {
                        (JoinType::Inner, vec![], vec![])
                    };
                        
                    let mut new_fields = current_schema.fields().to_vec();
                    new_fields.extend_from_slice(right_schema.fields());
                    let new_schema = Arc::new(Schema::new(new_fields));

                    current_stream = Box::new(VectorizedHashJoinStream::try_new(
                        new_schema.clone(),
                        current_stream,
                        right_batch,
                        join_type,
                        left_indices,
                        right_indices,
                    )?);
                    
                    let old_len = current_schema.fields().len();
                    current_schema = new_schema;
                    
                    for (idx, fid) in table_field_map[i+1].iter().enumerate() {
                        col_mapping.insert((i+1, *fid), old_len + idx);
                    }
                }
                
                // 4. Final Projection
                let output_fields: Vec<ArrowField> = multi.projections.iter().map(|proj| {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, alias } => {
                            let idx = col_mapping.get(&(*table_index, logical_field_id.field_id())).unwrap();
                            let field = current_schema.field(*idx);
                            ArrowField::new(alias.clone().unwrap_or_else(|| field.name().clone()), field.data_type().clone(), field.is_nullable())
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { alias, .. } => {
                            // Placeholder type, will be inferred or we assume Int64 for now
                            ArrowField::new(alias.clone(), arrow::datatypes::DataType::Int64, true)
                        }
                    }
                }).collect();
                let output_schema = Arc::new(Schema::new(output_fields));
                
                let output_schema_captured = output_schema.clone();
                let final_stream = current_stream.map(move |batch_res| {
                    let batch = batch_res?;
                    let mut columns = Vec::new();
                    
                    for proj in &multi.projections {
                        match proj {
                            llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, .. } => {
                                let idx = col_mapping.get(&(*table_index, logical_field_id.field_id()))
                                    .ok_or_else(|| Error::Internal("projection column missing".into()))?;
                                columns.push(batch.column(*idx).clone());
                            }
                            llkv_plan::logical_planner::ResolvedProjection::Computed { expr, .. } => {
                                let remapped = remap_scalar_expr(expr);
                                
                                let mut required_fields = FxHashSet::default();
                                ScalarEvaluator::collect_fields(&remapped, &mut required_fields);
                                
                                let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
                                for (tbl, fid) in required_fields {
                                    if let Some(idx) = col_mapping.get(&(tbl, fid)) {
                                        field_arrays.insert((tbl, fid), batch.column(*idx).clone());
                                    } else {
                                        return Err(Error::Internal(format!("Missing field {:?} in batch for computed column", (tbl, fid))));
                                    }
                                }
                                
                                let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
                                let result = ScalarEvaluator::evaluate_batch_simplified(&remapped, batch.num_rows(), &numeric_arrays)?;
                                columns.push(result);
                            }
                        }
                    }
                    RecordBatch::try_new(output_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
                });

                let trimmed = apply_offset_limit_stream(Box::new(final_stream), plan.offset, plan.limit);
                let table_name = multi.table_order.first().map(|t| t.qualified_name()).unwrap_or_default();
                Ok(SelectExecution::from_stream(table_name, output_schema, trimmed))
        }
    }
}
}

struct AggregateStream {
    states: Vec<AggregateState>,
    input: BatchIter,
    done: bool,
}

impl AggregateStream {
    fn new(
        input: BatchIter,
        plan: &SelectPlan,
        logical_schema: &llkv_plan::schema::PlanSchema,
        physical_schema: &SchemaRef,
    ) -> ExecutorResult<Self> {
        let states = build_aggregate_states(plan, logical_schema, physical_schema)?;
        Ok(Self {
            states,
            input,
            done: false,
        })
    }
}

impl Iterator for AggregateStream {
    type Item = ExecutorResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        for batch in self.input.by_ref() {
            let batch = match batch {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            for state in self.states.iter_mut() {
                if let Err(e) = state.update(&batch) {
                    return Some(Err(e));
                }
            }
        }

        let mut fields = Vec::with_capacity(self.states.len());
        let mut arrays = Vec::with_capacity(self.states.len());
        for state in self.states.drain(..) {
            let (field, array) = match state.finalize() {
                Ok(res) => res,
                Err(e) => return Some(Err(e)),
            };
            fields.push(field);
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = match RecordBatch::try_new(Arc::clone(&schema), arrays) {
            Ok(b) => b,
            Err(e) => return Some(Err(Error::Internal(e.to_string()))),
        };

        self.done = true;
        Some(Ok(batch))
    }
}

fn build_aggregate_states(
    plan: &SelectPlan,
    logical_schema: &llkv_plan::schema::PlanSchema,
    physical_schema: &SchemaRef,
) -> Result<Vec<AggregateState>, Error> {
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
                    Error::InvalidArgumentError(format!("unknown column '{}' in aggregate", column))
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

        let accumulator =
            AggregateAccumulator::new_with_projection_index(&spec, projection_idx, None)?;
        states.push(AggregateState::new(spec.alias.clone(), accumulator, None));
    }

    Ok(states)
}

struct OffsetLimitStream {
    input: BatchIter,
    remaining_offset: usize,
    remaining_limit: Option<usize>,
}

impl Iterator for OffsetLimitStream {
    type Item = ExecutorResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let batch = match self.input.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => return None,
            };

            if let Some(limit) = self.remaining_limit {
                if limit == 0 {
                    return None;
                }
            }

            let mut start = 0usize;
            let mut len = batch.num_rows();

            if self.remaining_offset > 0 {
                if self.remaining_offset >= len {
                    self.remaining_offset -= len;
                    continue;
                }
                start = self.remaining_offset;
                len -= self.remaining_offset;
                self.remaining_offset = 0;
            }

            if let Some(limit) = self.remaining_limit {
                if len > limit {
                    len = limit;
                }
                self.remaining_limit = Some(limit - len);
            }

            return Some(Ok(batch.slice(start, len)));
        }
    }
}

fn apply_offset_limit_stream(
    iter: BatchIter,
    offset: Option<usize>,
    limit: Option<usize>,
) -> BatchIter {
    Box::new(OffsetLimitStream {
        input: iter,
        remaining_offset: offset.unwrap_or(0),
        remaining_limit: limit,
    })
}

enum SelectSource {
    Stream(BatchIter),
    Materialized(Vec<RecordBatch>),
}

/// Streaming-friendly SELECT execution handle.
pub struct SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table_name: String,
    schema: SchemaRef,
    source: Arc<Mutex<SelectSource>>,
    _marker: PhantomData<P>,
}

impl<P> Clone for SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            table_name: self.table_name.clone(),
            schema: Arc::clone(&self.schema),
            source: Arc::clone(&self.source),
            _marker: PhantomData,
        }
    }
}

impl<P> fmt::Debug for SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectExecution")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .finish()
    }
}

impl<P> SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn new(table_name: String, schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self::from_materialized(table_name, schema, batches)
    }

    pub fn new_single_batch(table_name: String, schema: SchemaRef, batch: RecordBatch) -> Self {
        Self::from_materialized(table_name, schema, vec![batch])
    }

    pub fn from_batch(table_name: String, schema: SchemaRef, batch: RecordBatch) -> Self {
        Self::new_single_batch(table_name, schema, batch)
    }

    pub fn from_stream(table_name: String, schema: SchemaRef, iter: BatchIter) -> Self {
        Self {
            table_name,
            schema,
            source: Arc::new(Mutex::new(SelectSource::Stream(iter))),
            _marker: PhantomData,
        }
    }

    fn from_materialized(table_name: String, schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self {
            table_name,
            schema,
            source: Arc::new(Mutex::new(SelectSource::Materialized(batches))),
            _marker: PhantomData,
        }
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn collect(&self) -> ExecutorResult<Vec<RecordBatch>> {
        let mut guard = self
            .source
            .lock()
            .map_err(|_| Error::Internal("select stream poisoned".into()))?;
        match &mut *guard {
            SelectSource::Materialized(batches) => Ok(batches.clone()),
            SelectSource::Stream(iter) => {
                let mut collected = Vec::new();
                for batch in iter {
                    collected.push(batch?);
                }
                *guard = SelectSource::Materialized(collected.clone());
                Ok(collected)
            }
        }
    }

    pub fn stream<F>(&self, mut on_batch: F) -> ExecutorResult<()>
    where
        F: FnMut(RecordBatch) -> ExecutorResult<()>,
    {
        let mut guard = self
            .source
            .lock()
            .map_err(|_| Error::Internal("select stream poisoned".into()))?;
        match &mut *guard {
            SelectSource::Materialized(batches) => {
                for batch in batches.iter() {
                    on_batch(batch.clone())?;
                }
            }
            SelectSource::Stream(iter) => {
                for batch in iter {
                    on_batch(batch?)?;
                }
                *guard = SelectSource::Materialized(Vec::new());
            }
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
        let plan_schema = Arc::new(llkv_plan::schema::PlanSchema {
            columns,
            name_to_index,
        });
        Self { table, plan_schema }
    }

    fn executor_table(&self) -> Arc<ExecutorTable<P>> {
        Arc::clone(&self.table)
    }
}



struct DistinctStream {
    schema: SchemaRef,
    converter: RowConverter,
    seen: FxHashSet<OwnedRow>,
    input: BatchIter,
}

impl DistinctStream {
    fn new(schema: SchemaRef, input: BatchIter) -> ExecutorResult<Self> {
        let sort_fields: Vec<SortField> = schema
            .fields()
            .iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();
        let converter =
            RowConverter::new(sort_fields).map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            schema,
            converter,
            seen: FxHashSet::default(),
            input,
        })
    }
}

impl Iterator for DistinctStream {
    type Item = ExecutorResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(batch) = self.input.next() {
            let batch = match batch {
                Ok(b) => b,
                Err(e) => return Some(Err(e)),
            };

            let rows = match self.converter.convert_columns(batch.columns()) {
                Ok(r) => r,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };

            let mut unique_rows: Vec<Option<(usize, usize)>> = Vec::new();
            for row_idx in 0..batch.num_rows() {
                let owned = rows.row(row_idx).owned();
                if self.seen.insert(owned) {
                    unique_rows.push(Some((0, row_idx)));
                }
            }

            if unique_rows.is_empty() {
                continue;
            }

            let projection: Vec<usize> = (0..self.schema.fields().len()).collect();
            let arrays = match gather_optional_projected_indices_from_batches(
                &[batch],
                &unique_rows,
                &projection,
            ) {
                Ok(a) => a,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };

            let out = match RecordBatch::try_new(Arc::clone(&self.schema), arrays) {
                Ok(b) => b,
                Err(e) => return Some(Err(Error::Internal(e.to_string()))),
            };
            return Some(Ok(out));
        }
        None
    }
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
            operand: operand.as_deref().map(remap_scalar_expr).map(Box::new),
            branches: branches
                .iter()
                .map(|(when_expr, then_expr)| {
                    (remap_scalar_expr(when_expr), remap_scalar_expr(then_expr))
                })
                .collect(),
            else_expr: else_expr.as_deref().map(remap_scalar_expr).map(Box::new),
        },
        ScalarExpr::Coalesce(items) => {
            ScalarExpr::Coalesce(items.iter().map(remap_scalar_expr).collect())
        }
        ScalarExpr::Random => ScalarExpr::Random,
        ScalarExpr::ScalarSubquery(subquery) => ScalarExpr::ScalarSubquery(subquery.clone()),
    }
}

fn remap_filter_expr(
    expr: &LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
) -> ExecutorResult<ScalarExpr<(usize, FieldId)>> {
    fn combine_with_op(
        mut exprs: impl Iterator<Item = ExecutorResult<ScalarExpr<(usize, FieldId)>>>,
        op: llkv_expr::expr::BinaryOp,
    ) -> ExecutorResult<ScalarExpr<(usize, FieldId)>> {
        let first = exprs
            .next()
            .transpose()?
            .unwrap_or_else(|| ScalarExpr::Literal(Literal::Boolean(true)));
        exprs.try_fold(first, |acc, next| {
            let rhs = next?;
            Ok(ScalarExpr::Binary {
                left: Box::new(acc),
                op,
                right: Box::new(rhs),
            })
        })
    }

    match expr {
        LlkvExpr::And(list) => combine_with_op(list.iter().map(remap_filter_expr), llkv_expr::expr::BinaryOp::And),
        LlkvExpr::Or(list) => combine_with_op(list.iter().map(remap_filter_expr), llkv_expr::expr::BinaryOp::Or),
        LlkvExpr::Not(inner) => Ok(ScalarExpr::Not(Box::new(remap_filter_expr(inner)?))),
        LlkvExpr::Pred(filter) => predicate_to_scalar(filter),
        LlkvExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(remap_scalar_expr(left)),
            op: *op,
            right: Box::new(remap_scalar_expr(right)),
        }),
        LlkvExpr::InList {
            expr,
            list,
            negated,
        } => {
            let col = remap_scalar_expr(expr);
            let mut combined = ScalarExpr::Literal(Literal::Boolean(false));
            for lit in list {
                let eq = ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Eq,
                    right: Box::new(remap_scalar_expr(lit)),
                };
                combined = ScalarExpr::Binary {
                    left: Box::new(combined),
                    op: llkv_expr::expr::BinaryOp::Or,
                    right: Box::new(eq),
                };
            }
            if *negated {
                Ok(ScalarExpr::Not(Box::new(combined)))
            } else {
                Ok(combined)
            }
        }
        LlkvExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(remap_scalar_expr(expr)),
            negated: *negated,
        }),
        LlkvExpr::Literal(b) => Ok(ScalarExpr::Literal(Literal::Boolean(*b))),
        _ => Err(Error::InvalidArgumentError(
            "unsupported predicate in multi-table filter".into(),
        )),
    }
}

fn predicate_to_scalar(
    filter: &Filter<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
) -> ExecutorResult<ScalarExpr<(usize, FieldId)>> {
    let col = ScalarExpr::Column((
        filter.field_id.table_index,
        filter.field_id.logical_field_id.field_id(),
    ));

    let expr = match &filter.op {
        Operator::Equals(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::Eq,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::GreaterThan(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::Gt,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::GreaterThanOrEquals(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::GtEq,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::LessThan(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::Lt,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::LessThanOrEquals(lit) => ScalarExpr::Compare {
            left: Box::new(col),
            op: CompareOp::LtEq,
            right: Box::new(ScalarExpr::Literal(lit.clone())),
        },
        Operator::Range { lower, upper } => {
            let lower_expr = match lower {
                std::ops::Bound::Included(l) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::GtEq,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                }),
                std::ops::Bound::Excluded(l) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Gt,
                    right: Box::new(ScalarExpr::Literal(l.clone())),
                }),
                std::ops::Bound::Unbounded => None,
            };
            let upper_expr = match upper {
                std::ops::Bound::Included(u) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::LtEq,
                    right: Box::new(ScalarExpr::Literal(u.clone())),
                }),
                std::ops::Bound::Excluded(u) => Some(ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Lt,
                    right: Box::new(ScalarExpr::Literal(u.clone())),
                }),
                std::ops::Bound::Unbounded => None,
            };

            match (lower_expr, upper_expr) {
                (Some(l), Some(u)) => ScalarExpr::Binary {
                    left: Box::new(l),
                    op: llkv_expr::expr::BinaryOp::And,
                    right: Box::new(u),
                },
                (Some(l), None) => l,
                (None, Some(u)) => u,
                (None, None) => ScalarExpr::Literal(Literal::Boolean(true)),
            }
        }
        Operator::In(list) => {
            let mut combined = ScalarExpr::Literal(Literal::Boolean(false));
            for lit in *list {
                let eq = ScalarExpr::Compare {
                    left: Box::new(col.clone()),
                    op: CompareOp::Eq,
                    right: Box::new(ScalarExpr::Literal(lit.clone())),
                };
                combined = ScalarExpr::Binary {
                    left: Box::new(combined),
                    op: llkv_expr::expr::BinaryOp::Or,
                    right: Box::new(eq),
                };
            }
            combined
        }
        Operator::IsNull => ScalarExpr::IsNull {
            expr: Box::new(col),
            negated: false,
        },
        Operator::IsNotNull => ScalarExpr::IsNull {
            expr: Box::new(col),
            negated: true,
        },
        Operator::StartsWith { .. }
        | Operator::EndsWith { .. }
        | Operator::Contains { .. } => {
            return Err(Error::InvalidArgumentError(
                "string pattern predicates are not supported in multi-table execution".into(),
            ))
        }
    };

    Ok(expr)
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

fn extract_join_keys_and_filters(
    join: &ResolvedJoin,
) -> ExecutorResult<(
    Vec<JoinKey>,
    Vec<LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>>,
)> {
    let mut keys = Vec::new();
    let mut residuals = Vec::new();

    if let Some(on) = &join.on {
        collect_join_predicates(on, join.left_table_index, &mut keys, &mut residuals)?;
    }

    Ok((keys, residuals))
}

fn collect_join_predicates(
    expr: &LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>,
    left_table_index: usize,
    keys: &mut Vec<JoinKey>,
    residuals: &mut Vec<LlkvExpr<'static, llkv_plan::logical_planner::ResolvedFieldRef>>,
) -> ExecutorResult<()> {
    match expr {
        LlkvExpr::And(list) => {
            for e in list {
                collect_join_predicates(e, left_table_index, keys, residuals)?;
            }
            Ok(())
        }
        LlkvExpr::Compare { left, op, right } => {
            if matches!(op, CompareOp::Eq) {
                if let (ScalarExpr::Column(l), ScalarExpr::Column(r)) = (left, right) {
                    if (l.table_index == left_table_index
                        && r.table_index == left_table_index + 1)
                        || (r.table_index == left_table_index
                            && l.table_index == left_table_index + 1)
                    {
                        let (left_col, right_col) = if l.table_index == left_table_index {
                            (l, r)
                        } else {
                            (r, l)
                        };

                        keys.push(JoinKey {
                            left_field: left_col.logical_field_id.field_id(),
                            right_field: right_col.logical_field_id.field_id(),
                            null_equals_null: false,
                        });
                        return Ok(());
                    }
                }
            }

            residuals.push(expr.clone());
            Ok(())
        }
        _ => {
            residuals.push(expr.clone());
            Ok(())
        }
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
