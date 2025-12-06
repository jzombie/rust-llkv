use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, mpsc};

use arrow::compute::concat_batches;
use arrow::array::ArrayRef;
use arrow::datatypes::{Field as ArrowField, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{OwnedRow, RowConverter, SortField};
use llkv_aggregate::AggregateStream;
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
use llkv_plan::plans::{AggregateExpr, AggregateFunction};
use llkv_plan::schema::{PlanColumn, PlanSchema};
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

    fn execute_complex_aggregation(
        &self,
        base_stream: BatchIter,
        base_schema: Arc<Schema>,
        new_aggregates: Vec<AggregateExpr>,
        final_exprs: Vec<ScalarExpr<String>>,
        final_names: Vec<String>,
        plan: &SelectPlan,
        table_name: String,
    ) -> ExecutorResult<SelectExecution<P>> {
        if !plan.group_by.is_empty() {
            return Err(Error::InvalidArgumentError(
                "GROUP BY aggregates are not supported yet".into(),
            ));
        }

        // 1. Pre-aggregation stream is just base_stream
        // The base_stream is expected to yield batches containing the arguments for aggregation
        // (e.g. _agg_arg_0, _agg_arg_1, etc.) produced by the Scan.
        
        // We need to ensure the schema has the field_ids we expect (10000+i) so AggregateStream can match them.
        let mut fields_with_metadata = Vec::new();
        for (i, field) in base_schema.fields().iter().enumerate() {
            let mut metadata = field.metadata().clone();
            metadata.insert("field_id".to_string(), format!("{}", 10000 + i));
            let new_field = field.as_ref().clone().with_metadata(metadata);
            fields_with_metadata.push(new_field);
        }
        let pre_agg_schema = Arc::new(Schema::new(fields_with_metadata));
        
        let pre_agg_schema_captured = pre_agg_schema.clone();
        let pre_agg_stream = base_stream.map(move |batch_res| {
            let batch = batch_res?;
            // Just replace schema, data is same
            RecordBatch::try_new(pre_agg_schema_captured.clone(), batch.columns().to_vec())
                .map_err(|e| Error::Internal(e.to_string()))
        });

        // 2. Aggregation
        let mut agg_plan = plan.clone();
        agg_plan.aggregates = new_aggregates;
        
        let mut plan_columns = Vec::new();
        let mut name_to_index = FxHashMap::default();
        for (i, field) in pre_agg_schema.fields().iter().enumerate() {
            plan_columns.push(PlanColumn {
                name: field.name().clone(),
                data_type: field.data_type().clone(),
                field_id: 10000 + i as u32,
                is_nullable: field.is_nullable(),
                is_primary_key: false,
                is_unique: false,
                default_value: None,
                check_expr: None,
            });
            name_to_index.insert(field.name().clone(), i);
        }
        let pre_agg_plan_schema = PlanSchema { columns: plan_columns, name_to_index };

        let agg_iter = AggregateStream::new(Box::new(pre_agg_stream), &agg_plan, &pre_agg_plan_schema, &pre_agg_schema)?;
        let agg_schema = agg_iter.schema();

        // 3. Final Projection
        let mut final_output_fields = Vec::new();
        let mut col_mapping = FxHashMap::default();
        for (i, field) in agg_schema.fields().iter().enumerate() {
            col_mapping.insert((0, i as u32), i);
        }

        for (expr_str, name) in final_exprs.iter().zip(final_names.iter()) {
            let expr = resolve_scalar_expr_string(expr_str, &agg_schema)?;
            let dt = infer_type(&expr, &agg_schema, &col_mapping).unwrap_or(arrow::datatypes::DataType::Int64);
            final_output_fields.push(ArrowField::new(name, dt, true));
        }
        let final_output_schema = Arc::new(Schema::new(final_output_fields));
        let final_output_schema_captured = final_output_schema.clone();
        let agg_schema_captured = agg_schema.clone();

        let final_stream = agg_iter.map(move |batch_res| {
            let batch = batch_res?;
            let mut columns = Vec::new();
            
            // Prepare arrays for evaluation
            let mut field_arrays = FxHashMap::default();
            for (i, col) in batch.columns().iter().enumerate() {
                field_arrays.insert((0, i as u32), col.clone());
            }
            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
            
            for expr_str in &final_exprs {
                let expr = resolve_scalar_expr_string(expr_str, &agg_schema_captured)?;
                let result = ScalarEvaluator::evaluate_batch_simplified(&expr, batch.num_rows(), &numeric_arrays)?;
                columns.push(result);
            }
            
            RecordBatch::try_new(final_output_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
        });

        let mut iter: BatchIter = Box::new(final_stream);
        if plan.distinct {
             iter = Box::new(DistinctStream::new(final_output_schema.clone(), iter)?);
        }
        
        let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);
        Ok(SelectExecution::from_stream(table_name, final_output_schema, trimmed))
    }

    pub fn execute_select(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        self.execute_select_with_filter(plan, None)
    }

    pub fn execute_select_with_filter(
        &self,
        plan: SelectPlan,
        row_filter: Option<Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        // Check for complex aggregates in single-table queries BEFORE creating logical plan
        if plan.tables.len() == 1 {
            let table_ref = &plan.tables[0];
            let table_name = table_ref.qualified_name();
            
            // We need the schema to expand wildcards and check for aggregates
            if let Ok(schema) = self.logical_planner.get_table_schema(&table_name) {
                let mut string_projections = Vec::new();
                let mut final_names = Vec::new();
                
                for proj in &plan.projections {
                     match proj {
                        llkv_plan::plans::SelectProjection::AllColumns => {
                             for col in schema.columns.iter() {
                                 string_projections.push(ScalarExpr::Column(col.name.clone()));
                                 final_names.push(col.name.clone());
                             }
                        }
                        llkv_plan::plans::SelectProjection::AllColumnsExcept { exclude } => {
                             for col in schema.columns.iter() {
                                 if !exclude.contains(&col.name) {
                                     string_projections.push(ScalarExpr::Column(col.name.clone()));
                                     final_names.push(col.name.clone());
                                 }
                             }
                        }
                        llkv_plan::plans::SelectProjection::Column { name, alias } => {
                            string_projections.push(ScalarExpr::Column(name.clone()));
                            final_names.push(alias.clone().unwrap_or(name.clone()));
                        }
                        llkv_plan::plans::SelectProjection::Computed { expr, alias } => {
                            string_projections.push(expr.clone());
                            final_names.push(alias.clone());
                        }
                    }
                }

                let (new_aggregates, final_exprs, pre_agg_exprs) = extract_complex_aggregates(&string_projections);
                
                if !new_aggregates.is_empty() {
                     // Create pre-agg plan that projects the arguments for aggregation
                     let mut pre_agg_plan = plan.clone();
                     pre_agg_plan.projections = pre_agg_exprs.iter().enumerate().map(|(i, expr)| {
                         llkv_plan::plans::SelectProjection::Computed {
                             expr: expr.clone(),
                             alias: format!("_agg_arg_{}", i),
                         }
                     }).collect();
                     
                     // Create logical plan for pre-agg
                     // This should succeed because pre_agg_exprs do not contain aggregates
                     let logical_plan = self.logical_planner.create_logical_plan(&pre_agg_plan)?;
                     
                     match logical_plan {
                        LogicalPlan::Single(single) => {
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
                            
                            return self.execute_complex_aggregation(
                                base_iter,
                                schema,
                                new_aggregates,
                                final_exprs,
                                final_names,
                                &plan,
                                table_name,
                            );
                        }
                        _ => {
                            // Should not happen for single table plan
                            return Err(Error::Internal("Expected single table plan for pre-aggregation".into()));
                        }
                     }
                }
            }
        }

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
                    let agg_schema = agg_iter.schema();
                    let trimmed =
                        apply_offset_limit_stream(Box::new(agg_iter), plan.offset, plan.limit);
                    return Ok(SelectExecution::from_stream(table_name, agg_schema, trimmed));
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
                if table_count == 0 {
                     return Err(Error::Internal("Multi-table plan with no tables".into()));
                }
                
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
                
                // 3. Apply Filter (if any)
                if let Some(filter) = &multi.filter {
                    let predicate = simplify_predicate(&remap_filter_expr(filter)?);
                    
                    let col_mapping_captured = col_mapping.clone();
                    current_stream = Box::new(current_stream.map(move |batch_res| {
                        let batch = batch_res?;
                        
                        let mut required_fields = FxHashSet::default();
                        ScalarEvaluator::collect_fields(&predicate, &mut required_fields);
                        
                        let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
                        for (tbl, fid) in required_fields {
                            if let Some(idx) = col_mapping_captured.get(&(tbl, fid)) {
                                field_arrays.insert((tbl, fid), batch.column(*idx).clone());
                            }
                        }
                        
                        let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
                        let result = ScalarEvaluator::evaluate_batch_simplified(&predicate, batch.num_rows(), &numeric_arrays)?;
                        
                        let bool_array = result.as_any().downcast_ref::<arrow::array::BooleanArray>()
                            .ok_or_else(|| Error::Internal("Filter predicate must evaluate to boolean".into()))?;
                            
                        arrow::compute::filter_record_batch(&batch, bool_array).map_err(|e| Error::Internal(e.to_string()))
                    }));
                }
                
                // 4. Final Projection
                let mut plan_columns = Vec::new();
                let mut name_to_index = FxHashMap::default();

                let output_fields: Vec<ArrowField> = multi.projections.iter().enumerate().map(|(i, proj)| {
                    match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, alias } => {
                            let idx = col_mapping.get(&(*table_index, logical_field_id.field_id())).unwrap();
                            let field = current_schema.field(*idx);
                            
                            let name = alias.clone().unwrap_or_else(|| field.name().clone());
                            let mut metadata = HashMap::new();
                            metadata.insert("field_id".to_string(), logical_field_id.field_id().to_string());
                            
                            plan_columns.push(PlanColumn {
                                name: name.clone(),
                                data_type: field.data_type().clone(),
                                field_id: logical_field_id.field_id(),
                                is_nullable: field.is_nullable(),
                                is_primary_key: false,
                                is_unique: false,
                                default_value: None,
                                check_expr: None,
                            });
                            name_to_index.insert(name.to_ascii_lowercase(), i);

                            ArrowField::new(name, field.data_type().clone(), field.is_nullable())
                                .with_metadata(metadata)
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { alias, expr } => {
                            let name = alias.clone();
                            let dummy_fid = 999999 + i as u32;
                            
                            let remapped = remap_scalar_expr(expr);
                            let inferred_type = infer_type(&remapped, &current_schema, &col_mapping).unwrap_or(arrow::datatypes::DataType::Int64);
                            
                            let mut metadata = HashMap::new();
                            metadata.insert("field_id".to_string(), dummy_fid.to_string());

                            plan_columns.push(PlanColumn {
                                name: name.clone(),
                                data_type: inferred_type.clone(),
                                field_id: dummy_fid,
                                is_nullable: true,
                                is_primary_key: false,
                                is_unique: false,
                                default_value: None,
                                check_expr: None,
                            });
                            name_to_index.insert(name.to_ascii_lowercase(), i);

                            ArrowField::new(name, inferred_type, true)
                                .with_metadata(metadata)
                        }
                    }
                }).collect();
                let output_schema = Arc::new(Schema::new(output_fields));
                let logical_schema = PlanSchema { columns: plan_columns, name_to_index };
                
                // Check for aggregates
                let mut string_projections = Vec::new();
                for proj in &multi.projections {
                    let expr = match proj {
                        llkv_plan::logical_planner::ResolvedProjection::Column { table_index, logical_field_id, .. } => {
                            ScalarExpr::Column((*table_index, logical_field_id.field_id()))
                        }
                        llkv_plan::logical_planner::ResolvedProjection::Computed { expr, .. } => convert_resolved_expr(expr),
                    };
                    string_projections.push(map_expr_to_names(&expr, &col_mapping, &current_schema)?);
                }

                let (new_aggregates, final_exprs, pre_agg_exprs) = extract_complex_aggregates(&string_projections);
                let has_aggregates = !new_aggregates.is_empty();

                let mut iter: BatchIter = if has_aggregates {
                    if !plan.group_by.is_empty() {
                        return Err(Error::InvalidArgumentError(
                            "GROUP BY aggregates are not supported yet".into(),
                        ));
                    }

                    // 1. Pre-aggregation stream
                    let pre_agg_schema_fields: Vec<ArrowField> = pre_agg_exprs.iter().enumerate().map(|(i, _)| {
                        ArrowField::new(format!("_agg_arg_{}", i), arrow::datatypes::DataType::Int64, true)
                            .with_metadata(HashMap::from([("field_id".to_string(), format!("{}", 10000+i))]))
                    }).collect();
                    let pre_agg_schema = Arc::new(Schema::new(pre_agg_schema_fields));
                    
                    let pre_agg_schema_captured = pre_agg_schema.clone();
                    let current_schema_captured = current_schema.clone();
                    
                    let pre_agg_stream = current_stream.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();
                        
                        for expr_str in &pre_agg_exprs {
                            let expr = resolve_scalar_expr_string(expr_str, &current_schema_captured)?;
                            
                            let mut required_fields = FxHashSet::default();
                            ScalarEvaluator::collect_fields(&expr, &mut required_fields);
                            
                            let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
                            for (tbl, fid) in required_fields {
                                field_arrays.insert((tbl, fid), batch.column(fid as usize).clone());
                            }
                            
                            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
                            let result = ScalarEvaluator::evaluate_batch_simplified(&expr, batch.num_rows(), &numeric_arrays)?;
                            columns.push(result);
                        }
                        
                        if columns.is_empty() {
                            let options = arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                            RecordBatch::try_new_with_options(pre_agg_schema_captured.clone(), columns, &options).map_err(|e| Error::Internal(e.to_string()))
                        } else {
                            RecordBatch::try_new(pre_agg_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
                        }
                    });

                    // 2. Aggregate Stream
                    let mut dummy_plan = plan.clone();
                    dummy_plan.aggregates = new_aggregates;
                    
                    let mut dummy_cols = Vec::new();
                    let mut dummy_name_to_idx = FxHashMap::default();
                    for (i, field) in pre_agg_schema.fields().iter().enumerate() {
                        let name = field.name().clone();
                        let fid = field.metadata().get("field_id").unwrap().parse::<u32>().unwrap();
                        dummy_cols.push(PlanColumn {
                            name: name.clone(),
                            data_type: field.data_type().clone(),
                            field_id: fid,
                            is_nullable: true,
                            is_primary_key: false,
                            is_unique: false,
                            default_value: None,
                            check_expr: None,
                        });
                        dummy_name_to_idx.insert(name, i);
                    }
                    let dummy_logical_schema = PlanSchema { columns: dummy_cols, name_to_index: dummy_name_to_idx };

                    let agg_iter = AggregateStream::new(Box::new(pre_agg_stream), &dummy_plan, &dummy_logical_schema, &pre_agg_schema)?;
                    let agg_schema = agg_iter.schema();

                    // 3. Final Projection Stream
                    let output_schema_captured = output_schema.clone();
                    let final_stream = agg_iter.map(move |batch_res| {
                        let batch = batch_res?;
                        let mut columns = Vec::new();
                        
                        for expr_str in &final_exprs {
                            let expr = resolve_scalar_expr_string(expr_str, &agg_schema)?;
                            
                            let mut required_fields = FxHashSet::default();
                            ScalarEvaluator::collect_fields(&expr, &mut required_fields);
                            
                            let mut field_arrays: FxHashMap<(usize, FieldId), ArrayRef> = FxHashMap::default();
                            for (tbl, fid) in required_fields {
                                field_arrays.insert((tbl, fid), batch.column(fid as usize).clone());
                            }
                            
                            let numeric_arrays = ScalarEvaluator::prepare_numeric_arrays(&field_arrays, batch.num_rows());
                            let result = ScalarEvaluator::evaluate_batch_simplified(&expr, batch.num_rows(), &numeric_arrays)?;
                            columns.push(result);
                        }
                        RecordBatch::try_new(output_schema_captured.clone(), columns).map_err(|e| Error::Internal(e.to_string()))
                    });
                    
                    Box::new(final_stream)

                } else {
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
                    Box::new(final_stream)
                };

                if plan.distinct {
                    iter = Box::new(DistinctStream::new(output_schema.clone(), iter)?);
                }

                let trimmed = apply_offset_limit_stream(iter, plan.offset, plan.limit);
                let table_name = multi.table_order.first().map(|t| t.qualified_name()).unwrap_or_default();
                Ok(SelectExecution::from_stream(table_name, output_schema, trimmed))
        }
    }
}
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

fn simplify_predicate(expr: &ScalarExpr<(usize, FieldId)>) -> ScalarExpr<(usize, FieldId)> {
    match expr {
        ScalarExpr::IsNull { expr: inner, negated } => {
            let simplified_inner = simplify_predicate(inner);
            if let ScalarExpr::Literal(Literal::Null) = simplified_inner {
                ScalarExpr::Literal(Literal::Boolean(!*negated))
            } else {
                ScalarExpr::IsNull {
                    expr: Box::new(simplified_inner),
                    negated: *negated,
                }
            }
        }
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(simplify_predicate(left)),
            op: *op,
            right: Box::new(simplify_predicate(right)),
        },
        ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(simplify_predicate(e))),
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(simplify_predicate(left)),
            op: *op,
            right: Box::new(simplify_predicate(right)),
        },
        _ => expr.clone(),
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

struct AggVisitor {
    aggregates: Vec<AggregateExpr>,
    pre_agg_projections: Vec<ScalarExpr<String>>,
}

impl AggVisitor {
    fn new() -> Self {
        Self {
            aggregates: Vec::new(),
            pre_agg_projections: Vec::new(),
        }
    }

    fn visit(&mut self, expr: &ScalarExpr<String>) -> ScalarExpr<String> {
        match expr {
            ScalarExpr::Column(c) => ScalarExpr::Column(c.clone()),
            ScalarExpr::Literal(l) => ScalarExpr::Literal(l.clone()),
            ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
                left: Box::new(self.visit(left)),
                op: *op,
                right: Box::new(self.visit(right)),
            },
            ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(self.visit(e))),
            ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
                expr: Box::new(self.visit(expr)),
                negated: *negated,
            },
            ScalarExpr::Aggregate(call) => {
                let (arg_expr, distinct, func) = match call {
                    AggregateCall::CountStar => {
                         let alias = format!("_agg_res_{}", self.aggregates.len());
                         self.aggregates.push(AggregateExpr::CountStar {
                             alias: alias.clone(),
                             distinct: false,
                         });
                         return ScalarExpr::Column(alias);
                    }
                    AggregateCall::Count { expr, distinct } => (expr, *distinct, AggregateFunction::Count),
                    AggregateCall::Sum { expr, distinct } => (expr, *distinct, AggregateFunction::SumInt64),
                    AggregateCall::Total { expr, distinct } => (expr, *distinct, AggregateFunction::TotalInt64),
                    AggregateCall::Min(expr) => (expr, false, AggregateFunction::MinInt64),
                    AggregateCall::Max(expr) => (expr, false, AggregateFunction::MaxInt64),
                    AggregateCall::CountNulls(expr) => (expr, false, AggregateFunction::CountNulls),
                    AggregateCall::GroupConcat { expr, distinct, separator: _ } => (expr, *distinct, AggregateFunction::GroupConcat),
                    AggregateCall::Avg { expr, distinct } => {
                        let arg_idx = self.pre_agg_projections.len();
                        self.pre_agg_projections.push(*expr.clone());
                        let arg_col_name = format!("_agg_arg_{}", arg_idx);
                        
                        let sum_alias = format!("_agg_res_{}", self.aggregates.len());
                        self.aggregates.push(AggregateExpr::Column {
                            column: arg_col_name.clone(),
                            alias: sum_alias.clone(),
                            function: AggregateFunction::SumInt64,
                            distinct: *distinct,
                        });
                        
                        let count_alias = format!("_agg_res_{}", self.aggregates.len());
                        self.aggregates.push(AggregateExpr::Column {
                            column: arg_col_name,
                            alias: count_alias.clone(),
                            function: AggregateFunction::Count,
                            distinct: *distinct,
                        });
                        
                        return ScalarExpr::Binary {
                            left: Box::new(ScalarExpr::Column(sum_alias)),
                            op: llkv_expr::expr::BinaryOp::Divide,
                            right: Box::new(ScalarExpr::Column(count_alias)),
                        };
                    }
                };

                let arg_idx = self.pre_agg_projections.len();
                self.pre_agg_projections.push(*arg_expr.clone());
                let arg_col_name = format!("_agg_arg_{}", arg_idx);
                
                let alias = format!("_agg_res_{}", self.aggregates.len());
                self.aggregates.push(AggregateExpr::Column {
                    column: arg_col_name,
                    alias: alias.clone(),
                    function: func,
                    distinct,
                });
                
                ScalarExpr::Column(alias)
            }
            ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
                base: Box::new(self.visit(base)),
                field_name: field_name.clone(),
            },
            ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
                expr: Box::new(self.visit(expr)),
                data_type: data_type.clone(),
            },
            ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
                left: Box::new(self.visit(left)),
                op: *op,
                right: Box::new(self.visit(right)),
            },
            ScalarExpr::Coalesce(exprs) => ScalarExpr::Coalesce(
                exprs.iter().map(|e| self.visit(e)).collect()
            ),
            ScalarExpr::ScalarSubquery(s) => ScalarExpr::ScalarSubquery(s.clone()),
            ScalarExpr::Case { operand, branches, else_expr } => ScalarExpr::Case {
                operand: operand.as_ref().map(|e| Box::new(self.visit(e))),
                branches: branches.iter().map(|(w, t)| (self.visit(w), self.visit(t))).collect(),
                else_expr: else_expr.as_ref().map(|e| Box::new(self.visit(e))),
            },
            ScalarExpr::Random => ScalarExpr::Random,
        }
    }
}

fn extract_complex_aggregates(
    projections: &[ScalarExpr<String>],
) -> (Vec<AggregateExpr>, Vec<ScalarExpr<String>>, Vec<ScalarExpr<String>>) {
    let mut visitor = AggVisitor::new();
    let rewritten = projections.iter().map(|p| visitor.visit(p)).collect();
    (visitor.aggregates, rewritten, visitor.pre_agg_projections)
}

fn resolve_scalar_expr_string(
    expr: &ScalarExpr<String>,
    schema: &Schema,
) -> Result<ScalarExpr<(usize, FieldId)>, Error> {
    match expr {
        ScalarExpr::Column(name) => {
            let (idx, field) = schema.column_with_name(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("Column not found: {}", name))
            })?;
            // FieldId is usually not available in Arrow Schema directly as metadata unless we put it there.
            // But ScalarEvaluator uses (table_idx, field_idx) where field_idx is the index in the array list.
            // So we can just use  as FieldId.
            Ok(ScalarExpr::Column((0, idx as u32)))
        }
        ScalarExpr::Literal(l) => Ok(ScalarExpr::Literal(l.clone())),
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(resolve_scalar_expr_string(left, schema)?),
            op: *op,
            right: Box::new(resolve_scalar_expr_string(right, schema)?),
        }),
        ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(resolve_scalar_expr_string(e, schema)?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(resolve_scalar_expr_string(expr, schema)?),
            negated: *negated,
        }),
        ScalarExpr::Aggregate(_) => Err(Error::Internal("Nested aggregates not supported in resolution".into())),
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(resolve_scalar_expr_string(base, schema)?),
            field_name: field_name.clone(),
        }),
        ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
            expr: Box::new(resolve_scalar_expr_string(expr, schema)?),
            data_type: data_type.clone(),
        }),
        ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(resolve_scalar_expr_string(left, schema)?),
            op: *op,
            right: Box::new(resolve_scalar_expr_string(right, schema)?),
        }),
        ScalarExpr::Coalesce(exprs) => {
            let mut resolved = Vec::new();
            for e in exprs {
                resolved.push(resolve_scalar_expr_string(e, schema)?);
            }
            Ok(ScalarExpr::Coalesce(resolved))
        }
        ScalarExpr::ScalarSubquery(s) => Ok(ScalarExpr::ScalarSubquery(s.clone())),
        ScalarExpr::Case { operand, branches, else_expr } => {
            let op = if let Some(o) = operand {
                Some(Box::new(resolve_scalar_expr_string(o, schema)?))
            } else {
                None
            };
            let mut br = Vec::new();
            for (w, t) in branches {
                br.push((resolve_scalar_expr_string(w, schema)?, resolve_scalar_expr_string(t, schema)?));
            }
            let el = if let Some(e) = else_expr {
                Some(Box::new(resolve_scalar_expr_string(e, schema)?))
            } else {
                None
            };
            Ok(ScalarExpr::Case { operand: op, branches: br, else_expr: el })
        }
        ScalarExpr::Random => Ok(ScalarExpr::Random),
    }
}


use arrow::datatypes::DataType;

fn infer_type(
    expr: &ScalarExpr<(usize, FieldId)>, 
    schema: &Schema,
    col_mapping: &FxHashMap<(usize, FieldId), usize>
) -> Result<DataType, Error> {
    let res = match expr {
        ScalarExpr::Column(id) => {
            let idx = col_mapping.get(id).ok_or_else(|| Error::Internal("Column not found in mapping".into()))?;
            Ok(schema.field(*idx).data_type().clone())
        },
        ScalarExpr::Literal(l) => {
             match l {
                 Literal::Int128(_) => Ok(DataType::Int64),
                 Literal::Float64(_) => Ok(DataType::Float64),
                 Literal::Boolean(_) => Ok(DataType::Boolean),
                 Literal::String(_) => Ok(DataType::Utf8),
                 Literal::Null => Ok(DataType::Null),
                 Literal::Decimal128(d) => Ok(DataType::Decimal128(d.precision(), d.scale())),
                 _ => Ok(DataType::Int64),
             }
        },
        ScalarExpr::Binary { left, op, right } => {
            let l = infer_type(left, schema, col_mapping)?;
            let r = infer_type(right, schema, col_mapping)?;
            if matches!(l, DataType::Float64 | DataType::Float32) || matches!(r, DataType::Float64 | DataType::Float32) {
                Ok(DataType::Float64)
            } else if matches!(l, DataType::Decimal128(..)) || matches!(r, DataType::Decimal128(..)) {
                 // Simplified decimal inference
                 Ok(DataType::Decimal128(38, 10)) 
            } else {
                Ok(DataType::Int64)
            }
        },
        ScalarExpr::Cast { data_type, .. } => Ok(data_type.clone()),
        ScalarExpr::Case { else_expr, branches, .. } => {
             if let Some((_, t)) = branches.first() {
                 infer_type(t, schema, col_mapping)
             } else if let Some(e) = else_expr {
                 infer_type(e, schema, col_mapping)
             } else {
                 Ok(DataType::Null)
             }
        },
        ScalarExpr::Coalesce(items) => {
            if let Some(first) = items.first() {
                infer_type(first, schema, col_mapping)
            } else {
                Ok(DataType::Null)
            }
        },
        ScalarExpr::Not(_) | ScalarExpr::IsNull { .. } | ScalarExpr::Compare { .. } => Ok(DataType::Boolean),
        _ => Ok(DataType::Int64),
    };
    res
}

fn map_expr_to_names(
    expr: &ScalarExpr<(usize, FieldId)>,
    col_mapping: &FxHashMap<(usize, FieldId), usize>,
    schema: &Schema,
) -> Result<ScalarExpr<String>, Error> {
    match expr {
        ScalarExpr::Column(id) => {
            let idx = col_mapping.get(id).ok_or_else(|| {
                Error::Internal(format!("Column {:?} not found in mapping", id))
            })?;
            let field = schema.field(*idx);
            Ok(ScalarExpr::Column(field.name().clone()))
        }
        ScalarExpr::Literal(l) => Ok(ScalarExpr::Literal(l.clone())),
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(map_expr_to_names(left, col_mapping, schema)?),
            op: *op,
            right: Box::new(map_expr_to_names(right, col_mapping, schema)?),
        }),
        ScalarExpr::Not(e) => Ok(ScalarExpr::Not(Box::new(map_expr_to_names(e, col_mapping, schema)?))),
        ScalarExpr::IsNull { expr, negated } => Ok(ScalarExpr::IsNull {
            expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
            negated: *negated,
        }),
        ScalarExpr::Aggregate(call) => {
             // Recurse into aggregate arguments
             let new_call = match call {
                 AggregateCall::CountStar => AggregateCall::CountStar,
                 AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
                 AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
                 AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
                 AggregateCall::Min(expr) => AggregateCall::Min(Box::new(map_expr_to_names(expr, col_mapping, schema)?)),
                 AggregateCall::Max(expr) => AggregateCall::Max(Box::new(map_expr_to_names(expr, col_mapping, schema)?)),
                 AggregateCall::CountNulls(expr) => AggregateCall::CountNulls(Box::new(map_expr_to_names(expr, col_mapping, schema)?)),
                 AggregateCall::GroupConcat { expr, distinct, separator } => AggregateCall::GroupConcat {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                     separator: separator.clone(),
                 },
                 AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                     expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
                     distinct: *distinct,
                 },
             };
             Ok(ScalarExpr::Aggregate(new_call))
        },
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(map_expr_to_names(base, col_mapping, schema)?),
            field_name: field_name.clone(),
        }),
        ScalarExpr::Cast { expr, data_type } => Ok(ScalarExpr::Cast {
            expr: Box::new(map_expr_to_names(expr, col_mapping, schema)?),
            data_type: data_type.clone(),
        }),
        ScalarExpr::Compare { left, op, right } => Ok(ScalarExpr::Compare {
            left: Box::new(map_expr_to_names(left, col_mapping, schema)?),
            op: *op,
            right: Box::new(map_expr_to_names(right, col_mapping, schema)?),
        }),
        ScalarExpr::Coalesce(exprs) => {
            let mut mapped = Vec::new();
            for e in exprs {
                mapped.push(map_expr_to_names(e, col_mapping, schema)?);
            }
            Ok(ScalarExpr::Coalesce(mapped))
        },
        ScalarExpr::ScalarSubquery(s) => Ok(ScalarExpr::ScalarSubquery(s.clone())),
        ScalarExpr::Case { operand, branches, else_expr } => {
            let op = if let Some(o) = operand {
                Some(Box::new(map_expr_to_names(o, col_mapping, schema)?))
            } else {
                None
            };
            let mut br = Vec::new();
            for (w, t) in branches {
                br.push((map_expr_to_names(w, col_mapping, schema)?, map_expr_to_names(t, col_mapping, schema)?));
            }
            let el = if let Some(e) = else_expr {
                Some(Box::new(map_expr_to_names(e, col_mapping, schema)?))
            } else {
                None
            };
            Ok(ScalarExpr::Case { operand: op, branches: br, else_expr: el })
        },
        ScalarExpr::Random => Ok(ScalarExpr::Random),
    }
}

use llkv_plan::logical_planner::ResolvedFieldRef;

fn convert_resolved_expr(expr: &ScalarExpr<ResolvedFieldRef>) -> ScalarExpr<(usize, FieldId)> {
    match expr {
        ScalarExpr::Column(r) => ScalarExpr::Column((r.table_index, r.logical_field_id.field_id())),
        ScalarExpr::Literal(l) => ScalarExpr::Literal(l.clone()),
        ScalarExpr::Binary { left, op, right } => ScalarExpr::Binary {
            left: Box::new(convert_resolved_expr(left)),
            op: *op,
            right: Box::new(convert_resolved_expr(right)),
        },
        ScalarExpr::Not(e) => ScalarExpr::Not(Box::new(convert_resolved_expr(e))),
        ScalarExpr::IsNull { expr, negated } => ScalarExpr::IsNull {
            expr: Box::new(convert_resolved_expr(expr)),
            negated: *negated,
        },
        ScalarExpr::Aggregate(call) => {
             let new_call = match call {
                 AggregateCall::CountStar => AggregateCall::CountStar,
                 AggregateCall::Count { expr, distinct } => AggregateCall::Count {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
                 AggregateCall::Sum { expr, distinct } => AggregateCall::Sum {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
                 AggregateCall::Total { expr, distinct } => AggregateCall::Total {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
                 AggregateCall::Min(expr) => AggregateCall::Min(Box::new(convert_resolved_expr(expr))),
                 AggregateCall::Max(expr) => AggregateCall::Max(Box::new(convert_resolved_expr(expr))),
                 AggregateCall::CountNulls(expr) => AggregateCall::CountNulls(Box::new(convert_resolved_expr(expr))),
                 AggregateCall::GroupConcat { expr, distinct, separator } => AggregateCall::GroupConcat {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                     separator: separator.clone(),
                 },
                 AggregateCall::Avg { expr, distinct } => AggregateCall::Avg {
                     expr: Box::new(convert_resolved_expr(expr)),
                     distinct: *distinct,
                 },
             };
             ScalarExpr::Aggregate(new_call)
        },
        ScalarExpr::GetField { base, field_name } => ScalarExpr::GetField {
            base: Box::new(convert_resolved_expr(base)),
            field_name: field_name.clone(),
        },
        ScalarExpr::Cast { expr, data_type } => ScalarExpr::Cast {
            expr: Box::new(convert_resolved_expr(expr)),
            data_type: data_type.clone(),
        },
        ScalarExpr::Compare { left, op, right } => ScalarExpr::Compare {
            left: Box::new(convert_resolved_expr(left)),
            op: *op,
            right: Box::new(convert_resolved_expr(right)),
        },
        ScalarExpr::Coalesce(exprs) => ScalarExpr::Coalesce(
            exprs.iter().map(|e| convert_resolved_expr(e)).collect()
        ),
        ScalarExpr::ScalarSubquery(s) => ScalarExpr::ScalarSubquery(s.clone()),
        ScalarExpr::Case { operand, branches, else_expr } => ScalarExpr::Case {
            operand: operand.as_ref().map(|e| Box::new(convert_resolved_expr(e))),
            branches: branches.iter().map(|(w, t)| (convert_resolved_expr(w), convert_resolved_expr(t))).collect(),
            else_expr: else_expr.as_ref().map(|e| Box::new(convert_resolved_expr(e))),
        },
        ScalarExpr::Random => ScalarExpr::Random,
    }
}
