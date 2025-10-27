//! Query execution engine for LLKV.
//!
//! This crate provides the query execution layer that sits between the query planner
//! (`llkv-plan`) and the storage layer (`llkv-table`, `llkv-column-map`).
//!
//! # Module Organization
//!
//! - [`translation`]: Expression and projection translation utilities
//! - [`types`]: Core type definitions (tables, schemas, columns)  
//! - [`insert`]: INSERT operation support (value coercion)
//! - [`utils`]: Utility functions (time)
//!
//! The [`QueryExecutor`] and [`SelectExecution`] implementations are defined inline
//! in this module for now, but should be extracted to a dedicated `query` module
//! in a future refactoring.

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Array, Int64Array, Int64Builder,
    RecordBatch, StringArray, UInt32Array, new_null_array,
};
use arrow::compute::{
    SortColumn, SortOptions, cast, concat_batches, filter_record_batch, lexsort_to_indices, take,
};
use arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};
use llkv_column_map::store::Projection as StoreProjection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{AggregateCall, CompareOp, Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_expr::typed_predicate::{
    build_bool_predicate, build_fixed_width_predicate, build_var_width_predicate,
};
use llkv_join::cross_join_pair;
use llkv_plan::{
    AggregateExpr, AggregateFunction, CanonicalRow, OrderByPlan, OrderSortType, OrderTarget,
    PlanValue, SelectPlan, SelectProjection,
};
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{
    RowIdFilter, ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection,
    ScanStreamOptions,
};
use llkv_table::types::FieldId;
use llkv_table::{NumericArray, NumericArrayMap, NumericKernels, ROW_ID_FIELD_ID};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::Ordering;

// ============================================================================
// Module Declarations
// ============================================================================

pub mod insert;
pub mod translation;
pub mod types;
pub mod utils;

// ============================================================================
// Type Aliases and Re-exports
// ============================================================================

/// Result type for executor operations.
pub type ExecutorResult<T> = Result<T, Error>;

pub use insert::{
    build_array_for_column, normalize_insert_value_for_column, resolve_insert_columns,
};
pub use translation::{
    build_projected_columns, build_wildcard_projections, full_table_scan_filter,
    resolve_field_id_from_schema, schema_for_projections, translate_predicate,
    translate_predicate_with, translate_scalar, translate_scalar_with,
};
pub use types::{
    ExecutorColumn, ExecutorMultiColumnUnique, ExecutorRowBatch, ExecutorSchema, ExecutorTable,
    ExecutorTableProvider,
};
pub use utils::current_time_micros;

// ============================================================================
// Query Executor - Implementation
// ============================================================================
// TODO: Extract this implementation into a dedicated query/ module

/// Query executor that executes SELECT plans.
pub struct QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    provider: Arc<dyn ExecutorTableProvider<P>>,
}

impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn ExecutorTableProvider<P>>) -> Self {
        Self { provider }
    }

    pub fn execute_select(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        self.execute_select_with_filter(plan, None)
    }

    pub fn execute_select_with_filter(
        &self,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        // Handle SELECT without FROM clause (e.g., SELECT 42, SELECT {'a': 1})
        if plan.tables.is_empty() {
            return self.execute_select_without_table(plan);
        }

        // Handle multi-table queries (cross products/joins)
        if plan.tables.len() > 1 {
            return self.execute_cross_product(plan);
        }

        // Single table query
        let table_ref = &plan.tables[0];
        let table = self.provider.get_table(&table_ref.qualified_name())?;
        let display_name = table_ref.qualified_name();

        if !plan.aggregates.is_empty() {
            self.execute_aggregates(table, display_name, plan, row_filter)
        } else if self.has_computed_aggregates(&plan) {
            // Handle computed projections that contain embedded aggregates
            self.execute_computed_aggregates(table, display_name, plan, row_filter)
        } else {
            self.execute_projection(table, display_name, plan, row_filter)
        }
    }

    /// Check if any computed projections contain aggregate functions
    fn has_computed_aggregates(&self, plan: &SelectPlan) -> bool {
        plan.projections.iter().any(|proj| {
            if let SelectProjection::Computed { expr, .. } = proj {
                Self::expr_contains_aggregate(expr)
            } else {
                false
            }
        })
    }

    /// Recursively check if a scalar expression contains aggregates
    fn expr_contains_aggregate(expr: &ScalarExpr<String>) -> bool {
        match expr {
            ScalarExpr::Aggregate(_) => true,
            ScalarExpr::Binary { left, right, .. } => {
                Self::expr_contains_aggregate(left) || Self::expr_contains_aggregate(right)
            }
            ScalarExpr::Compare { left, right, .. } => {
                Self::expr_contains_aggregate(left) || Self::expr_contains_aggregate(right)
            }
            ScalarExpr::GetField { base, .. } => Self::expr_contains_aggregate(base),
            ScalarExpr::Cast { expr, .. } => Self::expr_contains_aggregate(expr),
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                operand
                    .as_deref()
                    .map(Self::expr_contains_aggregate)
                    .unwrap_or(false)
                    || branches.iter().any(|(when_expr, then_expr)| {
                        Self::expr_contains_aggregate(when_expr)
                            || Self::expr_contains_aggregate(then_expr)
                    })
                    || else_expr
                        .as_deref()
                        .map(Self::expr_contains_aggregate)
                        .unwrap_or(false)
            }
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) => false,
        }
    }

    /// Execute a SELECT without a FROM clause (e.g., SELECT 42, SELECT {'a': 1})
    fn execute_select_without_table(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        use arrow::array::ArrayRef;
        use arrow::datatypes::Field;

        // Build schema from computed projections
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for proj in &plan.projections {
            match proj {
                SelectProjection::Computed { expr, alias } => {
                    // Infer the data type from the expression
                    let (field_name, dtype, array) = match expr {
                        ScalarExpr::Literal(lit) => {
                            let (dtype, array) = Self::literal_to_array(lit)?;
                            (alias.clone(), dtype, array)
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(
                                "SELECT without FROM only supports literal expressions".into(),
                            ));
                        }
                    };

                    fields.push(Field::new(field_name, dtype, true));
                    arrays.push(array);
                }
                _ => {
                    return Err(Error::InvalidArgumentError(
                        "SELECT without FROM only supports computed projections".into(),
                    ));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
            .map_err(|e| Error::Internal(format!("failed to create record batch: {}", e)))?;

        if plan.distinct {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        let schema = batch.schema();

        Ok(SelectExecution::new_single_batch(
            String::new(), // No table name
            schema,
            batch,
        ))
    }

    /// Convert a Literal to an Arrow array (recursive for nested structs)
    fn literal_to_array(lit: &llkv_expr::literal::Literal) -> ExecutorResult<(DataType, ArrayRef)> {
        use arrow::array::{
            ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, StructArray,
            new_null_array,
        };
        use arrow::datatypes::{DataType, Field};
        use llkv_expr::literal::Literal;

        match lit {
            Literal::Integer(v) => {
                let val = i64::try_from(*v).unwrap_or(0);
                Ok((
                    DataType::Int64,
                    Arc::new(Int64Array::from(vec![val])) as ArrayRef,
                ))
            }
            Literal::Float(v) => Ok((
                DataType::Float64,
                Arc::new(Float64Array::from(vec![*v])) as ArrayRef,
            )),
            Literal::Boolean(v) => Ok((
                DataType::Boolean,
                Arc::new(BooleanArray::from(vec![*v])) as ArrayRef,
            )),
            Literal::String(v) => Ok((
                DataType::Utf8,
                Arc::new(StringArray::from(vec![v.clone()])) as ArrayRef,
            )),
            Literal::Null => Ok((DataType::Null, new_null_array(&DataType::Null, 1))),
            Literal::Struct(struct_fields) => {
                // Build a struct array recursively
                let mut inner_fields = Vec::new();
                let mut inner_arrays = Vec::new();

                for (field_name, field_lit) in struct_fields {
                    let (field_dtype, field_array) = Self::literal_to_array(field_lit)?;
                    inner_fields.push(Field::new(field_name.clone(), field_dtype, true));
                    inner_arrays.push(field_array);
                }

                let struct_array =
                    StructArray::try_new(inner_fields.clone().into(), inner_arrays, None).map_err(
                        |e| Error::Internal(format!("failed to create struct array: {}", e)),
                    )?;

                Ok((
                    DataType::Struct(inner_fields.into()),
                    Arc::new(struct_array) as ArrayRef,
                ))
            }
        }
    }

    /// Execute a cross product query (FROM table1, table2, ...)
    fn execute_cross_product(&self, plan: SelectPlan) -> ExecutorResult<SelectExecution<P>> {
        use arrow::compute::concat_batches;

        if plan.tables.len() < 2 {
            return Err(Error::InvalidArgumentError(
                "cross product requires at least 2 tables".into(),
            ));
        }

        // Acquire table handles and materialize the base batches for each table.
        let mut tables = Vec::with_capacity(plan.tables.len());
        for table_ref in &plan.tables {
            let qualified_name = table_ref.qualified_name();
            let table = self.provider.get_table(&qualified_name)?;
            tables.push((table_ref.clone(), table));
        }

        let mut staged: Vec<TableCrossProductData> = Vec::with_capacity(tables.len());
        for (table_ref, table) in &tables {
            staged.push(collect_table_data(table_ref, table.as_ref())?);
        }

        let mut staged_iter = staged.into_iter();
        let mut current = staged_iter
            .next()
            .ok_or_else(|| Error::Internal("cross product preparation yielded no tables".into()))?;

        for next in staged_iter {
            current = cross_join_table_batches(current, next)?;
        }

        let TableCrossProductData {
            schema: combined_schema,
            batches: mut combined_batches,
            column_counts,
        } = current;

        let column_lookup_map = build_cross_product_column_lookup(
            combined_schema.as_ref(),
            &plan.tables,
            &column_counts,
        );

        if let Some(filter_expr) = &plan.filter {
            let mut filter_context = CrossProductExpressionContext::new(
                combined_schema.as_ref(),
                column_lookup_map.clone(),
            )?;
            let translated_filter =
                translate_predicate(filter_expr.clone(), filter_context.schema(), |name| {
                    Error::InvalidArgumentError(format!(
                        "column '{}' not found in cross product result",
                        name
                    ))
                })?;

            let mut filtered_batches = Vec::with_capacity(combined_batches.len());
            for batch in combined_batches.into_iter() {
                filter_context.reset();
                let mask = filter_context.evaluate_predicate_mask(&translated_filter, &batch)?;
                let filtered = filter_record_batch(&batch, &mask).map_err(|err| {
                    Error::InvalidArgumentError(format!(
                        "failed to apply cross product filter: {err}"
                    ))
                })?;
                if filtered.num_rows() > 0 {
                    filtered_batches.push(filtered);
                }
            }
            combined_batches = filtered_batches;
        }

        let mut combined_batch = if combined_batches.is_empty() {
            RecordBatch::new_empty(Arc::clone(&combined_schema))
        } else if combined_batches.len() == 1 {
            combined_batches.pop().unwrap()
        } else {
            concat_batches(&combined_schema, &combined_batches).map_err(|e| {
                Error::Internal(format!(
                    "failed to concatenate cross product batches: {}",
                    e
                ))
            })?
        };

        // Apply SELECT projections if specified
        if !plan.projections.is_empty() {
            let mut selected_fields = Vec::new();
            let mut selected_columns = Vec::new();
            let mut expr_context: Option<CrossProductExpressionContext> = None;

            for proj in &plan.projections {
                match proj {
                    SelectProjection::AllColumns => {
                        // Keep all columns
                        selected_fields = combined_schema.fields().iter().cloned().collect();
                        selected_columns = combined_batch.columns().to_vec();
                        break;
                    }
                    SelectProjection::AllColumnsExcept { exclude } => {
                        // Keep all columns except the excluded ones
                        let exclude_lower: Vec<String> =
                            exclude.iter().map(|e| e.to_ascii_lowercase()).collect();

                        for (idx, field) in combined_schema.fields().iter().enumerate() {
                            let field_name_lower = field.name().to_ascii_lowercase();
                            if !exclude_lower.contains(&field_name_lower) {
                                selected_fields.push(field.clone());
                                selected_columns.push(combined_batch.column(idx).clone());
                            }
                        }
                        break;
                    }
                    SelectProjection::Column { name, alias } => {
                        // Find the column by qualified name
                        let col_name = name.to_ascii_lowercase();
                        if let Some(&idx) = column_lookup_map.get(&col_name) {
                            let field = combined_schema.field(idx);
                            let output_name = alias.as_ref().unwrap_or(name).clone();
                            selected_fields.push(Arc::new(arrow::datatypes::Field::new(
                                output_name,
                                field.data_type().clone(),
                                field.is_nullable(),
                            )));
                            selected_columns.push(combined_batch.column(idx).clone());
                        } else {
                            return Err(Error::InvalidArgumentError(format!(
                                "column '{}' not found in cross product result",
                                name
                            )));
                        }
                    }
                    SelectProjection::Computed { expr, alias } => {
                        if expr_context.is_none() {
                            expr_context = Some(CrossProductExpressionContext::new(
                                combined_schema.as_ref(),
                                column_lookup_map.clone(),
                            )?);
                        }
                        let context = expr_context
                            .as_mut()
                            .expect("projection context must be initialized");
                        let evaluated = context.evaluate(expr, &combined_batch)?;
                        let field = Arc::new(arrow::datatypes::Field::new(
                            alias.clone(),
                            evaluated.data_type().clone(),
                            true,
                        ));
                        selected_fields.push(field);
                        selected_columns.push(evaluated);
                    }
                }
            }

            let projected_schema = Arc::new(Schema::new(selected_fields));
            combined_batch = RecordBatch::try_new(projected_schema, selected_columns)
                .map_err(|e| Error::Internal(format!("failed to apply projections: {}", e)))?;
        }

        if plan.distinct {
            let mut state = DistinctState::default();
            let source_schema = combined_batch.schema();
            combined_batch = match distinct_filter_batch(combined_batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(source_schema),
            };
        }

        let schema = combined_batch.schema();

        let display_name = tables
            .iter()
            .map(|(table_ref, _)| table_ref.qualified_name())
            .collect::<Vec<_>>()
            .join(",");

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            combined_batch,
        ))
    }

    fn execute_projection(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let table_ref = table.as_ref();
        let projections = if plan.projections.is_empty() {
            build_wildcard_projections(table_ref)
        } else {
            build_projected_columns(table_ref, &plan.projections)?
        };
        let schema = schema_for_projections(table_ref, &projections)?;

        let (filter_expr, full_table_scan) = match plan.filter {
            Some(expr) => (
                crate::translation::expression::translate_predicate(
                    expr,
                    table_ref.schema.as_ref(),
                    |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
                )?,
                false,
            ),
            None => {
                let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                (
                    crate::translation::expression::full_table_scan_filter(field_id),
                    true,
                )
            }
        };

        let expanded_order = expand_order_targets(&plan.order_by, &projections)?;

        let mut physical_order: Option<ScanOrderSpec> = None;

        if let Some(first) = expanded_order.first() {
            match &first.target {
                OrderTarget::Column(name) => {
                    if table_ref.schema.resolve(name).is_some() {
                        physical_order = Some(resolve_scan_order(table_ref, &projections, first)?);
                    }
                }
                OrderTarget::Index(position) => match projections.get(*position) {
                    Some(ScanProjection::Column(_)) => {
                        physical_order = Some(resolve_scan_order(table_ref, &projections, first)?);
                    }
                    Some(ScanProjection::Computed { .. }) => {}
                    None => {
                        return Err(Error::InvalidArgumentError(format!(
                            "ORDER BY position {} is out of range",
                            position + 1
                        )));
                    }
                },
                OrderTarget::All => {}
            }
        }

        let options = if let Some(order_spec) = physical_order {
            if row_filter.is_some() {
                tracing::debug!("Applying MVCC row filter with ORDER BY");
            }
            ScanStreamOptions {
                include_nulls: true,
                order: Some(order_spec),
                row_id_filter: row_filter.clone(),
            }
        } else {
            if row_filter.is_some() {
                tracing::debug!("Applying MVCC row filter");
            }
            ScanStreamOptions {
                include_nulls: true,
                order: None,
                row_id_filter: row_filter.clone(),
            }
        };

        Ok(SelectExecution::new_projection(
            display_name,
            schema,
            table,
            projections,
            filter_expr,
            options,
            full_table_scan,
            expanded_order,
            plan.distinct,
        ))
    }

    fn execute_aggregates(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        let table_ref = table.as_ref();
        let distinct = plan.distinct;
        let mut specs: Vec<AggregateSpec> = Vec::with_capacity(plan.aggregates.len());
        for aggregate in plan.aggregates {
            match aggregate {
                AggregateExpr::CountStar { alias } => {
                    specs.push(AggregateSpec {
                        alias,
                        kind: AggregateKind::CountStar,
                    });
                }
                AggregateExpr::Column {
                    column,
                    alias,
                    function,
                    distinct,
                } => {
                    let col = table_ref.schema.resolve(&column).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column '{}' in aggregate",
                            column
                        ))
                    })?;

                    let kind = match function {
                        AggregateFunction::Count => {
                            if distinct {
                                AggregateKind::CountDistinctField {
                                    field_id: col.field_id,
                                }
                            } else {
                                AggregateKind::CountField {
                                    field_id: col.field_id,
                                }
                            }
                        }
                        AggregateFunction::SumInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "SUM currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::SumInt64 {
                                field_id: col.field_id,
                            }
                        }
                        AggregateFunction::MinInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MIN currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::MinInt64 {
                                field_id: col.field_id,
                            }
                        }
                        AggregateFunction::MaxInt64 => {
                            if col.data_type != DataType::Int64 {
                                return Err(Error::InvalidArgumentError(
                                    "MAX currently supports only INTEGER columns".into(),
                                ));
                            }
                            AggregateKind::MaxInt64 {
                                field_id: col.field_id,
                            }
                        }
                        AggregateFunction::CountNulls => {
                            if distinct {
                                return Err(Error::InvalidArgumentError(
                                    "DISTINCT is not supported for COUNT_NULLS".into(),
                                ));
                            }
                            AggregateKind::CountNulls {
                                field_id: col.field_id,
                            }
                        }
                    };
                    specs.push(AggregateSpec { alias, kind });
                }
            }
        }

        if specs.is_empty() {
            return Err(Error::InvalidArgumentError(
                "aggregate query requires at least one aggregate expression".into(),
            ));
        }

        let had_filter = plan.filter.is_some();
        let filter_expr = match plan.filter {
            Some(expr) => crate::translation::expression::translate_predicate(
                expr,
                table.schema.as_ref(),
                |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
            )?,
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform aggregate scan".into(),
                    )
                })?;
                crate::translation::expression::full_table_scan_filter(field_id)
            }
        };

        // Build projections and track which projection index each spec uses
        let mut projections = Vec::new();
        let mut spec_to_projection: Vec<Option<usize>> = Vec::with_capacity(specs.len());

        for spec in &specs {
            if let Some(field_id) = spec.kind.field_id() {
                let proj_idx = projections.len();
                spec_to_projection.push(Some(proj_idx));
                projections.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), field_id),
                    table
                        .schema
                        .column_by_field_id(field_id)
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("col{field_id}")),
                )));
            } else {
                spec_to_projection.push(None);
            }
        }

        if projections.is_empty() {
            let field_id = table.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError(
                    "table has no columns; cannot perform aggregate scan".into(),
                )
            })?;
            projections.push(ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), field_id),
                table
                    .schema
                    .column_by_field_id(field_id)
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| format!("col{field_id}")),
            )));
        }

        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: row_filter.clone(),
        };

        let mut states: Vec<AggregateState> = Vec::with_capacity(specs.len());
        // MVCC Note: We cannot use the total_rows shortcut when MVCC visibility filtering
        // is enabled, because some rows may be invisible due to uncommitted or aborted transactions.
        // Always scan to apply proper visibility rules.
        let mut count_star_override: Option<i64> = None;
        if !had_filter && row_filter.is_none() {
            // Only use shortcut if no filter AND no MVCC row filtering
            let total_rows = table.total_rows.load(Ordering::SeqCst);
            tracing::debug!(
                "[AGGREGATE] Using COUNT(*) shortcut: total_rows={}",
                total_rows
            );
            if total_rows > i64::MAX as u64 {
                return Err(Error::InvalidArgumentError(
                    "COUNT(*) result exceeds supported range".into(),
                ));
            }
            count_star_override = Some(total_rows as i64);
        } else {
            tracing::debug!(
                "[AGGREGATE] NOT using COUNT(*) shortcut: had_filter={}, has_row_filter={}",
                had_filter,
                row_filter.is_some()
            );
        }

        for (idx, spec) in specs.iter().enumerate() {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new_with_projection_index(
                    spec,
                    spec_to_projection[idx],
                    count_star_override,
                )?,
                override_value: match spec.kind {
                    AggregateKind::CountStar => {
                        tracing::debug!(
                            "[AGGREGATE] CountStar override_value={:?}",
                            count_star_override
                        );
                        count_star_override
                    }
                    _ => None,
                },
            });
        }

        let mut error: Option<Error> = None;
        match table.table.scan_stream(
            projections,
            &filter_expr,
            ScanStreamOptions {
                row_id_filter: row_filter.clone(),
                ..options
            },
            |batch| {
                if error.is_some() {
                    return;
                }
                for state in &mut states {
                    if let Err(err) = state.update(&batch) {
                        error = Some(err);
                        return;
                    }
                }
            },
        ) {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => {
                // Treat missing storage keys as an empty result set. This occurs
                // for freshly created tables that have no persisted chunks yet.
            }
            Err(err) => return Err(err),
        }
        if let Some(err) = error {
            return Err(err);
        }

        let mut fields = Vec::with_capacity(states.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(states.len());
        for state in states {
            let (field, array) = state.finalize()?;
            fields.push(field);
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        if distinct {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        let schema = batch.schema();

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            batch,
        ))
    }

    /// Execute a query where computed projections contain embedded aggregates
    /// This extracts aggregates, computes them, then evaluates the scalar expressions
    fn execute_computed_aggregates(
        &self,
        table: Arc<ExecutorTable<P>>,
        display_name: String,
        plan: SelectPlan,
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<SelectExecution<P>> {
        use arrow::array::Int64Array;
        use llkv_expr::expr::AggregateCall;

        let table_ref = table.as_ref();
        let distinct = plan.distinct;

        // First, extract all unique aggregates from the projections
        let mut aggregate_specs: Vec<(String, AggregateCall<String>)> = Vec::new();
        for proj in &plan.projections {
            if let SelectProjection::Computed { expr, .. } = proj {
                Self::collect_aggregates(expr, &mut aggregate_specs);
            }
        }

        // Compute the aggregates using the existing aggregate execution infrastructure
        let computed_aggregates = self.compute_aggregate_values(
            table.clone(),
            &plan.filter,
            &aggregate_specs,
            row_filter.clone(),
        )?;

        // Now build the final projections by evaluating expressions with aggregates substituted
        let mut fields = Vec::with_capacity(plan.projections.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(plan.projections.len());

        for proj in &plan.projections {
            match proj {
                SelectProjection::AllColumns | SelectProjection::AllColumnsExcept { .. } => {
                    return Err(Error::InvalidArgumentError(
                        "Wildcard projections not supported with computed aggregates".into(),
                    ));
                }
                SelectProjection::Column { name, alias } => {
                    let col = table_ref.schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", name))
                    })?;
                    let field_name = alias.as_ref().unwrap_or(name);
                    fields.push(arrow::datatypes::Field::new(
                        field_name,
                        col.data_type.clone(),
                        col.nullable,
                    ));
                    // For regular columns in an aggregate query, we'd need to handle GROUP BY
                    // For now, return an error as this is not supported
                    return Err(Error::InvalidArgumentError(
                        "Regular columns not supported in aggregate queries without GROUP BY"
                            .into(),
                    ));
                }
                SelectProjection::Computed { expr, alias } => {
                    // Evaluate the expression with aggregates substituted
                    let value = Self::evaluate_expr_with_aggregates(expr, &computed_aggregates)?;

                    fields.push(arrow::datatypes::Field::new(alias, DataType::Int64, false));

                    let array = Arc::new(Int64Array::from(vec![value])) as ArrayRef;
                    arrays.push(array);
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        if distinct {
            let mut state = DistinctState::default();
            batch = match distinct_filter_batch(batch, &mut state)? {
                Some(filtered) => filtered,
                None => RecordBatch::new_empty(Arc::clone(&schema)),
            };
        }

        let schema = batch.schema();

        Ok(SelectExecution::new_single_batch(
            display_name,
            schema,
            batch,
        ))
    }

    /// Collect all aggregate calls from an expression
    fn collect_aggregates(
        expr: &ScalarExpr<String>,
        aggregates: &mut Vec<(String, llkv_expr::expr::AggregateCall<String>)>,
    ) {
        match expr {
            ScalarExpr::Aggregate(agg) => {
                // Create a unique key for this aggregate
                let key = format!("{:?}", agg);
                if !aggregates.iter().any(|(k, _)| k == &key) {
                    aggregates.push((key, agg.clone()));
                }
            }
            ScalarExpr::Binary { left, right, .. } => {
                Self::collect_aggregates(left, aggregates);
                Self::collect_aggregates(right, aggregates);
            }
            ScalarExpr::Compare { left, right, .. } => {
                Self::collect_aggregates(left, aggregates);
                Self::collect_aggregates(right, aggregates);
            }
            ScalarExpr::GetField { base, .. } => {
                Self::collect_aggregates(base, aggregates);
            }
            ScalarExpr::Cast { expr, .. } => {
                Self::collect_aggregates(expr, aggregates);
            }
            ScalarExpr::Case {
                operand,
                branches,
                else_expr,
            } => {
                if let Some(inner) = operand.as_deref() {
                    Self::collect_aggregates(inner, aggregates);
                }
                for (when_expr, then_expr) in branches {
                    Self::collect_aggregates(when_expr, aggregates);
                    Self::collect_aggregates(then_expr, aggregates);
                }
                if let Some(inner) = else_expr.as_deref() {
                    Self::collect_aggregates(inner, aggregates);
                }
            }
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) => {}
        }
    }

    /// Compute the actual values for the aggregates
    fn compute_aggregate_values(
        &self,
        table: Arc<ExecutorTable<P>>,
        filter: &Option<llkv_expr::expr::Expr<'static, String>>,
        aggregate_specs: &[(String, llkv_expr::expr::AggregateCall<String>)],
        row_filter: Option<std::sync::Arc<dyn RowIdFilter<P>>>,
    ) -> ExecutorResult<FxHashMap<String, i64>> {
        use llkv_expr::expr::AggregateCall;

        let table_ref = table.as_ref();
        let mut results =
            FxHashMap::with_capacity_and_hasher(aggregate_specs.len(), Default::default());

        // Build aggregate specs for the aggregator
        let mut specs: Vec<AggregateSpec> = Vec::new();
        for (key, agg) in aggregate_specs {
            let kind = match agg {
                AggregateCall::CountStar => AggregateKind::CountStar,
                AggregateCall::Count(col_name) => {
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::CountField {
                        field_id: col.field_id,
                    }
                }
                AggregateCall::Sum(col_name) => {
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::SumInt64 {
                        field_id: col.field_id,
                    }
                }
                AggregateCall::Min(col_name) => {
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::MinInt64 {
                        field_id: col.field_id,
                    }
                }
                AggregateCall::Max(col_name) => {
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::MaxInt64 {
                        field_id: col.field_id,
                    }
                }
                AggregateCall::CountNulls(col_name) => {
                    let col = table_ref.schema.resolve(col_name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", col_name))
                    })?;
                    AggregateKind::CountNulls {
                        field_id: col.field_id,
                    }
                }
            };
            specs.push(AggregateSpec {
                alias: key.clone(),
                kind,
            });
        }

        // Prepare filter and projections
        let filter_expr = match filter {
            Some(expr) => crate::translation::expression::translate_predicate(
                expr.clone(),
                table_ref.schema.as_ref(),
                |name| Error::InvalidArgumentError(format!("unknown column '{}'", name)),
            )?,
            None => {
                let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform aggregate scan".into(),
                    )
                })?;
                crate::translation::expression::full_table_scan_filter(field_id)
            }
        };

        let mut projections: Vec<ScanProjection> = Vec::new();
        let mut spec_to_projection: Vec<Option<usize>> = Vec::with_capacity(specs.len());
        let count_star_override: Option<i64> = None;

        for spec in &specs {
            if let Some(field_id) = spec.kind.field_id() {
                spec_to_projection.push(Some(projections.len()));
                projections.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), field_id),
                    table
                        .schema
                        .column_by_field_id(field_id)
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("col{field_id}")),
                )));
            } else {
                spec_to_projection.push(None);
            }
        }

        if projections.is_empty() {
            let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                Error::InvalidArgumentError(
                    "table has no columns; cannot perform aggregate scan".into(),
                )
            })?;
            projections.push(ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), field_id),
                table
                    .schema
                    .column_by_field_id(field_id)
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| format!("col{field_id}")),
            )));
        }

        let base_options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: None,
        };

        let mut states: Vec<AggregateState> = Vec::with_capacity(specs.len());
        for (idx, spec) in specs.iter().enumerate() {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new_with_projection_index(
                    spec,
                    spec_to_projection[idx],
                    count_star_override,
                )?,
                override_value: match spec.kind {
                    AggregateKind::CountStar => count_star_override,
                    _ => None,
                },
            });
        }

        let mut error: Option<Error> = None;
        match table.table.scan_stream(
            projections,
            &filter_expr,
            ScanStreamOptions {
                row_id_filter: row_filter.clone(),
                ..base_options
            },
            |batch| {
                if error.is_some() {
                    return;
                }
                for state in &mut states {
                    if let Err(err) = state.update(&batch) {
                        error = Some(err);
                        return;
                    }
                }
            },
        ) {
            Ok(()) => {}
            Err(llkv_result::Error::NotFound) => {}
            Err(err) => return Err(err),
        }
        if let Some(err) = error {
            return Err(err);
        }

        // Extract the computed values
        for state in states {
            let alias = state.alias.clone();
            let (_field, array) = state.finalize()?;

            // Extract the i64 value from the array
            let int64_array = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| Error::Internal("Expected Int64Array from aggregate".into()))?;

            if int64_array.len() != 1 {
                return Err(Error::Internal(format!(
                    "Expected single value from aggregate, got {}",
                    int64_array.len()
                )));
            }

            let value = if int64_array.is_null(0) {
                0
            } else {
                int64_array.value(0)
            };

            results.insert(alias, value);
        }

        Ok(results)
    }

    /// Evaluate an expression by substituting aggregate values
    fn evaluate_expr_with_aggregates(
        expr: &ScalarExpr<String>,
        aggregates: &FxHashMap<String, i64>,
    ) -> ExecutorResult<i64> {
        use llkv_expr::expr::BinaryOp;
        use llkv_expr::literal::Literal;

        match expr {
            ScalarExpr::Literal(Literal::Integer(v)) => Ok(*v as i64),
            ScalarExpr::Literal(Literal::Float(v)) => Ok(*v as i64),
            ScalarExpr::Literal(Literal::Boolean(v)) => Ok(if *v { 1 } else { 0 }),
            ScalarExpr::Literal(Literal::String(_)) => Err(Error::InvalidArgumentError(
                "String literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Literal(Literal::Null) => Err(Error::InvalidArgumentError(
                "NULL literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Literal(Literal::Struct(_)) => Err(Error::InvalidArgumentError(
                "Struct literals not supported in aggregate expressions".into(),
            )),
            ScalarExpr::Column(_) => Err(Error::InvalidArgumentError(
                "Column references not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Compare { .. } => Err(Error::InvalidArgumentError(
                "Comparisons not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Aggregate(agg) => {
                let key = format!("{:?}", agg);
                aggregates.get(&key).copied().ok_or_else(|| {
                    Error::Internal(format!("Aggregate value not found for key: {}", key))
                })
            }
            ScalarExpr::Binary { left, op, right } => {
                let left_val = Self::evaluate_expr_with_aggregates(left, aggregates)?;
                let right_val = Self::evaluate_expr_with_aggregates(right, aggregates)?;

                let result = match op {
                    BinaryOp::Add => left_val.checked_add(right_val),
                    BinaryOp::Subtract => left_val.checked_sub(right_val),
                    BinaryOp::Multiply => left_val.checked_mul(right_val),
                    BinaryOp::Divide => {
                        if right_val == 0 {
                            return Err(Error::InvalidArgumentError("Division by zero".into()));
                        }
                        left_val.checked_div(right_val)
                    }
                    BinaryOp::Modulo => {
                        if right_val == 0 {
                            return Err(Error::InvalidArgumentError("Modulo by zero".into()));
                        }
                        left_val.checked_rem(right_val)
                    }
                };

                result.ok_or_else(|| {
                    Error::InvalidArgumentError("Arithmetic overflow in expression".into())
                })
            }
            ScalarExpr::Cast { .. } => Err(Error::InvalidArgumentError(
                "CAST is not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "GetField not supported in aggregate-only expressions".into(),
            )),
            ScalarExpr::Case { .. } => Err(Error::InvalidArgumentError(
                "CASE not supported in aggregate-only expressions".into(),
            )),
        }
    }
}

struct CrossProductExpressionContext {
    schema: Arc<ExecutorSchema>,
    field_id_to_index: FxHashMap<FieldId, usize>,
    numeric_cache: FxHashMap<FieldId, NumericArray>,
    column_cache: FxHashMap<FieldId, ColumnAccessor>,
}

#[derive(Clone)]
enum ColumnAccessor {
    Int64(Arc<Int64Array>),
    Float64(Arc<Float64Array>),
    Boolean(Arc<BooleanArray>),
    Utf8(Arc<StringArray>),
    Null(usize),
}

impl ColumnAccessor {
    fn from_array(array: &ArrayRef) -> ExecutorResult<Self> {
        match array.data_type() {
            DataType::Int64 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Internal("expected Int64 array".into()))?
                    .clone();
                Ok(Self::Int64(Arc::new(typed)))
            }
            DataType::Float64 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::Internal("expected Float64 array".into()))?
                    .clone();
                Ok(Self::Float64(Arc::new(typed)))
            }
            DataType::Boolean => {
                let typed = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| Error::Internal("expected Boolean array".into()))?
                    .clone();
                Ok(Self::Boolean(Arc::new(typed)))
            }
            DataType::Utf8 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::Internal("expected Utf8 array".into()))?
                    .clone();
                Ok(Self::Utf8(Arc::new(typed)))
            }
            DataType::Null => Ok(Self::Null(array.len())),
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported column type {:?} in cross product filter",
                other
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            ColumnAccessor::Int64(array) => array.len(),
            ColumnAccessor::Float64(array) => array.len(),
            ColumnAccessor::Boolean(array) => array.len(),
            ColumnAccessor::Utf8(array) => array.len(),
            ColumnAccessor::Null(len) => *len,
        }
    }

    fn is_null(&self, idx: usize) -> bool {
        match self {
            ColumnAccessor::Int64(array) => array.is_null(idx),
            ColumnAccessor::Float64(array) => array.is_null(idx),
            ColumnAccessor::Boolean(array) => array.is_null(idx),
            ColumnAccessor::Utf8(array) => array.is_null(idx),
            ColumnAccessor::Null(_) => true,
        }
    }

    fn as_array_ref(&self) -> ArrayRef {
        match self {
            ColumnAccessor::Int64(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Float64(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Boolean(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Utf8(array) => Arc::clone(array) as ArrayRef,
            ColumnAccessor::Null(len) => new_null_array(&DataType::Null, *len),
        }
    }
}

#[derive(Clone)]
enum ValueArray {
    Numeric(NumericArray),
    Boolean(Arc<BooleanArray>),
    Utf8(Arc<StringArray>),
    Null(usize),
}

impl ValueArray {
    fn from_array(array: ArrayRef) -> ExecutorResult<Self> {
        match array.data_type() {
            DataType::Boolean => {
                let typed = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| Error::Internal("expected Boolean array".into()))?
                    .clone();
                Ok(Self::Boolean(Arc::new(typed)))
            }
            DataType::Utf8 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::Internal("expected Utf8 array".into()))?
                    .clone();
                Ok(Self::Utf8(Arc::new(typed)))
            }
            DataType::Null => Ok(Self::Null(array.len())),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => {
                let numeric = NumericArray::try_from_arrow(&array)?;
                Ok(Self::Numeric(numeric))
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unsupported data type {:?} in cross product expression",
                other
            ))),
        }
    }

    fn len(&self) -> usize {
        match self {
            ValueArray::Numeric(array) => array.len(),
            ValueArray::Boolean(array) => array.len(),
            ValueArray::Utf8(array) => array.len(),
            ValueArray::Null(len) => *len,
        }
    }
}

fn truth_and(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
    match (lhs, rhs) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        (Some(true), None) | (None, Some(true)) | (None, None) => None,
    }
}

fn truth_or(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
    match (lhs, rhs) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        (Some(false), None) | (None, Some(false)) | (None, None) => None,
    }
}

fn truth_not(value: Option<bool>) -> Option<bool> {
    match value {
        Some(true) => Some(false),
        Some(false) => Some(true),
        None => None,
    }
}

fn compare_bool(op: CompareOp, lhs: bool, rhs: bool) -> bool {
    let l = lhs as u8;
    let r = rhs as u8;
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::NotEq => lhs != rhs,
        CompareOp::Lt => l < r,
        CompareOp::LtEq => l <= r,
        CompareOp::Gt => l > r,
        CompareOp::GtEq => l >= r,
    }
}

fn compare_str(op: CompareOp, lhs: &str, rhs: &str) -> bool {
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::NotEq => lhs != rhs,
        CompareOp::Lt => lhs < rhs,
        CompareOp::LtEq => lhs <= rhs,
        CompareOp::Gt => lhs > rhs,
        CompareOp::GtEq => lhs >= rhs,
    }
}

fn finalize_in_list_result(has_match: bool, saw_null: bool, negated: bool) -> Option<bool> {
    if has_match {
        Some(!negated)
    } else if saw_null {
        None
    } else if negated {
        Some(true)
    } else {
        Some(false)
    }
}

fn literal_to_constant_array(literal: &Literal, len: usize) -> ExecutorResult<ArrayRef> {
    match literal {
        Literal::Integer(v) => {
            let value = i64::try_from(*v).unwrap_or(0);
            let values = vec![value; len];
            Ok(Arc::new(Int64Array::from(values)) as ArrayRef)
        }
        Literal::Float(v) => {
            let values = vec![*v; len];
            Ok(Arc::new(Float64Array::from(values)) as ArrayRef)
        }
        Literal::Boolean(v) => {
            let values = vec![Some(*v); len];
            Ok(Arc::new(BooleanArray::from(values)) as ArrayRef)
        }
        Literal::String(v) => {
            let values: Vec<Option<String>> = (0..len).map(|_| Some(v.clone())).collect();
            Ok(Arc::new(StringArray::from(values)) as ArrayRef)
        }
        Literal::Null => Ok(new_null_array(&DataType::Null, len)),
        Literal::Struct(_) => Err(Error::InvalidArgumentError(
            "struct literals are not supported in cross product filters".into(),
        )),
    }
}

impl CrossProductExpressionContext {
    fn new(schema: &Schema, lookup: FxHashMap<String, usize>) -> ExecutorResult<Self> {
        let mut columns = Vec::with_capacity(schema.fields().len());
        let mut field_id_to_index = FxHashMap::default();
        let mut next_field_id: FieldId = 1;

        for (idx, field) in schema.fields().iter().enumerate() {
            if next_field_id == u32::MAX {
                return Err(Error::Internal(
                    "cross product projection exhausted FieldId space".into(),
                ));
            }

            let executor_column = ExecutorColumn {
                name: field.name().clone(),
                data_type: field.data_type().clone(),
                nullable: field.is_nullable(),
                primary_key: false,
                unique: false,
                field_id: next_field_id,
                check_expr: None,
            };
            let field_id = next_field_id;
            next_field_id = next_field_id.saturating_add(1);

            columns.push(executor_column);
            field_id_to_index.insert(field_id, idx);
        }

        Ok(Self {
            schema: Arc::new(ExecutorSchema { columns, lookup }),
            field_id_to_index,
            numeric_cache: FxHashMap::default(),
            column_cache: FxHashMap::default(),
        })
    }

    fn schema(&self) -> &ExecutorSchema {
        self.schema.as_ref()
    }

    fn reset(&mut self) {
        self.numeric_cache.clear();
        self.column_cache.clear();
    }

    fn evaluate(
        &mut self,
        expr: &ScalarExpr<String>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        let translated = translate_scalar(expr, self.schema.as_ref(), |name| {
            Error::InvalidArgumentError(format!(
                "column '{}' not found in cross product result",
                name
            ))
        })?;

        self.evaluate_numeric(&translated, batch)
    }

    fn evaluate_predicate_mask(
        &mut self,
        expr: &LlkvExpr<'static, FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<BooleanArray> {
        let truths = self.evaluate_predicate_truths(expr, batch)?;
        let mut builder = BooleanBuilder::with_capacity(truths.len());
        for value in truths {
            builder.append_value(value.unwrap_or(false));
        }
        Ok(builder.finish())
    }

    fn evaluate_predicate_truths(
        &mut self,
        expr: &LlkvExpr<'static, FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        match expr {
            LlkvExpr::Literal(value) => Ok(vec![Some(*value); batch.num_rows()]),
            LlkvExpr::And(children) => {
                if children.is_empty() {
                    return Ok(vec![Some(true); batch.num_rows()]);
                }
                let mut result = self.evaluate_predicate_truths(&children[0], batch)?;
                for child in &children[1..] {
                    let next = self.evaluate_predicate_truths(child, batch)?;
                    for (lhs, rhs) in result.iter_mut().zip(next.into_iter()) {
                        *lhs = truth_and(*lhs, rhs);
                    }
                }
                Ok(result)
            }
            LlkvExpr::Or(children) => {
                if children.is_empty() {
                    return Ok(vec![Some(false); batch.num_rows()]);
                }
                let mut result = self.evaluate_predicate_truths(&children[0], batch)?;
                for child in &children[1..] {
                    let next = self.evaluate_predicate_truths(child, batch)?;
                    for (lhs, rhs) in result.iter_mut().zip(next.into_iter()) {
                        *lhs = truth_or(*lhs, rhs);
                    }
                }
                Ok(result)
            }
            LlkvExpr::Not(inner) => {
                let mut values = self.evaluate_predicate_truths(inner, batch)?;
                for value in &mut values {
                    *value = truth_not(*value);
                }
                Ok(values)
            }
            LlkvExpr::Pred(filter) => self.evaluate_filter_truths(filter, batch),
            LlkvExpr::Compare { left, op, right } => {
                self.evaluate_compare_truths(left, *op, right, batch)
            }
            LlkvExpr::InList {
                expr: target,
                list,
                negated,
            } => self.evaluate_in_list_truths(target, list, *negated, batch),
        }
    }

    fn evaluate_filter_truths(
        &mut self,
        filter: &Filter<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let accessor = self.column_accessor(filter.field_id, batch)?;
        let len = accessor.len();

        match &filter.op {
            Operator::IsNull => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    out.push(Some(accessor.is_null(idx)));
                }
                Ok(out)
            }
            Operator::IsNotNull => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    out.push(Some(!accessor.is_null(idx)));
                }
                Ok(out)
            }
            _ => match accessor {
                ColumnAccessor::Int64(array) => {
                    let predicate = build_fixed_width_predicate::<Int64Type>(&filter.op)
                        .map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(&value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Float64(array) => {
                    let predicate = build_fixed_width_predicate::<Float64Type>(&filter.op)
                        .map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(&value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Boolean(array) => {
                    let predicate =
                        build_bool_predicate(&filter.op).map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(&value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Utf8(array) => {
                    let predicate =
                        build_var_width_predicate(&filter.op).map_err(Error::predicate_build)?;
                    let mut out = Vec::with_capacity(len);
                    for idx in 0..len {
                        if array.is_null(idx) {
                            out.push(None);
                        } else {
                            let value = array.value(idx);
                            out.push(Some(predicate.matches(value)));
                        }
                    }
                    Ok(out)
                }
                ColumnAccessor::Null(len) => Ok(vec![None; len]),
            },
        }
    }

    fn evaluate_compare_truths(
        &mut self,
        left: &ScalarExpr<FieldId>,
        op: CompareOp,
        right: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let left_values = self.materialize_value_array(left, batch)?;
        let right_values = self.materialize_value_array(right, batch)?;

        if left_values.len() != right_values.len() {
            return Err(Error::Internal(
                "mismatched compare operand lengths in cross product filter".into(),
            ));
        }

        let len = left_values.len();
        match (&left_values, &right_values) {
            (ValueArray::Null(_), _) | (_, ValueArray::Null(_)) => Ok(vec![None; len]),
            (ValueArray::Numeric(lhs), ValueArray::Numeric(rhs)) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    match (lhs.value(idx), rhs.value(idx)) {
                        (Some(lv), Some(rv)) => out.push(Some(NumericKernels::compare(op, lv, rv))),
                        _ => out.push(None),
                    }
                }
                Ok(out)
            }
            (ValueArray::Boolean(lhs), ValueArray::Boolean(rhs)) => {
                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if lhs.is_null(idx) || rhs.is_null(idx) {
                        out.push(None);
                    } else {
                        out.push(Some(compare_bool(op, lhs.value(idx), rhs.value(idx))));
                    }
                }
                Ok(out)
            }
            (ValueArray::Utf8(lhs), ValueArray::Utf8(rhs)) => {
                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if lhs.is_null(idx) || rhs.is_null(idx) {
                        out.push(None);
                    } else {
                        out.push(Some(compare_str(op, lhs.value(idx), rhs.value(idx))));
                    }
                }
                Ok(out)
            }
            _ => Err(Error::InvalidArgumentError(
                "unsupported comparison between mismatched types in cross product filter".into(),
            )),
        }
    }

    fn evaluate_in_list_truths(
        &mut self,
        target: &ScalarExpr<FieldId>,
        list: &[ScalarExpr<FieldId>],
        negated: bool,
        batch: &RecordBatch,
    ) -> ExecutorResult<Vec<Option<bool>>> {
        let target_values = self.materialize_value_array(target, batch)?;
        let list_values = list
            .iter()
            .map(|expr| self.materialize_value_array(expr, batch))
            .collect::<ExecutorResult<Vec<_>>>()?;

        let len = target_values.len();
        for values in &list_values {
            if values.len() != len {
                return Err(Error::Internal(
                    "mismatched IN list operand lengths in cross product filter".into(),
                ));
            }
        }

        match &target_values {
            ValueArray::Numeric(target_numeric) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    let target_value = match target_numeric.value(idx) {
                        Some(value) => value,
                        None => {
                            out.push(None);
                            continue;
                        }
                    };
                    let mut has_match = false;
                    let mut saw_null = false;
                    for candidate in &list_values {
                        match candidate {
                            ValueArray::Numeric(array) => match array.value(idx) {
                                Some(value) => {
                                    if NumericKernels::compare(CompareOp::Eq, target_value, value) {
                                        has_match = true;
                                        break;
                                    }
                                }
                                None => saw_null = true,
                            },
                            ValueArray::Null(_) => saw_null = true,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "type mismatch in IN list evaluation".into(),
                                ));
                            }
                        }
                    }
                    out.push(finalize_in_list_result(has_match, saw_null, negated));
                }
                Ok(out)
            }
            ValueArray::Boolean(target_bool) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if target_bool.is_null(idx) {
                        out.push(None);
                        continue;
                    }
                    let target_value = target_bool.value(idx);
                    let mut has_match = false;
                    let mut saw_null = false;
                    for candidate in &list_values {
                        match candidate {
                            ValueArray::Boolean(array) => {
                                if array.is_null(idx) {
                                    saw_null = true;
                                } else if array.value(idx) == target_value {
                                    has_match = true;
                                    break;
                                }
                            }
                            ValueArray::Null(_) => saw_null = true,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "type mismatch in IN list evaluation".into(),
                                ));
                            }
                        }
                    }
                    out.push(finalize_in_list_result(has_match, saw_null, negated));
                }
                Ok(out)
            }
            ValueArray::Utf8(target_utf8) => {
                let mut out = Vec::with_capacity(len);
                for idx in 0..len {
                    if target_utf8.is_null(idx) {
                        out.push(None);
                        continue;
                    }
                    let target_value = target_utf8.value(idx);
                    let mut has_match = false;
                    let mut saw_null = false;
                    for candidate in &list_values {
                        match candidate {
                            ValueArray::Utf8(array) => {
                                if array.is_null(idx) {
                                    saw_null = true;
                                } else if array.value(idx) == target_value {
                                    has_match = true;
                                    break;
                                }
                            }
                            ValueArray::Null(_) => saw_null = true,
                            _ => {
                                return Err(Error::InvalidArgumentError(
                                    "type mismatch in IN list evaluation".into(),
                                ));
                            }
                        }
                    }
                    out.push(finalize_in_list_result(has_match, saw_null, negated));
                }
                Ok(out)
            }
            ValueArray::Null(len) => Ok(vec![None; *len]),
        }
    }

    fn evaluate_numeric(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        let mut required = FxHashSet::default();
        collect_field_ids(expr, &mut required);

        let mut arrays = NumericArrayMap::default();
        for field_id in required {
            let numeric = self.numeric_array(field_id, batch)?;
            arrays.insert(field_id, numeric);
        }

        NumericKernels::evaluate_batch(expr, batch.num_rows(), &arrays)
    }

    fn numeric_array(
        &mut self,
        field_id: FieldId,
        batch: &RecordBatch,
    ) -> ExecutorResult<NumericArray> {
        if let Some(existing) = self.numeric_cache.get(&field_id) {
            return Ok(existing.clone());
        }

        let column_index = *self.field_id_to_index.get(&field_id).ok_or_else(|| {
            Error::Internal("field mapping missing during cross product evaluation".into())
        })?;

        let array_ref = batch.column(column_index).clone();
        let numeric = NumericArray::try_from_arrow(&array_ref)?;
        self.numeric_cache.insert(field_id, numeric.clone());
        Ok(numeric)
    }

    fn column_accessor(
        &mut self,
        field_id: FieldId,
        batch: &RecordBatch,
    ) -> ExecutorResult<ColumnAccessor> {
        if let Some(existing) = self.column_cache.get(&field_id) {
            return Ok(existing.clone());
        }

        let column_index = *self.field_id_to_index.get(&field_id).ok_or_else(|| {
            Error::Internal("field mapping missing during cross product evaluation".into())
        })?;

        let accessor = ColumnAccessor::from_array(batch.column(column_index))?;
        self.column_cache.insert(field_id, accessor.clone());
        Ok(accessor)
    }

    fn materialize_scalar_array(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ArrayRef> {
        match expr {
            ScalarExpr::Column(field_id) => {
                let accessor = self.column_accessor(*field_id, batch)?;
                Ok(accessor.as_array_ref())
            }
            ScalarExpr::Literal(literal) => literal_to_constant_array(literal, batch.num_rows()),
            ScalarExpr::Binary { .. } => self.evaluate_numeric(expr, batch),
            ScalarExpr::Compare { .. } => self.evaluate_numeric(expr, batch),
            ScalarExpr::Aggregate(_) => Err(Error::InvalidArgumentError(
                "aggregate expressions are not supported in cross product filters".into(),
            )),
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "struct field access is not supported in cross product filters".into(),
            )),
            ScalarExpr::Cast { expr, data_type } => {
                let source = self.materialize_scalar_array(expr.as_ref(), batch)?;
                let casted = cast(source.as_ref(), data_type).map_err(|err| {
                    Error::InvalidArgumentError(format!("failed to cast expression: {err}"))
                })?;
                Ok(casted)
            }
            ScalarExpr::Case { .. } => self.evaluate_numeric(expr, batch),
        }
    }

    fn materialize_value_array(
        &mut self,
        expr: &ScalarExpr<FieldId>,
        batch: &RecordBatch,
    ) -> ExecutorResult<ValueArray> {
        let array = self.materialize_scalar_array(expr, batch)?;
        ValueArray::from_array(array)
    }
}

// TODO: Move to llkv-aggregate?
fn collect_field_ids(expr: &ScalarExpr<FieldId>, out: &mut FxHashSet<FieldId>) {
    match expr {
        ScalarExpr::Column(fid) => {
            out.insert(*fid);
        }
        ScalarExpr::Binary { left, right, .. } => {
            collect_field_ids(left, out);
            collect_field_ids(right, out);
        }
        ScalarExpr::Compare { left, right, .. } => {
            collect_field_ids(left, out);
            collect_field_ids(right, out);
        }
        ScalarExpr::Aggregate(call) => match call {
            AggregateCall::CountStar => {}
            AggregateCall::Count(fid)
            | AggregateCall::Sum(fid)
            | AggregateCall::Min(fid)
            | AggregateCall::Max(fid)
            | AggregateCall::CountNulls(fid) => {
                out.insert(*fid);
            }
        },
        ScalarExpr::GetField { base, .. } => collect_field_ids(base, out),
        ScalarExpr::Cast { expr, .. } => collect_field_ids(expr, out),
        ScalarExpr::Case {
            operand,
            branches,
            else_expr,
        } => {
            if let Some(inner) = operand.as_deref() {
                collect_field_ids(inner, out);
            }
            for (when_expr, then_expr) in branches {
                collect_field_ids(when_expr, out);
                collect_field_ids(then_expr, out);
            }
            if let Some(inner) = else_expr.as_deref() {
                collect_field_ids(inner, out);
            }
        }
        ScalarExpr::Literal(_) => {}
    }
}

/// Streaming execution handle for SELECT queries.
#[derive(Clone)]
pub struct SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table_name: String,
    schema: Arc<Schema>,
    stream: SelectStream<P>,
}

#[derive(Clone)]
enum SelectStream<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    Projection {
        table: Arc<ExecutorTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        full_table_scan: bool,
        order_by: Vec<OrderByPlan>,
        distinct: bool,
    },
    Aggregation {
        batch: RecordBatch,
    },
}

impl<P> SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    fn new_projection(
        table_name: String,
        schema: Arc<Schema>,
        table: Arc<ExecutorTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        full_table_scan: bool,
        order_by: Vec<OrderByPlan>,
        distinct: bool,
    ) -> Self {
        Self {
            table_name,
            schema,
            stream: SelectStream::Projection {
                table,
                projections,
                filter_expr,
                options,
                full_table_scan,
                order_by,
                distinct,
            },
        }
    }

    pub fn new_single_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self {
            table_name,
            schema,
            stream: SelectStream::Aggregation { batch },
        }
    }

    pub fn from_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
        Self::new_single_batch(table_name, schema, batch)
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    pub fn stream(
        self,
        mut on_batch: impl FnMut(RecordBatch) -> ExecutorResult<()>,
    ) -> ExecutorResult<()> {
        let schema = Arc::clone(&self.schema);
        match self.stream {
            SelectStream::Projection {
                table,
                projections,
                filter_expr,
                options,
                full_table_scan,
                order_by,
                distinct,
            } => {
                // Early return for empty tables to avoid ColumnStore data_type() errors
                let total_rows = table.total_rows.load(Ordering::SeqCst);
                if total_rows == 0 {
                    // Empty table - return empty result with correct schema
                    return Ok(());
                }

                let mut error: Option<Error> = None;
                let mut produced = false;
                let mut produced_rows: u64 = 0;
                let capture_nulls_first = matches!(options.order, Some(spec) if spec.nulls_first);
                let needs_post_sort =
                    !order_by.is_empty() && (order_by.len() > 1 || options.order.is_none());
                let collect_batches = needs_post_sort || capture_nulls_first;
                let include_nulls = options.include_nulls;
                let has_row_id_filter = options.row_id_filter.is_some();
                let mut distinct_state = if distinct {
                    Some(DistinctState::default())
                } else {
                    None
                };
                let scan_options = options;
                let mut buffered_batches: Vec<RecordBatch> = Vec::new();
                table
                    .table
                    .scan_stream(projections, &filter_expr, scan_options, |batch| {
                        if error.is_some() {
                            return;
                        }
                        let mut batch = batch;
                        if let Some(state) = distinct_state.as_mut() {
                            match distinct_filter_batch(batch, state) {
                                Ok(Some(filtered)) => {
                                    batch = filtered;
                                }
                                Ok(None) => {
                                    return;
                                }
                                Err(err) => {
                                    error = Some(err);
                                    return;
                                }
                            }
                        }
                        produced = true;
                        produced_rows = produced_rows.saturating_add(batch.num_rows() as u64);
                        if collect_batches {
                            buffered_batches.push(batch);
                        } else if let Err(err) = on_batch(batch) {
                            error = Some(err);
                        }
                    })?;
                if let Some(err) = error {
                    return Err(err);
                }
                if !produced {
                    // Only synthesize null rows if this was a full table scan
                    // If there was a filter and it matched no rows, we should return empty results
                    if !distinct && full_table_scan && total_rows > 0 {
                        for batch in synthesize_null_scan(Arc::clone(&schema), total_rows)? {
                            on_batch(batch)?;
                        }
                    }
                    return Ok(());
                }
                let mut null_batches: Vec<RecordBatch> = Vec::new();
                // Only synthesize null rows if:
                // 1. include_nulls is true
                // 2. This is a full table scan
                // 3. We produced fewer rows than the total
                // 4. We DON'T have a row_id_filter (e.g., MVCC filter) that intentionally filtered rows
                if !distinct
                    && include_nulls
                    && full_table_scan
                    && produced_rows < total_rows
                    && !has_row_id_filter
                {
                    let missing = total_rows - produced_rows;
                    if missing > 0 {
                        null_batches = synthesize_null_scan(Arc::clone(&schema), missing)?;
                    }
                }

                if collect_batches {
                    if needs_post_sort {
                        if !null_batches.is_empty() {
                            buffered_batches.extend(null_batches);
                        }
                        if !buffered_batches.is_empty() {
                            let combined =
                                concat_batches(&schema, &buffered_batches).map_err(|err| {
                                    Error::InvalidArgumentError(format!(
                                        "failed to concatenate result batches for ORDER BY: {}",
                                        err
                                    ))
                                })?;
                            let sorted_batch =
                                sort_record_batch_with_order(&schema, &combined, &order_by)?;
                            on_batch(sorted_batch)?;
                        }
                    } else if capture_nulls_first {
                        for batch in null_batches {
                            on_batch(batch)?;
                        }
                        for batch in buffered_batches {
                            on_batch(batch)?;
                        }
                    }
                } else if !null_batches.is_empty() {
                    for batch in null_batches {
                        on_batch(batch)?;
                    }
                }
                Ok(())
            }
            SelectStream::Aggregation { batch } => on_batch(batch),
        }
    }

    pub fn collect(self) -> ExecutorResult<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        self.stream(|batch| {
            batches.push(batch);
            Ok(())
        })?;
        Ok(batches)
    }

    pub fn collect_rows(self) -> ExecutorResult<ExecutorRowBatch> {
        let schema = self.schema();
        let mut rows: Vec<Vec<PlanValue>> = Vec::new();
        self.stream(|batch| {
            for row_idx in 0..batch.num_rows() {
                let mut row: Vec<PlanValue> = Vec::with_capacity(batch.num_columns());
                for col_idx in 0..batch.num_columns() {
                    let value = llkv_plan::plan_value_from_array(batch.column(col_idx), row_idx)?;
                    row.push(value);
                }
                rows.push(row);
            }
            Ok(())
        })?;
        let columns = schema
            .fields()
            .iter()
            .map(|field| field.name().to_string())
            .collect();
        Ok(ExecutorRowBatch { columns, rows })
    }

    pub fn into_rows(self) -> ExecutorResult<Vec<Vec<PlanValue>>> {
        Ok(self.collect_rows()?.rows)
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

// ============================================================================
// Helper Functions
// ============================================================================

fn expand_order_targets(
    order_items: &[OrderByPlan],
    projections: &[ScanProjection],
) -> ExecutorResult<Vec<OrderByPlan>> {
    let mut expanded = Vec::new();

    for item in order_items {
        match &item.target {
            OrderTarget::All => {
                if projections.is_empty() {
                    return Err(Error::InvalidArgumentError(
                        "ORDER BY ALL requires at least one projection".into(),
                    ));
                }

                for (idx, projection) in projections.iter().enumerate() {
                    if matches!(projection, ScanProjection::Computed { .. }) {
                        return Err(Error::InvalidArgumentError(
                            "ORDER BY ALL cannot reference computed projections".into(),
                        ));
                    }

                    let mut clone = item.clone();
                    clone.target = OrderTarget::Index(idx);
                    expanded.push(clone);
                }
            }
            _ => expanded.push(item.clone()),
        }
    }

    Ok(expanded)
}

fn resolve_scan_order<P>(
    table: &ExecutorTable<P>,
    projections: &[ScanProjection],
    order_plan: &OrderByPlan,
) -> ExecutorResult<ScanOrderSpec>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let (column, field_id) = match &order_plan.target {
        OrderTarget::Column(name) => {
            let column = table.schema.resolve(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{}' in ORDER BY", name))
            })?;
            (column, column.field_id)
        }
        OrderTarget::Index(position) => {
            let projection = projections.get(*position).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "ORDER BY position {} is out of range",
                    position + 1
                ))
            })?;
            match projection {
                ScanProjection::Column(store_projection) => {
                    let field_id = store_projection.logical_field_id.field_id();
                    let column = table.schema.column_by_field_id(field_id).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column with field id {field_id} in ORDER BY"
                        ))
                    })?;
                    (column, field_id)
                }
                ScanProjection::Computed { .. } => {
                    return Err(Error::InvalidArgumentError(
                        "ORDER BY position referring to computed projection is not supported"
                            .into(),
                    ));
                }
            }
        }
        OrderTarget::All => {
            return Err(Error::InvalidArgumentError(
                "ORDER BY ALL should be expanded before execution".into(),
            ));
        }
    };

    let transform = match order_plan.sort_type {
        OrderSortType::Native => match column.data_type {
            DataType::Int64 => ScanOrderTransform::IdentityInteger,
            DataType::Utf8 => ScanOrderTransform::IdentityUtf8,
            ref other => {
                return Err(Error::InvalidArgumentError(format!(
                    "ORDER BY on column type {:?} is not supported",
                    other
                )));
            }
        },
        OrderSortType::CastTextToInteger => {
            if column.data_type != DataType::Utf8 {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY CAST expects a text column".into(),
                ));
            }
            ScanOrderTransform::CastUtf8ToInteger
        }
    };

    let direction = if order_plan.ascending {
        ScanOrderDirection::Ascending
    } else {
        ScanOrderDirection::Descending
    };

    Ok(ScanOrderSpec {
        field_id,
        direction,
        nulls_first: order_plan.nulls_first,
        transform,
    })
}

fn synthesize_null_scan(schema: Arc<Schema>, total_rows: u64) -> ExecutorResult<Vec<RecordBatch>> {
    let row_count = usize::try_from(total_rows).map_err(|_| {
        Error::InvalidArgumentError("table row count exceeds supported in-memory batch size".into())
    })?;

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        match field.data_type() {
            DataType::Int64 => {
                let mut builder = Int64Builder::with_capacity(row_count);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            DataType::Float64 => {
                let mut builder = arrow::array::Float64Builder::with_capacity(row_count);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            DataType::Utf8 => {
                let mut builder = arrow::array::StringBuilder::with_capacity(row_count, 0);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            DataType::Date32 => {
                let mut builder = arrow::array::Date32Builder::with_capacity(row_count);
                for _ in 0..row_count {
                    builder.append_null();
                }
                arrays.push(Arc::new(builder.finish()));
            }
            other => {
                return Err(Error::InvalidArgumentError(format!(
                    "unsupported data type in null synthesis: {other:?}"
                )));
            }
        }
    }

    let batch = RecordBatch::try_new(schema, arrays)?;
    Ok(vec![batch])
}

struct TableCrossProductData {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    column_counts: Vec<usize>,
}

fn collect_table_data<P>(
    table_ref: &llkv_plan::TableRef,
    table: &ExecutorTable<P>,
) -> ExecutorResult<TableCrossProductData>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    if table.schema.columns.is_empty() {
        return Err(Error::InvalidArgumentError(format!(
            "table '{}' has no columns; cross products require at least one column",
            table_ref.qualified_name()
        )));
    }

    let mut projections = Vec::with_capacity(table.schema.columns.len());
    let mut fields = Vec::with_capacity(table.schema.columns.len());

    for column in &table.schema.columns {
        let table_component = table_ref
            .alias
            .as_deref()
            .unwrap_or(table_ref.table.as_str());
        let qualified_name = format!("{}.{}.{}", table_ref.schema, table_component, column.name);
        projections.push(ScanProjection::from(StoreProjection::with_alias(
            LogicalFieldId::for_user(table.table.table_id(), column.field_id),
            qualified_name.clone(),
        )));
        fields.push(Field::new(
            qualified_name,
            column.data_type.clone(),
            column.nullable,
        ));
    }

    let schema = Arc::new(Schema::new(fields));

    let filter_field_id = table.schema.first_field_id().unwrap_or(ROW_ID_FIELD_ID);
    let filter_expr = crate::translation::expression::full_table_scan_filter(filter_field_id);

    let mut raw_batches = Vec::new();
    table.table.scan_stream(
        projections,
        &filter_expr,
        ScanStreamOptions {
            include_nulls: true,
            ..ScanStreamOptions::default()
        },
        |batch| {
            raw_batches.push(batch);
        },
    )?;

    let mut normalized_batches = Vec::with_capacity(raw_batches.len());
    for batch in raw_batches {
        let normalized = RecordBatch::try_new(Arc::clone(&schema), batch.columns().to_vec())
            .map_err(|err| {
                Error::Internal(format!(
                    "failed to align scan batch for table '{}': {}",
                    table_ref.qualified_name(),
                    err
                ))
            })?;
        normalized_batches.push(normalized);
    }

    Ok(TableCrossProductData {
        schema,
        batches: normalized_batches,
        column_counts: vec![table.schema.columns.len()],
    })
}

fn build_cross_product_column_lookup(
    schema: &Schema,
    tables: &[llkv_plan::TableRef],
    column_counts: &[usize],
) -> FxHashMap<String, usize> {
    debug_assert_eq!(tables.len(), column_counts.len());

    let mut column_occurrences: FxHashMap<String, usize> = FxHashMap::default();
    let mut table_column_counts: FxHashMap<String, usize> = FxHashMap::default();
    for field in schema.fields() {
        let column_name = extract_column_name(field.name());
        *column_occurrences.entry(column_name).or_insert(0) += 1;
        if let Some(pair) = table_column_suffix(field.name()) {
            *table_column_counts.entry(pair).or_insert(0) += 1;
        }
    }

    let mut base_table_totals: FxHashMap<String, usize> = FxHashMap::default();
    let mut base_table_unaliased: FxHashMap<String, usize> = FxHashMap::default();
    for table_ref in tables {
        let key = base_table_key(table_ref);
        *base_table_totals.entry(key.clone()).or_insert(0) += 1;
        if table_ref.alias.is_none() {
            *base_table_unaliased.entry(key).or_insert(0) += 1;
        }
    }

    let mut lookup = FxHashMap::default();

    if tables.is_empty() || column_counts.is_empty() {
        for (idx, field) in schema.fields().iter().enumerate() {
            let field_name_lower = field.name().to_ascii_lowercase();
            lookup.entry(field_name_lower).or_insert(idx);

            let trimmed_lower = field.name().trim_start_matches('.').to_ascii_lowercase();
            lookup.entry(trimmed_lower).or_insert(idx);

            if let Some(pair) = table_column_suffix(field.name())
                && table_column_counts.get(&pair).copied().unwrap_or(0) == 1
            {
                lookup.entry(pair).or_insert(idx);
            }

            let column_name = extract_column_name(field.name());
            if column_occurrences.get(&column_name).copied().unwrap_or(0) == 1 {
                lookup.entry(column_name).or_insert(idx);
            }
        }
        return lookup;
    }

    let mut offset = 0usize;
    for (table_ref, &count) in tables.iter().zip(column_counts.iter()) {
        let alias_lower = table_ref
            .alias
            .as_ref()
            .map(|alias| alias.to_ascii_lowercase());
        let table_lower = table_ref.table.to_ascii_lowercase();
        let schema_lower = table_ref.schema.to_ascii_lowercase();
        let base_key = base_table_key(table_ref);
        let total_refs = base_table_totals.get(&base_key).copied().unwrap_or(0);
        let unaliased_refs = base_table_unaliased.get(&base_key).copied().unwrap_or(0);

        let allow_base_mapping = if table_ref.alias.is_none() {
            unaliased_refs == 1
        } else {
            unaliased_refs == 0 && total_refs == 1
        };

        let mut table_keys: Vec<String> = Vec::new();

        if let Some(alias) = &alias_lower {
            table_keys.push(alias.clone());
            if !schema_lower.is_empty() {
                table_keys.push(format!("{}.{}", schema_lower, alias));
            }
        }

        if allow_base_mapping {
            table_keys.push(table_lower.clone());
            if !schema_lower.is_empty() {
                table_keys.push(format!("{}.{}", schema_lower, table_lower));
            }
        }

        for local_idx in 0..count {
            let field_index = offset + local_idx;
            let field = schema.field(field_index);
            let field_name_lower = field.name().to_ascii_lowercase();
            lookup.entry(field_name_lower).or_insert(field_index);

            let trimmed_lower = field.name().trim_start_matches('.').to_ascii_lowercase();
            lookup.entry(trimmed_lower).or_insert(field_index);

            let column_name = extract_column_name(field.name());
            for table_key in &table_keys {
                lookup
                    .entry(format!("{}.{}", table_key, column_name))
                    .or_insert(field_index);
            }

            if column_occurrences.get(&column_name).copied().unwrap_or(0) == 1 {
                lookup.entry(column_name.clone()).or_insert(field_index);
            }

            if table_keys.is_empty()
                && let Some(pair) = table_column_suffix(field.name())
                && table_column_counts.get(&pair).copied().unwrap_or(0) == 1
            {
                lookup.entry(pair).or_insert(field_index);
            }
        }

        offset = offset.saturating_add(count);
    }

    lookup
}

fn base_table_key(table_ref: &llkv_plan::TableRef) -> String {
    let schema_lower = table_ref.schema.to_ascii_lowercase();
    let table_lower = table_ref.table.to_ascii_lowercase();
    if schema_lower.is_empty() {
        table_lower
    } else {
        format!("{}.{}", schema_lower, table_lower)
    }
}

fn extract_column_name(name: &str) -> String {
    name.trim_start_matches('.')
        .rsplit('.')
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase()
}

fn table_column_suffix(name: &str) -> Option<String> {
    let trimmed = name.trim_start_matches('.');
    let mut parts: Vec<&str> = trimmed.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let column = parts.pop()?.to_ascii_lowercase();
    let table = parts.pop()?.to_ascii_lowercase();
    Some(format!("{}.{}", table, column))
}

fn cross_join_table_batches(
    left: TableCrossProductData,
    right: TableCrossProductData,
) -> ExecutorResult<TableCrossProductData> {
    let TableCrossProductData {
        schema: left_schema,
        batches: left_batches,
        column_counts: mut left_counts,
    } = left;
    let TableCrossProductData {
        schema: right_schema,
        batches: right_batches,
        column_counts: right_counts,
    } = right;

    let combined_fields: Vec<Field> = left_schema
        .fields()
        .iter()
        .chain(right_schema.fields().iter())
        .map(|field| field.as_ref().clone())
        .collect();

    let mut column_counts = Vec::with_capacity(left_counts.len() + right_counts.len());
    column_counts.append(&mut left_counts);
    column_counts.extend(right_counts);

    let combined_schema = Arc::new(Schema::new(combined_fields));

    let left_has_rows = left_batches.iter().any(|batch| batch.num_rows() > 0);
    let right_has_rows = right_batches.iter().any(|batch| batch.num_rows() > 0);

    if !left_has_rows || !right_has_rows {
        return Ok(TableCrossProductData {
            schema: combined_schema,
            batches: Vec::new(),
            column_counts,
        });
    }

    let mut output_batches = Vec::new();
    for left_batch in &left_batches {
        if left_batch.num_rows() == 0 {
            continue;
        }
        for right_batch in &right_batches {
            if right_batch.num_rows() == 0 {
                continue;
            }

            let batch =
                cross_join_pair(left_batch, right_batch, &combined_schema).map_err(|err| {
                    Error::Internal(format!("failed to build cross join batch: {err}"))
                })?;
            output_batches.push(batch);
        }
    }

    Ok(TableCrossProductData {
        schema: combined_schema,
        batches: output_batches,
        column_counts,
    })
}

#[derive(Default)]
struct DistinctState {
    seen: FxHashSet<CanonicalRow>,
}

impl DistinctState {
    fn insert(&mut self, row: CanonicalRow) -> bool {
        self.seen.insert(row)
    }
}

fn distinct_filter_batch(
    batch: RecordBatch,
    state: &mut DistinctState,
) -> ExecutorResult<Option<RecordBatch>> {
    if batch.num_rows() == 0 {
        return Ok(None);
    }

    let mut keep_flags = Vec::with_capacity(batch.num_rows());
    let mut keep_count = 0usize;

    for row_idx in 0..batch.num_rows() {
        let row = CanonicalRow::from_batch(&batch, row_idx)?;
        if state.insert(row) {
            keep_flags.push(true);
            keep_count += 1;
        } else {
            keep_flags.push(false);
        }
    }

    if keep_count == 0 {
        return Ok(None);
    }

    if keep_count == batch.num_rows() {
        return Ok(Some(batch));
    }

    let mut builder = BooleanBuilder::with_capacity(batch.num_rows());
    for flag in keep_flags {
        builder.append_value(flag);
    }
    let mask = Arc::new(builder.finish());

    let filtered = filter_record_batch(&batch, &mask).map_err(|err| {
        Error::InvalidArgumentError(format!("failed to apply DISTINCT filter: {err}"))
    })?;

    Ok(Some(filtered))
}

fn sort_record_batch_with_order(
    schema: &Arc<Schema>,
    batch: &RecordBatch,
    order_by: &[OrderByPlan],
) -> ExecutorResult<RecordBatch> {
    if order_by.is_empty() {
        return Ok(batch.clone());
    }

    let mut sort_columns: Vec<SortColumn> = Vec::with_capacity(order_by.len());

    for order in order_by {
        let column_index = match &order.target {
            OrderTarget::Column(name) => schema.index_of(name).map_err(|_| {
                Error::InvalidArgumentError(format!(
                    "ORDER BY references unknown column '{}'",
                    name
                ))
            })?,
            OrderTarget::Index(idx) => {
                if *idx >= batch.num_columns() {
                    return Err(Error::InvalidArgumentError(format!(
                        "ORDER BY position {} is out of bounds for {} columns",
                        idx + 1,
                        batch.num_columns()
                    )));
                }
                *idx
            }
            OrderTarget::All => {
                return Err(Error::InvalidArgumentError(
                    "ORDER BY ALL should be expanded before sorting".into(),
                ));
            }
        };

        let source_array = batch.column(column_index);

        let values: ArrayRef = match order.sort_type {
            OrderSortType::Native => Arc::clone(source_array),
            OrderSortType::CastTextToInteger => {
                let strings = source_array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "ORDER BY CAST expects the underlying column to be TEXT".into(),
                        )
                    })?;
                let mut builder = Int64Builder::with_capacity(strings.len());
                for i in 0..strings.len() {
                    if strings.is_null(i) {
                        builder.append_null();
                    } else {
                        match strings.value(i).parse::<i64>() {
                            Ok(value) => builder.append_value(value),
                            Err(_) => builder.append_null(),
                        }
                    }
                }
                Arc::new(builder.finish()) as ArrayRef
            }
        };

        let sort_options = SortOptions {
            descending: !order.ascending,
            nulls_first: order.nulls_first,
        };

        sort_columns.push(SortColumn {
            values,
            options: Some(sort_options),
        });
    }

    let indices = lexsort_to_indices(&sort_columns, None).map_err(|err| {
        Error::InvalidArgumentError(format!("failed to compute ORDER BY indices: {err}"))
    })?;

    let perm = indices
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Internal("ORDER BY sorting produced unexpected index type".into()))?;

    let mut reordered_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for col_idx in 0..batch.num_columns() {
        let reordered = take(batch.column(col_idx), perm, None).map_err(|err| {
            Error::InvalidArgumentError(format!(
                "failed to apply ORDER BY permutation to column {col_idx}: {err}"
            ))
        })?;
        reordered_columns.push(reordered);
    }

    RecordBatch::try_new(Arc::clone(schema), reordered_columns)
        .map_err(|err| Error::Internal(format!("failed to build reordered ORDER BY batch: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use llkv_expr::expr::BinaryOp;
    use std::sync::Arc;

    #[test]
    fn cross_product_context_evaluates_expressions() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("main.tab2.a", DataType::Int64, false),
            Field::new("main.tab2.b", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
            ],
        )
        .expect("valid batch");

        let lookup = build_cross_product_column_lookup(schema.as_ref(), &[], &[]);
        let mut ctx = CrossProductExpressionContext::new(schema.as_ref(), lookup)
            .expect("context builds from schema");

        let literal_expr: ScalarExpr<String> = ScalarExpr::literal(67);
        let literal = ctx
            .evaluate(&literal_expr, &batch)
            .expect("literal evaluation succeeds");
        let literal_array = literal
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 literal result");
        assert_eq!(literal_array.len(), 3);
        assert!(literal_array.iter().all(|value| value == Some(67)));

        let add_expr = ScalarExpr::binary(
            ScalarExpr::column("tab2.a".to_string()),
            BinaryOp::Add,
            ScalarExpr::literal(5),
        );
        let added = ctx
            .evaluate(&add_expr, &batch)
            .expect("column addition succeeds");
        let added_array = added
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 addition result");
        assert_eq!(added_array.values(), &[6, 7, 8]);
    }

    #[test]
    fn cross_product_handles_more_than_two_tables() {
        let schema_a = Arc::new(Schema::new(vec![Field::new(
            "main.t1.a",
            DataType::Int64,
            false,
        )]));
        let schema_b = Arc::new(Schema::new(vec![Field::new(
            "main.t2.b",
            DataType::Int64,
            false,
        )]));
        let schema_c = Arc::new(Schema::new(vec![Field::new(
            "main.t3.c",
            DataType::Int64,
            false,
        )]));

        let batch_a = RecordBatch::try_new(
            Arc::clone(&schema_a),
            vec![Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef],
        )
        .expect("valid batch");
        let batch_b = RecordBatch::try_new(
            Arc::clone(&schema_b),
            vec![Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef],
        )
        .expect("valid batch");
        let batch_c = RecordBatch::try_new(
            Arc::clone(&schema_c),
            vec![Arc::new(Int64Array::from(vec![100])) as ArrayRef],
        )
        .expect("valid batch");

        let data_a = TableCrossProductData {
            schema: schema_a,
            batches: vec![batch_a],
            column_counts: vec![1],
        };
        let data_b = TableCrossProductData {
            schema: schema_b,
            batches: vec![batch_b],
            column_counts: vec![1],
        };
        let data_c = TableCrossProductData {
            schema: schema_c,
            batches: vec![batch_c],
            column_counts: vec![1],
        };

        let ab = cross_join_table_batches(data_a, data_b).expect("two-table product");
        assert_eq!(ab.schema.fields().len(), 2);
        assert_eq!(ab.batches.len(), 1);
        assert_eq!(ab.batches[0].num_rows(), 6);

        let abc = cross_join_table_batches(ab, data_c).expect("three-table product");
        assert_eq!(abc.schema.fields().len(), 3);
        assert_eq!(abc.batches.len(), 1);

        let final_batch = &abc.batches[0];
        assert_eq!(final_batch.num_rows(), 6);

        let col_a = final_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("left column values");
        assert_eq!(col_a.values(), &[1, 1, 1, 2, 2, 2]);

        let col_b = final_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("middle column values");
        assert_eq!(col_b.values(), &[10, 20, 30, 10, 20, 30]);

        let col_c = final_batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("right column values");
        assert_eq!(col_c.values(), &[100, 100, 100, 100, 100, 100]);
    }
}
