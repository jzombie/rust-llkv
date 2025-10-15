use arrow::array::{Array, ArrayRef, Int64Builder, RecordBatch, StringArray, UInt32Array};
use arrow::compute::{SortColumn, SortOptions, concat_batches, lexsort_to_indices, take};
use arrow::datatypes::{DataType, Schema};
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec, AggregateState};
use llkv_column_map::store::Projection as StoreProjection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_plan::{
    AggregateExpr, AggregateFunction, OrderByPlan, OrderSortType, OrderTarget, PlanValue,
    SelectPlan, SelectProjection,
};
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{
    RowIdFilter, ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection,
    ScanStreamOptions,
};
use llkv_table::types::FieldId;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::fmt;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub type ExecutorResult<T> = Result<T, Error>;

mod projections;
mod schema;
pub use projections::{build_projected_columns, build_wildcard_projections};
pub use schema::schema_for_projections;

/// Trait for providing table access to the executor.
pub trait TableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> ExecutorResult<Arc<ExecutorTable<P>>>;
}

/// Query executor that executes SELECT plans.
pub struct QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    provider: Arc<dyn TableProvider<P>>,
}

impl<P> QueryExecutor<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(provider: Arc<dyn TableProvider<P>>) -> Self {
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
            ScalarExpr::GetField { base, .. } => Self::expr_contains_aggregate(base),
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
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
            .map_err(|e| Error::Internal(format!("failed to create record batch: {}", e)))?;

        Ok(SelectExecution::new_single_batch(
            String::new(), // No table name
            schema,
            batch,
        ))
    }

    /// Convert a Literal to an Arrow array (recursive for nested structs)
    fn literal_to_array(lit: &llkv_expr::literal::Literal) -> ExecutorResult<(DataType, ArrayRef)> {
        use arrow::array::{
            ArrayRef, Float64Array, Int64Array, StringArray, StructArray, new_null_array,
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

        // Get all tables
        let mut tables = Vec::new();
        for table_ref in &plan.tables {
            let qualified_name = table_ref.qualified_name();
            let table = self.provider.get_table(&qualified_name)?;
            tables.push((table_ref.clone(), table));
        }

        // For now, support only 2-table cross product
        if tables.len() > 2 {
            return Err(Error::InvalidArgumentError(
                "cross products with more than 2 tables not yet supported".into(),
            ));
        }

        let (left_ref, left_table) = &tables[0];
        let (right_ref, right_table) = &tables[1];

        // Build the cross product using llkv-join crate
        // For cross product, we pass empty join keys = Cartesian product
        use llkv_join::{JoinOptions, JoinType, TableJoinExt};

        let mut result_batches = Vec::new();
        left_table.table.join_stream(
            &right_table.table,
            &[], // Empty join keys = cross product
            &JoinOptions {
                join_type: JoinType::Inner,
                ..Default::default()
            },
            |batch| {
                result_batches.push(batch);
            },
        )?;

        // Build combined schema with qualified column names
        let mut combined_fields = Vec::new();

        // Add left table columns with schema.table.column prefix
        for col in &left_table.schema.columns {
            let qualified_name = format!("{}.{}.{}", left_ref.schema, left_ref.table, col.name);
            combined_fields.push(arrow::datatypes::Field::new(
                qualified_name,
                col.data_type.clone(),
                col.nullable,
            ));
        }

        // Add right table columns with schema.table.column prefix
        for col in &right_table.schema.columns {
            let qualified_name = format!("{}.{}.{}", right_ref.schema, right_ref.table, col.name);
            combined_fields.push(arrow::datatypes::Field::new(
                qualified_name,
                col.data_type.clone(),
                col.nullable,
            ));
        }

        let combined_schema = Arc::new(Schema::new(combined_fields));

        // Combine all result batches with the combined schema (renames columns)
        let mut combined_batch = if result_batches.is_empty() {
            RecordBatch::new_empty(Arc::clone(&combined_schema))
        } else if result_batches.len() == 1 {
            let batch = result_batches.into_iter().next().unwrap();
            // The batch from join has original column names, we need to apply our qualified schema
            RecordBatch::try_new(Arc::clone(&combined_schema), batch.columns().to_vec()).map_err(
                |e| {
                    Error::Internal(format!(
                        "failed to create batch with qualified names: {}",
                        e
                    ))
                },
            )?
        } else {
            // First concatenate with original schema
            let original_batch = concat_batches(&result_batches[0].schema(), &result_batches)
                .map_err(|e| Error::Internal(format!("failed to concatenate batches: {}", e)))?;
            // Then apply qualified schema
            RecordBatch::try_new(
                Arc::clone(&combined_schema),
                original_batch.columns().to_vec(),
            )
            .map_err(|e| {
                Error::Internal(format!(
                    "failed to create batch with qualified names: {}",
                    e
                ))
            })?
        };

        // Apply SELECT projections if specified
        if !plan.projections.is_empty() {
            let mut selected_fields = Vec::new();
            let mut selected_columns = Vec::new();

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
                        if let Some((idx, field)) = combined_schema
                            .fields()
                            .iter()
                            .enumerate()
                            .find(|(_, f)| f.name().to_ascii_lowercase() == col_name)
                        {
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
                        // Handle simple column references (like s1.t1.t)
                        if let ScalarExpr::Column(col_name) = expr {
                            let col_name_lower = col_name.to_ascii_lowercase();
                            if let Some((idx, field)) = combined_schema
                                .fields()
                                .iter()
                                .enumerate()
                                .find(|(_, f)| f.name().to_ascii_lowercase() == col_name_lower)
                            {
                                selected_fields.push(Arc::new(arrow::datatypes::Field::new(
                                    alias.clone(),
                                    field.data_type().clone(),
                                    field.is_nullable(),
                                )));
                                selected_columns.push(combined_batch.column(idx).clone());
                            } else {
                                return Err(Error::InvalidArgumentError(format!(
                                    "column '{}' not found in cross product result",
                                    col_name
                                )));
                            }
                        } else {
                            return Err(Error::InvalidArgumentError(
                                "complex computed projections not yet supported in cross products"
                                    .into(),
                            ));
                        }
                    }
                }
            }

            let projected_schema = Arc::new(Schema::new(selected_fields));
            combined_batch = RecordBatch::try_new(projected_schema, selected_columns)
                .map_err(|e| Error::Internal(format!("failed to apply projections: {}", e)))?;
        }

        Ok(SelectExecution::new_single_batch(
            format!(
                "{},{}",
                left_ref.qualified_name(),
                right_ref.qualified_name()
            ),
            combined_batch.schema(),
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
            Some(expr) => (translate_predicate(expr, table_ref.schema.as_ref())?, false),
            None => {
                let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                (full_table_scan_filter(field_id), true)
            }
        };

        let order_items = plan.order_by.clone();
        let physical_order = if let Some(first) = order_items.first() {
            Some(resolve_scan_order(table_ref, &projections, first)?)
        } else {
            None
        };

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
            order_items,
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
                } => {
                    let col = table_ref.schema.resolve(&column).ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "unknown column '{}' in aggregate",
                            column
                        ))
                    })?;
                    let kind = match function {
                        AggregateFunction::Count => AggregateKind::CountField {
                            field_id: col.field_id,
                        },
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
                        AggregateFunction::CountNulls => AggregateKind::CountNulls {
                            field_id: col.field_id,
                        },
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
            Some(expr) => translate_predicate(expr, table.schema.as_ref())?,
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform aggregate scan".into(),
                    )
                })?;
                full_table_scan_filter(field_id)
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
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
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
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
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
            ScalarExpr::GetField { base, .. } => {
                Self::collect_aggregates(base, aggregates);
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
            Some(expr) => translate_predicate(expr.clone(), table_ref.schema.as_ref())?,
            None => {
                let field_id = table_ref.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform aggregate scan".into(),
                    )
                })?;
                full_table_scan_filter(field_id)
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
            ScalarExpr::GetField { .. } => Err(Error::InvalidArgumentError(
                "GetField not supported in aggregate-only expressions".into(),
            )),
        }
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
                let needs_post_sort = order_by.len() > 1;
                let collect_batches = needs_post_sort || capture_nulls_first;
                let include_nulls = options.include_nulls;
                let has_row_id_filter = options.row_id_filter.is_some();
                let scan_options = options;
                let mut buffered_batches: Vec<RecordBatch> = Vec::new();
                table
                    .table
                    .scan_stream(projections, &filter_expr, scan_options, |batch| {
                        if error.is_some() {
                            return;
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
                    if total_rows > 0 {
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
                if include_nulls
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

    pub fn collect_rows(self) -> ExecutorResult<RowBatch> {
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
        Ok(RowBatch { columns, rows })
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

pub struct ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    // underlying physical table from llkv_table
    pub table: Arc<llkv_table::table::Table<P>>,
    pub schema: Arc<ExecutorSchema>,
    pub next_row_id: AtomicU64,
    pub total_rows: AtomicU64,
    pub multi_column_uniques: RwLock<Vec<ExecutorMultiColumnUnique>>,
}

pub struct ExecutorSchema {
    pub columns: Vec<ExecutorColumn>,
    pub lookup: FxHashMap<String, usize>,
}

impl ExecutorSchema {
    pub fn resolve(&self, name: &str) -> Option<&ExecutorColumn> {
        let normalized = name.to_ascii_lowercase();
        self.lookup
            .get(&normalized)
            .and_then(|idx| self.columns.get(*idx))
    }

    pub fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|col| col.field_id)
    }

    pub fn column_by_field_id(&self, field_id: FieldId) -> Option<&ExecutorColumn> {
        self.columns.iter().find(|col| col.field_id == field_id)
    }
}

#[derive(Clone)]
pub struct ExecutorColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub field_id: FieldId,
    pub check_expr: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorMultiColumnUnique {
    pub index_name: Option<String>,
    pub column_indices: Vec<usize>,
}

impl<P> ExecutorTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn multi_column_uniques(&self) -> Vec<ExecutorMultiColumnUnique> {
        self.multi_column_uniques.read().unwrap().clone()
    }

    pub fn set_multi_column_uniques(&self, uniques: Vec<ExecutorMultiColumnUnique>) {
        *self.multi_column_uniques.write().unwrap() = uniques;
    }

    pub fn add_multi_column_unique(&self, unique: ExecutorMultiColumnUnique) {
        let mut guard = self.multi_column_uniques.write().unwrap();
        if !guard
            .iter()
            .any(|existing| existing.column_indices == unique.column_indices)
        {
            guard.push(unique);
        }
    }
}

// Re-export from llkv-plan
// PlanValue is the plan-level value type; do not re-export higher-level prefixed symbols.

// Export executor-local types with explicit Exec-prefixed names to avoid
// colliding with Arrow / storage types imported above.
// Executor types are public `ExecutorColumn`, `ExecutorSchema`, `ExecutorTable`.
// No short `Exec*` aliases to avoid confusion.

pub struct RowBatch {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<PlanValue>>,
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

fn full_table_scan_filter(field_id: FieldId) -> LlkvExpr<'static, FieldId> {
    LlkvExpr::Pred(Filter {
        field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
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

// Translate predicate from column names to field IDs
fn translate_predicate(
    expr: llkv_expr::expr::Expr<'static, String>,
    schema: &ExecutorSchema,
) -> ExecutorResult<llkv_expr::expr::Expr<'static, FieldId>> {
    use llkv_expr::expr::Expr;
    match expr {
        Expr::And(exprs) => {
            let translated: Result<Vec<_>, _> = exprs
                .into_iter()
                .map(|e| translate_predicate(e, schema))
                .collect();
            Ok(Expr::And(translated?))
        }
        Expr::Or(exprs) => {
            let translated: Result<Vec<_>, _> = exprs
                .into_iter()
                .map(|e| translate_predicate(e, schema))
                .collect();
            Ok(Expr::Or(translated?))
        }
        Expr::Not(inner) => {
            let translated = translate_predicate(*inner, schema)?;
            Ok(Expr::Not(Box::new(translated)))
        }
        Expr::Pred(filter) => {
            let column = schema.resolve(&filter.field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{}'", filter.field_id))
            })?;
            Ok(Expr::Pred(Filter {
                field_id: column.field_id,
                op: filter.op,
            }))
        }
        Expr::Compare { left, op, right } => Ok(Expr::Compare {
            left: translate_scalar(&left, schema)?,
            op,
            right: translate_scalar(&right, schema)?,
        }),
    }
}

// Translate scalar expressions
fn translate_scalar(
    expr: &ScalarExpr<String>,
    schema: &ExecutorSchema,
) -> ExecutorResult<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Column(name) => {
            let column = schema
                .resolve(name)
                .ok_or_else(|| Error::InvalidArgumentError(format!("unknown column '{}'", name)))?;
            Ok(ScalarExpr::Column(column.field_id))
        }
        ScalarExpr::Binary { left, op, right } => Ok(ScalarExpr::Binary {
            left: Box::new(translate_scalar(left, schema)?),
            op: *op,
            right: Box::new(translate_scalar(right, schema)?),
        }),
        ScalarExpr::Aggregate(agg) => {
            // Translate column names in aggregate calls to field IDs
            use llkv_expr::expr::AggregateCall;
            let translated_agg = match agg {
                AggregateCall::CountStar => AggregateCall::CountStar,
                AggregateCall::Count(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", name))
                    })?;
                    AggregateCall::Count(column.field_id)
                }
                AggregateCall::Sum(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", name))
                    })?;
                    AggregateCall::Sum(column.field_id)
                }
                AggregateCall::Min(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", name))
                    })?;
                    AggregateCall::Min(column.field_id)
                }
                AggregateCall::Max(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", name))
                    })?;
                    AggregateCall::Max(column.field_id)
                }
                AggregateCall::CountNulls(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{}'", name))
                    })?;
                    AggregateCall::CountNulls(column.field_id)
                }
            };
            Ok(ScalarExpr::Aggregate(translated_agg))
        }
        ScalarExpr::GetField { base, field_name } => Ok(ScalarExpr::GetField {
            base: Box::new(translate_scalar(base, schema)?),
            field_name: field_name.clone(),
        }),
    }
}
