use arrow::array::{Array, ArrayRef, Int64Builder, RecordBatch};
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
use simd_r_drive_entry_handle::EntryHandle;
use std::collections::HashMap;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
        let table = self.provider.get_table(&plan.table)?;
        let display_name = plan.table.clone();

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
            ScalarExpr::Column(_) | ScalarExpr::Literal(_) => false,
        }
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

        let options = if let Some(order_plan) = &plan.order_by {
            let order_spec = resolve_scan_order(table_ref, &projections, order_plan)?;
            if row_filter.is_some() {
                println!("executor: applying row filter with order_by");
            }
            ScanStreamOptions {
                include_nulls: true,
                order: Some(order_spec),
                row_id_filter: row_filter.clone(),
            }
        } else {
            if row_filter.is_some() {
                println!("executor: applying row filter");
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
            row_id_filter: None,
        };

        let mut states: Vec<AggregateState> = Vec::with_capacity(specs.len());
        let mut count_star_override: Option<i64> = None;
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if !had_filter {
            if total_rows > i64::MAX as u64 {
                return Err(Error::InvalidArgumentError(
                    "COUNT(*) result exceeds supported range".into(),
                ));
            }
            count_star_override = Some(total_rows as i64);
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
                    AggregateKind::CountStar => count_star_override,
                    _ => None,
                },
            });
        }

        let mut error: Option<Error> = None;
        match table
            .table
            .scan_stream(
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
            }) {
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
                SelectProjection::AllColumns => {
                    return Err(Error::InvalidArgumentError(
                        "AllColumns projection not supported with computed aggregates".into(),
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
    ) -> ExecutorResult<HashMap<String, i64>> {
        use llkv_expr::expr::AggregateCall;

        let table_ref = table.as_ref();
        let mut results = HashMap::new();

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
        aggregates: &HashMap<String, i64>,
    ) -> ExecutorResult<i64> {
        use llkv_expr::expr::BinaryOp;
        use llkv_expr::literal::Literal;

        match expr {
            ScalarExpr::Literal(Literal::Integer(v)) => Ok(*v as i64),
            ScalarExpr::Literal(Literal::Float(v)) => Ok(*v as i64),
            ScalarExpr::Literal(Literal::String(_)) => Err(Error::InvalidArgumentError(
                "String literals not supported in aggregate expressions".into(),
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
    },
    Aggregation {
        batch: RecordBatch,
    },
}

impl<P> SelectExecution<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new_projection(
        table_name: String,
        schema: Arc<Schema>,
        table: Arc<ExecutorTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions<P>,
        full_table_scan: bool,
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
            },
        }
    }

    fn new_single_batch(table_name: String, schema: Arc<Schema>, batch: RecordBatch) -> Self {
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
                let include_nulls = options.include_nulls;
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
                        if capture_nulls_first {
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
                if include_nulls && full_table_scan && produced_rows < total_rows {
                    let missing = total_rows - produced_rows;
                    if missing > 0 {
                        null_batches = synthesize_null_scan(Arc::clone(&schema), missing)?;
                    }
                }

                if capture_nulls_first {
                    for batch in null_batches {
                        on_batch(batch)?;
                    }
                    for batch in buffered_batches {
                        on_batch(batch)?;
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
}

pub struct ExecutorSchema {
    pub columns: Vec<ExecutorColumn>,
    pub lookup: HashMap<String, usize>,
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
    pub field_id: FieldId,
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
    }
}
