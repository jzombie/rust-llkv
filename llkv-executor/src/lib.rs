use arrow::array::{
    Array, ArrayRef, Date32Array, Float64Array, Int64Array, Int64Builder, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_column_map::store::Projection as StoreProjection;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_plan::{
    AggregateExpr, AggregateFunction, OrderByPlan, OrderSortType, OrderTarget, SelectPlan,
};
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{
    ScanOrderDirection, ScanOrderSpec, ScanOrderTransform, ScanProjection, ScanStreamOptions, Table,
};
use llkv_table::types::{FieldId, TableId};
use simd_r_drive_entry_handle::EntryHandle;
use std::collections::HashMap;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub type DslResult<T> = Result<T, Error>;

mod projections;
mod schema;
pub use projections::{build_projected_columns, build_wildcard_projections};
pub use schema::schema_for_projections;

/// Trait for providing table access to the executor.
pub trait TableProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> DslResult<Arc<DslTable<P>>>;
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

    pub fn execute_select(&self, plan: SelectPlan) -> DslResult<SelectExecution<P>> {
        let table = self.provider.get_table(&plan.table)?;
        let display_name = plan.table.clone();

        if !plan.aggregates.is_empty() {
            self.execute_aggregates(table, display_name, plan)
        } else {
            self.execute_projection(table, display_name, plan)
        }
    }

    fn execute_projection(
        &self,
        table: Arc<DslTable<P>>,
        display_name: String,
        plan: SelectPlan,
    ) -> DslResult<SelectExecution<P>> {
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
            ScanStreamOptions {
                include_nulls: true,
                order: Some(order_spec),
            }
        } else {
            ScanStreamOptions {
                include_nulls: true,
                order: None,
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
        table: Arc<DslTable<P>>,
        display_name: String,
        plan: SelectPlan,
    ) -> DslResult<SelectExecution<P>> {
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
            ..Default::default()
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
            .scan_stream(projections, &filter_expr, options, |batch| {
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
        table: Arc<DslTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions,
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
        table: Arc<DslTable<P>>,
        projections: Vec<ScanProjection>,
        filter_expr: LlkvExpr<'static, FieldId>,
        options: ScanStreamOptions,
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

    pub fn stream(self, mut on_batch: impl FnMut(RecordBatch) -> DslResult<()>) -> DslResult<()> {
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
                let mut buffered_batches: Vec<RecordBatch> = Vec::new();
                table
                    .table
                    .scan_stream(projections, &filter_expr, options, |batch| {
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
                if options.include_nulls && full_table_scan && produced_rows < total_rows {
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

    pub fn collect(self) -> DslResult<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        self.stream(|batch| {
            batches.push(batch);
            Ok(())
        })?;
        Ok(batches)
    }

    pub fn collect_rows(self) -> DslResult<RowBatch> {
        let schema = self.schema();
        let mut rows: Vec<Vec<DslValue>> = Vec::new();
        self.stream(|batch| {
            for row_idx in 0..batch.num_rows() {
                let mut row: Vec<DslValue> = Vec::with_capacity(batch.num_columns());
                for col_idx in 0..batch.num_columns() {
                    let value = dsl_value_from_array(batch.column(col_idx), row_idx)?;
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

    pub fn into_rows(self) -> DslResult<Vec<Vec<DslValue>>> {
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

pub struct DslTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub table: Arc<Table<P>>,
    pub schema: Arc<DslSchema>,
    pub next_row_id: AtomicU64,
    pub total_rows: AtomicU64,
}

pub struct DslSchema {
    pub columns: Vec<DslColumn>,
    pub lookup: HashMap<String, usize>,
}

impl DslSchema {
    pub fn resolve(&self, name: &str) -> Option<&DslColumn> {
        let normalized = name.to_ascii_lowercase();
        self.lookup
            .get(&normalized)
            .and_then(|idx| self.columns.get(*idx))
    }

    pub fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|col| col.field_id)
    }

    pub fn column_by_field_id(&self, field_id: FieldId) -> Option<&DslColumn> {
        self.columns.iter().find(|col| col.field_id == field_id)
    }
}

#[derive(Clone)]
pub struct DslColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub field_id: FieldId,
}

// Re-export from llkv-plan
pub use llkv_plan::DslValue;

pub struct RowBatch {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<DslValue>>,
}

#[derive(Clone)]
struct AggregateSpec {
    alias: String,
    kind: AggregateKind,
}

#[derive(Clone)]
enum AggregateKind {
    CountStar,
    CountField { field_id: FieldId },
    SumInt64 { field_id: FieldId },
    MinInt64 { field_id: FieldId },
    MaxInt64 { field_id: FieldId },
    CountNulls { field_id: FieldId },
}

impl AggregateKind {
    fn field_id(&self) -> Option<FieldId> {
        match self {
            AggregateKind::CountStar => None,
            AggregateKind::CountField { field_id }
            | AggregateKind::SumInt64 { field_id }
            | AggregateKind::MinInt64 { field_id }
            | AggregateKind::MaxInt64 { field_id }
            | AggregateKind::CountNulls { field_id } => Some(*field_id),
        }
    }
}

struct AggregateState {
    alias: String,
    accumulator: AggregateAccumulator,
    override_value: Option<i64>,
}

enum AggregateAccumulator {
    CountStar {
        value: i64,
    },
    CountColumn {
        column_index: usize,
        value: i64,
    },
    SumInt64 {
        column_index: usize,
        value: i64,
        saw_value: bool,
    },
    MinInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    MaxInt64 {
        column_index: usize,
        value: Option<i64>,
    },
    CountNulls {
        column_index: usize,
        non_null_rows: i64,
        total_rows: i64,
    },
}

impl AggregateAccumulator {
    /// Create accumulator using projection index (position in the batch columns)
    /// instead of table schema position. projection_idx is None for CountStar.
    fn new_with_projection_index(
        spec: &AggregateSpec,
        projection_idx: Option<usize>,
        total_rows_hint: Option<i64>,
    ) -> DslResult<Self> {
        match spec.kind {
            AggregateKind::CountStar => Ok(AggregateAccumulator::CountStar { value: 0 }),
            AggregateKind::CountField { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("CountField aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::CountColumn {
                    column_index: idx,
                    value: 0,
                })
            }
            AggregateKind::SumInt64 { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("SumInt64 aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::SumInt64 {
                    column_index: idx,
                    value: 0,
                    saw_value: false,
                })
            }
            AggregateKind::MinInt64 { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("MinInt64 aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::MinInt64 {
                    column_index: idx,
                    value: None,
                })
            }
            AggregateKind::MaxInt64 { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("MaxInt64 aggregate requires projection index".into())
                })?;
                Ok(AggregateAccumulator::MaxInt64 {
                    column_index: idx,
                    value: None,
                })
            }
            AggregateKind::CountNulls { .. } => {
                let idx = projection_idx.ok_or_else(|| {
                    Error::Internal("CountNulls aggregate requires projection index".into())
                })?;
                let total_rows = total_rows_hint.ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM(CASE WHEN ... IS NULL ...) with WHERE clauses is not supported yet"
                            .into(),
                    )
                })?;
                Ok(AggregateAccumulator::CountNulls {
                    column_index: idx,
                    non_null_rows: 0,
                    total_rows,
                })
            }
        }
    }

    #[allow(dead_code)]
    fn new(
        schema: &DslSchema,
        spec: &AggregateSpec,
        total_rows_hint: Option<i64>,
    ) -> DslResult<Self> {
        match spec.kind {
            AggregateKind::CountStar => Ok(AggregateAccumulator::CountStar { value: 0 }),
            AggregateKind::CountField { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::CountColumn {
                    column_index: position,
                    value: 0,
                })
            }
            AggregateKind::SumInt64 { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::SumInt64 {
                    column_index: position,
                    value: 0,
                    saw_value: false,
                })
            }
            AggregateKind::MinInt64 { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::MinInt64 {
                    column_index: position,
                    value: None,
                })
            }
            AggregateKind::MaxInt64 { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                Ok(AggregateAccumulator::MaxInt64 {
                    column_index: position,
                    value: None,
                })
            }
            AggregateKind::CountNulls { field_id } => {
                let position = schema
                    .columns
                    .iter()
                    .position(|c| c.field_id == field_id)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "aggregate referenced unknown column field id".into(),
                        )
                    })?;
                let total_rows = total_rows_hint.ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM(CASE WHEN ... IS NULL ...) with WHERE clauses is not supported yet"
                            .into(),
                    )
                })?;
                Ok(AggregateAccumulator::CountNulls {
                    column_index: position,
                    non_null_rows: 0,
                    total_rows,
                })
            }
        }
    }

    fn update(&mut self, batch: &RecordBatch) -> DslResult<()> {
        match self {
            AggregateAccumulator::CountStar { value } => {
                let rows = i64::try_from(batch.num_rows()).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *value = value.checked_add(rows).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
            AggregateAccumulator::CountColumn {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                let non_null = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *value = value.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
            AggregateAccumulator::SumInt64 {
                column_index,
                value,
                saw_value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "SUM aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *value = value.checked_add(v).ok_or_else(|| {
                            Error::InvalidArgumentError(
                                "SUM aggregate result exceeds i64 range".into(),
                            )
                        })?;
                        *saw_value = true;
                    }
                }
            }
            AggregateAccumulator::MinInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MIN aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *value = Some(match *value {
                            Some(current) => current.min(v),
                            None => v,
                        });
                    }
                }
            }
            AggregateAccumulator::MaxInt64 {
                column_index,
                value,
            } => {
                let array = batch.column(*column_index);
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "MAX aggregate expected an INT column in execution".into(),
                    )
                })?;
                for i in 0..array.len() {
                    if array.is_valid(i) {
                        let v = array.value(i);
                        *value = Some(match *value {
                            Some(current) => current.max(v),
                            None => v,
                        });
                    }
                }
            }
            AggregateAccumulator::CountNulls {
                column_index,
                non_null_rows,
                total_rows: _,
            } => {
                let array = batch.column(*column_index);
                let non_null = (0..array.len()).filter(|idx| array.is_valid(*idx)).count();
                let non_null = i64::try_from(non_null).map_err(|_| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
                *non_null_rows = non_null_rows.checked_add(non_null).ok_or_else(|| {
                    Error::InvalidArgumentError("COUNT result exceeds i64 range".into())
                })?;
            }
        }
        Ok(())
    }

    fn finalize(self) -> DslResult<(Field, ArrayRef)> {
        match self {
            AggregateAccumulator::CountStar { value } => {
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count", DataType::Int64, false), array))
            }
            AggregateAccumulator::CountColumn { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(value);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count", DataType::Int64, false), array))
            }
            AggregateAccumulator::SumInt64 {
                value, saw_value, ..
            } => {
                let mut builder = Int64Builder::with_capacity(1);
                if saw_value {
                    builder.append_value(value);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("sum", DataType::Int64, true), array))
            }
            AggregateAccumulator::MinInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("min", DataType::Int64, true), array))
            }
            AggregateAccumulator::MaxInt64 { value, .. } => {
                let mut builder = Int64Builder::with_capacity(1);
                if let Some(v) = value {
                    builder.append_value(v);
                } else {
                    builder.append_null();
                }
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("max", DataType::Int64, true), array))
            }
            AggregateAccumulator::CountNulls {
                non_null_rows,
                total_rows,
                ..
            } => {
                let nulls = total_rows.checked_sub(non_null_rows).ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "NULL-count aggregate observed more non-null rows than total rows".into(),
                    )
                })?;
                let mut builder = Int64Builder::with_capacity(1);
                builder.append_value(nulls);
                let array = Arc::new(builder.finish()) as ArrayRef;
                Ok((Field::new("count_nulls", DataType::Int64, false), array))
            }
        }
    }
}

impl AggregateState {
    fn update(&mut self, batch: &RecordBatch) -> DslResult<()> {
        self.accumulator.update(batch)
    }

    fn finalize(self) -> DslResult<(Field, ArrayRef)> {
        let (mut field, array) = self.accumulator.finalize()?;
        field = field.with_name(self.alias);
        if let Some(value) = self.override_value {
            let mut builder = Int64Builder::with_capacity(1);
            builder.append_value(value);
            let array = Arc::new(builder.finish()) as ArrayRef;
            return Ok((field, array));
        }
        Ok((field, array))
    }
}

fn resolve_scan_order<P>(
    table: &DslTable<P>,
    projections: &[ScanProjection],
    order_plan: &OrderByPlan,
) -> DslResult<ScanOrderSpec>
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

fn synthesize_null_scan(schema: Arc<Schema>, total_rows: u64) -> DslResult<Vec<RecordBatch>> {
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

fn dsl_value_from_array(array: &ArrayRef, index: usize) -> DslResult<DslValue> {
    if array.is_null(index) {
        return Ok(DslValue::Null);
    }
    match array.data_type() {
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                Error::InvalidArgumentError("expected Int64 array in INSERT SELECT".into())
            })?;
            Ok(DslValue::Integer(values.value(index)))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Float64 array in INSERT SELECT".into())
                })?;
            Ok(DslValue::Float(values.value(index)))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Utf8 array in INSERT SELECT".into())
                })?;
            Ok(DslValue::String(values.value(index).to_string()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Date32 array in INSERT SELECT".into())
                })?;
            Ok(DslValue::Integer(values.value(index) as i64))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported data type in INSERT SELECT: {other:?}"
        ))),
    }
}

// Translate predicate from column names to field IDs
fn translate_predicate(
    expr: llkv_expr::expr::Expr<'static, String>,
    schema: &DslSchema,
) -> DslResult<llkv_expr::expr::Expr<'static, FieldId>> {
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
    schema: &DslSchema,
) -> DslResult<ScalarExpr<FieldId>> {
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
    }
}
