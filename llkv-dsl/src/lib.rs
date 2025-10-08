#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    Array, ArrayRef, Date32Array, Date32Builder, Float64Array, Float64Builder, Int64Array,
    Int64Builder, StringArray, StringBuilder, UInt64Builder, new_null_array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{Projection as StoreProjection, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_expr::literal::Literal;
use llkv_result::Error;
use llkv_storage::pager::Pager;
use llkv_table::table::{ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::{FieldId, TableId};
use llkv_table::{ColMeta, SysCatalog, TableMeta, CATALOG_TABLE_ID};
use simd_r_drive_entry_handle::EntryHandle;
use time::{Date, Month};

pub type DslResult<T> = llkv_result::Result<T>;

/// Result of running a DSL statement.
#[derive(Debug, Clone)]
pub enum StatementResult {
    CreateTable {
        table_name: String,
    },
    Insert {
        table_name: String,
        rows_inserted: usize,
    },
    Update {
        table_name: String,
        rows_updated: usize,
    },
    Select {
        table_name: String,
        batches: Vec<RecordBatch>,
    },
    Transaction {
        kind: TransactionKind,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionKind {
    Begin,
    Commit,
    Rollback,
}

/// Value literal used by the DSL.
#[derive(Clone, Debug, PartialEq)]
pub enum DslValue {
    Null,
    Integer(i64),
    Float(f64),
    String(String),
}

impl From<&str> for DslValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for DslValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<i64> for DslValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for DslValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

/// Specification for creating a table.
#[derive(Clone, Debug)]
pub struct CreateTablePlan {
    pub name: String,
    pub if_not_exists: bool,
    pub columns: Vec<ColumnSpec>,
    pub source: Option<CreateTableSource>,
}

impl CreateTablePlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            if_not_exists: false,
            columns: Vec::new(),
            source: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColumnSpec {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

impl ColumnSpec {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

#[derive(Clone, Debug)]
pub enum CreateTableSource {
    Batches {
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    },
}

/// Plan describing an insert operation.
#[derive(Clone, Debug)]
pub struct InsertPlan {
    pub table: String,
    pub columns: Vec<String>,
    pub source: InsertSource,
}

#[derive(Clone, Debug)]
pub enum InsertSource {
    Rows(Vec<Vec<DslValue>>),
    Batches(Vec<RecordBatch>),
}

/// Plan describing an update operation.
#[derive(Clone, Debug)]
pub struct UpdatePlan {
    pub table: String,
    pub assignments: Vec<ColumnAssignment>,
}

#[derive(Clone, Debug)]
pub struct ColumnAssignment {
    pub column: String,
    pub value: DslValue,
}

/// Logical query plan consumed by the DSL execution engine.
#[derive(Clone, Debug)]
pub struct SelectPlan {
    pub table: String,
    pub projections: Vec<SelectProjection>,
    pub filter: Option<LlkvExpr<'static, String>>,
    pub aggregates: Vec<AggregateExpr>,
}

impl SelectPlan {
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            projections: Vec::new(),
            filter: None,
            aggregates: Vec::new(),
        }
    }

    pub fn with_projections(mut self, projections: Vec<SelectProjection>) -> Self {
        self.projections = projections;
        self
    }

    pub fn with_filter(mut self, filter: Option<LlkvExpr<'static, String>>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_aggregates(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.aggregates = aggregates;
        self
    }
}

#[derive(Clone, Debug)]
pub enum SelectProjection {
    AllColumns,
    Column {
        name: String,
        alias: Option<String>,
    },
    Computed {
        expr: ScalarExpr<String>,
        alias: String,
    },
}

#[derive(Clone, Debug)]
pub enum AggregateExpr {
    CountStar {
        alias: String,
    },
    Column {
        column: String,
        alias: String,
        function: AggregateFunction,
    },
}

#[derive(Clone, Debug)]
pub enum AggregateFunction {
    Count,
    SumInt64,
    MinInt64,
    MaxInt64,
    CountNulls,
}

impl AggregateExpr {
    pub fn count_star(alias: impl Into<String>) -> Self {
        Self::CountStar {
            alias: alias.into(),
        }
    }

    pub fn count_column(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::Count,
        }
    }

    pub fn sum_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::SumInt64,
        }
    }

    pub fn min_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::MinInt64,
        }
    }

    pub fn max_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::MaxInt64,
        }
    }

    pub fn count_nulls(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::CountNulls,
        }
    }
}

/// In-memory execution context shared by DSL queries.
pub struct DslContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    tables: RwLock<HashMap<String, Arc<DslTable<P>>>>,
    transaction_active: Mutex<bool>,
}

impl<P> DslContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        Self {
            pager,
            tables: RwLock::new(HashMap::new()),
            transaction_active: Mutex::new(false),
        }
    }

    pub fn create_table(&self, plan: CreateTablePlan) -> DslResult<StatementResult> {
        if plan.columns.is_empty() && plan.source.is_none() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires explicit columns or a source".into(),
            ));
        }

        let (display_name, canonical_name) = canonical_table_name(&plan.name)?;
        let exists = {
            let tables = self.tables.read().unwrap();
            tables.contains_key(&canonical_name)
        };
        if exists {
            if plan.if_not_exists {
                return Ok(StatementResult::CreateTable { table_name: display_name });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }

        match plan.source {
            Some(CreateTableSource::Batches { schema, batches }) => {
                self.create_table_from_batches(
                    display_name,
                    canonical_name,
                    schema,
                    batches,
                    plan.if_not_exists,
                )
            }
            None => self.create_table_from_columns(
                display_name,
                canonical_name,
                plan.columns,
                plan.if_not_exists,
            ),
        }
    }

    pub fn insert(&self, plan: InsertPlan) -> DslResult<StatementResult> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.source {
            InsertSource::Rows(rows) => self.insert_rows(table.as_ref(), display_name, rows, plan.columns),
            InsertSource::Batches(batches) => {
                self.insert_batches(table.as_ref(), display_name, batches, plan.columns)
            }
        }
    }

    pub fn update(&self, plan: UpdatePlan) -> DslResult<StatementResult> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        self.update_all_rows(table.as_ref(), display_name, plan.assignments)
    }

    pub fn begin_transaction(&self) -> DslResult<StatementResult> {
        let mut active = self.transaction_active.lock().unwrap();
        if *active {
            return Err(Error::InvalidArgumentError(
                "a transaction is already in progress".into(),
            ));
        }
        *active = true;
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Begin,
        })
    }

    pub fn commit_transaction(&self) -> DslResult<StatementResult> {
        let mut active = self.transaction_active.lock().unwrap();
        if !*active {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress".into(),
            ));
        }
        *active = false;
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Commit,
        })
    }

    pub fn rollback_transaction(&self) -> DslResult<StatementResult> {
        let mut active = self.transaction_active.lock().unwrap();
        if !*active {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress".into(),
            ));
        }
        *active = false;
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    pub fn execute_select(&self, plan: SelectPlan) -> DslResult<SelectExecution> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        if !plan.aggregates.is_empty() {
            self.execute_aggregates(table.as_ref(), display_name, plan)
        } else {
            self.execute_projection(table.as_ref(), display_name, plan)
        }
    }

    fn create_table_from_columns(
        &self,
        display_name: String,
        canonical_name: String,
        columns: Vec<ColumnSpec>,
        if_not_exists: bool,
    ) -> DslResult<StatementResult> {
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        let mut column_defs: Vec<DslColumn> = Vec::with_capacity(columns.len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, column) in columns.iter().enumerate() {
            let normalized = column.name.to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }
            column_defs.push(DslColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                field_id: (idx + 1) as FieldId,
            });
        }

        let table_id = self.reserve_table_id()?;
        let table = Table::new(table_id, Arc::clone(&self.pager))?;
        table.put_table_meta(&TableMeta {
            table_id,
            name: Some(display_name.clone()),
            created_at_micros: current_time_micros(),
            flags: 0,
            epoch: 0,
        });

        for column in &column_defs {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let schema = Arc::new(DslSchema {
            columns: column_defs,
            lookup,
        });
        let table_entry = Arc::new(DslTable {
            table: Arc::new(table),
            schema,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
        });

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(StatementResult::CreateTable { table_name: display_name });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name, table_entry);
        Ok(StatementResult::CreateTable { table_name: display_name })
    }

    fn create_table_from_batches(
        &self,
        display_name: String,
        canonical_name: String,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        if_not_exists: bool,
    ) -> DslResult<StatementResult> {
        if schema.fields().is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one column".into(),
            ));
        }
        let mut column_defs: Vec<DslColumn> = Vec::with_capacity(schema.fields().len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, field) in schema.fields().iter().enumerate() {
            let data_type = match field.data_type() {
                DataType::Int64 | DataType::Float64 | DataType::Utf8 | DataType::Date32 => {
                    field.data_type().clone()
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported column type in CTAS result: {other:?}"
                    )));
                }
            };
            let normalized = field.name().to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in CTAS result",
                    field.name()
                )));
            }
            column_defs.push(DslColumn {
                name: field.name().to_string(),
                data_type,
                nullable: field.is_nullable(),
                field_id: (idx + 1) as FieldId,
            });
        }

        let table_id = self.reserve_table_id()?;
        let table = Table::new(table_id, Arc::clone(&self.pager))?;
        table.put_table_meta(&TableMeta {
            table_id,
            name: Some(display_name.clone()),
            created_at_micros: current_time_micros(),
            flags: 0,
            epoch: 0,
        });

        for column in &column_defs {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let schema_arc = Arc::new(DslSchema {
            columns: column_defs.clone(),
            lookup,
        });
        let table_entry = Arc::new(DslTable {
            table: Arc::new(table),
            schema: schema_arc,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
        });

        let mut next_row_id: u64 = 0;
        let mut total_rows: u64 = 0;
        for batch in batches {
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            if batch.num_columns() != column_defs.len() {
                return Err(Error::InvalidArgumentError(
                    "CTAS query returned unexpected column count".into(),
                ));
            }
            let start_row = next_row_id;
            next_row_id += row_count as u64;
            total_rows += row_count as u64;

            let mut row_builder = UInt64Builder::with_capacity(row_count);
            for offset in 0..row_count {
                row_builder.append_value(start_row + offset as u64);
            }

            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(row_count + 1);
            arrays.push(Arc::new(row_builder.finish()) as ArrayRef);

            let mut fields: Vec<Field> = Vec::with_capacity(column_defs.len() + 1);
            fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

            for (idx, column) in column_defs.iter().enumerate() {
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    column.field_id.to_string(),
                );
                let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata);
                fields.push(field);
                arrays.push(batch.column(idx).clone());
            }

            let append_schema = Arc::new(Schema::new(fields));
            let append_batch = RecordBatch::try_new(append_schema, arrays)?;
            table_entry.table.append(&append_batch)?;
        }

        table_entry
            .next_row_id
            .store(next_row_id, Ordering::SeqCst);
        table_entry
            .total_rows
            .store(total_rows, Ordering::SeqCst);

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(StatementResult::CreateTable { table_name: display_name });
            }
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name, table_entry);
        Ok(StatementResult::CreateTable { table_name: display_name })
    }

    fn insert_rows(
        &self,
        table: &DslTable<P>,
        display_name: String,
        rows: Vec<Vec<DslValue>>,
        columns: Vec<String>,
    ) -> DslResult<StatementResult> {
        if rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "INSERT requires at least one row".into(),
            ));
        }

        let column_order = resolve_insert_columns(&columns, table.schema.as_ref())?;
        let expected_len = column_order.len();
        for row in &rows {
            if row.len() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} values in INSERT row, found {}",
                    expected_len,
                    row.len()
                )));
            }
        }

        let row_count = rows.len();
        let mut column_values: Vec<Vec<DslValue>> = vec![Vec::with_capacity(row_count); table.schema.columns.len()];
        for row in rows {
            for (idx, value) in row.into_iter().enumerate() {
                let dest_index = column_order[idx];
                column_values[dest_index].push(value);
            }
        }

        let mut row_id_builder = UInt64Builder::with_capacity(row_count);
        let start_row = table.next_row_id.load(Ordering::SeqCst);
        for offset in 0..row_count {
            row_id_builder.append_value(start_row + offset as u64);
        }

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_values.len() + 1);
        arrays.push(Arc::new(row_id_builder.finish()));

        let mut fields: Vec<Field> = Vec::with_capacity(column_values.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (column, values) in table.schema.columns.iter().zip(column_values.into_iter()) {
            let array = build_array_for_column(&column.data_type, &values)?;
            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            arrays.push(array);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;
        table
            .next_row_id
            .store(start_row + row_count as u64, Ordering::SeqCst);
        table
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        Ok(StatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }

    fn insert_batches(
        &self,
        table: &DslTable<P>,
        display_name: String,
        batches: Vec<RecordBatch>,
        columns: Vec<String>,
    ) -> DslResult<StatementResult> {
        if batches.is_empty() {
            return Ok(StatementResult::Insert {
                table_name: display_name,
                rows_inserted: 0,
            });
        }

        let expected_len = if columns.is_empty() {
            table.schema.columns.len()
        } else {
            columns.len()
        };
        let mut total_rows_inserted = 0usize;

        for batch in batches {
            if batch.num_columns() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} columns in INSERT batch, found {}",
                    expected_len,
                    batch.num_columns()
                )));
            }
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            let mut rows: Vec<Vec<DslValue>> = Vec::with_capacity(row_count);
            for row_idx in 0..row_count {
                let mut row: Vec<DslValue> = Vec::with_capacity(expected_len);
                for col_idx in 0..expected_len {
                    let array = batch.column(col_idx);
                    row.push(dsl_value_from_array(array, row_idx)?);
                }
                rows.push(row);
            }

            match self.insert_rows(
                table,
                display_name.clone(),
                rows,
                columns.clone(),
            )? {
                StatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(StatementResult::Insert {
            table_name: display_name,
            rows_inserted: total_rows_inserted,
        })
    }

    fn update_all_rows(
        &self,
        table: &DslTable<P>,
        display_name: String,
        assignments: Vec<ColumnAssignment>,
    ) -> DslResult<StatementResult> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let total_rows_usize = usize::try_from(total_rows).map_err(|_| {
            Error::InvalidArgumentError("table row count exceeds supported range".into())
        })?;
        if total_rows_usize == 0 {
            return Ok(StatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let mut seen_columns: HashSet<String> = HashSet::new();
        let mut row_id_builder = UInt64Builder::with_capacity(total_rows_usize);
        for row_id in 0..total_rows {
            row_id_builder.append_value(row_id);
        }

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(assignments.len() + 1);
        arrays.push(Arc::new(row_id_builder.finish()));

        let mut fields: Vec<Field> = Vec::with_capacity(assignments.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table
                .schema
                .resolve(&assignment.column)
                .ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column '{}' in UPDATE",
                        assignment.column
                    ))
                })?;

            let values = vec![assignment.value.clone(); total_rows_usize];
            let array = build_array_for_column(&column.data_type, &values)?;
            arrays.push(array);

            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;

        Ok(StatementResult::Update {
            table_name: display_name,
            rows_updated: total_rows_usize,
        })
    }

    fn execute_projection(
        &self,
        table: &DslTable<P>,
        display_name: String,
        plan: SelectPlan,
    ) -> DslResult<SelectExecution> {
        let projections = if plan.projections.is_empty() {
            build_wildcard_projections(table)
        } else {
            build_projected_columns(table, &plan.projections)?
        };
        let schema = schema_for_projections(table, &projections)?;

        let filter_expr = match plan.filter {
            Some(expr) => translate_predicate(expr, table.schema.as_ref())?,
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                full_table_scan_filter(field_id)
            }
        };

        let mut batches: Vec<RecordBatch> = Vec::new();
        let options = ScanStreamOptions {
            include_nulls: true,
        };
        table
            .table
            .scan_stream(projections.clone(), &filter_expr, options, |batch| {
                batches.push(batch.clone());
            })?;

        if batches.is_empty() {
            let total_rows = table.total_rows.load(Ordering::SeqCst);
            if total_rows > 0 {
                batches = synthesize_null_scan(Arc::clone(&schema), total_rows)?;
            }
        }

        Ok(SelectExecution {
            table_name: display_name,
            schema,
            batches,
        })
    }

    fn execute_aggregates(
        &self,
        table: &DslTable<P>,
        display_name: String,
        plan: SelectPlan,
    ) -> DslResult<SelectExecution> {
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
                    let col = table.schema.resolve(&column).ok_or_else(|| {
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

        let mut projections = Vec::new();
        for spec in &specs {
            if let Some(field_id) = spec.kind.field_id() {
                projections.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), field_id),
                    table
                        .schema
                        .column_by_field_id(field_id)
                        .map(|c| c.name.clone())
                        .unwrap_or_else(|| format!("col{field_id}")),
                )));
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

        for spec in &specs {
            states.push(AggregateState {
                alias: spec.alias.clone(),
                accumulator: AggregateAccumulator::new(
                    table.schema.as_ref(),
                    spec,
                    count_star_override,
                )?,
                override_value: match spec.kind {
                    AggregateKind::CountStar => count_star_override,
                    _ => None,
                },
            });
        }

        let mut error: Option<Error> = None;
        table
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
            })?;
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
        Ok(SelectExecution {
            table_name: display_name,
            schema,
            batches: vec![batch],
        })
    }

    fn lookup_table(&self, canonical_name: &str) -> DslResult<Arc<DslTable<P>>> {
        let tables = self.tables.read().unwrap();
        tables
            .get(canonical_name)
            .cloned()
            .ok_or_else(|| Error::InvalidArgumentError(format!("unknown table '{canonical_name}'")))
    }

    fn reserve_table_id(&self) -> DslResult<TableId> {
        let store = ColumnStore::open(Arc::clone(&self.pager))?;
        let catalog = SysCatalog::new(&store);

        let mut next = match catalog.get_next_table_id()? {
            Some(value) => value,
            None => {
                let seed = catalog.max_table_id()?.unwrap_or(CATALOG_TABLE_ID);
                let initial = seed.checked_add(1).ok_or_else(|| {
                    Error::InvalidArgumentError("exhausted available table ids".into())
                })?;
                catalog.put_next_table_id(initial)?;
                initial
            }
        };

        if next == CATALOG_TABLE_ID {
            next = next.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        let mut following = next
            .checked_add(1)
            .ok_or_else(|| Error::InvalidArgumentError("exhausted available table ids".into()))?;
        if following == CATALOG_TABLE_ID {
            following = following.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }
        catalog.put_next_table_id(following)?;
        Ok(next)
    }
}

/// Lazily built logical plan.
pub struct LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
    plan: SelectPlan,
}

impl<P> LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn scan(context: Arc<DslContext<P>>, table: &str) -> DslResult<Self> {
        let (display, canonical) = canonical_table_name(table)?;
        context.lookup_table(&canonical)?;
        Ok(Self {
            context,
            plan: SelectPlan::new(display),
        })
    }

    pub fn filter(mut self, predicate: LlkvExpr<'static, String>) -> Self {
        self.plan.filter = Some(predicate);
        self
    }

    pub fn select(mut self, projections: Vec<SelectProjection>) -> Self {
        self.plan.projections = projections;
        self
    }

    pub fn aggregate(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.plan.aggregates = aggregates;
        self
    }

    pub fn collect(self) -> DslResult<SelectExecution> {
        self.context.execute_select(self.plan)
    }
}

/// Result of executing a select plan.
pub struct SelectExecution {
    pub table_name: String,
    pub schema: Arc<Schema>,
    pub batches: Vec<RecordBatch>,
}

struct DslTable<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table: Arc<Table<P>>,
    schema: Arc<DslSchema>,
    next_row_id: AtomicU64,
    total_rows: AtomicU64,
}

struct DslSchema {
    columns: Vec<DslColumn>,
    lookup: HashMap<String, usize>,
}

impl DslSchema {
    fn resolve(&self, name: &str) -> Option<&DslColumn> {
        let normalized = name.to_ascii_lowercase();
        self.lookup
            .get(&normalized)
            .and_then(|idx| self.columns.get(*idx))
    }

    fn first_field_id(&self) -> Option<FieldId> {
        self.columns.first().map(|col| col.field_id)
    }

    fn column_by_field_id(&self, field_id: FieldId) -> Option<&DslColumn> {
        self.columns.iter().find(|col| col.field_id == field_id)
    }
}

#[derive(Clone)]
struct DslColumn {
    name: String,
    data_type: DataType,
    nullable: bool,
    field_id: FieldId,
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
    CountStar { value: i64 },
    CountColumn { column_index: usize, value: i64 },
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
                let non_null =
                    i64::try_from(non_null).map_err(|_| {
                        Error::InvalidArgumentError(
                            "COUNT result exceeds i64 range".into(),
                        )
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
                let non_null =
                    i64::try_from(non_null).map_err(|_| {
                        Error::InvalidArgumentError(
                            "COUNT result exceeds i64 range".into(),
                        )
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

fn canonical_table_name(name: &str) -> DslResult<(String, String)> {
    if name.is_empty() {
        return Err(Error::InvalidArgumentError(
            "table name must not be empty".into(),
        ));
    }
    let display = name.to_string();
    let canonical = display.to_ascii_lowercase();
    Ok((display, canonical))
}

fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn resolve_insert_columns(
    columns: &[String],
    schema: &DslSchema,
) -> DslResult<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.columns.len()).collect());
    }
    let mut resolved = Vec::with_capacity(columns.len());
    for column in columns {
        let normalized = column.to_ascii_lowercase();
        let index = schema.lookup.get(&normalized).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown column '{}'", column))
        })?;
        resolved.push(*index);
    }
    Ok(resolved)
}

fn build_array_for_column(dtype: &DataType, values: &[DslValue]) -> DslResult<ArrayRef> {
    match dtype {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(v) => builder.append_value(*v),
                    DslValue::Float(v) => builder.append_value(*v as i64),
                    DslValue::String(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert string into INT column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(v) => builder.append_value(*v as f64),
                    DslValue::Float(v) => builder.append_value(*v),
                    DslValue::String(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert string into DOUBLE column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(v) => builder.append_value(v.to_string()),
                    DslValue::Float(v) => builder.append_value(v.to_string()),
                    DslValue::String(s) => builder.append_value(s),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Date32 => {
            let mut builder = Date32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    DslValue::Null => builder.append_null(),
                    DslValue::Integer(days) => {
                        let casted = i32::try_from(*days).map_err(|_| {
                            Error::InvalidArgumentError(
                                "integer literal out of range for DATE column".into(),
                            )
                        })?;
                        builder.append_value(casted);
                    }
                    DslValue::Float(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert float into DATE column".into(),
                        ));
                    }
                    DslValue::String(text) => {
                        let days = parse_date32_literal(text)?;
                        builder.append_value(days);
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type for INSERT: {other:?}"
        ))),
    }
}

fn parse_date32_literal(text: &str) -> DslResult<i32> {
    let mut parts = text.split('-');
    let year_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let month_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let day_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid DATE literal '{text}'"
        )));
    }

    let year = year_str.parse::<i32>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid year in DATE literal '{text}'"))
    })?;
    let month_num = month_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;
    let day = day_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid day in DATE literal '{text}'"))
    })?;

    let month = Month::try_from(month_num).map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;

    let date = Date::from_calendar_date(year, month, day).map_err(|err| {
        Error::InvalidArgumentError(format!("invalid DATE literal '{text}': {err}"))
    })?;
    let days = date.to_julian_day() - epoch_julian_day();
    Ok(days)
}

fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
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
            let values = array.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
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

fn build_wildcard_projections<P>(table: &DslTable<P>) -> Vec<ScanProjection>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    table
        .schema
        .columns
        .iter()
        .map(|column| {
            ScanProjection::from(StoreProjection::with_alias(
                LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                column.name.clone(),
            ))
        })
        .collect()
}

fn build_projected_columns<P>(
    table: &DslTable<P>,
    projections: &[SelectProjection],
) -> DslResult<Vec<ScanProjection>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut result = Vec::with_capacity(projections.len());
    for projection in projections.iter() {
        match projection {
            SelectProjection::AllColumns => {
                result.extend(build_wildcard_projections(table));
            }
            SelectProjection::Column { name, alias } => {
                let column = table.schema.resolve(name).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column '{}' in projection",
                        name
                    ))
                })?;
                let alias = alias.clone().unwrap_or_else(|| column.name.clone());
                result.push(ScanProjection::from(StoreProjection::with_alias(
                    LogicalFieldId::for_user(table.table.table_id(), column.field_id),
                    alias,
                )));
            }
            SelectProjection::Computed { expr, alias } => {
                let scalar = translate_scalar(expr, table.schema.as_ref())?;
                result.push(ScanProjection::computed(scalar, alias.clone()));
            }
        }
    }
    if result.is_empty() {
        return Err(Error::InvalidArgumentError(
            "projection must include at least one column".into(),
        ));
    }
    Ok(result)
}

fn schema_for_projections<P>(
    table: &DslTable<P>,
    projections: &[ScanProjection],
) -> DslResult<Arc<Schema>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    let mut fields: Vec<Field> = Vec::with_capacity(projections.len());
    for projection in projections {
        match projection {
            ScanProjection::Column(proj) => {
                let field_id = proj.logical_field_id.field_id();
                let column = table.schema.column_by_field_id(field_id).ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "unknown column with field id {} in projection",
                        field_id
                    ))
                })?;
                let name = proj.alias.clone().unwrap_or_else(|| column.name.clone());
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    column.field_id.to_string(),
                );
                let field = Field::new(&name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata);
                fields.push(field);
            }
            ScanProjection::Computed { alias, expr } => {
                let dtype = match expr {
                    ScalarExpr::Literal(Literal::Integer(_)) => DataType::Int64,
                    ScalarExpr::Literal(Literal::Float(_)) => DataType::Float64,
                    ScalarExpr::Literal(Literal::String(_)) => DataType::Utf8,
                    ScalarExpr::Column(field_id) => {
                        let column = table.schema.column_by_field_id(*field_id).ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "unknown column with field id {} in computed projection",
                                field_id
                            ))
                        })?;
                        column.data_type.clone()
                    }
                    ScalarExpr::Binary { .. } => DataType::Float64,
                };
                let field = Field::new(alias, dtype, true);
                fields.push(field);
            }
        }
    }
    Ok(Arc::new(Schema::new(fields)))
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

    if row_count == 0 {
        return Ok(Vec::new());
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        columns.push(new_null_array(field.data_type(), row_count));
    }

    let batch = RecordBatch::try_new(schema, columns)?;
    Ok(vec![batch])
}

fn translate_predicate(
    expr: LlkvExpr<'static, String>,
    schema: &DslSchema,
) -> DslResult<LlkvExpr<'static, FieldId>> {
    match expr {
        LlkvExpr::And(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::And(converted))
        }
        LlkvExpr::Or(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::Or(converted))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(translate_predicate(*inner, schema)?))),
        LlkvExpr::Pred(Filter { field_id, op }) => {
            let column = schema.resolve(&field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{field_id}' in filter"))
            })?;
            Ok(LlkvExpr::Pred(Filter {
                field_id: column.field_id,
                op,
            }))
        }
        LlkvExpr::Compare { left, op, right } => {
            let left = translate_scalar(&left, schema)?;
            let right = translate_scalar(&right, schema)?;
            Ok(LlkvExpr::Compare { left, op, right })
        }
    }
}

fn translate_scalar(expr: &ScalarExpr<String>, schema: &DslSchema) -> DslResult<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::Column(name) => {
            let column = schema.resolve(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{name}' in expression"))
            })?;
            Ok(ScalarExpr::column(column.field_id))
        }
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Binary { left, op, right } => {
            let left_expr = translate_scalar(left, schema)?;
            let right_expr = translate_scalar(right, schema)?;
            Ok(ScalarExpr::Binary {
                left: Box::new(left_expr),
                op: *op,
                right: Box::new(right_expr),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, StringArray};
    use llkv_storage::pager::MemPager;

    #[test]
    fn create_insert_select_roundtrip() {
        let pager = Arc::new(MemPager::default());
        let context = DslContext::new(pager);

        let plan = CreateTablePlan {
            name: "people".into(),
            if_not_exists: false,
            columns: vec![
                ColumnSpec::new("id", DataType::Int64, false),
                ColumnSpec::new("name", DataType::Utf8, false),
            ],
            source: None,
        };
        let result = context.create_table(plan).expect("create table");
        matches!(result, StatementResult::CreateTable { .. });

        let insert = InsertPlan {
            table: "people".into(),
            columns: vec!["id".into(), "name".into()],
            source: InsertSource::Rows(vec![
                vec![DslValue::Integer(1), DslValue::from("alice")],
                vec![DslValue::Integer(2), DslValue::from("bob")],
            ]),
        };
        let result = context.insert(insert).expect("insert rows");
        match result {
            StatementResult::Insert { rows_inserted, .. } => assert_eq!(rows_inserted, 2),
            other => panic!("expected insert result, got {other:?}"),
        }

        let select_plan =
            SelectPlan::new("people").with_projections(vec![SelectProjection::Column {
                name: "name".into(),
                alias: None,
            }]);
        let execution = context.execute_select(select_plan).expect("select rows");
        assert_eq!(execution.batches.len(), 1);
        let column = execution.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        assert_eq!(column.len(), 2);
    }

    #[test]
    fn aggregate_count_nulls() {
        let pager = Arc::new(MemPager::default());
        let context = DslContext::new(pager);

        context
            .create_table(CreateTablePlan {
                name: "ints".into(),
                if_not_exists: false,
                columns: vec![ColumnSpec::new("i", DataType::Int64, true)],
                source: None,
            })
            .expect("create table");

        context
            .insert(InsertPlan {
                table: "ints".into(),
                columns: vec!["i".into()],
                source: InsertSource::Rows(vec![
                    vec![DslValue::Null],
                    vec![DslValue::Integer(1)],
                    vec![DslValue::Null],
                ]),
            })
            .expect("insert rows");

        let plan = SelectPlan::new("ints").with_aggregates(vec![AggregateExpr::count_nulls(
            "i",
            "nulls",
        )]);
        let result = context.execute_select(plan).expect("select");
        let column = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int column");
        assert_eq!(column.value(0), 2);
    }
}
