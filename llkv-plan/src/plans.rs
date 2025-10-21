//! Logical query plan structures for LLKV.
//!
//! This module defines the plan structures that represent logical query operations
//! before they are executed. Plans are created by SQL parsers or fluent builders and
//! consumed by execution engines.

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use llkv_result::Error;

/// Result type for plan operations.
pub type PlanResult<T> = llkv_result::Result<T>;

// ============================================================================
// PlanValue Types
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub enum PlanValue {
    Null,
    Integer(i64),
    Float(f64),
    String(String),
    Struct(std::collections::HashMap<String, PlanValue>),
}

impl From<&str> for PlanValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for PlanValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<i64> for PlanValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for PlanValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for PlanValue {
    fn from(value: bool) -> Self {
        // Store booleans as integers for compatibility
        if value {
            Self::Integer(1)
        } else {
            Self::Integer(0)
        }
    }
}

impl From<i32> for PlanValue {
    fn from(value: i32) -> Self {
        Self::Integer(value as i64)
    }
}

// ============================================================================
// CREATE TABLE Plan
// ============================================================================

/// Multi-column unique constraint specification.
#[derive(Clone, Debug)]
pub struct MultiColumnUniqueSpec {
    /// Optional name for the unique constraint
    pub name: Option<String>,
    /// Column names participating in this UNIQUE constraint
    pub columns: Vec<String>,
}

/// Plan for creating a table.
#[derive(Clone, Debug)]
pub struct CreateTablePlan {
    pub name: String,
    pub if_not_exists: bool,
    pub or_replace: bool,
    pub columns: Vec<ColumnSpec>,
    pub source: Option<CreateTableSource>,
    /// Optional storage namespace for the table.
    pub namespace: Option<String>,
    pub foreign_keys: Vec<ForeignKeySpec>,
    pub multi_column_uniques: Vec<MultiColumnUniqueSpec>,
}

impl CreateTablePlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            if_not_exists: false,
            or_replace: false,
            columns: Vec::new(),
            source: None,
            namespace: None,
            foreign_keys: Vec::new(),
            multi_column_uniques: Vec::new(),
        }
    }
}

// ============================================================================
// DROP TABLE Plan
// ============================================================================

/// Plan for dropping a table.
#[derive(Clone, Debug)]
pub struct DropTablePlan {
    pub name: String,
    pub if_exists: bool,
}

impl DropTablePlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            if_exists: false,
        }
    }

    pub fn if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

// ============================================================================
// ALTER TABLE Plan Structures
// ============================================================================

/// Plan for ALTER TABLE operations.
#[derive(Clone, Debug, PartialEq)]
pub struct AlterTablePlan {
    pub table_name: String,
    pub if_exists: bool,
    pub operation: AlterTableOperation,
}

/// Specific ALTER TABLE operation to perform.
#[derive(Clone, Debug, PartialEq)]
pub enum AlterTableOperation {
    /// RENAME COLUMN old_name TO new_name
    RenameColumn {
        old_column_name: String,
        new_column_name: String,
    },
    /// ALTER COLUMN column_name SET DATA TYPE new_type
    SetColumnDataType {
        column_name: String,
        new_data_type: String, // SQL type string like "INTEGER", "VARCHAR", etc.
    },
    /// DROP COLUMN column_name
    DropColumn {
        column_name: String,
        if_exists: bool,
        cascade: bool,
    },
}

impl AlterTablePlan {
    pub fn new(table_name: impl Into<String>, operation: AlterTableOperation) -> Self {
        Self {
            table_name: table_name.into(),
            if_exists: false,
            operation,
        }
    }

    pub fn if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

// ============================================================================
// FOREIGN KEY Plan Structures
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForeignKeyAction {
    NoAction,
    Restrict,
}

impl Default for ForeignKeyAction {
    fn default() -> Self {
        Self::NoAction
    }
}

#[derive(Clone, Debug)]
pub struct ForeignKeySpec {
    pub name: Option<String>,
    pub columns: Vec<String>,
    pub referenced_table: String,
    pub referenced_columns: Vec<String>,
    pub on_delete: ForeignKeyAction,
    pub on_update: ForeignKeyAction,
}

// ============================================================================
// CREATE INDEX Plan
// ============================================================================

/// Column specification for CREATE INDEX statements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexColumnPlan {
    pub name: String,
    pub ascending: bool,
    pub nulls_first: bool,
}

impl IndexColumnPlan {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ascending: true,
            nulls_first: false,
        }
    }

    pub fn with_sort(mut self, ascending: bool, nulls_first: bool) -> Self {
        self.ascending = ascending;
        self.nulls_first = nulls_first;
        self
    }
}

/// Plan for creating an index on a table.
#[derive(Clone, Debug)]
pub struct CreateIndexPlan {
    pub name: Option<String>,
    pub table: String,
    pub unique: bool,
    pub if_not_exists: bool,
    pub columns: Vec<IndexColumnPlan>,
}

impl CreateIndexPlan {
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            name: None,
            table: table.into(),
            unique: false,
            if_not_exists: false,
            columns: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }

    pub fn with_unique(mut self, unique: bool) -> Self {
        self.unique = unique;
        self
    }

    pub fn with_if_not_exists(mut self, if_not_exists: bool) -> Self {
        self.if_not_exists = if_not_exists;
        self
    }

    pub fn with_columns(mut self, columns: Vec<IndexColumnPlan>) -> Self {
        self.columns = columns;
        self
    }
}

/// Column specification for CREATE TABLE.
#[derive(Clone, Debug)]
pub struct ColumnSpec {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    /// Optional CHECK constraint expression (SQL string).
    /// Example: "t.t=42" for CHECK(t.t=42)
    pub check_expr: Option<String>,
}

impl ColumnSpec {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
            primary_key: false,
            unique: false,
            check_expr: None,
        }
    }

    pub fn with_primary_key(mut self, primary_key: bool) -> Self {
        self.primary_key = primary_key;
        if primary_key {
            self.unique = true;
        }
        self
    }

    pub fn with_unique(mut self, unique: bool) -> Self {
        if unique {
            self.unique = true;
        }
        self
    }

    pub fn with_check(mut self, check_expr: Option<String>) -> Self {
        self.check_expr = check_expr;
        self
    }
}

/// Trait for types that can be converted into a ColumnSpec.
pub trait IntoColumnSpec {
    fn into_column_spec(self) -> ColumnSpec;
}

/// Column nullability specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnNullability {
    Nullable,
    NotNull,
}

impl ColumnNullability {
    pub fn is_nullable(self) -> bool {
        matches!(self, ColumnNullability::Nullable)
    }
}

/// Convenience constant for nullable columns.
#[allow(non_upper_case_globals)]
pub const Nullable: ColumnNullability = ColumnNullability::Nullable;

/// Convenience constant for non-null columns.
#[allow(non_upper_case_globals)]
pub const NotNull: ColumnNullability = ColumnNullability::NotNull;

impl IntoColumnSpec for ColumnSpec {
    fn into_column_spec(self) -> ColumnSpec {
        self
    }
}

impl<T> IntoColumnSpec for &T
where
    T: Clone + IntoColumnSpec,
{
    fn into_column_spec(self) -> ColumnSpec {
        self.clone().into_column_spec()
    }
}

impl IntoColumnSpec for (&str, DataType) {
    fn into_column_spec(self) -> ColumnSpec {
        ColumnSpec::new(self.0, self.1, true)
    }
}

impl IntoColumnSpec for (&str, DataType, bool) {
    fn into_column_spec(self) -> ColumnSpec {
        ColumnSpec::new(self.0, self.1, self.2)
    }
}

impl IntoColumnSpec for (&str, DataType, ColumnNullability) {
    fn into_column_spec(self) -> ColumnSpec {
        ColumnSpec::new(self.0, self.1, self.2.is_nullable())
    }
}

/// Source data for CREATE TABLE AS SELECT.
#[derive(Clone, Debug)]
pub enum CreateTableSource {
    Batches {
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    },
    Select {
        plan: Box<SelectPlan>,
    },
}

// ============================================================================
// INSERT Plan
// ============================================================================

/// Plan for inserting data into a table.
#[derive(Clone, Debug)]
pub struct InsertPlan {
    pub table: String,
    pub columns: Vec<String>,
    pub source: InsertSource,
}

/// Source data for INSERT operations.
#[derive(Clone, Debug)]
pub enum InsertSource {
    Rows(Vec<Vec<PlanValue>>),
    Batches(Vec<RecordBatch>),
    Select { plan: Box<SelectPlan> },
}

// ============================================================================
// UPDATE Plan
// ============================================================================

/// Plan for updating rows in a table.
#[derive(Clone, Debug)]
pub struct UpdatePlan {
    pub table: String,
    pub assignments: Vec<ColumnAssignment>,
    pub filter: Option<llkv_expr::expr::Expr<'static, String>>,
}

/// Value to assign in an UPDATE.
#[derive(Clone, Debug)]
pub enum AssignmentValue {
    Literal(PlanValue),
    Expression(llkv_expr::expr::ScalarExpr<String>),
}

/// Column assignment for UPDATE.
#[derive(Clone, Debug)]
pub struct ColumnAssignment {
    pub column: String,
    pub value: AssignmentValue,
}

// ============================================================================
// DELETE Plan
// ============================================================================

/// Plan for deleting rows from a table.
#[derive(Clone, Debug)]
pub struct DeletePlan {
    pub table: String,
    pub filter: Option<llkv_expr::expr::Expr<'static, String>>,
}

// ============================================================================
// SELECT Plan
// ============================================================================

/// Table reference in FROM clause.
#[derive(Clone, Debug)]
pub struct TableRef {
    pub schema: String,
    pub table: String,
}

impl TableRef {
    pub fn new(schema: impl Into<String>, table: impl Into<String>) -> Self {
        Self {
            schema: schema.into(),
            table: table.into(),
        }
    }

    /// Get fully qualified name as "schema.table"
    pub fn qualified_name(&self) -> String {
        if self.schema.is_empty() {
            self.table.clone()
        } else {
            format!("{}.{}", self.schema, self.table)
        }
    }
}

/// Logical query plan for SELECT operations.
#[derive(Clone, Debug)]
pub struct SelectPlan {
    /// Tables to query. Empty vec means no FROM clause (e.g., SELECT 42).
    /// Single element for simple queries, multiple for joins/cross products.
    pub tables: Vec<TableRef>,
    pub projections: Vec<SelectProjection>,
    pub filter: Option<llkv_expr::expr::Expr<'static, String>>,
    pub aggregates: Vec<AggregateExpr>,
    pub order_by: Vec<OrderByPlan>,
}

impl SelectPlan {
    /// Create a SelectPlan for a single table.
    pub fn new(table: impl Into<String>) -> Self {
        let table_name = table.into();
        let tables = if table_name.is_empty() {
            Vec::new()
        } else {
            // Parse "schema.table" or just "table"
            let parts: Vec<&str> = table_name.split('.').collect();
            if parts.len() >= 2 {
                let table_part = parts[1..].join(".");
                vec![TableRef::new(parts[0], table_part)]
            } else {
                vec![TableRef::new("", table_name)]
            }
        };

        Self {
            tables,
            projections: Vec::new(),
            filter: None,
            aggregates: Vec::new(),
            order_by: Vec::new(),
        }
    }

    /// Create a SelectPlan with multiple tables for cross product/joins.
    pub fn with_tables(tables: Vec<TableRef>) -> Self {
        Self {
            tables,
            projections: Vec::new(),
            filter: None,
            aggregates: Vec::new(),
            order_by: Vec::new(),
        }
    }

    pub fn with_projections(mut self, projections: Vec<SelectProjection>) -> Self {
        self.projections = projections;
        self
    }

    pub fn with_filter(mut self, filter: Option<llkv_expr::expr::Expr<'static, String>>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_aggregates(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.aggregates = aggregates;
        self
    }

    pub fn with_order_by(mut self, order_by: Vec<OrderByPlan>) -> Self {
        self.order_by = order_by;
        self
    }
}

/// Projection specification for SELECT.
#[derive(Clone, Debug)]
pub enum SelectProjection {
    AllColumns,
    AllColumnsExcept {
        exclude: Vec<String>,
    },
    Column {
        name: String,
        alias: Option<String>,
    },
    Computed {
        expr: llkv_expr::expr::ScalarExpr<String>,
        alias: String,
    },
}

// ============================================================================
// Aggregate Plans
// ============================================================================

/// Aggregate expression in SELECT.
#[derive(Clone, Debug)]
pub enum AggregateExpr {
    CountStar {
        alias: String,
    },
    Column {
        column: String,
        alias: String,
        function: AggregateFunction,
        distinct: bool,
    },
}

/// Supported aggregate functions.
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
            distinct: false,
        }
    }

    pub fn count_distinct_column(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::Count,
            distinct: true,
        }
    }

    pub fn sum_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::SumInt64,
            distinct: false,
        }
    }

    pub fn min_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::MinInt64,
            distinct: false,
        }
    }

    pub fn max_int64(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::MaxInt64,
            distinct: false,
        }
    }

    pub fn count_nulls(column: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Column {
            column: column.into(),
            alias: alias.into(),
            function: AggregateFunction::CountNulls,
            distinct: false,
        }
    }
}

/// Helper to convert an Arrow array cell into a plan-level Value.
pub fn plan_value_from_array(array: &ArrayRef, index: usize) -> PlanResult<PlanValue> {
    if array.is_null(index) {
        return Ok(PlanValue::Null);
    }
    match array.data_type() {
        DataType::Boolean => {
            let values = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Boolean array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::Integer(if values.value(index) { 1 } else { 0 }))
        }
        DataType::Int64 => {
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                Error::InvalidArgumentError("expected Int64 array in INSERT SELECT".into())
            })?;
            Ok(PlanValue::Integer(values.value(index)))
        }
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Float64 array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::Float(values.value(index)))
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Utf8 array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::String(values.value(index).to_string()))
        }
        DataType::Date32 => {
            let values = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| {
                    Error::InvalidArgumentError("expected Date32 array in INSERT SELECT".into())
                })?;
            Ok(PlanValue::Integer(values.value(index) as i64))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported data type in INSERT SELECT: {other:?}"
        ))),
    }
}

// ============================================================================
// ORDER BY Plan
// ============================================================================

/// ORDER BY specification.
#[derive(Clone, Debug)]
pub struct OrderByPlan {
    pub target: OrderTarget,
    pub sort_type: OrderSortType,
    pub ascending: bool,
    pub nulls_first: bool,
}

/// Sort type for ORDER BY.
#[derive(Clone, Debug)]
pub enum OrderSortType {
    Native,
    CastTextToInteger,
}

/// Target column/expression for ORDER BY.
#[derive(Clone, Debug)]
pub enum OrderTarget {
    Column(String),
    Index(usize),
    All,
}

// ============================================================================
// Operation Enum for Transaction Replay
// ============================================================================

/// Recordable plan operation for transaction replay.
#[derive(Clone, Debug)]
pub enum PlanOperation {
    CreateTable(CreateTablePlan),
    DropTable(DropTablePlan),
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    Select(SelectPlan),
}

/// Top-level plan statements that can be executed against a `Session`.
#[derive(Clone, Debug)]
pub enum PlanStatement {
    BeginTransaction,
    CommitTransaction,
    RollbackTransaction,
    CreateTable(CreateTablePlan),
    DropTable(DropTablePlan),
    AlterTable(AlterTablePlan),
    CreateIndex(CreateIndexPlan),
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    Select(SelectPlan),
}
