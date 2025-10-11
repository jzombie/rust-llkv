//! Logical query plan structures for LLKV.
//!
//! This module defines the plan structures that represent logical query operations
//! before they are executed. Plans are created by SQL parsers or fluent builders and
//! consumed by execution engines.

use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array, Float64Array, Int64Array, StringArray};
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

/// Plan for creating a table.
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

/// Column specification for CREATE TABLE.
#[derive(Clone, Debug)]
pub struct ColumnSpec {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
}

impl ColumnSpec {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
            primary_key: false,
        }
    }

    pub fn with_primary_key(mut self, primary_key: bool) -> Self {
        self.primary_key = primary_key;
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

/// Logical query plan for SELECT operations.
#[derive(Clone, Debug)]
pub struct SelectPlan {
    pub table: String,
    pub projections: Vec<SelectProjection>,
    pub filter: Option<llkv_expr::expr::Expr<'static, String>>,
    pub aggregates: Vec<AggregateExpr>,
    pub order_by: Option<OrderByPlan>,
}

impl SelectPlan {
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            projections: Vec::new(),
            filter: None,
            aggregates: Vec::new(),
            order_by: None,
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

    pub fn with_order_by(mut self, order_by: Option<OrderByPlan>) -> Self {
        self.order_by = order_by;
        self
    }
}

/// Projection specification for SELECT.
#[derive(Clone, Debug)]
pub enum SelectProjection {
    AllColumns,
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

/// Helper to convert an Arrow array cell into a plan-level Value.
pub fn plan_value_from_array(array: &ArrayRef, index: usize) -> PlanResult<PlanValue> {
    if array.is_null(index) {
        return Ok(PlanValue::Null);
    }
    match array.data_type() {
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
}

// ============================================================================
// Operation Enum for Transaction Replay
// ============================================================================

/// Recordable plan operation for transaction replay.
#[derive(Clone, Debug)]
pub enum PlanOperation {
    CreateTable(CreateTablePlan),
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
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    Select(SelectPlan),
}
