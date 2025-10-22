//! Query execution runtime for LLKV.
//!
//! This crate provides the runtime API (see [`RuntimeEngine`]) for executing SQL plans with full
//! transaction support. It coordinates between the transaction layer, storage layer,
//! and query executor to provide a complete database runtime.
//!
//! # Key Components
//!
//! - **[`RuntimeEngine`]**: Main execution engine for SQL operations
//! - **[`RuntimeSession`]**: Session-level interface with transaction management
//! - **[`TransactionContext`]**: Single-transaction execution context
//! - **Table Provider**: Integration with the query executor for table access
//!
//! # Transaction Support
//!
//! The runtime supports both:
//! - **Auto-commit**: Single-statement transactions (uses `TXN_ID_AUTO_COMMIT`)
//! - **Multi-statement**: Explicit BEGIN/COMMIT/ROLLBACK transactions
//!
//! # MVCC Integration
//!
//! All data modifications automatically include MVCC metadata:
//! - `row_id`: Unique row identifier
//! - `created_by`: Transaction ID that created the row
//! - `deleted_by`: Transaction ID that deleted the row (or `TXN_ID_NONE`)
//!
//! The runtime ensures these columns are injected and managed consistently.

#![forbid(unsafe_code)]

pub mod storage_namespace;

mod runtime_statement_result;
pub use runtime_statement_result::RuntimeStatementResult;

mod runtime_transaction_context;
pub use runtime_transaction_context::RuntimeTransactionContext;

mod runtime_session;
pub use runtime_session::RuntimeSession;

mod runtime_engine;
pub use runtime_engine::RuntimeEngine;

mod runtime_constraints;
pub(crate) use runtime_constraints::validate_alter_table_operation;

mod runtime_lazy_frame;
pub use runtime_lazy_frame::RuntimeLazyFrame;

mod runtime_table;
pub use runtime_table::{
    row,
    IntoInsertRow,
    RuntimeCreateTableBuilder,
    RuntimeInsertRowKind,
    RuntimeRow,
    RuntimeTableHandle,
};

use std::marker::PhantomData;
use std::mem;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use rustc_hash::{FxHashMap, FxHashSet};

use arrow::array::{
    Array, ArrayRef, BooleanBuilder, Date32Builder, Float64Builder, Int64Builder, StringBuilder,
    UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{GatherNullPolicy, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_executor::{
    ExecutorColumn, ExecutorMultiColumnUnique, ExecutorSchema, ExecutorTable, QueryExecutor,
    RowBatch, TableProvider,
};
pub use llkv_executor::SelectExecution;
pub use llkv_plan::{
    AggregateExpr, AlterTablePlan, AssignmentValue, ColumnAssignment, ColumnSpec, CreateIndexPlan,
    CreateTablePlan, CreateTableSource, DeletePlan, DropIndexPlan, DropTablePlan, ForeignKeyAction,
    ForeignKeySpec, IndexColumnPlan, InsertPlan, InsertSource, IntoColumnSpec,
    MultiColumnUniqueSpec, OrderByPlan, OrderSortType, OrderTarget, PlanOperation, PlanStatement,
    PlanValue, RenameTablePlan, SelectPlan, SelectProjection, UpdatePlan,
};
use llkv_result::{Error, Result};
use llkv_storage::pager::{MemPager, Pager};
use llkv_table::{
    build_composite_unique_key, canonical_table_name, CatalogDdl, CatalogManager,
    ConstraintColumnInfo, ConstraintKind, ConstraintService, CreateTableResult, FieldId,
    ForeignKeyColumn, ForeignKeyTableInfo, ForeignKeyView, InsertColumnConstraint,
    InsertMultiColumnUnique, InsertUniqueColumn, MetadataManager, MultiColumnUniqueEntryMeta,
    MultiColumnUniqueRegistration, RowId, SingleColumnIndexDescriptor, SingleColumnIndexRegistration,
    SysCatalog, Table, TableConstraintSummaryView, TableId, TableView, UniqueKey, ROW_ID_FIELD_ID,
};
use llkv_table::catalog::{MvccColumnBuilder, TableCatalog};
use llkv_table::resolvers::{FieldConstraints, FieldDefinition};
use llkv_table::table::{RowIdFilter, ScanProjection, ScanStreamOptions};
use llkv_table::{ensure_multi_column_unique, ensure_single_column_unique};
pub use llkv_transaction::{
    TransactionContext, TransactionKind, TransactionManager, TransactionResult,
    TransactionSession, TransactionSnapshot, TxnId, TxnIdManager, TXN_ID_AUTO_COMMIT,
    TXN_ID_NONE,
};
use llkv_transaction::mvcc::{self, RowVersion};
use sqlparser::ast::{
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, GroupByExpr, ObjectName, ObjectNamePart,
    Select, SelectItem, SelectItemQualifiedWildcardKind, TableAlias, TableFactor, UnaryOperator,
    Value, ValueWithSpan,
};
use time::{Date, Month};
use simd_r_drive_entry_handle::EntryHandle;


fn is_index_not_found_error(err: &Error) -> bool {
    matches!(err, Error::CatalogError(message) if message.contains("does not exist"))
}

fn is_table_missing_error(err: &Error) -> bool {
    matches!(err, Error::CatalogError(message) if message.contains("does not exist"))
}

pub fn statement_table_name(statement: &PlanStatement) -> Option<&str> {
    match statement {
        PlanStatement::CreateTable(plan) => Some(plan.name.as_str()),
        PlanStatement::DropTable(plan) => Some(plan.name.as_str()),
        PlanStatement::AlterTable(plan) => Some(plan.table_name.as_str()),
        PlanStatement::CreateIndex(plan) => Some(plan.table.as_str()),
        PlanStatement::Insert(plan) => Some(plan.table.as_str()),
        PlanStatement::Update(plan) => Some(plan.table.as_str()),
        PlanStatement::Delete(plan) => Some(plan.table.as_str()),
        PlanStatement::Select(plan) => plan
            .tables
            .first()
            .map(|table_ref| table_ref.table.as_str()),
        PlanStatement::DropIndex(_) => None,
        PlanStatement::BeginTransaction
        | PlanStatement::CommitTransaction
        | PlanStatement::RollbackTransaction => None,
    }
}

/// Represents how a column assignment should be materialized during UPDATE/INSERT.
enum PreparedAssignmentValue {
    Literal(PlanValue),
    Expression { expr_index: usize },
}

struct TransactionMvccBuilder;

impl MvccColumnBuilder for TransactionMvccBuilder {
    fn build_insert_columns(
        &self,
        row_count: usize,
        start_row_id: RowId,
        creator_txn_id: u64,
        deleted_marker: u64,
    ) -> (ArrayRef, ArrayRef, ArrayRef) {
        mvcc::build_insert_mvcc_columns(row_count, start_row_id, creator_txn_id, deleted_marker)
    }

    fn mvcc_fields(&self) -> Vec<Field> {
        mvcc::build_mvcc_fields()
    }

    fn field_with_metadata(
        &self,
        name: &str,
        data_type: DataType,
        nullable: bool,
        field_id: FieldId,
    ) -> Field {
        mvcc::build_field_with_metadata(name, data_type, nullable, field_id)
    }
}

#[derive(Debug, Clone)]
struct TableConstraintContext {
    schema_field_ids: Vec<FieldId>,
    column_constraints: Vec<InsertColumnConstraint>,
    unique_columns: Vec<InsertUniqueColumn>,
    multi_column_uniques: Vec<InsertMultiColumnUnique>,
    primary_key: Option<InsertMultiColumnUnique>,
}

mod runtime_context;
pub use runtime_context::{RuntimeContext, filter_row_ids_for_snapshot, resolve_insert_columns, build_array_for_column};

fn plan_value_from_sql_expr(expr: &SqlExpr) -> Result<PlanValue> {
    match expr {
        SqlExpr::Value(value) => plan_value_from_sql_value(value),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match plan_value_from_sql_expr(expr)? {
            PlanValue::Integer(v) => Ok(PlanValue::Integer(-v)),
            PlanValue::Float(v) => Ok(PlanValue::Float(-v)),
            PlanValue::Null | PlanValue::String(_) | PlanValue::Struct(_) => Err(
                Error::InvalidArgumentError("cannot negate non-numeric literal".into()),
            ),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => plan_value_from_sql_expr(expr),
        SqlExpr::Nested(inner) => plan_value_from_sql_expr(inner),
        SqlExpr::Dictionary(fields) => {
            let mut map = std::collections::HashMap::new();
            for field in fields {
                let key = field.key.value.clone();
                let value = plan_value_from_sql_expr(&field.value)?;
                map.insert(key, value);
            }
            Ok(PlanValue::Struct(map))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported literal expression: {other:?}"
        ))),
    }
}

fn plan_value_from_sql_value(value: &ValueWithSpan) -> Result<PlanValue> {
    match &value.value {
        Value::Null => Ok(PlanValue::Null),
        Value::Number(text, _) => {
            if text.contains(['.', 'e', 'E']) {
                let parsed = text.parse::<f64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid float literal: {err}"))
                })?;
                Ok(PlanValue::Float(parsed))
            } else {
                let parsed = text.parse::<i64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
                })?;
                Ok(PlanValue::Integer(parsed))
            }
        }
        Value::Boolean(_) => Err(Error::InvalidArgumentError(
            "BOOLEAN literals are not supported yet".into(),
        )),
        other => {
            if let Some(text) = other.clone().into_string() {
                Ok(PlanValue::String(text))
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unsupported literal: {other:?}"
                )))
            }
        }
    }
}

fn group_by_is_empty(expr: &GroupByExpr) -> bool {
    matches!(
        expr,
        GroupByExpr::Expressions(exprs, modifiers)
            if exprs.is_empty() && modifiers.is_empty()
    )
}

#[derive(Clone)]
pub struct RuntimeRangeSelectRows {
    rows: Vec<Vec<PlanValue>>,
}

impl RuntimeRangeSelectRows {
    pub fn into_rows(self) -> Vec<Vec<PlanValue>> {
        self.rows
    }
}

#[derive(Clone)]
enum RangeProjection {
    Column,
    Literal(PlanValue),
}

#[derive(Clone)]
pub struct RuntimeRangeSpec {
    start: i64,
    #[allow(dead_code)] // Used for validation, computed into row_count
    end: i64,
    row_count: usize,
    column_name_lower: String,
    table_alias_lower: Option<String>,
}

impl RuntimeRangeSpec {
    fn matches_identifier(&self, ident: &str) -> bool {
        let lower = ident.to_ascii_lowercase();
        lower == self.column_name_lower || lower == "range"
    }

    fn matches_table_alias(&self, ident: &str) -> bool {
        let lower = ident.to_ascii_lowercase();
        match &self.table_alias_lower {
            Some(alias) => lower == *alias,
            None => lower == "range",
        }
    }

    fn matches_object_name(&self, name: &ObjectName) -> bool {
        if name.0.len() != 1 {
            return false;
        }
        match &name.0[0] {
            ObjectNamePart::Identifier(ident) => self.matches_table_alias(&ident.value),
            _ => false,
        }
    }
}

pub fn extract_rows_from_range(select: &Select) -> Result<Option<RuntimeRangeSelectRows>> {
    let spec = match parse_range_spec(select)? {
        Some(spec) => spec,
        None => return Ok(None),
    };

    if select.selection.is_some() {
        return Err(Error::InvalidArgumentError(
            "WHERE clauses are not supported for range() SELECT statements".into(),
        ));
    }
    if select.having.is_some()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
        || select.distinct.is_some()
        || select.top.is_some()
        || select.into.is_some()
        || select.prewhere.is_some()
        || !select.lateral_views.is_empty()
        || select.value_table_mode.is_some()
        || !group_by_is_empty(&select.group_by)
    {
        return Err(Error::InvalidArgumentError(
            "advanced SELECT clauses are not supported for range() SELECT statements".into(),
        ));
    }

    let mut projections: Vec<RangeProjection> = Vec::with_capacity(select.projection.len());

    // If projection is empty, treat it as SELECT * (implicit wildcard)
    if select.projection.is_empty() {
        projections.push(RangeProjection::Column);
    } else {
        for item in &select.projection {
            let projection = match item {
                SelectItem::Wildcard(_) => RangeProjection::Column,
                SelectItem::QualifiedWildcard(kind, _) => match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(object_name) => {
                        if spec.matches_object_name(object_name) {
                            RangeProjection::Column
                        } else {
                            return Err(Error::InvalidArgumentError(
                                "qualified wildcard must reference the range() source".into(),
                            ));
                        }
                    }
                    SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(Error::InvalidArgumentError(
                            "expression-qualified wildcards are not supported for range() SELECT statements".into(),
                        ));
                    }
                },
                SelectItem::UnnamedExpr(expr) => build_range_projection_expr(expr, &spec)?,
                SelectItem::ExprWithAlias { expr, .. } => build_range_projection_expr(expr, &spec)?,
            };
            projections.push(projection);
        }
    }

    let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(spec.row_count);
    for idx in 0..spec.row_count {
        let mut row: Vec<PlanValue> = Vec::with_capacity(projections.len());
        let value = spec.start + (idx as i64);
        for projection in &projections {
            match projection {
                RangeProjection::Column => row.push(PlanValue::Integer(value)),
                RangeProjection::Literal(value) => row.push(value.clone()),
            }
        }
        rows.push(row);
    }

    Ok(Some(RuntimeRangeSelectRows { rows }))
}

fn build_range_projection_expr(expr: &SqlExpr, spec: &RuntimeRangeSpec) -> Result<RangeProjection> {
    match expr {
        SqlExpr::Identifier(ident) => {
            if spec.matches_identifier(&ident.value) {
                Ok(RangeProjection::Column)
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unknown column '{}' in range() SELECT",
                    ident.value
                )))
            }
        }
        SqlExpr::CompoundIdentifier(parts) => {
            if parts.len() == 2
                && spec.matches_table_alias(&parts[0].value)
                && spec.matches_identifier(&parts[1].value)
            {
                Ok(RangeProjection::Column)
            } else {
                Err(Error::InvalidArgumentError(
                    "compound identifiers must reference the range() source".into(),
                ))
            }
        }
        SqlExpr::Wildcard(_) | SqlExpr::QualifiedWildcard(_, _) => unreachable!(),
        other => Ok(RangeProjection::Literal(plan_value_from_sql_expr(other)?)),
    }
}

fn parse_range_spec(select: &Select) -> Result<Option<RuntimeRangeSpec>> {
    if select.from.len() != 1 {
        return Ok(None);
    }
    let item = &select.from[0];
    if !item.joins.is_empty() {
        return Err(Error::InvalidArgumentError(
            "JOIN clauses are not supported for range() SELECT statements".into(),
        ));
    }

    match &item.relation {
        TableFactor::Function {
            lateral,
            name,
            args,
            alias,
        } => {
            if *lateral {
                return Err(Error::InvalidArgumentError(
                    "LATERAL range() is not supported".into(),
                ));
            }
            parse_range_spec_from_args(name, args, alias)
        }
        TableFactor::Table {
            name,
            alias,
            args: Some(table_args),
            with_ordinality,
            ..
        } => {
            if *with_ordinality {
                return Err(Error::InvalidArgumentError(
                    "WITH ORDINALITY is not supported for range()".into(),
                ));
            }
            if table_args.settings.is_some() {
                return Err(Error::InvalidArgumentError(
                    "range() SETTINGS clause is not supported".into(),
                ));
            }
            parse_range_spec_from_args(name, &table_args.args, alias)
        }
        _ => Ok(None),
    }
}

fn parse_range_spec_from_args(
    name: &ObjectName,
    args: &[FunctionArg],
    alias: &Option<TableAlias>,
) -> Result<Option<RuntimeRangeSpec>> {
    if name.0.len() != 1 {
        return Ok(None);
    }
    let func_name = match &name.0[0] {
        ObjectNamePart::Identifier(ident) => ident.value.to_ascii_lowercase(),
        _ => return Ok(None),
    };
    if func_name != "range" {
        return Ok(None);
    }

    if args.is_empty() || args.len() > 2 {
        return Err(Error::InvalidArgumentError(
            "range() requires one or two arguments".into(),
        ));
    }

    // Helper to extract integer from argument
    let extract_int = |arg: &FunctionArg| -> Result<i64> {
        let arg_expr = match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
            FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_))
            | FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                return Err(Error::InvalidArgumentError(
                    "range() argument must be an integer literal".into(),
                ));
            }
            FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => {
                return Err(Error::InvalidArgumentError(
                    "named arguments are not supported for range()".into(),
                ));
            }
        };

        let value = plan_value_from_sql_expr(arg_expr)?;
        match value {
            PlanValue::Integer(v) => Ok(v),
            _ => Err(Error::InvalidArgumentError(
                "range() argument must be an integer literal".into(),
            )),
        }
    };

    let (start, end, row_count) = if args.len() == 1 {
        // range(count) - generate [0, count)
        let count = extract_int(&args[0])?;
        if count < 0 {
            return Err(Error::InvalidArgumentError(
                "range() argument must be non-negative".into(),
            ));
        }
        (0, count, count as usize)
    } else {
        // range(start, end) - generate [start, end)
        let start = extract_int(&args[0])?;
        let end = extract_int(&args[1])?;
        if end < start {
            return Err(Error::InvalidArgumentError(
                "range() end must be >= start".into(),
            ));
        }
        let row_count = (end - start) as usize;
        (start, end, row_count)
    };

    let column_name_lower = alias
        .as_ref()
        .and_then(|a| {
            a.columns
                .first()
                .map(|col| col.name.value.to_ascii_lowercase())
        })
        .unwrap_or_else(|| "range".to_string());
    let table_alias_lower = alias.as_ref().map(|a| a.name.value.to_ascii_lowercase());

    Ok(Some(RuntimeRangeSpec {
        start,
        end,
        row_count,
        column_name_lower,
        table_alias_lower,
    }))
}

/// Parse a SQL data type string to an Arrow DataType.
/// Supports common SQL types like TEXT, VARCHAR, INTEGER, etc.
pub(crate) fn sql_type_to_arrow(type_str: &str) -> Result<DataType> {
    let normalized = type_str.trim().to_uppercase();

    // Remove precision/scale for simplicity (VARCHAR(100) -> VARCHAR)
    let base_type = if let Some(paren_pos) = normalized.find('(') {
        &normalized[..paren_pos]
    } else {
        &normalized
    };

    match base_type {
        "TEXT" | "VARCHAR" | "CHAR" | "STRING" => Ok(DataType::Utf8),
        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(DataType::Int64),
        "FLOAT" | "REAL" => Ok(DataType::Float64),
        "DOUBLE" | "DOUBLE PRECISION" => Ok(DataType::Float64),
        "DECIMAL" | "NUMERIC" => Ok(DataType::Float64),
        "BOOLEAN" | "BOOL" => Ok(DataType::Boolean),
        "DATE" => Ok(DataType::Date32),
        _ => Err(Error::InvalidArgumentError(format!(
            "unsupported SQL data type: '{}'",
            type_str
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use llkv_storage::pager::MemPager;
    use llkv_plan::{NotNull, Nullable};
    use std::sync::Arc;

    #[test]
    fn create_insert_select_roundtrip() {
        let pager = Arc::new(MemPager::default());
        let context = Arc::new(RuntimeContext::new(pager));

        let table = context
            .create_table(
                "people",
                [
                    ("id", DataType::Int64, NotNull),
                    ("name", DataType::Utf8, Nullable),
                ],
            )
            .expect("create table");
        table
            .insert_rows([(1_i64, "alice"), (2_i64, "bob")])
            .expect("insert rows");

        let execution = table.lazy().expect("lazy scan");
        let select = execution.collect().expect("build select execution");
        let batches = select.collect().expect("collect batches");
        assert_eq!(batches.len(), 1);
        let column = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string column");
        assert_eq!(column.len(), 2);
    }

    #[test]
    fn aggregate_count_nulls() {
        let pager = Arc::new(MemPager::default());
        let context = Arc::new(RuntimeContext::new(pager));

        let table = context
            .create_table("ints", [("i", DataType::Int64)])
            .expect("create table");
        table
            .insert_rows([
                (PlanValue::Null,),
                (PlanValue::Integer(1),),
                (PlanValue::Null,),
            ])
            .expect("insert rows");

        let plan =
            SelectPlan::new("ints").with_aggregates(vec![AggregateExpr::count_nulls("i", "nulls")]);
        let execution = context.execute_select(plan).expect("select");
        let batches = execution.collect().expect("collect batches");
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int column");
        assert_eq!(column.value(0), 2);
    }
}
