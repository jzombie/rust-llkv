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

mod runtime_lazy_frame;
pub use runtime_lazy_frame::RuntimeLazyFrame;

mod runtime_table;
pub use runtime_table::{
    IntoInsertRow, RuntimeCreateTableBuilder, RuntimeInsertRowKind, RuntimeRow, RuntimeTableHandle,
    row,
};

pub use llkv_executor::SelectExecution;
pub use llkv_plan::{
    AggregateExpr, AlterTablePlan, AssignmentValue, ColumnAssignment, CreateIndexPlan,
    CreateTablePlan, CreateTableSource, DeletePlan, DropIndexPlan, DropTablePlan, ForeignKeyAction,
    ForeignKeySpec, IndexColumnPlan, InsertPlan, InsertSource, IntoPlanColumnSpec, PlanColumnSpec,
    MultiColumnUniqueSpec, OrderByPlan, OrderSortType, OrderTarget, PlanOperation, PlanStatement,
    PlanValue, RenameTablePlan, SelectPlan, SelectProjection, UpdatePlan,
};
use llkv_result::{Error, Result};
use llkv_table::{CatalogDdl, canonical_table_name};
pub use llkv_transaction::{
    TXN_ID_AUTO_COMMIT, TXN_ID_NONE, TransactionContext, TransactionKind, TransactionManager,
    TransactionResult, TransactionSession, TransactionSnapshot, TxnId, TxnIdManager,
};
use sqlparser::ast::Select;

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

mod runtime_context;
pub use runtime_context::RuntimeContext;

// Re-export range SELECT parsing from llkv-plan
// llkv-sql talks to llkv-runtime, which delegates to llkv-plan
pub use llkv_plan::RangeSelectRows as RuntimeRangeSelectRows;

/// Extract rows from a range() SELECT statement.
///
/// This is a thin wrapper around llkv_plan::extract_rows_from_range that maintains
/// the llkv-runtime API boundary. llkv-sql should call this function, not directly
/// access llkv-plan.
///
/// # Examples
///
/// ```ignore
/// use llkv_runtime::extract_rows_from_range;
/// let select = /* parse SELECT */;
/// if let Some(rows) = extract_rows_from_range(&select)? {
///     // Handle range query
/// }
/// ```
pub fn extract_rows_from_range(select: &Select) -> Result<Option<RuntimeRangeSelectRows>> {
    llkv_plan::extract_rows_from_range(select)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::datatypes::DataType;
    use llkv_plan::{NotNull, Nullable};
    use llkv_storage::pager::MemPager;
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
        let snapshot = context.default_snapshot();
        let execution = context.execute_select(plan, snapshot).expect("select");
        let batches = execution.collect().expect("collect batches");
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int column");
        assert_eq!(column.value(0), 2);
    }
}
