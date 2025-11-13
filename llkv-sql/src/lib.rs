//! SQL interface for LLKV.
//!
//! This crate provides the [`SqlEngine`], which parses SQL statements and executes
//! them using the LLKV runtime. It serves as the primary user-facing interface for
//! interacting with LLKV databases.
//!
//! ```text
//! SQL String → sqlparser → AST → Plan → Runtime → Storage
//! ```
//!
//! The SQL engine:
//! 1. Parses SQL using [`sqlparser`]
//! 2. Converts AST to execution plans
//! 3. Delegates to `llkv-runtime` for execution
//! 4. Returns results as Arrow `RecordBatch` instances or row counts
//!
//! # Transactions
//!
//! By default, each statement executes in its own auto-commit transaction. Use
//! explicit transaction control for multi-statement transactions:
//!
//! # Type System
//!
//! SQL types are mapped to Arrow data types (the following is a non-exhaustive list):
//! - `INT`, `INTEGER`, `BIGINT` → `Int64`
//! - `FLOAT`, `DOUBLE`, `REAL` → `Float64`
//! - `TEXT`, `VARCHAR` → `Utf8`
//! - `DATE` → `Date32`
pub type SqlResult<T> = llkv_result::Result<T>;

mod sql_engine;
pub use sql_engine::{
    PreparedStatement, SqlEngine, SqlParamValue, StatementExpectation,
    clear_pending_statement_expectations, register_statement_expectation,
};

pub mod tpch;

mod interval;

mod sql_value;
use sql_value::SqlValue;

mod ddl_utils;
pub use ddl_utils::{
    canonical_table_ident, normalize_table_constraint, order_create_tables_by_foreign_keys,
};

pub use llkv_runtime::{RuntimeStatementResult, SelectExecution};
pub use llkv_transaction::TransactionKind;
