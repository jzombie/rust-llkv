//! Plan-driven query execution for LLKV.
//!
//! This crate bridges `llkv-plan` physical plans to the storage layer, providing
//! a streaming executor that stays MVCC-aware via optional row-id filters. It also
//! hosts DML helpers used by the runtime (value coercion, insert column mapping).

mod query;
pub mod insert;
pub mod types;

pub use query::{QueryExecutor, SelectExecution};
pub use types::{ExecutorColumn, ExecutorMultiColumnUnique, ExecutorRowBatch, ExecutorSchema, ExecutorTable, ExecutorTableProvider, StorageTable, TableStorageAdapter};
pub use insert::{build_array_for_column, normalize_insert_value_for_column, resolve_insert_columns};

/// Current timestamp in microseconds. Kept here to avoid duplicating utility code.
pub use llkv_compute::time::current_time_micros;

/// Common result alias for executor operations.
pub type ExecutorResult<T> = llkv_result::Result<T>;
