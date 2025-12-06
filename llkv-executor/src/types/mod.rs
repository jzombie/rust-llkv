//! Executor-visible type definitions.

pub mod executor_types;
pub mod provider;
pub mod storage;

pub use executor_types::{
    ExecutorColumn, ExecutorMultiColumnUnique, ExecutorRowBatch, ExecutorSchema, ExecutorTable,
};
pub use provider::ExecutorTableProvider;
pub use storage::{StorageTable, TableStorageAdapter};
