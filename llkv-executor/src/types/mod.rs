//! Type definitions for the executor.
//!
//! This module contains the core types used throughout the executor, including:
//! - Table and schema representations
//! - Column metadata
//! - Result types (RowBatch)
//! - Provider trait for table access

pub mod executor_types;
pub mod provider;
pub mod storage;

pub use executor_types::{
    ExecutorColumn, ExecutorMultiColumnUnique, ExecutorRowBatch, ExecutorSchema, ExecutorTable,
};
pub use provider::ExecutorTableProvider;
pub use storage::{StorageTable, TableStorageAdapter};
