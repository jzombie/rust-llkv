//! Error types and result definitions for the LLKV database system.
//!
//! Ideally, this crate would provide a unified error type ([`Error`]) and result type alias ([`Result<T>`])
//! used throughout all LLKV crates. All operations that could fail return `Result<T>`, where
//! the error variant contains detailed information about what went wrong. However, some of the crates do
//! still have their own error types, which is a work in progress to unify.
//!
//! # Error Philosophy (still a work in progress to implement fully across all crates)
//!
//! LLKV uses a single error enum ([`Error`]) rather than crate-specific error types.
//!
//! This approach:
//! - Simplifies error handling across crate boundaries
//! - Allows errors to propagate naturally with `?` operator
//! - Provides clear error messages for end users
//! - Enables structured error matching for programmatic handling
//!
//! # Error Categories
//!
//! Errors are organized into several categories:
//!
//! - **I/O errors** ([`Error::Io`]): Disk operations, file access
//! - **Data format errors** ([`Error::Arrow`]): Arrow/Parquet serialization issues
//! - **Lookup failures** ([`Error::NotFound`]): Missing tables, columns, rows
//! - **User input errors** ([`Error::InvalidArgumentError`]): Invalid SQL, bad parameters
//! - **Constraint violations** ([`Error::ConstraintError`]): Primary key conflicts, type mismatches
//! - **Transaction errors** ([`Error::TransactionContextError`]): Isolation violations, conflicts
//! - **Catalog errors** ([`Error::CatalogError`]): Metadata corruption or inconsistency
//! - **Internal errors** ([`Error::Internal`]): Bugs or unexpected states

pub mod error;
pub mod result;
pub mod table_error;

pub use error::Error;
pub use result::Result;
