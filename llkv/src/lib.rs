//! LLKV: Arrow-Native SQL over Key-Value Storage
//!
//! This crate serves as the primary entrypoint for the LLKV database toolkit.
//! It re-exports the high-level SQL engine and storage abstractions from the
//! underlying `llkv-*` crates, providing a unified API surface for users.
//!
//! # Quick Start
//!
//! Create an in-memory SQL engine and execute queries:
//!
//! ```rust
//! use std::sync::Arc;
//! use llkv::{SqlEngine, MemPager};
//!
//! let engine = SqlEngine::new(Arc::new(MemPager::default()));
//! let results = engine.execute("SELECT 42 AS answer").unwrap();
//! ```
//!
//! # Architecture
//!
//! LLKV is organized as a layered workspace:
//!
//! - **SQL Interface** (`llkv-sql`): Parses and executes SQL statements.
//! - **Query Planning** (`llkv-plan`, `llkv-expr`): Defines logical plans and expression ASTs.
//! - **Runtime** (`llkv-runtime`, `llkv-transaction`): Coordinates MVCC transactions and statement execution.
//! - **Execution** (`llkv-executor`, `llkv-aggregate`, `llkv-join`): Evaluates queries and streams Arrow batches.
//! - **Storage** (`llkv-table`, `llkv-column-map`, `llkv-storage`): Manages columnar storage and pager abstractions.
//!
//! # Re-exports
//!
//! This crate re-exports the following modules for convenient access:
//!
//! - [`SqlEngine`]: The main SQL execution engine.
//! - [`storage`]: Pager abstractions and implementations.

// Re-export the SQL engine as the primary user-facing API
pub use llkv_sql::SqlEngine;

// Re-export storage pager abstractions
pub mod storage {
    //! Storage layer abstractions and pager implementations.
    //!
    //! This module provides the `Pager` trait and concrete implementations
    //! for both in-memory and persistent storage backends.

    pub use llkv_storage::pager::{MemPager, Pager};

    // SimdRDrivePager is only available when llkv-storage is built with simd-r-drive-support
    #[cfg(feature = "simd-r-drive-support")]
    pub use llkv_storage::pager::SimdRDrivePager;
}

// Re-export result types for error handling
pub use llkv_result::{Error, Result};

// Re-export runtime types that users might need when working with statement results
pub use llkv_runtime::RuntimeStatementResult;
