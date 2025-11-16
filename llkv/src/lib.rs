//! LLKV: Arrow-Native SQL over Key-Value Storage
//!
//! This crate serves as the primary entrypoint for the LLKV database toolkit.
//! It re-exports the high-level SQL engine and storage abstractions from the
//! underlying `llkv-*` crates so downstream applications see a single surface
//! for planning, execution, and storage.
//!
//! # Why LLKV Exists
//!
//! LLKV explores what an [Apache Arrow](https://arrow.apache.org/)-first SQL stack can look like when layered
//! on top of key-value pagers instead of a purpose-built storage format. The
//! project targets columnar OLAP workloads while insisting on compatibility with
//! the reference SQLite [`sqllogictest`](https://sqlite.org/sqllogictest/doc/trunk/about.wiki) suite. Today every published SQLite test
//! runs unmodified against LLKV, giving the engine a broad SQL regression net as
//! new features land.
//!
//! LLKV also ingests a growing set of DuckDB `sqllogictest` cases. Those tests help
//! keep transaction semantics honest as dual-dialect support takes shape, but the
//! DuckDB integration is early and still expanding.
//!
//! # Story So Far
//!
//! LLKV is an experimental SQL database that layers Arrow columnar storage, a streaming execution
//! engine, and MVCC transaction management on top of key-value pagers. Every crate in this workspace
//! serves that goal, keeping [`arrow::record_batch::RecordBatch`] as the interchange format from
//! storage through execution.
//!
//! The surface begins with [llkv-sql](https://docs.rs/llkv-sql/latest/llkv_sql/), which parses statements via
//! [`sqlparser`](https://docs.rs/sqlparser) and lowers them into execution plans. Those plans feed
//! into [llkv-runtime](https://docs.rs/llkv-runtime/latest/llkv_runtime/), the orchestration layer that injects MVCC
//! metadata, coordinates transactions, and dispatches work across the execution and storage stacks.
//! Query evaluation lives in [llkv-executor](https://docs.rs/llkv-executor/latest/llkv_executor/), which streams Arrow
//! `RecordBatch` results without owning MVCC state, while [llkv-table](https://docs.rs/llkv-table/latest/llkv_table/)
//! enforces schema rules and logical field tracking on top of the column store.
//!
//! At the storage layer, [llkv-column-map](https://docs.rs/llkv-column-map/latest/llkv_column_map/) persists column chunks as
//! Arrow-serialized blobs keyed by pager-managed physical IDs. That layout lets backends such as
//! [`simd-r-drive`] provide zero-copy buffers, and it keeps higher layers working against Arrow data
//! structures end-to-end. Concurrency remains synchronous by default, leaning on [Rayon] and
//! [Crossbeam] while still embedding inside [Tokio] when async orchestration is required.
//!
//! Compatibility is measured continuously. [`llkv_slt_tester`](https://docs.rs/llkv-slt-tester/latest/llkv_slt_tester/)
//! executes SQLite and DuckDB `sqllogictest` suites via the
//! [`sqllogictest` crate](https://crates.io/crates/sqllogictest), CI spans Linux, macOS, and Windows,
//! and the
//! documentation here stays synchronized with the rest of the repository so the rustdoc narrative
//! matches the public design record.
//!
//! # Crate Topology
//!
//! LLKV ships as a layered Cargo workspace where higher crates depend on the ones below while sharing
//! Arrow as their interchange format. The main strata are:
//!
//! - **SQL Interface**: [`llkv-sql`](https://docs.rs/llkv-sql/latest/llkv_sql/) exposes the SQL entry point and delegates
//!   planning and execution.
//! - **Query Planning**: [`llkv-plan`](https://docs.rs/llkv-plan/latest/llkv_plan/) and [`llkv-expr`](https://docs.rs/llkv-expr/latest/llkv_expr/)
//!   define logical plans and expression ASTs.
//! - **Runtime & Transactions**: [`llkv-runtime`](https://docs.rs/llkv-runtime/latest/llkv_runtime/) orchestrates sessions and
//!   MVCC, while [`llkv-transaction`](https://docs.rs/llkv-transaction/latest/llkv_transaction/) manages transaction IDs and
//!   snapshot isolation.
//! - **Query Execution**: [`llkv-executor`](https://docs.rs/llkv-executor/latest/llkv_executor/) streams Arrow `RecordBatch`
//!   results with help from [`llkv-aggregate`](https://docs.rs/llkv-aggregate/latest/llkv_aggregate/) and
//!   [`llkv-join`](https://docs.rs/llkv-join/latest/llkv_join/).
//! - **Table & Metadata**: [`llkv-table`](https://docs.rs/llkv-table/latest/llkv_table/) enforces schemas and catalogs atop
//!   [`llkv-column-map`](https://docs.rs/llkv-column-map/latest/llkv_column_map/), the columnar storage engine.
//! - **Storage & I/O**: [`llkv-storage`](https://docs.rs/llkv-storage/latest/llkv_storage/) provides the pager abstraction and
//!   integrates with [`simd-r-drive`](https://crates.io/crates/simd-r-drive) backends.
//! - **Supporting crates**: [`llkv-result`](https://docs.rs/llkv-result/latest/llkv_result/) unifies error handling,
//!   [`llkv-csv`](https://docs.rs/llkv-csv/latest/llkv_csv/) handles CSV ingestion, [`llkv-test-utils`](https://docs.rs/llkv-test-utils/latest/llkv_test_utils/)
//!   supplies testing helpers, and [`llkv-slt-tester`](https://docs.rs/llkv-slt-tester/latest/llkv_slt_tester/) drives SQL logic tests.
//!
//! # MVCC Snapshot Isolation
//!
//! LLKV tracks visibility with MVCC metadata injected into every table: hidden `row_id`,
//! `created_by`, and `deleted_by` columns are managed by the runtime and storage layers. Transactions
//! obtain 64-bit IDs from the [`llkv_transaction`](https://docs.rs/llkv-transaction/latest/llkv_transaction/) stack, capture a
//! snapshot of the last committed transaction, and tag new or modified rows accordingly. Rows are
//! visible when `created_by` is at or below the snapshot watermark and `deleted_by` is absent or
//! greater than that watermark. `UPDATE` and `DELETE` use soft deletes (`deleted_by = txn_id`), so
//! old versions remain until future compaction work lands, and auto-commit statements reuse a fast
//! path that tags rows with the reserved auto-commit ID.
//!
//! Each transaction operates with both a base context (existing tables) and a staging context (new
//! tables created during the transaction). On commit, staged operations replay into the base pager
//! once the transaction watermark advances, preserving snapshot isolation without copying entire
//! tables during the unit of work.
//!
//! # Roadmap Signals
//!
//! Active work centers on extending the transaction lifecycle (the
//! [`TxnIdManager`](https://docs.rs/llkv-transaction/latest/llkv_transaction/mvcc/struct.TxnIdManager.html) still carries TODOs for next-ID
//! management), expanding the constraint system across primary, unique, foreign-key, and check
//! metadata, and tightening performance around Arrow batch sizing and columnar access patterns. The
//! crates in this workspace continue to evolve together, keeping documentation and implementation in
//! lockstep.
//!
//! # Dialect and Tooling Outlook
//!
//! - **SQLite compatibility**: LLKV parses SQLite-flavored SQL, batches `INSERT`
//!   statements for throughput, and surfaces results in Arrow form. Passing the
//!   upstream `sqllogictest` suites establishes a baseline but does not yet make
//!   LLKV a drop-in SQLite replacement.
//! - **DuckDB coverage**: Early DuckDB suites exercise MVCC and typed
//!   transaction flows. They chart the roadmap rather than guarantee full DuckDB
//!   parity today.
//! - **Tokio-friendly, synchronous core**: Queries execute synchronously by
//!   default, delegating concurrency to [Rayon] and [Crossbeam]. Embedders can still
//!   tuck the engine inside [Tokio], which is how the SQL Logic Test runner drives
//!   concurrent sessions.
//!
//! See [dev-docs/high-level-crate-linkage.md](../dev-docs/high-level-crate-linkage.md)
//! and the [DeepWiki documentation](https://deepwiki.com/jzombie/rust-llkv)
//! for diagrams and extended commentary.
//!
//! # Quick Start
//!
//! Create an in-memory SQL engine and execute queries:
//!
//! ```rust
//! use std::sync::Arc;
//! use llkv::{SqlEngine, storage::MemPager};
//!
//! let engine = SqlEngine::new(Arc::new(MemPager::default()));
//! let results = engine.execute("SELECT 42 AS answer").unwrap();
//! ```
//!
//! # Architecture
//!
//! LLKV is organized as a layered workspace:
//!
//! - **SQL Interface** ([llkv-sql](https://docs.rs/llkv-sql/latest/llkv_sql/)): Parses and executes SQL statements.
//! - **Query Planning** ([llkv-plan](https://docs.rs/llkv-plan/latest/llkv_plan/), [llkv-expr](https://docs.rs/llkv-expr/latest/llkv_expr/)): Defines logical plans and expression ASTs.
//! - **Runtime** ([llkv-runtime](https://docs.rs/llkv-runtime/latest/llkv_runtime/), [llkv-transaction](https://docs.rs/llkv-transaction/latest/llkv_transaction/)): Coordinates MVCC transactions and statement execution.
//! - **Execution** ([llkv-executor](https://docs.rs/llkv-executor/latest/llkv_executor/), [llkv-aggregate](https://docs.rs/llkv-aggregate/latest/llkv_aggregate/), [llkv-join](https://docs.rs/llkv-join/latest/llkv_join/)): Streams Arrow batches through operators.
//! - **Storage** ([llkv-table](https://docs.rs/llkv-table/latest/llkv_table/), [llkv-column-map](https://docs.rs/llkv-column-map/latest/llkv_column_map/), [llkv-storage](https://docs.rs/llkv-storage/latest/llkv_storage/)): Manages columnar storage and pager abstractions.
//!
//! # Re-exports
//!
//! This crate re-exports the following modules for convenient access:
//!
//! - [`SqlEngine`]: The main SQL execution engine.
//! - [`storage`]: Pager abstractions and implementations.
//!
//! [Rayon]: https://docs.rs/rayon
//! [Crossbeam]: https://docs.rs/crossbeam
//! [Tokio]: https://docs.rs/tokio
//! [`simd-r-drive`]: https://crates.io/crates/simd-r-drive

// Re-export the SQL engine as the primary user-facing API
pub use llkv_sql::SqlEngine;

// Re-export storage pager abstractions
pub mod storage {
    //! Storage layer abstractions and pager implementations.
    //!
    //! This module provides the `Pager` trait and concrete implementations
    //! for both in-memory and persistent storage backends.

    pub use llkv_storage::pager::{
        InstrumentedPager, IoStats, IoStatsSnapshot, MemPager, Pager, PagerDiagnostics,
    };

    // SimdRDrivePager is only available when llkv-storage is built with simd-r-drive-support
    #[cfg(feature = "simd-r-drive-support")]
    pub use llkv_storage::pager::SimdRDrivePager;
}

// Re-export result types for error handling
pub use llkv_result::{Error, Result};

// Re-export runtime types that users might need when working with statement results
pub use llkv_runtime::RuntimeStatementResult;
