//! Transaction management and MVCC (Multi-Version Concurrency Control) for LLKV.
//!
//! This crate provides transaction isolation using MVCC semantics. Each transaction
//! operates with a consistent snapshot of the database, determined by its transaction
//! ID and snapshot timestamp.
//!
//! # Module Organization
//!
//! - `core`: Core MVCC primitives (TxnIdManager, TransactionSnapshot, RowVersion) - *currently `mvcc` module*
//! - `table`: Table-level MVCC integration (row filtering, builders) - *currently `helpers` module*
//! - [`context`]: Transaction context types (SessionTransaction, TransactionSession, TransactionManager)
//! - [`types`]: Data type definitions (TransactionResult, TransactionKind, TransactionCatalogSnapshot)
//!
//! # Key Concepts
//!
//! - **Transaction ID ([`TxnId`])**: Unique 64-bit identifier for each transaction
//! - **Snapshot Isolation**: Transactions see a consistent view of data as of their start time
//! - **Row Versioning**: Each row tracks when it was created and deleted via `created_by` and `deleted_by` columns
//! - **[`TransactionSnapshot`]**: Captures transaction ID and snapshot timestamp
//!
//! # Reserved Transaction IDs
//!
//! - **[`TXN_ID_NONE`] (u64::MAX)**: Indicates no transaction (uninitialized state)
//! - **[`TXN_ID_AUTO_COMMIT`] (1)**: Used for auto-commit (single-statement) transactions
//! - **IDs 2+**: Multi-statement transactions (allocated by [`TxnIdManager`])
//!
//! # Visibility Rules
//!
//! A row is visible to a transaction if:
//! 1. It was created before the transaction's snapshot (`created_by <= snapshot_id`)
//! 2. It was not deleted, or deleted after the snapshot (`deleted_by == TXN_ID_NONE || deleted_by > snapshot_id`)
//!
//! # Architecture
//!
//! - **[`TxnIdManager`]**: Allocates transaction IDs and tracks commit status
//! - **[`TransactionSnapshot`]**: Immutable view of transaction state for visibility checks
//! - **[`context::TransactionContext`]**: Main interface for executing operations within a transaction
//! - **[`context::SessionTransaction`]**: Per-transaction state machine
//! - **[`context::TransactionSession`]**: Session-level transaction management
//! - **[`context::TransactionManager`]**: Cross-session transaction coordinator
//! - **[`RowVersion`]**: Metadata tracking which transaction created/deleted a row
//! - **[`types::TransactionCatalogSnapshot`]**: Catalog snapshot interface for table lookups

// ============================================================================
// Module Declarations
// ============================================================================

pub mod context;
pub mod helpers;
pub mod mvcc;
pub mod types;

pub use context::{
    TransactionSessionId, SessionTransaction, TransactionContext, TransactionManager, TransactionSession,
};
pub use helpers::{MvccRowIdFilter, TransactionMvccBuilder, filter_row_ids_for_snapshot};
pub use mvcc::{
    RowVersion, TXN_ID_AUTO_COMMIT, TXN_ID_NONE, TransactionSnapshot, TxnId, TxnIdManager,
};
pub use types::{TransactionCatalogSnapshot, TransactionKind, TransactionResult};
