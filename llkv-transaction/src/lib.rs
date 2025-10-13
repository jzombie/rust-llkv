//! Transaction management and MVCC (Multi-Version Concurrency Control) for LLKV.
//!
//! This crate provides transaction isolation using MVCC semantics. Each transaction
//! operates with a consistent snapshot of the database, determined by its transaction
//! ID and snapshot timestamp.
//!
//! # Key Concepts
//!
//! - **Transaction ID ([`TxnId`])**: Unique 64-bit identifier for each transaction
//! - **Snapshot Isolation**: Transactions see a consistent view of data as of their start time
//! - **Row Versioning**: Each row tracks when it was created and deleted via `created_by` and `deleted_by` columns
//! - **Transaction Snapshot ([`TransactionSnapshot`])**: Captures transaction ID and snapshot timestamp
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
//! - **[`TransactionContext`]**: Main interface for executing operations within a transaction
//! - **[`RowVersion`]**: Metadata tracking which transaction created/deleted a row
pub mod mvcc;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;

pub use mvcc::{
    RowVersion, TXN_ID_AUTO_COMMIT, TXN_ID_NONE, TransactionSnapshot, TxnId, TxnIdManager,
};

/// Session identifier type.
///
/// Session IDs track client sessions that may spawn multiple transactions.
/// They are distinct from transaction IDs and managed separately.
pub type SessionId = u64;

use llkv_expr::expr::Expr as LlkvExpr;
use llkv_plan::plans::{
    ColumnSpec, CreateTablePlan, DeletePlan, InsertPlan, PlanOperation, PlanValue, SelectPlan,
    UpdatePlan,
};
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use llkv_executor::SelectExecution;

// ============================================================================
// Type Definitions
// ============================================================================

/// Simplified row batch for export/import
pub struct RowBatch {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<PlanValue>>,
}

/// Transaction kind enum
#[derive(Clone, Debug)]
pub enum TransactionKind {
    Begin,
    Commit,
    Rollback,
}

/// Transaction result enum (simplified version for transaction module)
#[allow(clippy::large_enum_variant)] // TODO: Consider refactoring large variants
#[derive(Clone, Debug)]
pub enum TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    CreateTable {
        table_name: String,
    },
    Insert {
        rows_inserted: usize,
    },
    Update {
        rows_matched: usize,
        rows_updated: usize,
    },
    Delete {
        rows_deleted: usize,
    },
    Select {
        table_name: String,
        schema: Arc<Schema>,
        execution: SelectExecution<P>,
    },
    Transaction {
        kind: TransactionKind,
    },
}

impl<P> TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Convert pager type for compatibility
    pub fn convert_pager_type<P2>(self) -> LlkvResult<TransactionResult<P2>>
    where
        P2: Pager<Blob = EntryHandle> + Send + Sync + 'static,
    {
        match self {
            TransactionResult::CreateTable { table_name } => {
                Ok(TransactionResult::CreateTable { table_name })
            }
            TransactionResult::Insert { rows_inserted } => {
                Ok(TransactionResult::Insert { rows_inserted })
            }
            TransactionResult::Update {
                rows_matched,
                rows_updated,
            } => Ok(TransactionResult::Update {
                rows_matched,
                rows_updated,
            }),
            TransactionResult::Delete { rows_deleted } => {
                Ok(TransactionResult::Delete { rows_deleted })
            }
            TransactionResult::Transaction { kind } => Ok(TransactionResult::Transaction { kind }),
            TransactionResult::Select { .. } => Err(Error::Internal(
                "cannot convert SELECT TransactionResult between pager types".into(),
            )),
        }
    }
}

// ============================================================================
// Transaction Management Types
// ============================================================================

/// A trait for transaction context operations.
/// This allows SessionTransaction to work with any context that implements these operations.
/// The associated type P specifies the pager type this context uses.
pub trait TransactionContext: Send + Sync {
    /// The pager type used by this context
    type Pager: Pager<Blob = EntryHandle> + Send + Sync + 'static;

    /// Update the snapshot used for MVCC visibility decisions.
    fn set_snapshot(&self, snapshot: mvcc::TransactionSnapshot);

    /// Get the snapshot currently associated with this context.
    fn snapshot(&self) -> mvcc::TransactionSnapshot;

    /// Get table column specifications
    fn table_column_specs(&self, table_name: &str) -> LlkvResult<Vec<ColumnSpec>>;

    /// Export table rows for snapshotting
    fn export_table_rows(&self, table_name: &str) -> LlkvResult<RowBatch>;

    /// Get batches with row IDs for seeding updates
    fn get_batches_with_row_ids(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
    ) -> LlkvResult<Vec<RecordBatch>>;

    /// Execute a SELECT plan with this context's pager type
    fn execute_select(&self, plan: SelectPlan) -> LlkvResult<SelectExecution<Self::Pager>>;

    /// Create a table from plan
    fn create_table_plan(
        &self,
        plan: CreateTablePlan,
    ) -> LlkvResult<TransactionResult<Self::Pager>>;

    /// Insert rows
    fn insert(&self, plan: InsertPlan) -> LlkvResult<TransactionResult<Self::Pager>>;

    /// Update rows
    fn update(&self, plan: UpdatePlan) -> LlkvResult<TransactionResult<Self::Pager>>;

    /// Delete rows
    fn delete(&self, plan: DeletePlan) -> LlkvResult<TransactionResult<Self::Pager>>;

    /// Append batches with row IDs
    fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> LlkvResult<usize>;

    /// Get table names for catalog snapshot
    fn table_names(&self) -> Vec<String>;

    /// Get table ID for a given table name (for conflict detection)
    fn table_id(&self, table_name: &str) -> LlkvResult<llkv_table::types::TableId>;

    /// Get an immutable catalog snapshot for transaction isolation
    fn catalog_snapshot(&self) -> llkv_table::catalog::TableCatalogSnapshot;
}

/// Transaction state for the runtime context.
pub struct SessionTransaction<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    /// Transaction snapshot (contains txn id + snapshot watermark)
    snapshot: mvcc::TransactionSnapshot,
    /// Staging context with MemPager for isolation (only used for tables created in this txn).
    staging: Arc<StagingCtx>,
    /// Operations to replay on commit.
    operations: Vec<PlanOperation>,
    /// Tables that have been verified to exist (either in base or staging).
    staged_tables: HashSet<String>,
    /// Tables created within this transaction (live in staging until commit).
    new_tables: HashSet<String>,
    /// Tables known to be missing.
    missing_tables: HashSet<String>,
    /// Immutable catalog snapshot at transaction start (for isolation).
    /// Contains table name→ID mappings. Replaces separate HashSet and HashMap.
    catalog_snapshot: llkv_table::catalog::TableCatalogSnapshot,
    /// Base context for reading existing tables with MVCC visibility.
    base_context: Arc<BaseCtx>,
    /// Whether this transaction has been aborted due to an error.
    is_aborted: bool,
    /// Transaction ID manager (shared across all transactions)
    txn_manager: Arc<TxnIdManager>,
    /// Tables accessed (names only) by this transaction
    accessed_tables: HashSet<String>,
}

impl<BaseCtx, StagingCtx> SessionTransaction<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    pub fn new(
        base_context: Arc<BaseCtx>,
        staging: Arc<StagingCtx>,
        txn_manager: Arc<TxnIdManager>,
    ) -> Self {
        // Get immutable catalog snapshot for transaction isolation
        // This replaces the previous HashSet<String> and HashMap<String, TableId>
        let catalog_snapshot = base_context.catalog_snapshot();

        let snapshot = txn_manager.begin_transaction();
        tracing::debug!(
            "[SESSION_TX] new() created transaction with txn_id={}, snapshot_id={}",
            snapshot.txn_id,
            snapshot.snapshot_id
        );
        TransactionContext::set_snapshot(&*base_context, snapshot);
        TransactionContext::set_snapshot(&*staging, snapshot);

        Self {
            staging,
            operations: Vec::new(),
            staged_tables: HashSet::new(),
            new_tables: HashSet::new(),
            missing_tables: HashSet::new(),
            catalog_snapshot,
            base_context,
            is_aborted: false,
            accessed_tables: HashSet::new(),
            snapshot,
            txn_manager,
        }
    }

    /// Ensure a table exists and is visible to this transaction.
    /// NO COPYING - just check if table exists in base or was created in this transaction.
    fn ensure_table_exists(&mut self, table_name: &str) -> LlkvResult<()> {
        tracing::trace!(
            "[ENSURE] ensure_table_exists called for table='{}'",
            table_name
        );

        // If we already checked this table, return early
        if self.staged_tables.contains(table_name) {
            tracing::trace!("[ENSURE] table already verified to exist");
            return Ok(());
        }

        // Check if table exists in catalog snapshot OR was created in this transaction
        if !self.catalog_snapshot.table_exists(table_name) && !self.new_tables.contains(table_name)
        {
            self.missing_tables.insert(table_name.to_string());
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{table_name}' does not exist"
            )));
        }

        if self.missing_tables.contains(table_name) {
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{table_name}' does not exist"
            )));
        }

        // For tables created in this transaction, also create in staging for isolation
        if self.new_tables.contains(table_name) {
            tracing::trace!("[ENSURE] Table was created in this transaction");
            // Check if it exists in staging
            match self.staging.table_column_specs(table_name) {
                Ok(_) => {
                    self.staged_tables.insert(table_name.to_string());
                    return Ok(());
                }
                Err(_) => {
                    return Err(Error::CatalogError(format!(
                        "Catalog Error: Table '{table_name}' was created but not found in staging"
                    )));
                }
            }
        }

        // Table exists in base - mark as verified
        tracing::trace!(
            "[ENSURE] Table exists in base, no copying needed (MVCC will handle visibility)"
        );
        self.staged_tables.insert(table_name.to_string());
        Ok(())
    }

    /// Execute a SELECT query within transaction isolation.
    /// For tables created in this transaction: read from staging.
    /// For existing tables: read from BASE with MVCC visibility filtering.
    pub fn execute_select(
        &mut self,
        plan: SelectPlan,
    ) -> LlkvResult<SelectExecution<StagingCtx::Pager>> {
        // Ensure table exists
        self.ensure_table_exists(&plan.table)?;

        // If table was created in this transaction, read from staging
        if self.new_tables.contains(&plan.table) {
            tracing::trace!(
                "[SELECT] Reading from staging for new table '{}'",
                plan.table
            );
            return self.staging.execute_select(plan);
        }

        // Track access to existing table for conflict detection
        self.accessed_tables.insert(plan.table.clone());

        // Otherwise read from BASE with MVCC visibility
        // The base context already has the snapshot set in SessionTransaction::new()
        tracing::trace!(
            "[SELECT] Reading from BASE with MVCC for existing table '{}'",
            plan.table
        );
        let table_name = plan.table.clone();
        self.base_context.execute_select(plan).and_then(|exec| {
            // Convert pager type from BaseCtx to StagingCtx
            // This is a limitation of the current type system
            // In practice, we're just collecting and re-packaging
            let schema = exec.schema();
            let batches = exec.collect().unwrap_or_default();
            let combined = if batches.is_empty() {
                RecordBatch::new_empty(Arc::clone(&schema))
            } else if batches.len() == 1 {
                batches.into_iter().next().unwrap()
            } else {
                let refs: Vec<&RecordBatch> = batches.iter().collect();
                arrow::compute::concat_batches(&schema, refs).map_err(|err| {
                    Error::Internal(format!("failed to concatenate batches: {err}"))
                })?
            };
            Ok(SelectExecution::from_batch(
                table_name,
                Arc::clone(&schema),
                combined,
            ))
        })
    }

    /// Execute an operation in the transaction staging context
    pub fn execute_operation(
        &mut self,
        operation: PlanOperation,
    ) -> LlkvResult<TransactionResult<StagingCtx::Pager>> {
        tracing::trace!(
            "[TX] SessionTransaction::execute_operation called, operation={:?}",
            match &operation {
                PlanOperation::Insert(p) => format!("INSERT({})", p.table),
                PlanOperation::Update(p) => format!("UPDATE({})", p.table),
                PlanOperation::Delete(p) => format!("DELETE({})", p.table),
                PlanOperation::CreateTable(p) => format!("CREATE_TABLE({})", p.name),
                _ => "OTHER".to_string(),
            }
        );
        // Check if transaction is aborted
        if self.is_aborted {
            return Err(Error::TransactionContextError(
                "TransactionContext Error: transaction is aborted".into(),
            ));
        }

        // Execute operation and catch errors to mark transaction as aborted
        let result = match operation {
            PlanOperation::CreateTable(ref plan) => {
                match self.staging.create_table_plan(plan.clone()) {
                    Ok(result) => {
                        // Track new table so it's visible to subsequent operations in this transaction
                        self.new_tables.insert(plan.name.clone());
                        self.missing_tables.remove(&plan.name);
                        self.staged_tables.insert(plan.name.clone());
                        // Track for commit replay
                        self.operations
                            .push(PlanOperation::CreateTable(plan.clone()));
                        result.convert_pager_type()?
                    }
                    Err(e) => {
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            PlanOperation::Insert(ref plan) => {
                tracing::trace!(
                    "[TX] SessionTransaction::execute_operation INSERT for table='{}'",
                    plan.table
                );
                // Ensure table exists
                if let Err(e) = self.ensure_table_exists(&plan.table) {
                    self.is_aborted = true;
                    return Err(e);
                }

                // If table was created in this transaction, insert into staging
                // Otherwise insert directly into BASE (with transaction ID tagging)
                let is_new_table = self.new_tables.contains(&plan.table);
                // Track access to existing table for conflict detection
                if !is_new_table {
                    self.accessed_tables.insert(plan.table.clone());
                }
                let result = if is_new_table {
                    tracing::trace!("[TX] INSERT into staging for new table");
                    self.staging.insert(plan.clone())
                } else {
                    tracing::trace!(
                        "[TX] INSERT directly into BASE with txn_id={}",
                        self.snapshot.txn_id
                    );
                    // Insert into base - MVCC tagging happens automatically in insert_rows()
                    self.base_context
                        .insert(plan.clone())
                        .and_then(|r| r.convert_pager_type())
                };

                match result {
                    Ok(result) => {
                        // Only track operations for NEW tables - they need replay on commit
                        // For existing tables, changes are already in BASE with MVCC tags
                        if is_new_table {
                            tracing::trace!(
                                "[TX] INSERT to new table - tracking for commit replay"
                            );
                            self.operations.push(PlanOperation::Insert(plan.clone()));
                        } else {
                            tracing::trace!(
                                "[TX] INSERT to existing table - already in BASE, no replay needed"
                            );
                        }
                        result
                    }
                    Err(e) => {
                        tracing::trace!(
                            "DEBUG SessionTransaction::execute_operation INSERT failed: {:?}",
                            e
                        );
                        tracing::trace!("DEBUG setting is_aborted=true");
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            PlanOperation::Update(ref plan) => {
                if let Err(e) = self.ensure_table_exists(&plan.table) {
                    self.is_aborted = true;
                    return Err(e);
                }

                // If table was created in this transaction, update in staging
                // Otherwise update directly in BASE (with MVCC soft-delete + insert)
                let is_new_table = self.new_tables.contains(&plan.table);
                // Track access to existing table for conflict detection
                if !is_new_table {
                    self.accessed_tables.insert(plan.table.clone());
                }
                let result = if is_new_table {
                    tracing::trace!("[TX] UPDATE in staging for new table");
                    self.staging.update(plan.clone())
                } else {
                    tracing::trace!(
                        "[TX] UPDATE directly in BASE with txn_id={}",
                        self.snapshot.txn_id
                    );
                    self.base_context
                        .update(plan.clone())
                        .and_then(|r| r.convert_pager_type())
                };

                match result {
                    Ok(result) => {
                        // Only track operations for NEW tables - they need replay on commit
                        if is_new_table {
                            tracing::trace!(
                                "[TX] UPDATE to new table - tracking for commit replay"
                            );
                            self.operations.push(PlanOperation::Update(plan.clone()));
                        } else {
                            tracing::trace!(
                                "[TX] UPDATE to existing table - already in BASE, no replay needed"
                            );
                        }
                        result
                    }
                    Err(e) => {
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            PlanOperation::Delete(ref plan) => {
                tracing::debug!("[DELETE] Starting delete for table '{}'", plan.table);
                if let Err(e) = self.ensure_table_exists(&plan.table) {
                    tracing::debug!("[DELETE] ensure_table_exists failed: {}", e);
                    self.is_aborted = true;
                    return Err(e);
                }

                // If table was created in this transaction, delete from staging
                // Otherwise delete directly in BASE (with MVCC soft-delete)
                let is_new_table = self.new_tables.contains(&plan.table);
                tracing::debug!("[DELETE] is_new_table={}", is_new_table);
                // Track access to existing table for conflict detection
                if !is_new_table {
                    tracing::debug!(
                        "[DELETE] Tracking access to existing table '{}'",
                        plan.table
                    );
                    self.accessed_tables.insert(plan.table.clone());
                }
                let result = if is_new_table {
                    tracing::debug!("[DELETE] Deleting from staging for new table");
                    self.staging.delete(plan.clone())
                } else {
                    tracing::debug!(
                        "[DELETE] Deleting from BASE with txn_id={}",
                        self.snapshot.txn_id
                    );
                    self.base_context
                        .delete(plan.clone())
                        .and_then(|r| r.convert_pager_type())
                };

                tracing::debug!(
                    "[DELETE] Result: {:?}",
                    result.as_ref().map(|_| "Ok").map_err(|e| format!("{}", e))
                );
                match result {
                    Ok(result) => {
                        // Only track operations for NEW tables - they need replay on commit
                        if is_new_table {
                            tracing::trace!(
                                "[TX] DELETE from new table - tracking for commit replay"
                            );
                            self.operations.push(PlanOperation::Delete(plan.clone()));
                        } else {
                            tracing::trace!(
                                "[TX] DELETE from existing table - already in BASE, no replay needed"
                            );
                        }
                        result
                    }
                    Err(e) => {
                        self.is_aborted = true;
                        return Err(e);
                    }
                }
            }
            PlanOperation::Select(ref plan) => {
                // SELECT is read-only, not tracked for replay
                // But still fails if transaction is aborted (already checked above)
                let table_name = plan.table.clone();
                match self.execute_select(plan.clone()) {
                    Ok(staging_execution) => {
                        // Collect staging execution into batches
                        let schema = staging_execution.schema();
                        let batches = staging_execution.collect().unwrap_or_default();

                        // Combine into single batch
                        let combined = if batches.is_empty() {
                            RecordBatch::new_empty(Arc::clone(&schema))
                        } else if batches.len() == 1 {
                            batches.into_iter().next().unwrap()
                        } else {
                            let refs: Vec<&RecordBatch> = batches.iter().collect();
                            arrow::compute::concat_batches(&schema, refs).map_err(|err| {
                                Error::Internal(format!("failed to concatenate batches: {err}"))
                            })?
                        };

                        // Return execution with combined batch
                        let execution = SelectExecution::from_batch(
                            table_name.clone(),
                            Arc::clone(&schema),
                            combined,
                        );

                        TransactionResult::Select {
                            table_name,
                            schema,
                            execution,
                        }
                    }
                    Err(e) => {
                        // Don't abort on SELECT errors (like table not found)
                        // Only Session layer aborts on constraint violations
                        return Err(e);
                    }
                }
            }
        };

        Ok(result)
    }

    /// Get the operations queued for commit
    pub fn operations(&self) -> &[PlanOperation] {
        &self.operations
    }
}

/// A session handle for transaction management.
/// When dropped, automatically rolls back any active transaction.
pub struct TransactionSession<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    context: Arc<BaseCtx>,
    session_id: SessionId,
    transactions: Arc<Mutex<HashMap<SessionId, SessionTransaction<BaseCtx, StagingCtx>>>>,
    txn_manager: Arc<TxnIdManager>,
}

impl<BaseCtx, StagingCtx> TransactionSession<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    pub fn new(
        context: Arc<BaseCtx>,
        session_id: SessionId,
        transactions: Arc<Mutex<HashMap<SessionId, SessionTransaction<BaseCtx, StagingCtx>>>>,
        txn_manager: Arc<TxnIdManager>,
    ) -> Self {
        Self {
            context,
            session_id,
            transactions,
            txn_manager,
        }
    }

    /// Clone this session (reuses the same session_id and shared transaction map).
    /// This is necessary to maintain transaction state across Engine clones.
    pub fn clone_session(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            session_id: self.session_id,
            transactions: Arc::clone(&self.transactions),
            txn_manager: Arc::clone(&self.txn_manager),
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the underlying context (for advanced use).
    pub fn context(&self) -> &Arc<BaseCtx> {
        &self.context
    }

    /// Check if this session has an active transaction.
    pub fn has_active_transaction(&self) -> bool {
        self.transactions
            .lock()
            .expect("transactions lock poisoned")
            .contains_key(&self.session_id)
    }

    /// Check if the current transaction has been aborted due to an error.
    pub fn is_aborted(&self) -> bool {
        self.transactions
            .lock()
            .expect("transactions lock poisoned")
            .get(&self.session_id)
            .map(|tx| tx.is_aborted)
            .unwrap_or(false)
    }

    /// Mark the current transaction as aborted due to an error.
    /// This should be called when any error occurs during a transaction.
    pub fn abort_transaction(&self) {
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        if let Some(tx) = guard.get_mut(&self.session_id) {
            tx.is_aborted = true;
        }
    }

    /// Begin a transaction in this session.
    pub fn begin_transaction(
        &self,
        staging: Arc<StagingCtx>,
    ) -> LlkvResult<TransactionResult<BaseCtx::Pager>> {
        tracing::debug!(
            "[BEGIN] begin_transaction called for session_id={}",
            self.session_id
        );
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        tracing::debug!(
            "[BEGIN] session_id={}, transactions map has {} entries",
            self.session_id,
            guard.len()
        );
        if guard.contains_key(&self.session_id) {
            return Err(Error::InvalidArgumentError(
                "a transaction is already in progress in this session".into(),
            ));
        }
        guard.insert(
            self.session_id,
            SessionTransaction::new(
                Arc::clone(&self.context),
                staging,
                Arc::clone(&self.txn_manager),
            ),
        );
        tracing::debug!(
            "[BEGIN] session_id={}, inserted transaction, map now has {} entries",
            self.session_id,
            guard.len()
        );
        Ok(TransactionResult::Transaction {
            kind: TransactionKind::Begin,
        })
    }

    /// Commit the transaction in this session.
    /// If the transaction is aborted, this acts as a ROLLBACK instead.
    pub fn commit_transaction(
        &self,
    ) -> LlkvResult<(TransactionResult<BaseCtx::Pager>, Vec<PlanOperation>)> {
        tracing::trace!(
            "[COMMIT] commit_transaction called for session {:?}",
            self.session_id
        );
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        tracing::trace!("[COMMIT] commit_transaction got lock, checking for transaction...");
        let tx_opt = guard.remove(&self.session_id);
        tracing::trace!(
            "[COMMIT] commit_transaction remove returned: {}",
            tx_opt.is_some()
        );
        let tx = tx_opt.ok_or_else(|| {
            tracing::trace!("[COMMIT] commit_transaction: no transaction found!");
            Error::InvalidArgumentError(
                "no transaction is currently in progress in this session".into(),
            )
        })?;
        tracing::trace!("DEBUG commit_transaction: is_aborted={}", tx.is_aborted);

        // If transaction is aborted, commit becomes a rollback (no operations to replay)
        if tx.is_aborted {
            tx.txn_manager.mark_aborted(tx.snapshot.txn_id);
            // Reset context snapshot to auto-commit view (aborted txn's writes should be invisible)
            let auto_commit_snapshot = TransactionSnapshot {
                txn_id: TXN_ID_AUTO_COMMIT,
                snapshot_id: tx.txn_manager.last_committed(),
            };
            TransactionContext::set_snapshot(&*self.context, auto_commit_snapshot);
            tracing::trace!("DEBUG commit_transaction: returning Rollback with 0 operations");
            return Ok((
                TransactionResult::Transaction {
                    kind: TransactionKind::Rollback,
                },
                Vec::new(),
            ));
        }

        // Check for write-write conflicts: detect if any accessed tables have been dropped or replaced
        // We captured (table_name, table_id) pairs at transaction start
        tracing::debug!(
            "[COMMIT CONFLICT CHECK] Transaction {} accessed {} tables",
            tx.snapshot.txn_id,
            tx.accessed_tables.len()
        );
        for accessed_table_name in &tx.accessed_tables {
            tracing::debug!(
                "[COMMIT CONFLICT CHECK] Checking table '{}'",
                accessed_table_name
            );
            // Get the table ID from our catalog snapshot at transaction start
            if let Some(snapshot_table_id) = tx.catalog_snapshot.table_id(accessed_table_name) {
                // Check current table state
                match self.context.table_id(accessed_table_name) {
                    Ok(current_table_id) => {
                        // If table ID changed, it was dropped and recreated
                        if current_table_id != snapshot_table_id {
                            tx.txn_manager.mark_aborted(tx.snapshot.txn_id);
                            let auto_commit_snapshot = TransactionSnapshot {
                                txn_id: TXN_ID_AUTO_COMMIT,
                                snapshot_id: tx.txn_manager.last_committed(),
                            };
                            TransactionContext::set_snapshot(&*self.context, auto_commit_snapshot);
                            return Err(Error::TransactionContextError(
                                "another transaction has dropped this table".into(),
                            ));
                        }
                    }
                    Err(_) => {
                        // Table no longer exists - it was dropped
                        tx.txn_manager.mark_aborted(tx.snapshot.txn_id);
                        let auto_commit_snapshot = TransactionSnapshot {
                            txn_id: TXN_ID_AUTO_COMMIT,
                            snapshot_id: tx.txn_manager.last_committed(),
                        };
                        TransactionContext::set_snapshot(&*self.context, auto_commit_snapshot);
                        return Err(Error::TransactionContextError(
                            "another transaction has dropped this table".into(),
                        ));
                    }
                }
            }
        }

        let operations = tx.operations;
        tracing::trace!(
            "DEBUG commit_transaction: returning Commit with {} operations",
            operations.len()
        );

        tx.txn_manager.mark_committed(tx.snapshot.txn_id);
        TransactionContext::set_snapshot(&*self.context, tx.snapshot);

        Ok((
            TransactionResult::Transaction {
                kind: TransactionKind::Commit,
            },
            operations,
        ))
    }

    /// Rollback the transaction in this session.
    pub fn rollback_transaction(&self) -> LlkvResult<TransactionResult<BaseCtx::Pager>> {
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        if let Some(tx) = guard.remove(&self.session_id) {
            tx.txn_manager.mark_aborted(tx.snapshot.txn_id);
            // Reset context snapshot to auto-commit view (rolled-back txn's writes should be invisible)
            let auto_commit_snapshot = TransactionSnapshot {
                txn_id: TXN_ID_AUTO_COMMIT,
                snapshot_id: tx.txn_manager.last_committed(),
            };
            TransactionContext::set_snapshot(&*self.context, auto_commit_snapshot);
        } else {
            return Err(Error::InvalidArgumentError(
                "no transaction is currently in progress in this session".into(),
            ));
        }
        Ok(TransactionResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    /// Execute an operation in this session's transaction, or directly if no transaction is active.
    pub fn execute_operation(
        &self,
        operation: PlanOperation,
    ) -> LlkvResult<TransactionResult<StagingCtx::Pager>> {
        tracing::debug!(
            "[EXECUTE_OP] execute_operation called for session_id={}",
            self.session_id
        );
        if !self.has_active_transaction() {
            // No transaction - caller must handle direct execution
            return Err(Error::InvalidArgumentError(
                "execute_operation called without active transaction".into(),
            ));
        }

        // In transaction - add to transaction and execute on staging context
        let mut guard = self
            .transactions
            .lock()
            .expect("transactions lock poisoned");
        tracing::debug!(
            "[EXECUTE_OP] session_id={}, transactions map has {} entries",
            self.session_id,
            guard.len()
        );
        let tx = guard
            .get_mut(&self.session_id)
            .ok_or_else(|| Error::Internal("transaction disappeared during execution".into()))?;
        tracing::debug!(
            "[EXECUTE_OP] session_id={}, found transaction with txn_id={}, accessed_tables={}",
            self.session_id,
            tx.snapshot.txn_id,
            tx.accessed_tables.len()
        );

        let result = tx.execute_operation(operation);
        if let Err(ref e) = result {
            tracing::trace!("DEBUG TransactionSession::execute_operation error: {:?}", e);
            tracing::trace!("DEBUG Transaction is_aborted={}", tx.is_aborted);
        }
        result
    }
}

impl<BaseCtx, StagingCtx> Drop for TransactionSession<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext,
    StagingCtx: TransactionContext,
{
    fn drop(&mut self) {
        // Auto-rollback on drop if transaction is active
        // Handle poisoned mutex gracefully to avoid panic during cleanup
        match self.transactions.lock() {
            Ok(mut guard) => {
                if guard.remove(&self.session_id).is_some() {
                    eprintln!(
                        "Warning: TransactionSession dropped with active transaction - auto-rolling back"
                    );
                }
            }
            Err(_) => {
                // Mutex is poisoned, likely due to a panic elsewhere
                // Don't panic again during cleanup
                tracing::trace!(
                    "Warning: TransactionSession dropped with poisoned transaction mutex"
                );
            }
        }
    }
}

/// Transaction manager for coordinating sessions
pub struct TransactionManager<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    transactions: Arc<Mutex<HashMap<SessionId, SessionTransaction<BaseCtx, StagingCtx>>>>,
    next_session_id: AtomicU64,
    txn_manager: Arc<TxnIdManager>,
}

impl<BaseCtx, StagingCtx> TransactionManager<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    pub fn new() -> Self {
        Self {
            transactions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: AtomicU64::new(1),
            txn_manager: Arc::new(TxnIdManager::new()),
        }
    }

    /// Create a new TransactionManager with a custom initial transaction ID.
    pub fn new_with_initial_txn_id(next_txn_id: TxnId) -> Self {
        Self {
            transactions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: AtomicU64::new(1),
            txn_manager: Arc::new(TxnIdManager::new_with_initial_txn_id(next_txn_id)),
        }
    }

    /// Create a new TransactionManager with custom initial state.
    pub fn new_with_initial_state(next_txn_id: TxnId, last_committed: TxnId) -> Self {
        Self {
            transactions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: AtomicU64::new(1),
            txn_manager: Arc::new(TxnIdManager::new_with_initial_state(
                next_txn_id,
                last_committed,
            )),
        }
    }

    /// Create a new session for transaction management.
    pub fn create_session(&self, context: Arc<BaseCtx>) -> TransactionSession<BaseCtx, StagingCtx> {
        let session_id = self.next_session_id.fetch_add(1, Ordering::SeqCst);
        tracing::debug!(
            "[TX_MANAGER] create_session: allocated session_id={}",
            session_id
        );
        TransactionSession::new(
            context,
            session_id,
            Arc::clone(&self.transactions),
            Arc::clone(&self.txn_manager),
        )
    }

    /// Obtain the shared transaction ID manager.
    pub fn txn_manager(&self) -> Arc<TxnIdManager> {
        Arc::clone(&self.txn_manager)
    }

    /// Check if there's an active transaction (checks if ANY session has a transaction).
    pub fn has_active_transaction(&self) -> bool {
        !self
            .transactions
            .lock()
            .expect("transactions lock poisoned")
            .is_empty()
    }
}

impl<BaseCtx, StagingCtx> Default for TransactionManager<BaseCtx, StagingCtx>
where
    BaseCtx: TransactionContext + 'static,
    StagingCtx: TransactionContext + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}
