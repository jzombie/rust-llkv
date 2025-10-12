#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    Array, ArrayRef, Date32Builder, Float64Builder, Int64Builder, StringBuilder, UInt64Array,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{
    CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, GatherNullPolicy, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
// Literal is not used at top-level; keep it out to avoid unused import warnings.
use llkv_result::Error;
use llkv_storage::pager::{MemPager, Pager};
use llkv_table::table::{RowIdFilter, ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::{FieldId, ROW_ID_FIELD_ID, TableId};
use llkv_table::{CATALOG_TABLE_ID, ColMeta, SysCatalog, TableMeta};
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, GroupByExpr, ObjectName, ObjectNamePart, Select,
    SelectItem, SelectItemQualifiedWildcardKind, TableAlias, TableFactor, UnaryOperator, Value,
    ValueWithSpan,
};
use time::{Date, Month};

pub type Result<T> = llkv_result::Result<T>;

// Re-export plan structures from llkv-plan
pub use llkv_plan::{
    AggregateExpr, AggregateFunction, AssignmentValue, ColumnAssignment, ColumnNullability,
    ColumnSpec, CreateTablePlan, CreateTableSource, DeletePlan, InsertPlan, InsertSource,
    IntoColumnSpec, NotNull, Nullable, OrderByPlan, OrderSortType, OrderTarget, PlanOperation,
    PlanStatement, PlanValue, SelectPlan, SelectProjection, UpdatePlan,
};

// Execution structures from llkv-executor
use llkv_executor::{ExecutorColumn, ExecutorSchema, ExecutorTable};
pub use llkv_executor::{QueryExecutor, RowBatch, SelectExecution, TableProvider};

// Import transaction structures from llkv-transaction for internal use.
pub use llkv_transaction::TransactionKind;
use llkv_transaction::{
    RowVersion, TXN_ID_AUTO_COMMIT, TXN_ID_NONE, TransactionContext, TransactionManager,
    TransactionResult, TxnId, TxnIdManager, mvcc::TransactionSnapshot,
};

// Internal low-level transaction session type (from llkv-transaction)
use llkv_transaction::TransactionSession;

// Note: Session is the high-level wrapper that users should use instead of the lower-level session API

/// Result of running a plan statement.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum StatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    CreateTable {
        table_name: String,
    },
    NoOp,
    Insert {
        table_name: String,
        rows_inserted: usize,
    },
    Update {
        table_name: String,
        rows_updated: usize,
    },
    Delete {
        table_name: String,
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

impl<P> fmt::Debug for StatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatementResult::CreateTable { table_name } => f
                .debug_struct("CreateTable")
                .field("table_name", table_name)
                .finish(),
            StatementResult::NoOp => f.debug_struct("NoOp").finish(),
            StatementResult::Insert {
                table_name,
                rows_inserted,
            } => f
                .debug_struct("Insert")
                .field("table_name", table_name)
                .field("rows_inserted", rows_inserted)
                .finish(),
            StatementResult::Update {
                table_name,
                rows_updated,
            } => f
                .debug_struct("Update")
                .field("table_name", table_name)
                .field("rows_updated", rows_updated)
                .finish(),
            StatementResult::Delete {
                table_name,
                rows_deleted,
            } => f
                .debug_struct("Delete")
                .field("table_name", table_name)
                .field("rows_deleted", rows_deleted)
                .finish(),
            StatementResult::Select {
                table_name, schema, ..
            } => f
                .debug_struct("Select")
                .field("table_name", table_name)
                .field("schema", schema)
                .finish(),
            StatementResult::Transaction { kind } => {
                f.debug_struct("Transaction").field("kind", kind).finish()
            }
        }
    }
}

impl<P> StatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Convert a StatementResult from one pager type to another.
    /// Only works for non-SELECT results (CreateTable, Insert, Update, Delete, NoOp, Transaction).
    #[allow(dead_code)]
    pub(crate) fn convert_pager_type<Q>(self) -> Result<StatementResult<Q>>
    where
        Q: Pager<Blob = EntryHandle> + Send + Sync,
    {
        match self {
            StatementResult::CreateTable { table_name } => {
                Ok(StatementResult::CreateTable { table_name })
            }
            StatementResult::NoOp => Ok(StatementResult::NoOp),
            StatementResult::Insert {
                table_name,
                rows_inserted,
            } => Ok(StatementResult::Insert {
                table_name,
                rows_inserted,
            }),
            StatementResult::Update {
                table_name,
                rows_updated,
            } => Ok(StatementResult::Update {
                table_name,
                rows_updated,
            }),
            StatementResult::Delete {
                table_name,
                rows_deleted,
            } => Ok(StatementResult::Delete {
                table_name,
                rows_deleted,
            }),
            StatementResult::Transaction { kind } => Ok(StatementResult::Transaction { kind }),
            StatementResult::Select { .. } => Err(Error::Internal(
                "Cannot convert SELECT result between pager types in transaction".into(),
            )),
        }
    }
}

/// Return the table name referenced by a plan statement, if any.
///
/// This is a small helper used by higher-level engines (for example the
/// SQL front-end) to provide better error messages when a statement fails
/// with a table-related error. It intentionally returns an `Option<&str>` so
/// callers can decide how to report missing table context.
pub fn statement_table_name(statement: &PlanStatement) -> Option<&str> {
    match statement {
        PlanStatement::CreateTable(plan) => Some(&plan.name),
        PlanStatement::Insert(plan) => Some(&plan.table),
        PlanStatement::Update(plan) => Some(&plan.table),
        PlanStatement::Delete(plan) => Some(&plan.table),
        PlanStatement::Select(plan) => Some(&plan.table),
        PlanStatement::BeginTransaction
        | PlanStatement::CommitTransaction
        | PlanStatement::RollbackTransaction => None,
    }
}

// ============================================================================
// Plan Structures (now in llkv-plan and re-exported above)
// ============================================================================
//
// The following types are defined in llkv-plan and re-exported:
// - plan values, CreateTablePlan, ColumnSpec, IntoColumnSpec
// - InsertPlan, InsertSource, UpdatePlan, DeletePlan
// - SelectPlan, SelectProjection, AggregateExpr, AggregateFunction
// - OrderByPlan, OrderSortType, OrderTarget
// - PlanOperation
//
// This separation allows plans to be used independently of execution logic.
// ============================================================================

// Transaction management is now handled by llkv-transaction crate
// The SessionTransaction and TableDeltaState types are re-exported from there

/// Wrapper for Context that implements TransactionContext
pub struct ContextWrapper<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    ctx: Arc<Context<P>>,
    snapshot: RwLock<TransactionSnapshot>,
}

impl<P> ContextWrapper<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new(ctx: Arc<Context<P>>) -> Self {
        let snapshot = ctx.default_snapshot();
        Self {
            ctx,
            snapshot: RwLock::new(snapshot),
        }
    }

    fn update_snapshot(&self, snapshot: TransactionSnapshot) {
        let mut guard = self.snapshot.write().expect("snapshot lock poisoned");
        *guard = snapshot;
    }

    fn current_snapshot(&self) -> TransactionSnapshot {
        *self.snapshot.read().expect("snapshot lock poisoned")
    }

    fn context(&self) -> &Arc<Context<P>> {
        &self.ctx
    }

    fn ctx(&self) -> &Context<P> {
        &self.ctx
    }
}

/// A session for executing operations with optional transaction support.
///
/// This is a high-level wrapper around the transaction machinery that provides
/// a clean API for users. Operations can be executed directly or within a transaction.
pub struct Session<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    // TODO: Allow generic pager type
    inner: TransactionSession<ContextWrapper<P>, ContextWrapper<MemPager>>,
}

impl<P> Session<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Clone this session (reuses the same underlying TransactionSession).
    /// This is necessary to maintain transaction state across Engine clones.
    pub(crate) fn clone_session(&self) -> Self {
        Self {
            inner: self.inner.clone_session(),
        }
    }

    /// Begin a transaction in this session.
    /// Creates an isolated staging context automatically.
    pub fn begin_transaction(&self) -> Result<StatementResult<P>> {
        let staging_pager = Arc::new(MemPager::default());
        tracing::trace!(
            "BEGIN_TRANSACTION: Created staging pager at {:p}",
            &*staging_pager
        );
        let staging_ctx = Arc::new(Context::new(staging_pager));

        // Copy table metadata from the main context to the staging context,
        // but create new Table instances that use the staging pager
        self.inner
            .context()
            .ctx()
            .copy_tables_to_staging(&staging_ctx)?;

        let staging_wrapper = Arc::new(ContextWrapper::new(staging_ctx));

        self.inner.begin_transaction(staging_wrapper)?;
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Begin,
        })
    }

    /// Mark the current transaction as aborted due to an error.
    /// This should be called when any error occurs during a transaction.
    pub fn abort_transaction(&self) {
        self.inner.abort_transaction();
    }

    /// Check if this session has an active transaction.
    pub fn has_active_transaction(&self) -> bool {
        let result = self.inner.has_active_transaction();
        tracing::trace!("SESSION: has_active_transaction() = {}", result);
        result
    }

    /// Check if the current transaction has been aborted due to an error.
    pub fn is_aborted(&self) -> bool {
        self.inner.is_aborted()
    }

    /// Commit the current transaction and apply changes to the base context.
    /// If the transaction was aborted, this acts as a ROLLBACK instead.
    pub fn commit_transaction(&self) -> Result<StatementResult<P>> {
        tracing::trace!("Session::commit_transaction called");
        let (tx_result, operations) = self.inner.commit_transaction()?;
        tracing::trace!(
            "Session::commit_transaction got {} operations",
            operations.len()
        );

        if !operations.is_empty() {
            let dropped_tables = self
                .inner
                .context()
                .ctx()
                .dropped_tables
                .read()
                .unwrap()
                .clone();
            if !dropped_tables.is_empty() {
                for operation in &operations {
                    let table_name_opt = match operation {
                        PlanOperation::Insert(plan) => Some(plan.table.as_str()),
                        PlanOperation::Update(plan) => Some(plan.table.as_str()),
                        PlanOperation::Delete(plan) => Some(plan.table.as_str()),
                        _ => None,
                    };
                    if let Some(table_name) = table_name_opt {
                        let (_, canonical) = canonical_table_name(table_name)?;
                        if dropped_tables.contains(&canonical) {
                            self.abort_transaction();
                            return Err(Error::TransactionContextError(
                                "another transaction has dropped this table".into(),
                            ));
                        }
                    }
                }
            }
        }

        // Extract the transaction kind from the transaction module's result
        let kind = match tx_result {
            TransactionResult::Transaction { kind } => kind,
            _ => {
                return Err(Error::Internal(
                    "commit_transaction returned non-transaction result".into(),
                ));
            }
        };
        tracing::trace!("Session::commit_transaction kind={:?}", kind);

        // Only replay operations if there are any (empty if transaction was aborted)
        for operation in operations {
            match operation {
                PlanOperation::CreateTable(plan) => {
                    TransactionContext::create_table_plan(&**self.inner.context(), plan)?;
                }
                PlanOperation::Insert(plan) => {
                    TransactionContext::insert(&**self.inner.context(), plan)?;
                }
                PlanOperation::Update(plan) => {
                    TransactionContext::update(&**self.inner.context(), plan)?;
                }
                PlanOperation::Delete(plan) => {
                    TransactionContext::delete(&**self.inner.context(), plan)?;
                }
                _ => {}
            }
        }

        // Reset the base context snapshot to the default auto-commit view now that
        // the transaction has been replayed onto the base tables.
        let base_ctx = self.inner.context();
        let default_snapshot = base_ctx.ctx().default_snapshot();
        TransactionContext::set_snapshot(&**base_ctx, default_snapshot);

        // Return a StatementResult with the correct kind (Commit or Rollback)
        Ok(StatementResult::Transaction { kind })
    }

    /// Rollback the current transaction, discarding all changes.
    pub fn rollback_transaction(&self) -> Result<StatementResult<P>> {
        self.inner.rollback_transaction()?;
        let base_ctx = self.inner.context();
        let default_snapshot = base_ctx.ctx().default_snapshot();
        TransactionContext::set_snapshot(&**base_ctx, default_snapshot);
        Ok(StatementResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    fn materialize_create_table_plan(&self, mut plan: CreateTablePlan) -> Result<CreateTablePlan> {
        if let Some(CreateTableSource::Select { plan: select_plan }) = plan.source.take() {
            let select_result = self.select(*select_plan)?;
            let (schema, batches) = match select_result {
                StatementResult::Select {
                    schema, execution, ..
                } => {
                    let batches = execution.collect()?;
                    (schema, batches)
                }
                _ => {
                    return Err(Error::Internal(
                        "expected SELECT result while executing CREATE TABLE AS SELECT".into(),
                    ));
                }
            };
            plan.source = Some(CreateTableSource::Batches { schema, batches });
        }
        Ok(plan)
    }

    /// Create a table (outside or inside transaction).
    pub fn create_table_plan(&self, plan: CreateTablePlan) -> Result<StatementResult<P>> {
        let plan = self.materialize_create_table_plan(plan)?;
        if self.has_active_transaction() {
            let table_name = plan.name.clone();
            match self
                .inner
                .execute_operation(PlanOperation::CreateTable(plan))
            {
                Ok(_) => Ok(StatementResult::CreateTable { table_name }),
                Err(e) => {
                    // If an error occurs during a transaction, abort it
                    self.abort_transaction();
                    Err(e)
                }
            }
        } else {
            // Call via TransactionContext trait
            let table_name = plan.name.clone();
            TransactionContext::create_table_plan(&**self.inner.context(), plan)?;
            Ok(StatementResult::CreateTable { table_name })
        }
    }

    fn normalize_insert_plan(&self, plan: InsertPlan) -> Result<(InsertPlan, usize)> {
        let InsertPlan {
            table,
            columns,
            source,
        } = plan;

        match source {
            InsertSource::Rows(rows) => {
                let count = rows.len();
                Ok((
                    InsertPlan {
                        table,
                        columns,
                        source: InsertSource::Rows(rows),
                    },
                    count,
                ))
            }
            InsertSource::Batches(batches) => {
                let count = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
                Ok((
                    InsertPlan {
                        table,
                        columns,
                        source: InsertSource::Batches(batches),
                    },
                    count,
                ))
            }
            InsertSource::Select { plan: select_plan } => {
                let select_result = self.select(*select_plan)?;
                let rows = match select_result {
                    StatementResult::Select { execution, .. } => execution.into_rows()?,
                    _ => {
                        return Err(Error::Internal(
                            "expected Select result when executing INSERT ... SELECT".into(),
                        ));
                    }
                };
                let count = rows.len();
                Ok((
                    InsertPlan {
                        table,
                        columns,
                        source: InsertSource::Rows(rows),
                    },
                    count,
                ))
            }
        }
    }

    /// Insert rows (outside or inside transaction).
    pub fn insert(&self, plan: InsertPlan) -> Result<StatementResult<P>> {
        tracing::trace!("Session::insert called for table={}", plan.table);
        let (plan, rows_inserted) = self.normalize_insert_plan(plan)?;
        let table_name = plan.table.clone();

        if self.has_active_transaction() {
            match self.inner.execute_operation(PlanOperation::Insert(plan)) {
                Ok(_) => {
                    tracing::trace!("Session::insert succeeded for table={}", table_name);
                    Ok(StatementResult::Insert {
                        rows_inserted,
                        table_name,
                    })
                }
                Err(e) => {
                    tracing::trace!(
                        "Session::insert failed for table={}, error={:?}",
                        table_name,
                        e
                    );
                    // Only abort transaction on constraint violations
                    if matches!(e, Error::ConstraintError(_)) {
                        tracing::trace!("Transaction is_aborted=true");
                        self.abort_transaction();
                    }
                    Err(e)
                }
            }
        } else {
            // Call via TransactionContext trait
            let context = self.inner.context();
            let default_snapshot = context.ctx().default_snapshot();
            TransactionContext::set_snapshot(&**context, default_snapshot);
            TransactionContext::insert(&**context, plan)?;
            Ok(StatementResult::Insert {
                rows_inserted,
                table_name,
            })
        }
    }

    /// Select rows (outside or inside transaction).
    pub fn select(&self, plan: SelectPlan) -> Result<StatementResult<P>> {
        if self.has_active_transaction() {
            let tx_result = match self
                .inner
                .execute_operation(PlanOperation::Select(plan.clone()))
            {
                Ok(result) => result,
                Err(e) => {
                    // Only abort transaction on specific errors (constraint violations, etc.)
                    // Don't abort on catalog errors (table doesn't exist) or similar
                    if matches!(e, Error::ConstraintError(_)) {
                        self.abort_transaction();
                    }
                    return Err(e);
                }
            };
            match tx_result {
                TransactionResult::Select {
                    table_name,
                    schema,
                    execution: staging_execution,
                } => {
                    // Convert from staging (MemPager) execution to base pager execution
                    // by collecting batches and rebuilding
                    let batches = staging_execution.collect().unwrap_or_default();
                    let combined = if batches.is_empty() {
                        RecordBatch::new_empty(Arc::clone(&schema))
                    } else if batches.len() == 1 {
                        batches.into_iter().next().unwrap()
                    } else {
                        let refs: Vec<&RecordBatch> = batches.iter().collect();
                        arrow::compute::concat_batches(&schema, refs)?
                    };

                    let execution = SelectExecution::from_batch(
                        table_name.clone(),
                        Arc::clone(&schema),
                        combined,
                    );

                    Ok(StatementResult::Select {
                        execution,
                        table_name,
                        schema,
                    })
                }
                _ => Err(Error::Internal("expected Select result".into())),
            }
        } else {
            // Call via TransactionContext trait
            let context = self.inner.context();
            let default_snapshot = context.ctx().default_snapshot();
            TransactionContext::set_snapshot(&**context, default_snapshot);
            let table_name = plan.table.clone();
            let execution = TransactionContext::execute_select(&**context, plan)?;
            let schema = execution.schema();
            Ok(StatementResult::Select {
                execution,
                table_name,
                schema,
            })
        }
    }

    /// Convenience helper to fetch all rows from a table within this session.
    pub fn table_rows(&self, table: &str) -> Result<Vec<Vec<PlanValue>>> {
        let plan =
            SelectPlan::new(table.to_string()).with_projections(vec![SelectProjection::AllColumns]);
        match self.select(plan)? {
            StatementResult::Select { execution, .. } => Ok(execution.collect_rows()?.rows),
            other => Err(Error::Internal(format!(
                "expected Select result when reading table '{table}', got {:?}",
                other
            ))),
        }
    }

    /// Update rows (outside or inside transaction).
    pub fn update(&self, plan: UpdatePlan) -> Result<StatementResult<P>> {
        if self.has_active_transaction() {
            let table_name = plan.table.clone();
            let result = match self.inner.execute_operation(PlanOperation::Update(plan)) {
                Ok(result) => result,
                Err(e) => {
                    // If an error occurs during a transaction, abort it
                    self.abort_transaction();
                    return Err(e);
                }
            };
            match result {
                TransactionResult::Update {
                    rows_matched: _,
                    rows_updated,
                } => Ok(StatementResult::Update {
                    rows_updated,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Update result".into())),
            }
        } else {
            // Call via TransactionContext trait
            let context = self.inner.context();
            let default_snapshot = context.ctx().default_snapshot();
            TransactionContext::set_snapshot(&**context, default_snapshot);
            let table_name = plan.table.clone();
            let result = TransactionContext::update(&**context, plan)?;
            match result {
                TransactionResult::Update {
                    rows_matched: _,
                    rows_updated,
                } => Ok(StatementResult::Update {
                    rows_updated,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Update result".into())),
            }
        }
    }

    /// Delete rows (outside or inside transaction).
    pub fn delete(&self, plan: DeletePlan) -> Result<StatementResult<P>> {
        if self.has_active_transaction() {
            let table_name = plan.table.clone();
            let result = match self.inner.execute_operation(PlanOperation::Delete(plan)) {
                Ok(result) => result,
                Err(e) => {
                    // If an error occurs during a transaction, abort it
                    self.abort_transaction();
                    return Err(e);
                }
            };
            match result {
                TransactionResult::Delete { rows_deleted } => Ok(StatementResult::Delete {
                    rows_deleted,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Delete result".into())),
            }
        } else {
            // Call via TransactionContext trait
            let context = self.inner.context();
            let default_snapshot = context.ctx().default_snapshot();
            TransactionContext::set_snapshot(&**context, default_snapshot);
            let table_name = plan.table.clone();
            let result = TransactionContext::delete(&**context, plan)?;
            match result {
                TransactionResult::Delete { rows_deleted } => Ok(StatementResult::Delete {
                    rows_deleted,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Delete result".into())),
            }
        }
    }
}

pub struct Engine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    context: Arc<Context<P>>,
    session: Session<P>,
}

impl<P> Clone for Engine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        // IMPORTANT: Reuse the same session to maintain transaction state!
        // Creating a new session would break multi-statement transactions.
        tracing::debug!("[ENGINE] Engine::clone() called - reusing same session");
        Self {
            context: Arc::clone(&self.context),
            session: self.session.clone_session(),
        }
    }
}

impl<P> Engine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        let context = Arc::new(Context::new(pager));
        Self::from_context(context)
    }

    pub fn from_context(context: Arc<Context<P>>) -> Self {
        tracing::debug!("[ENGINE] Engine::from_context - creating new session");
        let session = context.create_session();
        tracing::debug!("[ENGINE] Engine::from_context - created session");
        Self { context, session }
    }

    pub fn context(&self) -> Arc<Context<P>> {
        Arc::clone(&self.context)
    }

    pub fn session(&self) -> &Session<P> {
        &self.session
    }

    pub fn execute_statement(&self, statement: PlanStatement) -> Result<StatementResult<P>> {
        match statement {
            PlanStatement::BeginTransaction => self.session.begin_transaction(),
            PlanStatement::CommitTransaction => self.session.commit_transaction(),
            PlanStatement::RollbackTransaction => self.session.rollback_transaction(),
            PlanStatement::CreateTable(plan) => self.session.create_table_plan(plan),
            PlanStatement::Insert(plan) => self.session.insert(plan),
            PlanStatement::Update(plan) => self.session.update(plan),
            PlanStatement::Delete(plan) => self.session.delete(plan),
            PlanStatement::Select(plan) => self.session.select(plan),
        }
    }

    pub fn execute_all<I>(&self, statements: I) -> Result<Vec<StatementResult<P>>>
    where
        I: IntoIterator<Item = PlanStatement>,
    {
        let mut results = Vec::new();
        for statement in statements {
            results.push(self.execute_statement(statement)?);
        }
        Ok(results)
    }
}

/// In-memory execution context shared by plan-based queries.
pub struct Context<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    tables: RwLock<HashMap<String, Arc<ExecutorTable<P>>>>,
    dropped_tables: RwLock<HashSet<String>>,
    // Transaction manager for session-based transactions
    transaction_manager: TransactionManager<ContextWrapper<P>, ContextWrapper<MemPager>>,
    txn_manager: Arc<TxnIdManager>,
}

impl<P> Context<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        tracing::trace!("Context::new called, pager={:p}", &*pager);
        let transaction_manager = TransactionManager::new();
        let txn_manager = transaction_manager.txn_manager();
        Self {
            pager,
            tables: RwLock::new(HashMap::new()),
            dropped_tables: RwLock::new(HashSet::new()),
            transaction_manager,
            txn_manager,
        }
    }

    /// Return the transaction ID manager shared with sessions.
    pub fn txn_manager(&self) -> Arc<TxnIdManager> {
        Arc::clone(&self.txn_manager)
    }

    /// Construct the default snapshot for auto-commit operations.
    pub fn default_snapshot(&self) -> TransactionSnapshot {
        TransactionSnapshot {
            txn_id: TXN_ID_AUTO_COMMIT,
            snapshot_id: self.txn_manager.last_committed(),
        }
    }

    /// Create a new session for transaction management.
    /// Each session can have its own independent transaction.
    pub fn create_session(self: &Arc<Self>) -> Session<P> {
        tracing::debug!("[SESSION] Context::create_session called");
        let wrapper = ContextWrapper::new(Arc::clone(self));
        let inner = self.transaction_manager.create_session(Arc::new(wrapper));
        tracing::debug!("[SESSION] Created TransactionSession with session_id (will be logged by transaction manager)");
        Session { inner }
    }

    /// Get a handle to an existing table by name.
    pub fn table(self: &Arc<Self>, name: &str) -> Result<TableHandle<P>> {
        TableHandle::new(Arc::clone(self), name)
    }

    /// Check if there's an active transaction (legacy - checks if ANY session has a transaction).
    #[deprecated(note = "Use session-based transactions instead")]
    pub fn has_active_transaction(&self) -> bool {
        self.transaction_manager.has_active_transaction()
    }

    pub fn create_table<C, I>(self: &Arc<Self>, name: &str, columns: I) -> Result<TableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, false)
    }

    pub fn create_table_if_not_exists<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
    ) -> Result<TableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, true)
    }

    pub fn create_table_plan(&self, plan: CreateTablePlan) -> Result<StatementResult<P>> {
        if plan.columns.is_empty() && plan.source.is_none() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires explicit columns or a source".into(),
            ));
        }

        let (display_name, canonical_name) = canonical_table_name(&plan.name)?;
        tracing::trace!(
            "DEBUG create_table_plan: table='{}' if_not_exists={} columns={}",
            display_name,
            plan.if_not_exists,
            plan.columns.len()
        );
        for (idx, col) in plan.columns.iter().enumerate() {
            tracing::trace!(
                "  plan column[{}]: name='{}' primary_key={}",
                idx,
                col.name,
                col.primary_key
            );
        }
        let exists = {
            let tables = self.tables.read().unwrap();
            tables.contains_key(&canonical_name)
        };
        tracing::trace!("DEBUG create_table_plan: exists={}", exists);
        if exists {
            if plan.or_replace {
                tracing::trace!(
                    "DEBUG create_table_plan: table '{}' exists and or_replace=true, removing existing table before recreation",
                    display_name
                );
                self.remove_table_entry(&canonical_name);
            } else if plan.if_not_exists {
                tracing::trace!(
                    "DEBUG create_table_plan: table '{}' exists and if_not_exists=true, returning early WITHOUT creating",
                    display_name
                );
                return Ok(StatementResult::CreateTable {
                    table_name: display_name,
                });
            } else {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' already exists",
                    display_name
                )));
            }
        }

        self.dropped_tables.write().unwrap().remove(&canonical_name);

        match plan.source {
            Some(CreateTableSource::Batches { schema, batches }) => self.create_table_from_batches(
                display_name,
                canonical_name,
                schema,
                batches,
                plan.if_not_exists,
            ),
            Some(CreateTableSource::Select { .. }) => Err(Error::Internal(
                "CreateTableSource::Select should be materialized before reaching Context::create_table_plan"
                    .into(),
            )),
            None => self.create_table_from_columns(
                display_name,
                canonical_name,
                plan.columns,
                plan.if_not_exists,
            ),
        }
    }

    pub fn table_names(self: &Arc<Self>) -> Vec<String> {
        let tables = self.tables.read().unwrap();
        tables.keys().cloned().collect()
    }

    fn filter_visible_row_ids(
        &self,
        table: &ExecutorTable<P>,
        row_ids: Vec<u64>,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<u64>> {
        filter_row_ids_for_snapshot(
            table.table.store(),
            table.table.table_id(),
            row_ids,
            &self.txn_manager,
            snapshot,
        )
    }

    pub fn create_table_builder(&self, name: &str) -> CreateTableBuilder<'_, P> {
        CreateTableBuilder {
            ctx: self,
            plan: CreateTablePlan::new(name),
        }
    }

    pub fn table_column_specs(self: &Arc<Self>, name: &str) -> Result<Vec<ColumnSpec>> {
        let (_, canonical_name) = canonical_table_name(name)?;
        let table = self.lookup_table(&canonical_name)?;
        Ok(table
            .schema
            .columns
            .iter()
            .map(|column| {
                ColumnSpec::new(
                    column.name.clone(),
                    column.data_type.clone(),
                    column.nullable,
                )
                .with_primary_key(column.primary_key)
            })
            .collect())
    }

    /// Copy table metadata from this context to a staging context.
    /// This creates new Table instances that use the staging context's pager.
    fn copy_tables_to_staging<Q>(&self, staging: &Context<Q>) -> Result<()>
    where
        Q: Pager<Blob = EntryHandle> + Send + Sync + 'static,
    {
        let base_store = ColumnStore::open(Arc::clone(&self.pager))?;
        let base_catalog = SysCatalog::new(&base_store);

        let staging_store = ColumnStore::open(Arc::clone(&staging.pager))?;
        let staging_catalog = SysCatalog::new(&staging_store);

        let mut next_table_id = match base_catalog.get_next_table_id()? {
            Some(value) => value,
            None => {
                let seed = base_catalog.max_table_id()?.unwrap_or(CATALOG_TABLE_ID);
                seed.checked_add(1).ok_or_else(|| {
                    Error::InvalidArgumentError("exhausted available table ids".into())
                })?
            }
        };
        if next_table_id == CATALOG_TABLE_ID {
            next_table_id = next_table_id.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        staging_catalog.put_next_table_id(next_table_id)?;

        let source_tables: Vec<(String, Arc<ExecutorTable<P>>)> = {
            let guard = self.tables.read().unwrap();
            guard
                .iter()
                .map(|(name, table)| (name.clone(), Arc::clone(table)))
                .collect()
        };

        for (table_name, source_table) in source_tables.iter() {
            tracing::trace!(
                "!!! COPY_TABLES_TO_STAGING: Copying table '{}' with {} columns:",
                table_name,
                source_table.schema.columns.len()
            );
            for (idx, col) in source_table.schema.columns.iter().enumerate() {
                tracing::trace!(
                    "    source column[{}]: name='{}' primary_key={}",
                    idx,
                    col.name,
                    col.primary_key
                );
            }

            // Create a new Table instance with the same table_id but using the staging pager
            let new_table = Table::new(source_table.table.table_id(), Arc::clone(&staging.pager))?;

            // Create a new ExecutorTable with the new table but same schema
            // Start with fresh row counters (0) for transaction isolation
            let new_executor_table = Arc::new(ExecutorTable {
                table: Arc::new(new_table),
                schema: source_table.schema.clone(),
                next_row_id: AtomicU64::new(0),
                total_rows: AtomicU64::new(0),
            });

            tracing::trace!(
                "!!! COPY_TABLES_TO_STAGING: After copy, {} columns in new executor table:",
                new_executor_table.schema.columns.len()
            );
            for (idx, col) in new_executor_table.schema.columns.iter().enumerate() {
                tracing::trace!(
                    "    new column[{}]: name='{}' primary_key={}",
                    idx,
                    col.name,
                    col.primary_key
                );
            }

            {
                let mut staging_tables = staging.tables.write().unwrap();
                staging_tables.insert(table_name.clone(), Arc::clone(&new_executor_table));
            }

            // Snapshot existing rows (with row_ids) into staging for transaction isolation.
            // This provides full REPEATABLE READ isolation by copying data at BEGIN.
            let batches = match self.get_batches_with_row_ids(table_name, None) {
                Ok(batches) => batches,
                Err(Error::NotFound) => {
                    // Table exists but has no data yet (empty table)
                    Vec::new()
                }
                Err(e) => return Err(e),
            };
            if !batches.is_empty() {
                for batch in batches {
                    new_executor_table.table.append(&batch)?;
                }
            }

            let next_row_id = source_table.next_row_id.load(Ordering::SeqCst);
            new_executor_table
                .next_row_id
                .store(next_row_id, Ordering::SeqCst);
            let total_rows = source_table.total_rows.load(Ordering::SeqCst);
            new_executor_table
                .total_rows
                .store(total_rows, Ordering::SeqCst);
        }

        let staging_count = staging.tables.read().unwrap().len();
        tracing::trace!(
            "!!! COPY_TABLES_TO_STAGING: Copied {} tables to staging context",
            staging_count
        );
        Ok(())
    }

    pub fn export_table_rows(self: &Arc<Self>, name: &str) -> Result<RowBatch> {
        let handle = TableHandle::new(Arc::clone(self), name)?;
        handle.lazy()?.collect_rows()
    }

    fn execute_create_table(&self, plan: CreateTablePlan) -> Result<StatementResult<P>> {
        self.create_table_plan(plan)
    }

    fn create_table_with_options<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
        if_not_exists: bool,
    ) -> Result<TableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        let mut plan = CreateTablePlan::new(name);
        plan.if_not_exists = if_not_exists;
        plan.columns = columns
            .into_iter()
            .map(|column| column.into_column_spec())
            .collect();
        let result = self.create_table_plan(plan)?;
        match result {
            StatementResult::CreateTable { .. } => TableHandle::new(Arc::clone(self), name),
            other => Err(Error::InvalidArgumentError(format!(
                "unexpected statement result {other:?} when creating table"
            ))),
        }
    }

    pub fn insert(&self, plan: InsertPlan) -> Result<StatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        let result = self.insert_with_snapshot(plan, snapshot)?;
        self.txn_manager.mark_committed(snapshot.txn_id);
        Ok(result)
    }

    pub fn insert_with_snapshot(
        &self,
        plan: InsertPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<StatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;

        // Targeted debug for 'keys' table only
        if display_name == "keys" {
            tracing::trace!(
                "\n[KEYS] INSERT starting - table_id={}, context_pager={:p}",
                table.table.table_id(),
                &*self.pager
            );
            tracing::trace!(
                "[KEYS] Table has {} columns, primary_key columns: {:?}",
                table.schema.columns.len(),
                table
                    .schema
                    .columns
                    .iter()
                    .filter(|c| c.primary_key)
                    .map(|c| &c.name)
                    .collect::<Vec<_>>()
            );
        }

        let result = match plan.source {
            InsertSource::Rows(rows) => self.insert_rows(
                table.as_ref(),
                display_name.clone(),
                rows,
                plan.columns,
                snapshot,
            ),
            InsertSource::Batches(batches) => self.insert_batches(
                table.as_ref(),
                display_name.clone(),
                batches,
                plan.columns,
                snapshot,
            ),
            InsertSource::Select { .. } => Err(Error::Internal(
                "InsertSource::Select should be materialized before reaching Context::insert"
                    .into(),
            )),
        };

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] INSERT completed: {:?}",
                result
                    .as_ref()
                    .map(|_| "OK")
                    .map_err(|e| format!("{:?}", e))
            );
        }

        result
    }

    /// Get raw batches from a table including row_ids, optionally filtered.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    pub fn get_batches_with_row_ids(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
    ) -> Result<Vec<RecordBatch>> {
        self.get_batches_with_row_ids_with_snapshot(table_name, filter, self.default_snapshot())
    }

    pub fn get_batches_with_row_ids_with_snapshot(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<RecordBatch>> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let filter_expr = match filter {
            Some(expr) => translate_predicate(expr, table.schema.as_ref())?,
            None => {
                let field_id = table.schema.first_field_id().ok_or_else(|| {
                    Error::InvalidArgumentError(
                        "table has no columns; cannot perform wildcard scan".into(),
                    )
                })?;
                full_table_scan_filter(field_id)
            }
        };

        // First, get the row_ids that match the filter
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let visible_row_ids = self.filter_visible_row_ids(table.as_ref(), row_ids, snapshot)?;
        if visible_row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Scan to get the column data
        let table_id = table.table.table_id();

        let mut fields: Vec<Field> = Vec::with_capacity(table.schema.columns.len() + 1);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(table.schema.columns.len() + 1);

        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
        arrays.push(Arc::new(UInt64Array::from(visible_row_ids.clone())));

        for column in &table.schema.columns {
            let logical_field_id = LogicalFieldId::for_user(table_id, column.field_id);
            let gathered = table.table.store().gather_rows(
                &[logical_field_id],
                &visible_row_ids,
                GatherNullPolicy::IncludeNulls,
            )?;
            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            fields.push(
                Field::new(&column.name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata),
            );
            arrays.push(gathered.column(0).clone());
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        Ok(vec![batch])
    }

    /// Append batches directly to a table, preserving row_ids from the batches.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    pub fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> Result<usize> {
        let (_, canonical_name) = canonical_table_name(table_name)?;
        let table = self.lookup_table(&canonical_name)?;

        let mut total_rows = 0;
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            // Verify the batch has a row_id column
            let _row_id_idx = batch.schema().index_of(ROW_ID_COLUMN_NAME).map_err(|_| {
                Error::InvalidArgumentError(
                    "batch must contain row_id column for direct append".into(),
                )
            })?;

            // Append the batch directly to the underlying table
            table.table.append(&batch)?;
            total_rows += batch.num_rows();
        }

        Ok(total_rows)
    }

    pub fn update(&self, plan: UpdatePlan) -> Result<StatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        let result = self.update_with_snapshot(plan, snapshot)?;
        self.txn_manager.mark_committed(snapshot.txn_id);
        Ok(result)
    }

    pub fn update_with_snapshot(
        &self,
        plan: UpdatePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<StatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.filter {
            Some(filter) => self.update_filtered_rows(
                table.as_ref(),
                display_name,
                plan.assignments,
                filter,
                snapshot,
            ),
            None => self.update_all_rows(table.as_ref(), display_name, plan.assignments, snapshot),
        }
    }

    pub fn delete(&self, plan: DeletePlan) -> Result<StatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        let result = self.delete_with_snapshot(plan, snapshot)?;
        self.txn_manager.mark_committed(snapshot.txn_id);
        Ok(result)
    }

    pub fn delete_with_snapshot(
        &self,
        plan: DeletePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<StatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.filter {
            Some(filter) => self.delete_filtered_rows(
                table.as_ref(),
                display_name,
                filter,
                snapshot,
                snapshot.txn_id,
            ),
            None => self.delete_all_rows(table.as_ref(), display_name, snapshot, snapshot.txn_id),
        }
    }

    pub fn table_handle(self: &Arc<Self>, name: &str) -> Result<TableHandle<P>> {
        TableHandle::new(Arc::clone(self), name)
    }

    pub fn execute_select(self: &Arc<Self>, plan: SelectPlan) -> Result<SelectExecution<P>> {
        let (_display_name, canonical_name) = canonical_table_name(&plan.table)?;
        // Verify table exists
        let _table = self.lookup_table(&canonical_name)?;

        // Create a plan with canonical table name
        let mut canonical_plan = plan.clone();
        canonical_plan.table = canonical_name;

        // Use the QueryExecutor from llkv-executor
        let provider: Arc<dyn TableProvider<P>> = Arc::new(ContextProvider {
            context: Arc::clone(self),
        });
        let executor = QueryExecutor::new(provider);
        executor.execute_select(canonical_plan)
    }

    pub fn execute_select_with_snapshot(
        self: &Arc<Self>,
        plan: SelectPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<SelectExecution<P>> {
        let (_display_name, canonical_name) = canonical_table_name(&plan.table)?;
        self.lookup_table(&canonical_name)?;

        let mut canonical_plan = plan.clone();
        canonical_plan.table = canonical_name;

        let provider: Arc<dyn TableProvider<P>> = Arc::new(ContextProvider {
            context: Arc::clone(self),
        });
        let executor = QueryExecutor::new(provider);
        let row_filter: Arc<dyn RowIdFilter<P>> = Arc::new(MvccRowIdFilter::new(
            Arc::clone(&self.txn_manager),
            snapshot,
        ));
        executor.execute_select_with_filter(canonical_plan, Some(row_filter))
    }

    fn create_table_from_columns(
        &self,
        display_name: String,
        canonical_name: String,
        columns: Vec<ColumnSpec>,
        if_not_exists: bool,
    ) -> Result<StatementResult<P>> {
        tracing::trace!(
            "\n=== CREATE_TABLE_FROM_COLUMNS: table='{}' columns={} ===",
            display_name,
            columns.len()
        );
        for (idx, col) in columns.iter().enumerate() {
            tracing::trace!(
                "  input column[{}]: name='{}' primary_key={}",
                idx,
                col.name,
                col.primary_key
            );
        }
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires at least one column".into(),
            ));
        }

        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(columns.len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, column) in columns.iter().enumerate() {
            let normalized = column.name.to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }
            tracing::trace!(
                "DEBUG create_table_from_columns[{}]: name='{}' data_type={:?} nullable={} primary_key={}",
                idx,
                column.name,
                column.data_type,
                column.nullable,
                column.primary_key
            );
            column_defs.push(ExecutorColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                field_id: (idx + 1) as FieldId,
            });
            let pushed = column_defs.last().unwrap();
            tracing::trace!(
                "DEBUG create_table_from_columns[{}]: pushed ExecutorColumn name='{}' primary_key={}",
                idx,
                pushed.name,
                pushed.primary_key
            );
        }

        let table_id = self.reserve_table_id()?;
        tracing::trace!(
            "=== TABLE '{}' CREATED WITH table_id={} pager={:p} ===",
            display_name,
            table_id,
            &*self.pager
        );
        let table = Table::new(table_id, Arc::clone(&self.pager))?;
        table.put_table_meta(&TableMeta {
            table_id,
            name: Some(display_name.clone()),
            created_at_micros: current_time_micros(),
            flags: 0,
            epoch: 0,
        });

        for column in &column_defs {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let schema = Arc::new(ExecutorSchema {
            columns: column_defs,
            lookup,
        });
        let table_entry = Arc::new(ExecutorTable {
            table: Arc::new(table),
            schema,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
        });

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(StatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name, table_entry);
        Ok(StatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn create_table_from_batches(
        &self,
        display_name: String,
        canonical_name: String,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        if_not_exists: bool,
    ) -> Result<StatementResult<P>> {
        if schema.fields().is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one column".into(),
            ));
        }
        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(schema.fields().len());
        let mut lookup: HashMap<String, usize> = HashMap::new();
        for (idx, field) in schema.fields().iter().enumerate() {
            let data_type = match field.data_type() {
                DataType::Int64 | DataType::Float64 | DataType::Utf8 | DataType::Date32 => {
                    field.data_type().clone()
                }
                other => {
                    return Err(Error::InvalidArgumentError(format!(
                        "unsupported column type in CTAS result: {other:?}"
                    )));
                }
            };
            let normalized = field.name().to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in CTAS result",
                    field.name()
                )));
            }
            column_defs.push(ExecutorColumn {
                name: field.name().to_string(),
                data_type,
                nullable: field.is_nullable(),
                primary_key: false, // CTAS does not preserve PRIMARY KEY constraints
                field_id: (idx + 1) as FieldId,
            });
        }

        let table_id = self.reserve_table_id()?;
        let table = Table::new(table_id, Arc::clone(&self.pager))?;
        table.put_table_meta(&TableMeta {
            table_id,
            name: Some(display_name.clone()),
            created_at_micros: current_time_micros(),
            flags: 0,
            epoch: 0,
        });

        for column in &column_defs {
            table.put_col_meta(&ColMeta {
                col_id: column.field_id,
                name: Some(column.name.clone()),
                flags: 0,
                default: None,
            });
        }

        let schema_arc = Arc::new(ExecutorSchema {
            columns: column_defs.clone(),
            lookup,
        });
        let table_entry = Arc::new(ExecutorTable {
            table: Arc::new(table),
            schema: schema_arc,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
        });

        let mut next_row_id: u64 = 0;
        let mut total_rows: u64 = 0;
        let creator_snapshot = self.txn_manager.begin_transaction();
        let creator_txn_id = creator_snapshot.txn_id;
        for batch in batches {
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            if batch.num_columns() != column_defs.len() {
                return Err(Error::InvalidArgumentError(
                    "CTAS query returned unexpected column count".into(),
                ));
            }
            let start_row = next_row_id;
            next_row_id += row_count as u64;
            total_rows += row_count as u64;

            let mut row_builder = UInt64Builder::with_capacity(row_count);
            for offset in 0..row_count {
                row_builder.append_value(start_row + offset as u64);
            }

            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_defs.len() + 3);
            arrays.push(Arc::new(row_builder.finish()) as ArrayRef);

            let mut created_builder = UInt64Builder::with_capacity(row_count);
            let mut deleted_builder = UInt64Builder::with_capacity(row_count);
            for _ in 0..row_count {
                created_builder.append_value(creator_txn_id);
                deleted_builder.append_value(TXN_ID_NONE);
            }
            arrays.push(Arc::new(created_builder.finish()) as ArrayRef);
            arrays.push(Arc::new(deleted_builder.finish()) as ArrayRef);

            let mut fields: Vec<Field> = Vec::with_capacity(column_defs.len() + 3);
            fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
            fields.push(Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false));
            fields.push(Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false));

            for (idx, column) in column_defs.iter().enumerate() {
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    column.field_id.to_string(),
                );
                let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                    .with_metadata(metadata);
                fields.push(field);
                arrays.push(batch.column(idx).clone());
            }

            let append_schema = Arc::new(Schema::new(fields));
            let append_batch = RecordBatch::try_new(append_schema, arrays)?;
            table_entry.table.append(&append_batch)?;
        }

        self.txn_manager.mark_committed(creator_txn_id);

        table_entry.next_row_id.store(next_row_id, Ordering::SeqCst);
        table_entry.total_rows.store(total_rows, Ordering::SeqCst);

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(StatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }
        tracing::trace!(
            "=== INSERTING TABLE '{}' INTO TABLES MAP (pager={:p}) ===",
            canonical_name,
            &*self.pager
        );
        for (idx, col) in table_entry.schema.columns.iter().enumerate() {
            tracing::trace!(
                "  inserting column[{}]: name='{}' primary_key={}",
                idx,
                col.name,
                col.primary_key
            );
        }
        tables.insert(canonical_name.clone(), table_entry);
        Ok(StatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn check_primary_key_constraints(
        &self,
        table: &ExecutorTable<P>,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        let _table_id = table.table.table_id();
        // Find columns with PRIMARY KEY constraint
        let primary_key_columns: Vec<(usize, &ExecutorColumn)> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, col)| col.primary_key)
            .collect();

        if primary_key_columns.is_empty() {
            return Ok(());
        }

        // For each PRIMARY KEY column, check for duplicates
        for (col_idx, column) in primary_key_columns {
            // Get existing values for this column from the table
            let field_id = column.field_id;
            let existing_values = self.scan_column_values(table, field_id, snapshot)?;

            tracing::trace!(
                "[PK_CHECK] snapshot(txn={}, snap_id={}) column '{}': found {} existing VISIBLE values: {:?}",
                snapshot.txn_id,
                snapshot.snapshot_id,
                column.name,
                existing_values.len(),
                existing_values
            );

            // Check each new row value against existing values
            for row in rows {
                // Find which position in the INSERT statement corresponds to this column
                let insert_position = column_order
                    .iter()
                    .position(|&dest_idx| dest_idx == col_idx);

                if let Some(pos) = insert_position {
                    let new_value = &row[pos];

                    // Skip NULL values (PRIMARY KEY typically requires NOT NULL, but we check anyway)
                    if matches!(new_value, PlanValue::Null) {
                        continue;
                    }

                    // Check if this value already exists
                    if existing_values.contains(new_value) {
                        return Err(Error::ConstraintError(format!(
                            "constraint violation on column '{}'",
                            column.name
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    fn scan_column_values(
        &self,
        table: &ExecutorTable<P>,
        field_id: FieldId,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<PlanValue>> {
        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        // Create a filter that matches all rows (unbounded range)
        let match_all_filter = Filter {
            field_id,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        // Get all matching row_ids first
        let row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        
        // Apply MVCC filtering manually using filter_row_ids_for_snapshot
        let row_ids = filter_row_ids_for_snapshot(
            table.table.store(),
            table_id,
            row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Gather the column values for visible rows
        let logical_field_id = LogicalFieldId::for_user(table_id, field_id);
        let batch = match table.table.store().gather_rows(
            &[logical_field_id],
            &row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(b) => b,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut values = Vec::with_capacity(row_ids.len());
        if batch.num_columns() > 0 {
            let array = batch.column(0);
            for row_idx in 0..batch.num_rows() {
                if let Ok(value) = llkv_plan::plan_value_from_array(array, row_idx) {
                    values.push(value);
                }
            }
        }

        Ok(values)
    }

    fn insert_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        rows: Vec<Vec<PlanValue>>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<StatementResult<P>> {
        if rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "INSERT requires at least one row".into(),
            ));
        }

        let column_order = resolve_insert_columns(&columns, table.schema.as_ref())?;
        let expected_len = column_order.len();
        for row in &rows {
            if row.len() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} values in INSERT row, found {}",
                    expected_len,
                    row.len()
                )));
            }
        }

        // Check PRIMARY KEY constraints  
        self.check_primary_key_constraints(table, &rows, &column_order, snapshot)?;

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] Checking PRIMARY KEY constraints - {} rows to insert",
                rows.len()
            );
            for (i, row) in rows.iter().enumerate() {
                tracing::trace!("[KEYS]   row[{}]: {:?}", i, row);
            }
        }

        let constraint_result = self.check_primary_key_constraints(table, &rows, &column_order, snapshot);

        if display_name == "keys" {
            match &constraint_result {
                Ok(_) => tracing::trace!("[KEYS] PRIMARY KEY check PASSED"),
                Err(e) => tracing::trace!("[KEYS] PRIMARY KEY check FAILED: {:?}", e),
            }
        }

        constraint_result?;

        let row_count = rows.len();
        let mut column_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_count); table.schema.columns.len()];
        for row in rows {
            for (idx, value) in row.into_iter().enumerate() {
                let dest_index = column_order[idx];
                column_values[dest_index].push(value);
            }
        }

        let mut row_id_builder = UInt64Builder::with_capacity(row_count);
        let start_row = table.next_row_id.load(Ordering::SeqCst);
        for offset in 0..row_count {
            row_id_builder.append_value(start_row + offset as u64);
        }

        // MVCC: Build transaction ID columns
        let mut created_by_builder = UInt64Builder::with_capacity(row_count);
        let mut deleted_by_builder = UInt64Builder::with_capacity(row_count);
        for _ in 0..row_count {
            created_by_builder.append_value(snapshot.txn_id);
            deleted_by_builder.append_value(TXN_ID_NONE);
        }

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_values.len() + 3); // +3 for row_id, created_by, deleted_by
        let row_id_array: ArrayRef = Arc::new(row_id_builder.finish());
        let created_by_array: ArrayRef = Arc::new(created_by_builder.finish());
        let deleted_by_array: ArrayRef = Arc::new(deleted_by_builder.finish());

        arrays.push(Arc::clone(&row_id_array));
        arrays.push(Arc::clone(&created_by_array));
        arrays.push(Arc::clone(&deleted_by_array));

        let mut fields: Vec<Field> = Vec::with_capacity(column_values.len() + 3);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        fields.push(Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false));
        fields.push(Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false));

        for (column, values) in table.schema.columns.iter().zip(column_values.into_iter()) {
            let array = build_array_for_column(&column.data_type, &values)?;
            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            arrays.push(array);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;
        table
            .next_row_id
            .store(start_row + row_count as u64, Ordering::SeqCst);
        table
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        Ok(StatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }

    fn insert_batches(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        batches: Vec<RecordBatch>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<StatementResult<P>> {
        if batches.is_empty() {
            return Ok(StatementResult::Insert {
                table_name: display_name,
                rows_inserted: 0,
            });
        }

        let expected_len = if columns.is_empty() {
            table.schema.columns.len()
        } else {
            columns.len()
        };
        let mut total_rows_inserted = 0usize;

        for batch in batches {
            if batch.num_columns() != expected_len {
                return Err(Error::InvalidArgumentError(format!(
                    "expected {} columns in INSERT batch, found {}",
                    expected_len,
                    batch.num_columns()
                )));
            }
            let row_count = batch.num_rows();
            if row_count == 0 {
                continue;
            }
            let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(row_count);
            for row_idx in 0..row_count {
                let mut row: Vec<PlanValue> = Vec::with_capacity(expected_len);
                for col_idx in 0..expected_len {
                    let array = batch.column(col_idx);
                    row.push(llkv_plan::plan_value_from_array(array, row_idx)?);
                }
                rows.push(row);
            }

            match self.insert_rows(table, display_name.clone(), rows, columns.clone(), snapshot)? {
                StatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(StatementResult::Insert {
            table_name: display_name,
            rows_inserted: total_rows_inserted,
        })
    }

    fn update_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        assignments: Vec<ColumnAssignment>,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
    ) -> Result<StatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;

        // TODO: Dedupe
        enum PreparedValue {
            Literal(PlanValue),
            Expression { expr_index: usize },
        }

        let mut seen_columns: HashSet<String> = HashSet::new();
        let mut prepared: Vec<(ExecutorColumn, PreparedValue)> =
            Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedValue::Literal(value)));
                }
                AssignmentValue::Expression(expr) => {
                    let translated = translate_scalar(&expr, schema)?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    prepared.push((column.clone(), PreparedValue::Expression { expr_index }));
                }
            }
        }

        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(StatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let table_id = table.table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let gathered = table.table.store().gather_rows(
            &logical_fields,
            &row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        for (col_idx, _column) in table.schema.columns.iter().enumerate() {
            let array = gathered.column(col_idx);
            for (row_idx, row) in new_rows.iter_mut().enumerate().take(row_count) {
                let value = llkv_plan::plan_value_from_array(array, row_idx)?;
                row.push(value);
            }
        }

        let column_positions: HashMap<FieldId, usize> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, column)| (column.field_id, idx))
            .collect();

        for (column, value) in prepared {
            let column_index =
                column_positions
                    .get(&column.field_id)
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' missing in table schema during UPDATE",
                            column.name
                        ))
                    })?;

            let values = match value {
                PreparedValue::Literal(lit) => vec![lit; row_count],
                PreparedValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };

            for (row_idx, new_value) in values.into_iter().enumerate() {
                if let Some(row) = new_rows.get_mut(row_idx) {
                    row[column_index] = new_value;
                }
            }
        }

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            row_ids.clone(),
            snapshot.txn_id,
        )?;

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(StatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn update_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        assignments: Vec<ColumnAssignment>,
        snapshot: TransactionSnapshot,
    ) -> Result<StatementResult<P>> {
        if assignments.is_empty() {
            return Err(Error::InvalidArgumentError(
                "UPDATE requires at least one assignment".into(),
            ));
        }

        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let total_rows_usize = usize::try_from(total_rows).map_err(|_| {
            Error::InvalidArgumentError("table row count exceeds supported range".into())
        })?;
        if total_rows_usize == 0 {
            return Ok(StatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let schema = table.schema.as_ref();

        // TODO: Dedupe
        enum PreparedValue {
            Literal(PlanValue),
            Expression { expr_index: usize },
        }

        let mut seen_columns: HashSet<String> = HashSet::new();
        let mut prepared: Vec<(ExecutorColumn, PreparedValue)> =
            Vec::with_capacity(assignments.len());
        let mut scalar_exprs: Vec<ScalarExpr<FieldId>> = Vec::new();
        let mut first_field_id: Option<FieldId> = None;

        for assignment in assignments {
            let normalized = assignment.column.to_ascii_lowercase();
            if !seen_columns.insert(normalized.clone()) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in UPDATE assignments",
                    assignment.column
                )));
            }
            let column = table.schema.resolve(&assignment.column).ok_or_else(|| {
                Error::InvalidArgumentError(format!(
                    "unknown column '{}' in UPDATE",
                    assignment.column
                ))
            })?;
            if first_field_id.is_none() {
                first_field_id = Some(column.field_id);
            }

            match assignment.value {
                AssignmentValue::Literal(value) => {
                    prepared.push((column.clone(), PreparedValue::Literal(value)));
                }
                AssignmentValue::Expression(expr) => {
                    let translated = translate_scalar(&expr, schema)?;
                    let expr_index = scalar_exprs.len();
                    scalar_exprs.push(translated);
                    prepared.push((column.clone(), PreparedValue::Expression { expr_index }));
                }
            }
        }

        let anchor_field = first_field_id.ok_or_else(|| {
            Error::InvalidArgumentError("UPDATE requires at least one target column".into())
        })?;

        let filter_expr = full_table_scan_filter(anchor_field);
        let (row_ids, mut expr_values) =
            self.collect_update_rows(table, &filter_expr, &scalar_exprs, snapshot)?;

        if row_ids.is_empty() {
            return Ok(StatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let table_id = table.table.table_id();
        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table_id, column.field_id))
            .collect();

        let gathered = table.table.store().gather_rows(
            &logical_fields,
            &row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        for (col_idx, _column) in table.schema.columns.iter().enumerate() {
            let array = gathered.column(col_idx);
            for (row_idx, row) in new_rows.iter_mut().enumerate().take(row_count) {
                let value = llkv_plan::plan_value_from_array(array, row_idx)?;
                row.push(value);
            }
        }

        let column_positions: HashMap<FieldId, usize> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, column)| (column.field_id, idx))
            .collect();

        for (column, value) in prepared {
            let column_index =
                column_positions
                    .get(&column.field_id)
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "column '{}' missing in table schema during UPDATE",
                            column.name
                        ))
                    })?;

            let values = match value {
                PreparedValue::Literal(lit) => vec![lit; row_count],
                PreparedValue::Expression { expr_index } => {
                    let column_values = expr_values.get_mut(expr_index).ok_or_else(|| {
                        Error::InvalidArgumentError(
                            "expression assignment value missing during UPDATE".into(),
                        )
                    })?;
                    if column_values.len() != row_count {
                        return Err(Error::InvalidArgumentError(
                            "expression result count did not match targeted row count".into(),
                        ));
                    }
                    mem::take(column_values)
                }
            };

            for (row_idx, new_value) in values.into_iter().enumerate() {
                if let Some(row) = new_rows.get_mut(row_idx) {
                    row[column_index] = new_value;
                }
            }
        }

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            row_ids.clone(),
            snapshot.txn_id,
        )?;

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(StatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn delete_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
        txn_id: TxnId,
    ) -> Result<StatementResult<P>> {
        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        tracing::trace!(
            table = %display_name,
            rows = row_ids.len(),
            "delete_filtered_rows collected row ids"
        );
        self.apply_delete(table, display_name, row_ids, txn_id)
    }

    fn delete_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        snapshot: TransactionSnapshot,
        txn_id: TxnId,
    ) -> Result<StatementResult<P>> {
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if total_rows == 0 {
            return Ok(StatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let anchor_field = table.schema.first_field_id().ok_or_else(|| {
            Error::InvalidArgumentError("DELETE requires a table with at least one column".into())
        })?;
        let filter_expr = full_table_scan_filter(anchor_field);
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        self.apply_delete(table, display_name, row_ids, txn_id)
    }

    fn apply_delete(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        row_ids: Vec<u64>,
        txn_id: TxnId,
    ) -> Result<StatementResult<P>> {
        if row_ids.is_empty() {
            return Ok(StatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let removed = row_ids.len();

        let mut fields = Vec::with_capacity(2);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(2);

        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));
        let row_id_array: ArrayRef = Arc::new(UInt64Array::from(row_ids.clone()));
        arrays.push(row_id_array);

        fields.push(Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false));
        let deleted_array: ArrayRef = Arc::new(UInt64Array::from(vec![txn_id; removed]));
        arrays.push(deleted_array);

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;

        let removed_u64 = u64::try_from(removed)
            .map_err(|_| Error::InvalidArgumentError("row count exceeds supported range".into()))?;
        table.total_rows.fetch_sub(removed_u64, Ordering::SeqCst);

        Ok(StatementResult::Delete {
            table_name: display_name,
            rows_deleted: removed,
        })
    }

    fn collect_update_rows(
        &self,
        table: &ExecutorTable<P>,
        filter_expr: &LlkvExpr<'static, FieldId>,
        expressions: &[ScalarExpr<FieldId>],
        snapshot: TransactionSnapshot,
    ) -> Result<(Vec<u64>, Vec<Vec<PlanValue>>)> {
        let row_ids = table.table.filter_row_ids(filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        if row_ids.is_empty() {
            return Ok((row_ids, vec![Vec::new(); expressions.len()]));
        }

        if expressions.is_empty() {
            return Ok((row_ids, Vec::new()));
        }

        let mut projections: Vec<ScanProjection> = Vec::with_capacity(expressions.len());
        for (idx, expr) in expressions.iter().enumerate() {
            let alias = format!("__expr_{idx}");
            projections.push(ScanProjection::computed(expr.clone(), alias));
        }

        let mut expr_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_ids.len()); expressions.len()];
        let mut error: Option<Error> = None;
        let row_filter: Arc<dyn RowIdFilter<P>> = Arc::new(MvccRowIdFilter::new(
            Arc::clone(&self.txn_manager),
            snapshot,
        ));
        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
            row_id_filter: Some(row_filter),
        };

        table
            .table
            .scan_stream_with_exprs(&projections, filter_expr, options, |batch| {
                if error.is_some() {
                    return;
                }
                if let Err(err) = Self::collect_expression_values(&mut expr_values, batch) {
                    error = Some(err);
                }
            })?;

        if let Some(err) = error {
            return Err(err);
        }

        for values in &expr_values {
            if values.len() != row_ids.len() {
                return Err(Error::InvalidArgumentError(
                    "expression result count did not match targeted row count".into(),
                ));
            }
        }

        Ok((row_ids, expr_values))
    }

    fn collect_expression_values(
        expr_values: &mut [Vec<PlanValue>],
        batch: RecordBatch,
    ) -> Result<()> {
        for row_idx in 0..batch.num_rows() {
            for (expr_index, values) in expr_values.iter_mut().enumerate() {
                let value = llkv_plan::plan_value_from_array(batch.column(expr_index), row_idx)?;
                values.push(value);
            }
        }

        Ok(())
    }

    fn lookup_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        let tables = self.tables.read().unwrap();
        let table = tables.get(canonical_name).cloned().ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{canonical_name}'"))
        })?;
        tracing::trace!(
            "=== LOOKUP_TABLE '{}' table_id={} columns={} context_pager={:p} ===",
            canonical_name,
            table.table.table_id(),
            table.schema.columns.len(),
            &*self.pager
        );
        for (idx, col) in table.schema.columns.iter().enumerate() {
            tracing::trace!(
                "  column[{}]: name='{}' primary_key={}",
                idx,
                col.name,
                col.primary_key
            );
        }
        Ok(table)
    }

    fn remove_table_entry(&self, canonical_name: &str) {
        let mut tables = self.tables.write().unwrap();
        if tables.remove(canonical_name).is_some() {
            tracing::trace!(
                "remove_table_entry: removed table '{}' from context cache",
                canonical_name
            );
        }
    }

    pub fn drop_table_immediate(&self, name: &str, if_exists: bool) -> Result<()> {
        let (display_name, canonical_name) = canonical_table_name(name)?;
        let tables = self.tables.read().unwrap();
        if !tables.contains_key(&canonical_name) {
            if if_exists {
                return Ok(());
            } else {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' does not exist",
                    display_name
                )));
            }
        }
        drop(tables);
        self.dropped_tables.write().unwrap().insert(canonical_name);
        Ok(())
    }

    pub fn is_table_marked_dropped(&self, canonical_name: &str) -> bool {
        self.dropped_tables.read().unwrap().contains(canonical_name)
    }

    fn reserve_table_id(&self) -> Result<TableId> {
        let store = ColumnStore::open(Arc::clone(&self.pager))?;
        let catalog = SysCatalog::new(&store);

        let mut next = match catalog.get_next_table_id()? {
            Some(value) => value,
            None => {
                let seed = catalog.max_table_id()?.unwrap_or(CATALOG_TABLE_ID);
                let initial = seed.checked_add(1).ok_or_else(|| {
                    Error::InvalidArgumentError("exhausted available table ids".into())
                })?;
                catalog.put_next_table_id(initial)?;
                initial
            }
        };

        if next == CATALOG_TABLE_ID {
            next = next.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        let mut following = next
            .checked_add(1)
            .ok_or_else(|| Error::InvalidArgumentError("exhausted available table ids".into()))?;
        if following == CATALOG_TABLE_ID {
            following = following.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }
        catalog.put_next_table_id(following)?;
        Ok(next)
    }
}

// Implement TransactionContext for ContextWrapper to enable llkv-transaction integration
impl<P> TransactionContext for ContextWrapper<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type Pager = P;

    fn set_snapshot(&self, snapshot: TransactionSnapshot) {
        self.update_snapshot(snapshot);
    }

    fn snapshot(&self) -> TransactionSnapshot {
        self.current_snapshot()
    }

    fn table_column_specs(&self, table_name: &str) -> llkv_result::Result<Vec<ColumnSpec>> {
        Context::table_column_specs(self.context(), table_name)
    }

    fn export_table_rows(
        &self,
        table_name: &str,
    ) -> llkv_result::Result<llkv_transaction::RowBatch> {
        let batch = Context::export_table_rows(self.context(), table_name)?;
        // Convert from llkv_executor::RowBatch to llkv_transaction::RowBatch
        Ok(llkv_transaction::RowBatch {
            columns: batch.columns,
            rows: batch.rows,
        })
    }

    fn get_batches_with_row_ids(
        &self,
        table_name: &str,
        filter: Option<LlkvExpr<'static, String>>,
    ) -> llkv_result::Result<Vec<RecordBatch>> {
        Context::get_batches_with_row_ids_with_snapshot(
            self.context(),
            table_name,
            filter,
            self.snapshot(),
        )
    }

    fn execute_select(
        &self,
        plan: SelectPlan,
    ) -> llkv_result::Result<SelectExecution<Self::Pager>> {
        Context::execute_select_with_snapshot(self.context(), plan, self.snapshot())
    }

    fn create_table_plan(
        &self,
        plan: CreateTablePlan,
    ) -> llkv_result::Result<TransactionResult<P>> {
        let result = Context::create_table_plan(self.context(), plan)?;
        Ok(convert_statement_result(result))
    }

    fn insert(&self, plan: InsertPlan) -> llkv_result::Result<TransactionResult<P>> {
        tracing::trace!(
            "[WRAPPER] TransactionContext::insert called - plan.table='{}', wrapper_context_pager={:p}",
            plan.table,
            &*self.ctx.pager
        );
        let snapshot = self.current_snapshot();
        let result = if snapshot.txn_id == TXN_ID_AUTO_COMMIT {
            self.ctx().insert(plan)?
        } else {
            Context::insert_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn update(&self, plan: UpdatePlan) -> llkv_result::Result<TransactionResult<P>> {
        let snapshot = self.current_snapshot();
        let result = if snapshot.txn_id == TXN_ID_AUTO_COMMIT {
            self.ctx().update(plan)?
        } else {
            Context::update_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn delete(&self, plan: DeletePlan) -> llkv_result::Result<TransactionResult<P>> {
        let snapshot = self.current_snapshot();
        let result = if snapshot.txn_id == TXN_ID_AUTO_COMMIT {
            self.ctx().delete(plan)?
        } else {
            Context::delete_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> llkv_result::Result<usize> {
        Context::append_batches_with_row_ids(self.context(), table_name, batches)
    }

    fn table_names(&self) -> Vec<String> {
        Context::table_names(self.context())
    }

    fn table_id(&self, table_name: &str) -> llkv_result::Result<llkv_table::types::TableId> {
        // Check CURRENT state: if table is marked as dropped, return error
        // This is used by conflict detection to detect if a table was dropped
        let ctx = self.context();
        if ctx.is_table_marked_dropped(table_name) {
            return Err(Error::InvalidArgumentError(format!(
                "table '{}' has been dropped",
                table_name
            )));
        }
        
        let table = ctx.lookup_table(table_name)?;
        Ok(table.table.table_id())
    }
}

// Helper to convert StatementResult between types (legacy)
fn convert_statement_result<P>(result: StatementResult<P>) -> TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    use llkv_transaction::TransactionResult as TxResult;
    match result {
        StatementResult::CreateTable { table_name } => TxResult::CreateTable { table_name },
        StatementResult::Insert { rows_inserted, .. } => TxResult::Insert { rows_inserted },
        StatementResult::Update { rows_updated, .. } => TxResult::Update {
            rows_matched: rows_updated,
            rows_updated,
        },
        StatementResult::Delete { rows_deleted, .. } => TxResult::Delete { rows_deleted },
        StatementResult::Transaction { kind } => TxResult::Transaction { kind },
        _ => panic!("unsupported StatementResult conversion"),
    }
}

fn filter_row_ids_for_snapshot<P>(
    store: &ColumnStore<P>,
    table_id: TableId,
    row_ids: Vec<u64>,
    txn_manager: &TxnIdManager,
    snapshot: TransactionSnapshot,
) -> Result<Vec<u64>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    tracing::debug!("[FILTER_ROWS] Filtering {} row IDs for snapshot txn_id={}, snapshot_id={}", 
        row_ids.len(), snapshot.txn_id, snapshot.snapshot_id);
    
    if row_ids.is_empty() {
        return Ok(row_ids);
    }

    let created_lfid = LogicalFieldId::for_mvcc_created_by(table_id);
    let deleted_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);

    let version_batch = match store.gather_rows(
        &[created_lfid, deleted_lfid],
        &row_ids,
        GatherNullPolicy::IncludeNulls,
    ) {
        Ok(batch) => batch,
        Err(Error::NotFound) => {
            tracing::trace!("[FILTER_ROWS] gather_rows returned NotFound for MVCC columns, treating all {} rows as visible (committed)", row_ids.len());
            return Ok(row_ids);
        }
        Err(err) => {
            tracing::error!("[FILTER_ROWS] gather_rows error: {:?}", err);
            return Err(err);
        }
    };

    if version_batch.num_columns() < 2 {
        tracing::debug!("[FILTER_ROWS] version_batch has < 2 columns, returning all {} rows", row_ids.len());
        return Ok(row_ids);
    }

    let created_column = version_batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>();
    let deleted_column = version_batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>();

    if created_column.is_none() || deleted_column.is_none() {
        tracing::debug!("[FILTER_ROWS] Failed to downcast columns, returning all {} rows", row_ids.len());
        return Ok(row_ids);
    }

    let created_column = created_column.unwrap();
    let deleted_column = deleted_column.unwrap();

    let mut visible = Vec::with_capacity(row_ids.len());
    for (idx, row_id) in row_ids.iter().enumerate() {
        let created_by = if created_column.is_null(idx) {
            TXN_ID_AUTO_COMMIT
        } else {
            created_column.value(idx)
        };
        let deleted_by = if deleted_column.is_null(idx) {
            TXN_ID_NONE
        } else {
            deleted_column.value(idx)
        };

        let version = RowVersion {
            created_by,
            deleted_by,
        };
        let is_visible = version.is_visible_for(txn_manager, snapshot);
        tracing::trace!(
            "[FILTER_ROWS] row_id={}: created_by={}, deleted_by={}, is_visible={}",
            row_id,
            created_by,
            deleted_by,
            is_visible
        );
        if is_visible {
            visible.push(*row_id);
        }
    }

    tracing::debug!("[FILTER_ROWS] Filtered from {} to {} visible rows", row_ids.len(), visible.len());
    Ok(visible)
}

struct MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    txn_manager: Arc<TxnIdManager>,
    snapshot: TransactionSnapshot,
    _marker: PhantomData<fn(P)>,
}

impl<P> MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new(txn_manager: Arc<TxnIdManager>, snapshot: TransactionSnapshot) -> Self {
        Self {
            txn_manager,
            snapshot,
            _marker: PhantomData,
        }
    }
}

impl<P> RowIdFilter<P> for MvccRowIdFilter<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn filter(&self, table: &Table<P>, row_ids: Vec<u64>) -> Result<Vec<u64>> {
        tracing::trace!(
            "[MVCC_FILTER] filter() called with {} row_ids, snapshot txn={}, snapshot_id={}",
            row_ids.len(),
            self.snapshot.txn_id,
            self.snapshot.snapshot_id
        );
        filter_row_ids_for_snapshot(
            table.store(),
            table.table_id(),
            row_ids,
            &self.txn_manager,
            self.snapshot,
        )
    }
}

// Wrapper to implement TableProvider for Context
struct ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<Context<P>>,
}

impl<P> TableProvider<P> for ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        self.context.lookup_table(canonical_name)
    }
}

/// Lazily built logical plan (thin wrapper over SelectPlan).
pub struct LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<Context<P>>,
    plan: SelectPlan,
}

impl<P> LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn scan(context: Arc<Context<P>>, table: &str) -> Result<Self> {
        let (display, canonical) = canonical_table_name(table)?;
        context.lookup_table(&canonical)?;
        Ok(Self {
            context,
            plan: SelectPlan::new(display),
        })
    }

    pub fn filter(mut self, predicate: LlkvExpr<'static, String>) -> Self {
        self.plan.filter = Some(predicate);
        self
    }

    pub fn select_all(mut self) -> Self {
        self.plan.projections = vec![SelectProjection::AllColumns];
        self
    }

    pub fn select_columns<S>(mut self, columns: impl IntoIterator<Item = S>) -> Self
    where
        S: AsRef<str>,
    {
        self.plan.projections = columns
            .into_iter()
            .map(|name| SelectProjection::Column {
                name: name.as_ref().to_string(),
                alias: None,
            })
            .collect();
        self
    }

    pub fn select(mut self, projections: Vec<SelectProjection>) -> Self {
        self.plan.projections = projections;
        self
    }

    pub fn aggregate(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.plan.aggregates = aggregates;
        self
    }

    pub fn collect(self) -> Result<SelectExecution<P>> {
        self.context.execute_select(self.plan)
    }

    pub fn collect_rows(self) -> Result<RowBatch> {
        let execution = self.context.execute_select(self.plan)?;
        execution.collect_rows()
    }

    pub fn collect_rows_vec(self) -> Result<Vec<Vec<PlanValue>>> {
        Ok(self.collect_rows()?.rows)
    }
}

pub fn canonical_table_name(name: &str) -> Result<(String, String)> {
    if name.is_empty() {
        return Err(Error::InvalidArgumentError(
            "table name must not be empty".into(),
        ));
    }
    let display = name.to_string();
    let canonical = display.to_ascii_lowercase();
    Ok((display, canonical))
}

fn current_time_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

pub fn resolve_insert_columns(columns: &[String], schema: &ExecutorSchema) -> Result<Vec<usize>> {
    if columns.is_empty() {
        return Ok((0..schema.columns.len()).collect());
    }
    let mut resolved = Vec::with_capacity(columns.len());
    for column in columns {
        let normalized = column.to_ascii_lowercase();
        let index = schema
            .lookup
            .get(&normalized)
            .ok_or_else(|| Error::InvalidArgumentError(format!("unknown column '{}'", column)))?;
        resolved.push(*index);
    }
    Ok(resolved)
}

pub fn build_array_for_column(dtype: &DataType, values: &[PlanValue]) -> Result<ArrayRef> {
    match dtype {
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v),
                    PlanValue::Float(v) => builder.append_value(*v as i64),
                    PlanValue::String(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert string into INT column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v as f64),
                    PlanValue::Float(v) => builder.append_value(*v),
                    PlanValue::String(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert string into DOUBLE column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(v.to_string()),
                    PlanValue::Float(v) => builder.append_value(v.to_string()),
                    PlanValue::String(s) => builder.append_value(s),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Date32 => {
            let mut builder = Date32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(days) => {
                        let casted = i32::try_from(*days).map_err(|_| {
                            Error::InvalidArgumentError(
                                "integer literal out of range for DATE column".into(),
                            )
                        })?;
                        builder.append_value(casted);
                    }
                    PlanValue::Float(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert float into DATE column".into(),
                        ));
                    }
                    PlanValue::String(text) => {
                        let days = parse_date32_literal(text)?;
                        builder.append_value(days);
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type for INSERT: {other:?}"
        ))),
    }
}

fn parse_date32_literal(text: &str) -> Result<i32> {
    let mut parts = text.split('-');
    let year_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let month_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    let day_str = parts
        .next()
        .ok_or_else(|| Error::InvalidArgumentError(format!("invalid DATE literal '{text}'")))?;
    if parts.next().is_some() {
        return Err(Error::InvalidArgumentError(format!(
            "invalid DATE literal '{text}'"
        )));
    }

    let year = year_str.parse::<i32>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid year in DATE literal '{text}'"))
    })?;
    let month_num = month_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;
    let day = day_str.parse::<u8>().map_err(|_| {
        Error::InvalidArgumentError(format!("invalid day in DATE literal '{text}'"))
    })?;

    let month = Month::try_from(month_num).map_err(|_| {
        Error::InvalidArgumentError(format!("invalid month in DATE literal '{text}'"))
    })?;

    let date = Date::from_calendar_date(year, month, day).map_err(|err| {
        Error::InvalidArgumentError(format!("invalid DATE literal '{text}': {err}"))
    })?;
    let days = date.to_julian_day() - epoch_julian_day();
    Ok(days)
}

fn epoch_julian_day() -> i32 {
    Date::from_calendar_date(1970, Month::January, 1)
        .expect("1970-01-01 is a valid date")
        .to_julian_day()
}

// Array -> PlanValue conversion is provided by llkv-plan::plan_value_from_array

fn full_table_scan_filter(field_id: FieldId) -> LlkvExpr<'static, FieldId> {
    LlkvExpr::Pred(Filter {
        field_id,
        op: Operator::Range {
            lower: Bound::Unbounded,
            upper: Bound::Unbounded,
        },
    })
}

fn resolve_field_id_from_schema(schema: &ExecutorSchema, name: &str) -> Result<FieldId> {
    if name.eq_ignore_ascii_case(ROW_ID_COLUMN_NAME) {
        return Ok(ROW_ID_FIELD_ID);
    }

    schema
        .resolve(name)
        .map(|column| column.field_id)
        .ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown column '{name}' in expression"))
        })
}

fn translate_predicate(
    expr: LlkvExpr<'static, String>,
    schema: &ExecutorSchema,
) -> Result<LlkvExpr<'static, FieldId>> {
    match expr {
        LlkvExpr::And(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::And(converted))
        }
        LlkvExpr::Or(list) => {
            let mut converted = Vec::with_capacity(list.len());
            for item in list {
                converted.push(translate_predicate(item, schema)?);
            }
            Ok(LlkvExpr::Or(converted))
        }
        LlkvExpr::Not(inner) => Ok(LlkvExpr::Not(Box::new(translate_predicate(
            *inner, schema,
        )?))),
        LlkvExpr::Pred(Filter { field_id, op }) => {
            let resolved = resolve_field_id_from_schema(schema, &field_id)?;
            Ok(LlkvExpr::Pred(Filter {
                field_id: resolved,
                op,
            }))
        }
        LlkvExpr::Compare { left, op, right } => {
            let left = translate_scalar(&left, schema)?;
            let right = translate_scalar(&right, schema)?;
            Ok(LlkvExpr::Compare { left, op, right })
        }
    }
}

fn translate_scalar(
    expr: &ScalarExpr<String>,
    schema: &ExecutorSchema,
) -> Result<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::Column(name) => {
            let field_id = resolve_field_id_from_schema(schema, name)?;
            Ok(ScalarExpr::column(field_id))
        }
        ScalarExpr::Literal(lit) => Ok(ScalarExpr::Literal(lit.clone())),
        ScalarExpr::Binary { left, op, right } => {
            let left_expr = translate_scalar(left, schema)?;
            let right_expr = translate_scalar(right, schema)?;
            Ok(ScalarExpr::Binary {
                left: Box::new(left_expr),
                op: *op,
                right: Box::new(right_expr),
            })
        }
        ScalarExpr::Aggregate(agg) => {
            // Translate column names in aggregate calls to field IDs
            use llkv_expr::expr::AggregateCall;
            let translated_agg = match agg {
                AggregateCall::CountStar => AggregateCall::CountStar,
                AggregateCall::Count(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Count(field_id)
                }
                AggregateCall::Sum(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Sum(field_id)
                }
                AggregateCall::Min(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Min(field_id)
                }
                AggregateCall::Max(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Max(field_id)
                }
                AggregateCall::CountNulls(name) => {
                    let field_id = resolve_field_id_from_schema(schema, name).map_err(|_| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::CountNulls(field_id)
                }
            };
            Ok(ScalarExpr::Aggregate(translated_agg))
        }
    }
}

fn plan_value_from_sql_expr(expr: &SqlExpr) -> Result<PlanValue> {
    match expr {
        SqlExpr::Value(value) => plan_value_from_sql_value(value),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match plan_value_from_sql_expr(expr)? {
            PlanValue::Integer(v) => Ok(PlanValue::Integer(-v)),
            PlanValue::Float(v) => Ok(PlanValue::Float(-v)),
            PlanValue::Null | PlanValue::String(_) => Err(Error::InvalidArgumentError(
                "cannot negate non-numeric literal".into(),
            )),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => plan_value_from_sql_expr(expr),
        SqlExpr::Nested(inner) => plan_value_from_sql_expr(inner),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported literal expression: {other:?}"
        ))),
    }
}

fn plan_value_from_sql_value(value: &ValueWithSpan) -> Result<PlanValue> {
    match &value.value {
        Value::Null => Ok(PlanValue::Null),
        Value::Number(text, _) => {
            if text.contains(['.', 'e', 'E']) {
                let parsed = text.parse::<f64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid float literal: {err}"))
                })?;
                Ok(PlanValue::Float(parsed))
            } else {
                let parsed = text.parse::<i64>().map_err(|err| {
                    Error::InvalidArgumentError(format!("invalid integer literal: {err}"))
                })?;
                Ok(PlanValue::Integer(parsed))
            }
        }
        Value::Boolean(_) => Err(Error::InvalidArgumentError(
            "BOOLEAN literals are not supported yet".into(),
        )),
        other => {
            if let Some(text) = other.clone().into_string() {
                Ok(PlanValue::String(text))
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unsupported literal: {other:?}"
                )))
            }
        }
    }
}

fn group_by_is_empty(expr: &GroupByExpr) -> bool {
    matches!(
        expr,
        GroupByExpr::Expressions(exprs, modifiers)
            if exprs.is_empty() && modifiers.is_empty()
    )
}

#[derive(Clone)]
pub struct RangeSelectRows {
    rows: Vec<Vec<PlanValue>>,
}

impl RangeSelectRows {
    pub fn into_rows(self) -> Vec<Vec<PlanValue>> {
        self.rows
    }
}

#[derive(Clone)]
enum RangeProjection {
    Column,
    Literal(PlanValue),
}

#[derive(Clone)]
pub struct RangeSpec {
    start: i64,
    #[allow(dead_code)] // Used for validation, computed into row_count
    end: i64,
    row_count: usize,
    column_name_lower: String,
    table_alias_lower: Option<String>,
}

impl RangeSpec {
    fn matches_identifier(&self, ident: &str) -> bool {
        let lower = ident.to_ascii_lowercase();
        lower == self.column_name_lower || lower == "range"
    }

    fn matches_table_alias(&self, ident: &str) -> bool {
        let lower = ident.to_ascii_lowercase();
        match &self.table_alias_lower {
            Some(alias) => lower == *alias,
            None => lower == "range",
        }
    }

    fn matches_object_name(&self, name: &ObjectName) -> bool {
        if name.0.len() != 1 {
            return false;
        }
        match &name.0[0] {
            ObjectNamePart::Identifier(ident) => self.matches_table_alias(&ident.value),
            _ => false,
        }
    }
}

pub fn extract_rows_from_range(select: &Select) -> Result<Option<RangeSelectRows>> {
    let spec = match parse_range_spec(select)? {
        Some(spec) => spec,
        None => return Ok(None),
    };

    if select.selection.is_some() {
        return Err(Error::InvalidArgumentError(
            "WHERE clauses are not supported for range() SELECT statements".into(),
        ));
    }
    if select.having.is_some()
        || !select.named_window.is_empty()
        || select.qualify.is_some()
        || select.distinct.is_some()
        || select.top.is_some()
        || select.into.is_some()
        || select.prewhere.is_some()
        || !select.lateral_views.is_empty()
        || select.value_table_mode.is_some()
        || !group_by_is_empty(&select.group_by)
    {
        return Err(Error::InvalidArgumentError(
            "advanced SELECT clauses are not supported for range() SELECT statements".into(),
        ));
    }

    let mut projections: Vec<RangeProjection> = Vec::with_capacity(select.projection.len());

    // If projection is empty, treat it as SELECT * (implicit wildcard)
    if select.projection.is_empty() {
        projections.push(RangeProjection::Column);
    } else {
        for item in &select.projection {
            let projection = match item {
                SelectItem::Wildcard(_) => RangeProjection::Column,
                SelectItem::QualifiedWildcard(kind, _) => match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(object_name) => {
                        if spec.matches_object_name(object_name) {
                            RangeProjection::Column
                        } else {
                            return Err(Error::InvalidArgumentError(
                                "qualified wildcard must reference the range() source".into(),
                            ));
                        }
                    }
                    SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(Error::InvalidArgumentError(
                            "expression-qualified wildcards are not supported for range() SELECT statements".into(),
                        ));
                    }
                },
                SelectItem::UnnamedExpr(expr) => build_range_projection_expr(expr, &spec)?,
                SelectItem::ExprWithAlias { expr, .. } => build_range_projection_expr(expr, &spec)?,
            };
            projections.push(projection);
        }
    }

    let mut rows: Vec<Vec<PlanValue>> = Vec::with_capacity(spec.row_count);
    for idx in 0..spec.row_count {
        let mut row: Vec<PlanValue> = Vec::with_capacity(projections.len());
        let value = spec.start + (idx as i64);
        for projection in &projections {
            match projection {
                RangeProjection::Column => row.push(PlanValue::Integer(value)),
                RangeProjection::Literal(value) => row.push(value.clone()),
            }
        }
        rows.push(row);
    }

    Ok(Some(RangeSelectRows { rows }))
}

fn build_range_projection_expr(expr: &SqlExpr, spec: &RangeSpec) -> Result<RangeProjection> {
    match expr {
        SqlExpr::Identifier(ident) => {
            if spec.matches_identifier(&ident.value) {
                Ok(RangeProjection::Column)
            } else {
                Err(Error::InvalidArgumentError(format!(
                    "unknown column '{}' in range() SELECT",
                    ident.value
                )))
            }
        }
        SqlExpr::CompoundIdentifier(parts) => {
            if parts.len() == 2
                && spec.matches_table_alias(&parts[0].value)
                && spec.matches_identifier(&parts[1].value)
            {
                Ok(RangeProjection::Column)
            } else {
                Err(Error::InvalidArgumentError(
                    "compound identifiers must reference the range() source".into(),
                ))
            }
        }
        SqlExpr::Wildcard(_) | SqlExpr::QualifiedWildcard(_, _) => unreachable!(),
        other => Ok(RangeProjection::Literal(plan_value_from_sql_expr(other)?)),
    }
}

fn parse_range_spec(select: &Select) -> Result<Option<RangeSpec>> {
    if select.from.len() != 1 {
        return Ok(None);
    }
    let item = &select.from[0];
    if !item.joins.is_empty() {
        return Err(Error::InvalidArgumentError(
            "JOIN clauses are not supported for range() SELECT statements".into(),
        ));
    }

    match &item.relation {
        TableFactor::Function {
            lateral,
            name,
            args,
            alias,
        } => {
            if *lateral {
                return Err(Error::InvalidArgumentError(
                    "LATERAL range() is not supported".into(),
                ));
            }
            parse_range_spec_from_args(name, args, alias)
        }
        TableFactor::Table {
            name,
            alias,
            args: Some(table_args),
            with_ordinality,
            ..
        } => {
            if *with_ordinality {
                return Err(Error::InvalidArgumentError(
                    "WITH ORDINALITY is not supported for range()".into(),
                ));
            }
            if table_args.settings.is_some() {
                return Err(Error::InvalidArgumentError(
                    "range() SETTINGS clause is not supported".into(),
                ));
            }
            parse_range_spec_from_args(name, &table_args.args, alias)
        }
        _ => Ok(None),
    }
}

fn parse_range_spec_from_args(
    name: &ObjectName,
    args: &[FunctionArg],
    alias: &Option<TableAlias>,
) -> Result<Option<RangeSpec>> {
    if name.0.len() != 1 {
        return Ok(None);
    }
    let func_name = match &name.0[0] {
        ObjectNamePart::Identifier(ident) => ident.value.to_ascii_lowercase(),
        _ => return Ok(None),
    };
    if func_name != "range" {
        return Ok(None);
    }

    if args.is_empty() || args.len() > 2 {
        return Err(Error::InvalidArgumentError(
            "range() requires one or two arguments".into(),
        ));
    }

    // Helper to extract integer from argument
    let extract_int = |arg: &FunctionArg| -> Result<i64> {
        let arg_expr = match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => expr,
            FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_))
            | FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                return Err(Error::InvalidArgumentError(
                    "range() argument must be an integer literal".into(),
                ));
            }
            FunctionArg::Named { .. } | FunctionArg::ExprNamed { .. } => {
                return Err(Error::InvalidArgumentError(
                    "named arguments are not supported for range()".into(),
                ));
            }
        };

        let value = plan_value_from_sql_expr(arg_expr)?;
        match value {
            PlanValue::Integer(v) => Ok(v),
            _ => Err(Error::InvalidArgumentError(
                "range() argument must be an integer literal".into(),
            )),
        }
    };

    let (start, end, row_count) = if args.len() == 1 {
        // range(count) - generate [0, count)
        let count = extract_int(&args[0])?;
        if count < 0 {
            return Err(Error::InvalidArgumentError(
                "range() argument must be non-negative".into(),
            ));
        }
        (0, count, count as usize)
    } else {
        // range(start, end) - generate [start, end)
        let start = extract_int(&args[0])?;
        let end = extract_int(&args[1])?;
        if end < start {
            return Err(Error::InvalidArgumentError(
                "range() end must be >= start".into(),
            ));
        }
        let row_count = (end - start) as usize;
        (start, end, row_count)
    };

    let column_name_lower = alias
        .as_ref()
        .and_then(|a| {
            a.columns
                .first()
                .map(|col| col.name.value.to_ascii_lowercase())
        })
        .unwrap_or_else(|| "range".to_string());
    let table_alias_lower = alias.as_ref().map(|a| a.name.value.to_ascii_lowercase());

    Ok(Some(RangeSpec {
        start,
        end,
        row_count,
        column_name_lower,
        table_alias_lower,
    }))
}

pub struct CreateTableBuilder<'ctx, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    ctx: &'ctx Context<P>,
    plan: CreateTablePlan,
}

impl<'ctx, P> CreateTableBuilder<'ctx, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub fn if_not_exists(mut self) -> Self {
        self.plan.if_not_exists = true;
        self
    }

    pub fn or_replace(mut self) -> Self {
        self.plan.or_replace = true;
        self
    }

    pub fn with_column(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.plan
            .columns
            .push(ColumnSpec::new(name.into(), data_type, true));
        self
    }

    pub fn with_not_null_column(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.plan
            .columns
            .push(ColumnSpec::new(name.into(), data_type, false));
        self
    }

    pub fn with_column_spec(mut self, spec: ColumnSpec) -> Self {
        self.plan.columns.push(spec);
        self
    }

    pub fn finish(self) -> Result<StatementResult<P>> {
        self.ctx.execute_create_table(self.plan)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Row {
    values: Vec<(String, PlanValue)>,
}

impl Row {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn with(mut self, name: impl Into<String>, value: impl Into<PlanValue>) -> Self {
        self.set(name, value);
        self
    }

    pub fn set(&mut self, name: impl Into<String>, value: impl Into<PlanValue>) -> &mut Self {
        let name = name.into();
        let value = value.into();
        if let Some((_, existing)) = self.values.iter_mut().find(|(n, _)| *n == name) {
            *existing = value;
        } else {
            self.values.push((name, value));
        }
        self
    }

    fn columns(&self) -> Vec<String> {
        self.values.iter().map(|(n, _)| n.clone()).collect()
    }

    fn values_for_columns(&self, columns: &[String]) -> Result<Vec<PlanValue>> {
        let mut out = Vec::with_capacity(columns.len());
        for column in columns {
            let value = self
                .values
                .iter()
                .find(|(name, _)| name == column)
                .ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "insert row missing value for column '{}'",
                        column
                    ))
                })?;
            out.push(value.1.clone());
        }
        Ok(out)
    }
}

pub fn row() -> Row {
    Row::new()
}

#[doc(hidden)]
pub enum InsertRowKind {
    Named {
        columns: Vec<String>,
        values: Vec<PlanValue>,
    },
    Positional(Vec<PlanValue>),
}

pub trait IntoInsertRow {
    fn into_insert_row(self) -> Result<InsertRowKind>;
}

impl IntoInsertRow for Row {
    fn into_insert_row(self) -> Result<InsertRowKind> {
        let row = self;
        if row.values.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        let columns = row.columns();
        let values = row.values_for_columns(&columns)?;
        Ok(InsertRowKind::Named { columns, values })
    }
}

// Remove the generic impl for `&T` which caused unconditional-recursion
// and noop-clone clippy warnings. Callers can pass owned values or use
// the provided tuple/array/Vec implementations.

impl<T> IntoInsertRow for Vec<T>
where
    T: Into<PlanValue>,
{
    fn into_insert_row(self) -> Result<InsertRowKind> {
        if self.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(InsertRowKind::Positional(
            self.into_iter().map(Into::into).collect(),
        ))
    }
}

impl<T, const N: usize> IntoInsertRow for [T; N]
where
    T: Into<PlanValue>,
{
    fn into_insert_row(self) -> Result<InsertRowKind> {
        if N == 0 {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(InsertRowKind::Positional(
            self.into_iter().map(Into::into).collect(),
        ))
    }
}

macro_rules! impl_into_insert_row_tuple {
    ($($type:ident => $value:ident),+) => {
        impl<$($type,)+> IntoInsertRow for ($($type,)+)
        where
            $($type: Into<PlanValue>,)+
        {
            fn into_insert_row(self) -> Result<InsertRowKind> {
                let ($($value,)+) = self;
                Ok(InsertRowKind::Positional(vec![$($value.into(),)+]))
            }
        }
    };
}

impl_into_insert_row_tuple!(T1 => v1);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4, T5 => v5);
impl_into_insert_row_tuple!(T1 => v1, T2 => v2, T3 => v3, T4 => v4, T5 => v5, T6 => v6);
impl_into_insert_row_tuple!(
    T1 => v1,
    T2 => v2,
    T3 => v3,
    T4 => v4,
    T5 => v5,
    T6 => v6,
    T7 => v7
);
impl_into_insert_row_tuple!(
    T1 => v1,
    T2 => v2,
    T3 => v3,
    T4 => v4,
    T5 => v5,
    T6 => v6,
    T7 => v7,
    T8 => v8
);

pub struct TableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<Context<P>>,
    display_name: String,
    _canonical_name: String,
}

impl<P> TableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(context: Arc<Context<P>>, name: &str) -> Result<Self> {
        let (display_name, canonical_name) = canonical_table_name(name)?;
        context.lookup_table(&canonical_name)?;
        Ok(Self {
            context,
            display_name,
            _canonical_name: canonical_name,
        })
    }

    pub fn lazy(&self) -> Result<LazyFrame<P>> {
        LazyFrame::scan(Arc::clone(&self.context), &self.display_name)
    }

    pub fn insert_rows<R>(&self, rows: impl IntoIterator<Item = R>) -> Result<StatementResult<P>>
    where
        R: IntoInsertRow,
    {
        enum InsertMode {
            Named,
            Positional,
        }

        let table = self.context.lookup_table(&self._canonical_name)?;
        let schema = table.schema.as_ref();
        let schema_column_names: Vec<String> =
            schema.columns.iter().map(|col| col.name.clone()).collect();
        let mut normalized_rows: Vec<Vec<PlanValue>> = Vec::new();
        let mut mode: Option<InsertMode> = None;
        let mut column_names: Option<Vec<String>> = None;
        let mut row_count = 0usize;

        for row in rows.into_iter() {
            row_count += 1;
            match row.into_insert_row()? {
                InsertRowKind::Named { columns, values } => {
                    if let Some(existing) = &mode {
                        if !matches!(existing, InsertMode::Named) {
                            return Err(Error::InvalidArgumentError(
                                "cannot mix positional and named insert rows".into(),
                            ));
                        }
                    } else {
                        mode = Some(InsertMode::Named);
                        let mut seen = HashSet::new();
                        for column in &columns {
                            if !seen.insert(column.clone()) {
                                return Err(Error::InvalidArgumentError(format!(
                                    "duplicate column '{}' in insert row",
                                    column
                                )));
                            }
                        }
                        column_names = Some(columns.clone());
                    }

                    let expected = column_names
                        .as_ref()
                        .expect("column names must be initialized for named insert");
                    if columns != *expected {
                        return Err(Error::InvalidArgumentError(
                            "insert rows must specify the same columns".into(),
                        ));
                    }
                    if values.len() != expected.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "insert row expected {} values, found {}",
                            expected.len(),
                            values.len()
                        )));
                    }
                    normalized_rows.push(values);
                }
                InsertRowKind::Positional(values) => {
                    if let Some(existing) = &mode {
                        if !matches!(existing, InsertMode::Positional) {
                            return Err(Error::InvalidArgumentError(
                                "cannot mix positional and named insert rows".into(),
                            ));
                        }
                    } else {
                        mode = Some(InsertMode::Positional);
                        column_names = Some(schema_column_names.clone());
                    }

                    if values.len() != schema.columns.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "insert row expected {} values, found {}",
                            schema.columns.len(),
                            values.len()
                        )));
                    }
                    normalized_rows.push(values);
                }
            }
        }

        if row_count == 0 {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one row".into(),
            ));
        }

        let columns = column_names.unwrap_or_else(|| schema_column_names.clone());
        self.insert_row_batch(RowBatch {
            columns,
            rows: normalized_rows,
        })
    }

    pub fn insert_row_batch(&self, batch: RowBatch) -> Result<StatementResult<P>> {
        if batch.rows.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one row".into(),
            ));
        }
        if batch.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        for row in &batch.rows {
            if row.len() != batch.columns.len() {
                return Err(Error::InvalidArgumentError(
                    "insert rows must have values for every column".into(),
                ));
            }
        }

        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: batch.columns,
            source: InsertSource::Rows(batch.rows),
        };
        self.context.insert(plan)
    }

    pub fn insert_batches(&self, batches: Vec<RecordBatch>) -> Result<StatementResult<P>> {
        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: Vec::new(),
            source: InsertSource::Batches(batches),
        };
        self.context.insert(plan)
    }

    pub fn insert_lazy(&self, frame: LazyFrame<P>) -> Result<StatementResult<P>> {
        let RowBatch { columns, rows } = frame.collect_rows()?;
        self.insert_row_batch(RowBatch { columns, rows })
    }

    pub fn name(&self) -> &str {
        &self.display_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use llkv_storage::pager::MemPager;
    use std::sync::Arc;

    #[test]
    fn create_insert_select_roundtrip() {
        let pager = Arc::new(MemPager::default());
        let context = Arc::new(Context::new(pager));

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
        let context = Arc::new(Context::new(pager));

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
        let execution = context.execute_select(plan).expect("select");
        let batches = execution.collect().expect("collect batches");
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int column");
        assert_eq!(column.value(0), 2);
    }
}
