#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::mem;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    ArrayRef, Date32Builder, Float64Builder, Int64Builder, StringBuilder, UInt64Array,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::ROW_ID_COLUMN_NAME;
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
// Literal is not used at top-level; keep it out to avoid unused import warnings.
use llkv_result::Error;
use llkv_storage::pager::{MemPager, Pager};
use llkv_table::table::{ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::{FieldId, TableId};
use llkv_table::{CATALOG_TABLE_ID, ColMeta, SysCatalog, TableMeta};
use simd_r_drive_entry_handle::EntryHandle;
use sqlparser::ast::{
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, GroupByExpr, ObjectName, ObjectNamePart, Select,
    SelectItem, SelectItemQualifiedWildcardKind, TableAlias, TableFactor, UnaryOperator, Value,
    ValueWithSpan,
};
use time::{Date, Month};

pub type DslResult<T> = llkv_result::Result<T>;

// Re-export plan structures from llkv-plan
pub use llkv_plan::{
    AggregateExpr, AggregateFunction, AssignmentValue, ColumnAssignment, ColumnNullability,
    ColumnSpec, CreateTablePlan, CreateTableSource, DeletePlan, DslOperation, DslStatement,
    InsertPlan, InsertSource, IntoColumnSpec, NotNull, Nullable, OrderByPlan, OrderSortType,
    OrderTarget, PlanValue, SelectPlan, SelectProjection, UpdatePlan,
};

// Execution structures from llkv-executor
// Keep DSL-prefixed types private to this crate so they are not exposed as
// part of the llkv-dsl public API. Publicly re-export only the generic
// execution APIs that are not DSL-prefixed.
use llkv_executor::{ExecutorColumn, ExecutorSchema, ExecutorTable};
pub use llkv_executor::{QueryExecutor, RowBatch, SelectExecution, TableProvider};

// Import transaction structures from llkv-transaction for internal use.
// NOTE: we intentionally do NOT re-export these types from the DSL crate so
// non-DSL crates do not get `Dsl*` symbols pulled into their public API.
use llkv_transaction::{
    TransactionContext, TransactionKind, TransactionManager, TransactionResult,
};

// Internal low-level transaction session type (from llkv-transaction)
use llkv_transaction::TransactionSession;

// Note: Session is the high-level wrapper that users should use instead of raw DslSession

/// Result of running a DSL statement.
#[derive(Clone)]
pub enum DslStatementResult<P>
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

impl<P> fmt::Debug for DslStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslStatementResult::CreateTable { table_name } => f
                .debug_struct("CreateTable")
                .field("table_name", table_name)
                .finish(),
            DslStatementResult::NoOp => f.debug_struct("NoOp").finish(),
            DslStatementResult::Insert {
                table_name,
                rows_inserted,
            } => f
                .debug_struct("Insert")
                .field("table_name", table_name)
                .field("rows_inserted", rows_inserted)
                .finish(),
            DslStatementResult::Update {
                table_name,
                rows_updated,
            } => f
                .debug_struct("Update")
                .field("table_name", table_name)
                .field("rows_updated", rows_updated)
                .finish(),
            DslStatementResult::Delete {
                table_name,
                rows_deleted,
            } => f
                .debug_struct("Delete")
                .field("table_name", table_name)
                .field("rows_deleted", rows_deleted)
                .finish(),
            DslStatementResult::Select {
                table_name, schema, ..
            } => f
                .debug_struct("Select")
                .field("table_name", table_name)
                .field("schema", schema)
                .finish(),
            DslStatementResult::Transaction { kind } => {
                f.debug_struct("Transaction").field("kind", kind).finish()
            }
        }
    }
}

impl<P> DslStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Convert a StatementResult from one pager type to another.
    /// Only works for non-SELECT results (CreateTable, Insert, Update, Delete, NoOp, Transaction).
    #[allow(dead_code)]
    pub(crate) fn convert_pager_type<Q>(self) -> DslResult<DslStatementResult<Q>>
    where
        Q: Pager<Blob = EntryHandle> + Send + Sync,
    {
        match self {
            DslStatementResult::CreateTable { table_name } => {
                Ok(DslStatementResult::CreateTable { table_name })
            }
            DslStatementResult::NoOp => Ok(DslStatementResult::NoOp),
            DslStatementResult::Insert {
                table_name,
                rows_inserted,
            } => Ok(DslStatementResult::Insert {
                table_name,
                rows_inserted,
            }),
            DslStatementResult::Update {
                table_name,
                rows_updated,
            } => Ok(DslStatementResult::Update {
                table_name,
                rows_updated,
            }),
            DslStatementResult::Delete {
                table_name,
                rows_deleted,
            } => Ok(DslStatementResult::Delete {
                table_name,
                rows_deleted,
            }),
            DslStatementResult::Transaction { kind } => {
                Ok(DslStatementResult::Transaction { kind })
            }
            DslStatementResult::Select { .. } => Err(Error::Internal(
                "Cannot convert SELECT result between pager types in transaction".into(),
            )),
        }
    }
}

/// Return the table name referenced by a DSL statement, if any.
///
/// This is a small helper used by higher-level engines (for example the
/// SQL front-end) to provide better error messages when a statement fails
/// with a table-related error. It intentionally returns an `Option<&str>` so
/// callers can decide how to report missing table context.
pub fn statement_table_name(statement: &DslStatement) -> Option<&str> {
    match statement {
        DslStatement::CreateTable(plan) => Some(&plan.name),
        DslStatement::Insert(plan) => Some(&plan.table),
        DslStatement::Update(plan) => Some(&plan.table),
        DslStatement::Delete(plan) => Some(&plan.table),
        DslStatement::Select(plan) => Some(&plan.table),
        DslStatement::BeginTransaction
        | DslStatement::CommitTransaction
        | DslStatement::RollbackTransaction => None,
    }
}

// ============================================================================
// Plan Structures (now in llkv-plan and re-exported above)
// ============================================================================
//
// The following types are defined in llkv-plan and re-exported:
// - DslValue, CreateTablePlan, ColumnSpec, IntoColumnSpec
// - InsertPlan, InsertSource, UpdatePlan, DeletePlan
// - SelectPlan, SelectProjection, AggregateExpr, AggregateFunction
// - OrderByPlan, OrderSortType, OrderTarget
// - DslOperation
//
// This separation allows plans to be used independently of execution logic.
// ============================================================================

// Transaction management is now handled by llkv-transaction crate
// The DslTransaction and TableDeltaState types are re-exported from there

/// Wrapper for DslContext that implements TransactionContext
pub struct DslContextWrapper<P>(Arc<DslContext<P>>)
where
    P: Pager<Blob = EntryHandle> + Send + Sync;

impl<P> DslContextWrapper<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new(ctx: Arc<DslContext<P>>) -> Self {
        Self(ctx)
    }
}

// TODO: Rename to `DslSession`
/// A session for executing operations with optional transaction support.
///
/// This is a high-level wrapper around the transaction machinery that provides
/// a clean API for users. Operations can be executed directly or within a transaction.
pub struct Session<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    // TODO: Allow generic pager type
    inner: TransactionSession<DslContextWrapper<P>, DslContextWrapper<MemPager>>,
}

impl<P> Session<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Begin a transaction in this session.
    /// Creates an isolated staging context automatically.
    pub fn begin_transaction(&self) -> DslResult<DslStatementResult<P>> {
        let staging_pager = Arc::new(MemPager::default());
        tracing::trace!(
            "BEGIN_TRANSACTION: Created staging pager at {:p}",
            &*staging_pager
        );
        let staging_ctx = Arc::new(DslContext::new(staging_pager));

        // Copy table metadata from the main context to the staging context,
        // but create new Table instances that use the staging pager
        self.inner
            .context()
            .0
            .copy_tables_to_staging(&staging_ctx)?;

        let staging_wrapper = Arc::new(DslContextWrapper::new(staging_ctx));

        self.inner.begin_transaction(staging_wrapper)?;
        Ok(DslStatementResult::Transaction {
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
    pub fn commit_transaction(&self) -> DslResult<DslStatementResult<P>> {
        tracing::trace!("Session::commit_transaction called");
        let (tx_result, operations) = self.inner.commit_transaction()?;
        tracing::trace!(
            "Session::commit_transaction got {} operations",
            operations.len()
        );

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
                DslOperation::CreateTable(plan) => {
                    TransactionContext::create_table_plan(&**self.inner.context(), plan)?;
                }
                DslOperation::Insert(plan) => {
                    TransactionContext::insert(&**self.inner.context(), plan)?;
                }
                DslOperation::Update(plan) => {
                    TransactionContext::update(&**self.inner.context(), plan)?;
                }
                DslOperation::Delete(plan) => {
                    TransactionContext::delete(&**self.inner.context(), plan)?;
                }
                _ => {}
            }
        }

        // Return the DSL's StatementResult with the correct kind (Commit or Rollback)
        Ok(DslStatementResult::Transaction { kind })
    }

    /// Rollback the current transaction, discarding all changes.
    pub fn rollback_transaction(&self) -> DslResult<DslStatementResult<P>> {
        self.inner.rollback_transaction()?;
        Ok(DslStatementResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    fn materialize_create_table_plan(
        &self,
        mut plan: CreateTablePlan,
    ) -> DslResult<CreateTablePlan> {
        if let Some(CreateTableSource::Select { plan: select_plan }) = plan.source.take() {
            let select_result = self.select(*select_plan)?;
            let (schema, batches) = match select_result {
                DslStatementResult::Select {
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
    pub fn create_table_plan(&self, plan: CreateTablePlan) -> DslResult<DslStatementResult<P>> {
        let plan = self.materialize_create_table_plan(plan)?;
        if self.has_active_transaction() {
            let table_name = plan.name.clone();
            match self
                .inner
                .execute_operation(DslOperation::CreateTable(plan))
            {
                Ok(_) => Ok(DslStatementResult::CreateTable { table_name }),
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
            Ok(DslStatementResult::CreateTable { table_name })
        }
    }

    fn normalize_insert_plan(&self, plan: InsertPlan) -> DslResult<(InsertPlan, usize)> {
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
                    DslStatementResult::Select { execution, .. } => execution.into_rows()?,
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
    pub fn insert(&self, plan: InsertPlan) -> DslResult<DslStatementResult<P>> {
        tracing::trace!("Session::insert called for table={}", plan.table);
        let (plan, rows_inserted) = self.normalize_insert_plan(plan)?;
        let table_name = plan.table.clone();

        if self.has_active_transaction() {
            match self.inner.execute_operation(DslOperation::Insert(plan)) {
                Ok(_) => {
                    tracing::trace!("Session::insert succeeded for table={}", table_name);
                    Ok(DslStatementResult::Insert {
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
            TransactionContext::insert(&**self.inner.context(), plan)?;
            Ok(DslStatementResult::Insert {
                rows_inserted,
                table_name,
            })
        }
    }

    /// Select rows (outside or inside transaction).
    pub fn select(&self, plan: SelectPlan) -> DslResult<DslStatementResult<P>> {
        if self.has_active_transaction() {
            let tx_result = match self
                .inner
                .execute_operation(DslOperation::Select(plan.clone()))
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
                    execution,
                } => Ok(DslStatementResult::Select {
                    execution,
                    table_name,
                    schema,
                }),
                _ => Err(Error::Internal("expected Select result".into())),
            }
        } else {
            // Call via TransactionContext trait
            let table_name = plan.table.clone();
            let execution = TransactionContext::execute_select(&**self.inner.context(), plan)?;
            let schema = execution.schema();
            Ok(DslStatementResult::Select {
                execution,
                table_name,
                schema,
            })
        }
    }

    /// Update rows (outside or inside transaction).
    pub fn update(&self, plan: UpdatePlan) -> DslResult<DslStatementResult<P>> {
        if self.has_active_transaction() {
            let table_name = plan.table.clone();
            let result = match self.inner.execute_operation(DslOperation::Update(plan)) {
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
                } => Ok(DslStatementResult::Update {
                    rows_updated,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Update result".into())),
            }
        } else {
            // Call via TransactionContext trait
            let table_name = plan.table.clone();
            let result = TransactionContext::update(&**self.inner.context(), plan)?;
            match result {
                TransactionResult::Update {
                    rows_matched: _,
                    rows_updated,
                } => Ok(DslStatementResult::Update {
                    rows_updated,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Update result".into())),
            }
        }
    }

    /// Delete rows (outside or inside transaction).
    pub fn delete(&self, plan: DeletePlan) -> DslResult<DslStatementResult<P>> {
        if self.has_active_transaction() {
            let table_name = plan.table.clone();
            let result = match self.inner.execute_operation(DslOperation::Delete(plan)) {
                Ok(result) => result,
                Err(e) => {
                    // If an error occurs during a transaction, abort it
                    self.abort_transaction();
                    return Err(e);
                }
            };
            match result {
                TransactionResult::Delete { rows_deleted } => Ok(DslStatementResult::Delete {
                    rows_deleted,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Delete result".into())),
            }
        } else {
            // Call via TransactionContext trait
            let table_name = plan.table.clone();
            let result = TransactionContext::delete(&**self.inner.context(), plan)?;
            match result {
                TransactionResult::Delete { rows_deleted } => Ok(DslStatementResult::Delete {
                    rows_deleted,
                    table_name,
                }),
                _ => Err(Error::Internal("expected Delete result".into())),
            }
        }
    }
}

pub struct DslEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    context: Arc<DslContext<P>>,
    session: Session<P>,
}

impl<P> Clone for DslEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            session: self.context.create_session(),
        }
    }
}

impl<P> DslEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        let context = Arc::new(DslContext::new(pager));
        Self::from_context(context)
    }

    pub fn from_context(context: Arc<DslContext<P>>) -> Self {
        let session = context.create_session();
        Self { context, session }
    }

    pub fn context(&self) -> Arc<DslContext<P>> {
        Arc::clone(&self.context)
    }

    pub fn session(&self) -> &Session<P> {
        &self.session
    }

    pub fn execute_statement(&self, statement: DslStatement) -> DslResult<DslStatementResult<P>> {
        match statement {
            DslStatement::BeginTransaction => self.session.begin_transaction(),
            DslStatement::CommitTransaction => self.session.commit_transaction(),
            DslStatement::RollbackTransaction => self.session.rollback_transaction(),
            DslStatement::CreateTable(plan) => self.session.create_table_plan(plan),
            DslStatement::Insert(plan) => self.session.insert(plan),
            DslStatement::Update(plan) => self.session.update(plan),
            DslStatement::Delete(plan) => self.session.delete(plan),
            DslStatement::Select(plan) => self.session.select(plan),
        }
    }

    pub fn execute_all<I>(&self, statements: I) -> DslResult<Vec<DslStatementResult<P>>>
    where
        I: IntoIterator<Item = DslStatement>,
    {
        let mut results = Vec::new();
        for statement in statements {
            results.push(self.execute_statement(statement)?);
        }
        Ok(results)
    }
}

/// In-memory execution context shared by DSL queries.
pub struct DslContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    tables: RwLock<HashMap<String, Arc<ExecutorTable<P>>>>,
    // Transaction manager for session-based transactions
    transaction_manager: TransactionManager<DslContextWrapper<P>, DslContextWrapper<MemPager>>,
}

impl<P> DslContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        tracing::trace!("DslContext::new called, pager={:p}", &*pager);
        Self {
            pager,
            tables: RwLock::new(HashMap::new()),
            transaction_manager: TransactionManager::new(),
        }
    }

    /// Create a new session for transaction management.
    /// Each session can have its own independent transaction.
    pub fn create_session(self: &Arc<Self>) -> Session<P> {
        tracing::trace!(
            "DslContext::create_session called, pager={:p}",
            &*self.pager
        );
        let wrapper = DslContextWrapper::new(Arc::clone(self));
        tracing::trace!(
            "Created DslContextWrapper, wrapper pager={:p}",
            &*self.pager
        );
        let inner = self.transaction_manager.create_session(Arc::new(wrapper));
        tracing::trace!("Created TransactionSession");
        Session { inner }
    }

    /// Check if there's an active transaction (legacy - checks if ANY session has a transaction).
    #[deprecated(note = "Use session-based transactions instead")]
    pub fn has_active_transaction(&self) -> bool {
        self.transaction_manager.has_active_transaction()
    }

    pub fn create_table<C, I>(self: &Arc<Self>, name: &str, columns: I) -> DslResult<TableHandle<P>>
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
    ) -> DslResult<TableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, true)
    }

    pub fn create_table_plan(&self, plan: CreateTablePlan) -> DslResult<DslStatementResult<P>> {
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
            if plan.if_not_exists {
                tracing::trace!(
                    "DEBUG create_table_plan: table '{}' exists and if_not_exists=true, returning early WITHOUT creating",
                    display_name
                );
                return Ok(DslStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }

        match plan.source {
            Some(CreateTableSource::Batches { schema, batches }) => self.create_table_from_batches(
                display_name,
                canonical_name,
                schema,
                batches,
                plan.if_not_exists,
            ),
            Some(CreateTableSource::Select { .. }) => Err(Error::Internal(
                "CreateTableSource::Select should be materialized before reaching DslContext::create_table_plan"
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

    pub fn create_table_builder(&self, name: &str) -> CreateTableBuilder<'_, P> {
        CreateTableBuilder {
            ctx: self,
            plan: CreateTablePlan::new(name),
        }
    }

    pub fn table_column_specs(self: &Arc<Self>, name: &str) -> DslResult<Vec<ColumnSpec>> {
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
    fn copy_tables_to_staging<Q>(&self, staging: &DslContext<Q>) -> DslResult<()>
    where
        Q: Pager<Blob = EntryHandle> + Send + Sync + 'static,
    {
        let source_tables = self.tables.read().unwrap();
        let mut staging_tables = staging.tables.write().unwrap();

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

            staging_tables.insert(table_name.clone(), new_executor_table);
        }

        tracing::trace!(
            "!!! COPY_TABLES_TO_STAGING: Copied {} tables to staging context",
            staging_tables.len()
        );
        Ok(())
    }

    pub fn export_table_rows(self: &Arc<Self>, name: &str) -> DslResult<RowBatch> {
        let handle = TableHandle::new(Arc::clone(self), name)?;
        handle.lazy()?.collect_rows()
    }

    fn execute_create_table(&self, plan: CreateTablePlan) -> DslResult<DslStatementResult<P>> {
        self.create_table_plan(plan)
    }

    fn create_table_with_options<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
        if_not_exists: bool,
    ) -> DslResult<TableHandle<P>>
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
            DslStatementResult::CreateTable { .. } => TableHandle::new(Arc::clone(self), name),
            other => Err(Error::InvalidArgumentError(format!(
                "unexpected statement result {other:?} when creating table"
            ))),
        }
    }

    pub fn insert(&self, plan: InsertPlan) -> DslResult<DslStatementResult<P>> {
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
            InsertSource::Rows(rows) => {
                self.insert_rows(table.as_ref(), display_name.clone(), rows, plan.columns)
            }
            InsertSource::Batches(batches) => {
                self.insert_batches(table.as_ref(), display_name.clone(), batches, plan.columns)
            }
            InsertSource::Select { .. } => Err(Error::Internal(
                "InsertSource::Select should be materialized before reaching DslContext::insert"
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
    ) -> DslResult<Vec<RecordBatch>> {
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

        // Scan to get the column data
        let mut batches_without_rowid = Vec::new();
        let table_id = table.table.table_id();
        let projections: Vec<ScanProjection> = table
            .schema
            .columns
            .iter()
            .map(|col| {
                let logical_field_id = LogicalFieldId::for_user(table_id, col.field_id);
                ScanProjection::column((logical_field_id, col.name.clone()))
            })
            .collect();

        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
        };

        table
            .table
            .scan_stream_with_exprs(&projections, &filter_expr, options, |batch| {
                batches_without_rowid.push(batch.clone());
            })?;

        // Now add row_id column to each batch
        let mut batches_with_rowid = Vec::new();
        let mut row_id_offset = 0usize;

        for batch in batches_without_rowid {
            let batch_size = batch.num_rows();
            let batch_row_ids: Vec<u64> =
                row_ids[row_id_offset..row_id_offset + batch_size].to_vec();
            row_id_offset += batch_size;

            // Create a new batch with row_id as the first column
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() + 1);
            arrays.push(Arc::new(UInt64Array::from(batch_row_ids)));
            for i in 0..batch.num_columns() {
                arrays.push(batch.column(i).clone());
            }

            let mut fields: Vec<Field> = Vec::with_capacity(batch.schema().fields().len() + 1);
            fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

            // Reconstruct fields with proper field_id metadata
            for (idx, _field) in batch.schema().fields().iter().enumerate() {
                let col = &table.schema.columns[idx];
                let mut metadata = HashMap::new();
                metadata.insert(
                    llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                    col.field_id.to_string(),
                );
                let field_with_metadata =
                    Field::new(&col.name, col.data_type.clone(), col.nullable)
                        .with_metadata(metadata);
                fields.push(field_with_metadata);
            }

            let schema_with_rowid = Arc::new(Schema::new(fields));
            let batch_with_rowid = RecordBatch::try_new(schema_with_rowid, arrays)?;
            batches_with_rowid.push(batch_with_rowid);
        }

        Ok(batches_with_rowid)
    }

    /// Append batches directly to a table, preserving row_ids from the batches.
    /// This is used for transaction seeding where we need to preserve existing row_ids.
    pub fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> DslResult<usize> {
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

    pub fn update(&self, plan: UpdatePlan) -> DslResult<DslStatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.filter {
            Some(filter) => {
                self.update_filtered_rows(table.as_ref(), display_name, plan.assignments, filter)
            }
            None => self.update_all_rows(table.as_ref(), display_name, plan.assignments),
        }
    }

    pub fn delete(&self, plan: DeletePlan) -> DslResult<DslStatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;
        match plan.filter {
            Some(filter) => self.delete_filtered_rows(table.as_ref(), display_name, filter),
            None => self.delete_all_rows(table.as_ref(), display_name),
        }
    }

    pub fn table_handle(self: &Arc<Self>, name: &str) -> DslResult<TableHandle<P>> {
        TableHandle::new(Arc::clone(self), name)
    }

    pub fn execute_select(self: &Arc<Self>, plan: SelectPlan) -> DslResult<SelectExecution<P>> {
        let (_display_name, canonical_name) = canonical_table_name(&plan.table)?;
        // Verify table exists
        let _table = self.lookup_table(&canonical_name)?;

        // Create a plan with canonical table name
        let mut canonical_plan = plan.clone();
        canonical_plan.table = canonical_name;

        // Use the QueryExecutor from llkv-executor
        let provider: Arc<dyn TableProvider<P>> = Arc::new(DslContextProvider {
            context: Arc::clone(self),
        });
        let executor = QueryExecutor::new(provider);
        executor.execute_select(canonical_plan)
    }

    fn create_table_from_columns(
        &self,
        display_name: String,
        canonical_name: String,
        columns: Vec<ColumnSpec>,
        if_not_exists: bool,
    ) -> DslResult<DslStatementResult<P>> {
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
                return Ok(DslStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name, table_entry);
        Ok(DslStatementResult::CreateTable {
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
    ) -> DslResult<DslStatementResult<P>> {
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

            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(row_count + 1);
            arrays.push(Arc::new(row_builder.finish()) as ArrayRef);

            let mut fields: Vec<Field> = Vec::with_capacity(column_defs.len() + 1);
            fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

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

        table_entry.next_row_id.store(next_row_id, Ordering::SeqCst);
        table_entry.total_rows.store(total_rows, Ordering::SeqCst);

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            if if_not_exists {
                return Ok(DslStatementResult::CreateTable {
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
        Ok(DslStatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn check_primary_key_constraints(
        &self,
        table: &ExecutorTable<P>,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
    ) -> DslResult<()> {
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
            let existing_values = self.scan_column_values(table, field_id)?;

            tracing::trace!(
                "[DEBUG] PK check on column '{}': found {} existing values: {:?}",
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
    ) -> DslResult<Vec<PlanValue>> {
        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        let mut values = Vec::new();

        let logical_field_id = LogicalFieldId::for_user(table_id, field_id);
        let projection = ScanProjection::column((logical_field_id, "value".to_string()));

        // Create a filter that matches all rows (unbounded range)
        let match_all_filter = Filter {
            field_id,
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        let scan_result = table.table.scan_stream(
            &[projection],
            &filter_expr,
            ScanStreamOptions::default(),
            |batch| {
                if batch.num_columns() > 0 {
                    let array = batch.column(0);
                    for row_idx in 0..batch.num_rows() {
                        if let Ok(value) = llkv_plan::plan_value_from_array(array, row_idx) {
                            values.push(value);
                        }
                    }
                }
            },
        );

        // Handle NotFound error - return empty (table has no data yet)
        match scan_result {
            Ok(_) => Ok(values),
            Err(Error::NotFound) => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    fn insert_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        rows: Vec<Vec<PlanValue>>,
        columns: Vec<String>,
    ) -> DslResult<DslStatementResult<P>> {
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
        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] Checking PRIMARY KEY constraints - {} rows to insert",
                rows.len()
            );
            for (i, row) in rows.iter().enumerate() {
                tracing::trace!("[KEYS]   row[{}]: {:?}", i, row);
            }
        }

        let constraint_result = self.check_primary_key_constraints(table, &rows, &column_order);

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

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_values.len() + 1);
        arrays.push(Arc::new(row_id_builder.finish()));

        let mut fields: Vec<Field> = Vec::with_capacity(column_values.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

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

        Ok(DslStatementResult::Insert {
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
    ) -> DslResult<DslStatementResult<P>> {
        if batches.is_empty() {
            return Ok(DslStatementResult::Insert {
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

            match self.insert_rows(table, display_name.clone(), rows, columns.clone())? {
                DslStatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(DslStatementResult::Insert {
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
    ) -> DslResult<DslStatementResult<P>> {
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
            self.collect_update_rows(table, &filter_expr, &scalar_exprs)?;

        if row_ids.is_empty() {
            return Ok(DslStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(prepared.len() + 1);
        arrays.push(Arc::new(UInt64Array::from(row_ids.clone())));

        let mut fields: Vec<Field> = Vec::with_capacity(prepared.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (column, value) in prepared {
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
            let array = build_array_for_column(&column.data_type, &values)?;
            arrays.push(array);

            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;

        Ok(DslStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn update_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        assignments: Vec<ColumnAssignment>,
    ) -> DslResult<DslStatementResult<P>> {
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
            return Ok(DslStatementResult::Update {
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
            self.collect_update_rows(table, &filter_expr, &scalar_exprs)?;

        if row_ids.is_empty() {
            return Ok(DslStatementResult::Update {
                table_name: display_name,
                rows_updated: 0,
            });
        }

        let row_count = row_ids.len();
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(prepared.len() + 1);
        arrays.push(Arc::new(UInt64Array::from(row_ids.clone())));

        let mut fields: Vec<Field> = Vec::with_capacity(prepared.len() + 1);
        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for (column, value) in prepared {
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
            let array = build_array_for_column(&column.data_type, &values)?;
            arrays.push(array);

            let mut metadata = HashMap::new();
            metadata.insert(
                llkv_table::constants::FIELD_ID_META_KEY.to_string(),
                column.field_id.to_string(),
            );
            let field = Field::new(&column.name, column.data_type.clone(), column.nullable)
                .with_metadata(metadata);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        table.table.append(&batch)?;

        Ok(DslStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn delete_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        filter: LlkvExpr<'static, String>,
    ) -> DslResult<DslStatementResult<P>> {
        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        self.apply_delete(table, display_name, row_ids)
    }

    fn delete_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
    ) -> DslResult<DslStatementResult<P>> {
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if total_rows == 0 {
            return Ok(DslStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let anchor_field = table.schema.first_field_id().ok_or_else(|| {
            Error::InvalidArgumentError("DELETE requires a table with at least one column".into())
        })?;
        let filter_expr = full_table_scan_filter(anchor_field);
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        self.apply_delete(table, display_name, row_ids)
    }

    fn apply_delete(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        row_ids: Vec<u64>,
    ) -> DslResult<DslStatementResult<P>> {
        if row_ids.is_empty() {
            return Ok(DslStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let logical_fields: Vec<LogicalFieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| LogicalFieldId::for_user(table.table.table_id(), column.field_id))
            .collect();

        if logical_fields.is_empty() {
            return Err(Error::InvalidArgumentError(
                "DELETE requires a table with at least one column".into(),
            ));
        }

        table.table.store().delete_rows(&logical_fields, &row_ids)?;

        let removed = row_ids.len();
        let removed_u64 = u64::try_from(removed)
            .map_err(|_| Error::InvalidArgumentError("row count exceeds supported range".into()))?;
        table.total_rows.fetch_sub(removed_u64, Ordering::SeqCst);

        Ok(DslStatementResult::Delete {
            table_name: display_name,
            rows_deleted: removed,
        })
    }

    fn collect_update_rows(
        &self,
        table: &ExecutorTable<P>,
        filter_expr: &LlkvExpr<'static, FieldId>,
        expressions: &[ScalarExpr<FieldId>],
    ) -> DslResult<(Vec<u64>, Vec<Vec<PlanValue>>)> {
        let row_ids = table.table.filter_row_ids(filter_expr)?;
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
        let options = ScanStreamOptions {
            include_nulls: true,
            order: None,
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
    ) -> DslResult<()> {
        for row_idx in 0..batch.num_rows() {
            for (expr_index, values) in expr_values.iter_mut().enumerate() {
                let value = llkv_plan::plan_value_from_array(batch.column(expr_index), row_idx)?;
                values.push(value);
            }
        }

        Ok(())
    }

    fn lookup_table(&self, canonical_name: &str) -> DslResult<Arc<ExecutorTable<P>>> {
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

    fn reserve_table_id(&self) -> DslResult<TableId> {
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

// Implement TransactionContext for DslContextWrapper to enable llkv-transaction integration
impl<P> TransactionContext for DslContextWrapper<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type Pager = P;

    fn table_column_specs(&self, table_name: &str) -> llkv_result::Result<Vec<ColumnSpec>> {
        DslContext::table_column_specs(&self.0, table_name)
    }

    fn export_table_rows(
        &self,
        table_name: &str,
    ) -> llkv_result::Result<llkv_transaction::RowBatch> {
        let batch = DslContext::export_table_rows(&self.0, table_name)?;
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
        DslContext::get_batches_with_row_ids(&self.0, table_name, filter)
    }

    fn execute_select(
        &self,
        plan: SelectPlan,
    ) -> llkv_result::Result<SelectExecution<Self::Pager>> {
        DslContext::execute_select(&self.0, plan)
    }

    fn create_table_plan(
        &self,
        plan: CreateTablePlan,
    ) -> llkv_result::Result<TransactionResult<MemPager>> {
        let result = DslContext::create_table_plan(&self.0, plan)?;
        Ok(convert_statement_result(result))
    }

    fn insert(&self, plan: InsertPlan) -> llkv_result::Result<TransactionResult<MemPager>> {
        tracing::trace!(
            "[WRAPPER] TransactionContext::insert called - plan.table='{}', wrapper_context_pager={:p}",
            plan.table,
            &*self.0.pager
        );
        let result = DslContext::insert(&self.0, plan)?;
        Ok(convert_statement_result(result))
    }

    fn update(&self, plan: UpdatePlan) -> llkv_result::Result<TransactionResult<MemPager>> {
        let result = DslContext::update(&self.0, plan)?;
        Ok(convert_statement_result(result))
    }

    fn delete(&self, plan: DeletePlan) -> llkv_result::Result<TransactionResult<MemPager>> {
        let result = DslContext::delete(&self.0, plan)?;
        Ok(convert_statement_result(result))
    }

    fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> llkv_result::Result<usize> {
        DslContext::append_batches_with_row_ids(&self.0, table_name, batches)
    }

    fn table_names(&self) -> Vec<String> {
        DslContext::table_names(&self.0)
    }
}

// Helper to convert StatementResult between types (legacy)
fn convert_statement_result<P>(result: DslStatementResult<P>) -> TransactionResult<MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    use llkv_transaction::TransactionResult as TxResult;
    match result {
        DslStatementResult::CreateTable { table_name } => TxResult::CreateTable { table_name },
        DslStatementResult::Insert { rows_inserted, .. } => TxResult::Insert { rows_inserted },
        DslStatementResult::Update { rows_updated, .. } => TxResult::Update {
            rows_matched: rows_updated,
            rows_updated,
        },
        DslStatementResult::Delete { rows_deleted, .. } => TxResult::Delete { rows_deleted },
        DslStatementResult::Transaction { kind } => TxResult::Transaction { kind },
        _ => panic!("unsupported DslStatementResult conversion"),
    }
}

// Wrapper to implement TableProvider for DslContext
struct DslContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
}

impl<P> TableProvider<P> for DslContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> DslResult<Arc<ExecutorTable<P>>> {
        self.context.lookup_table(canonical_name)
    }
}

/// Lazily built logical plan (thin wrapper over SelectPlan).
pub struct LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<DslContext<P>>,
    plan: SelectPlan,
}

impl<P> LazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn scan(context: Arc<DslContext<P>>, table: &str) -> DslResult<Self> {
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

    pub fn select(mut self, projections: Vec<SelectProjection>) -> Self {
        self.plan.projections = projections;
        self
    }

    pub fn aggregate(mut self, aggregates: Vec<AggregateExpr>) -> Self {
        self.plan.aggregates = aggregates;
        self
    }

    pub fn collect(self) -> DslResult<SelectExecution<P>> {
        self.context.execute_select(self.plan)
    }

    pub fn collect_rows(self) -> DslResult<RowBatch> {
        let execution = self.context.execute_select(self.plan)?;
        execution.collect_rows()
    }
}

fn canonical_table_name(name: &str) -> DslResult<(String, String)> {
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

fn resolve_insert_columns(columns: &[String], schema: &ExecutorSchema) -> DslResult<Vec<usize>> {
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

fn build_array_for_column(dtype: &DataType, values: &[PlanValue]) -> DslResult<ArrayRef> {
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

fn parse_date32_literal(text: &str) -> DslResult<i32> {
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

fn translate_predicate(
    expr: LlkvExpr<'static, String>,
    schema: &ExecutorSchema,
) -> DslResult<LlkvExpr<'static, FieldId>> {
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
            let column = schema.resolve(&field_id).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{field_id}' in filter"))
            })?;
            Ok(LlkvExpr::Pred(Filter {
                field_id: column.field_id,
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
) -> DslResult<ScalarExpr<FieldId>> {
    match expr {
        ScalarExpr::Column(name) => {
            let column = schema.resolve(name).ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown column '{name}' in expression"))
            })?;
            Ok(ScalarExpr::column(column.field_id))
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
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Count(column.field_id)
                }
                AggregateCall::Sum(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Sum(column.field_id)
                }
                AggregateCall::Min(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Min(column.field_id)
                }
                AggregateCall::Max(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::Max(column.field_id)
                }
                AggregateCall::CountNulls(name) => {
                    let column = schema.resolve(name).ok_or_else(|| {
                        Error::InvalidArgumentError(format!("unknown column '{name}' in aggregate"))
                    })?;
                    AggregateCall::CountNulls(column.field_id)
                }
            };
            Ok(ScalarExpr::Aggregate(translated_agg))
        }
    }
}

fn dsl_value_from_sql_expr(expr: &SqlExpr) -> DslResult<PlanValue> {
    match expr {
        SqlExpr::Value(value) => dsl_value_from_sql_value(value),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match dsl_value_from_sql_expr(expr)? {
            PlanValue::Integer(v) => Ok(PlanValue::Integer(-v)),
            PlanValue::Float(v) => Ok(PlanValue::Float(-v)),
            PlanValue::Null | PlanValue::String(_) => Err(Error::InvalidArgumentError(
                "cannot negate non-numeric literal".into(),
            )),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => dsl_value_from_sql_expr(expr),
        SqlExpr::Nested(inner) => dsl_value_from_sql_expr(inner),
        other => Err(Error::InvalidArgumentError(format!(
            "unsupported literal expression: {other:?}"
        ))),
    }
}

fn dsl_value_from_sql_value(value: &ValueWithSpan) -> DslResult<PlanValue> {
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

pub fn extract_rows_from_range(select: &Select) -> DslResult<Option<RangeSelectRows>> {
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

fn build_range_projection_expr(expr: &SqlExpr, spec: &RangeSpec) -> DslResult<RangeProjection> {
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
        other => Ok(RangeProjection::Literal(dsl_value_from_sql_expr(other)?)),
    }
}

fn parse_range_spec(select: &Select) -> DslResult<Option<RangeSpec>> {
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
) -> DslResult<Option<RangeSpec>> {
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
    let extract_int = |arg: &FunctionArg| -> DslResult<i64> {
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

        let value = dsl_value_from_sql_expr(arg_expr)?;
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
    ctx: &'ctx DslContext<P>,
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

    pub fn finish(self) -> DslResult<DslStatementResult<P>> {
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

    fn values_for_columns(&self, columns: &[String]) -> DslResult<Vec<PlanValue>> {
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
    fn into_insert_row(self) -> DslResult<InsertRowKind>;
}

impl IntoInsertRow for Row {
    fn into_insert_row(self) -> DslResult<InsertRowKind> {
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
    fn into_insert_row(self) -> DslResult<InsertRowKind> {
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
    fn into_insert_row(self) -> DslResult<InsertRowKind> {
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
            fn into_insert_row(self) -> DslResult<InsertRowKind> {
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
    context: Arc<DslContext<P>>,
    display_name: String,
    _canonical_name: String,
}

impl<P> TableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(context: Arc<DslContext<P>>, name: &str) -> DslResult<Self> {
        let (display_name, canonical_name) = canonical_table_name(name)?;
        context.lookup_table(&canonical_name)?;
        Ok(Self {
            context,
            display_name,
            _canonical_name: canonical_name,
        })
    }

    pub fn lazy(&self) -> DslResult<LazyFrame<P>> {
        LazyFrame::scan(Arc::clone(&self.context), &self.display_name)
    }

    pub fn insert_rows<R>(
        &self,
        rows: impl IntoIterator<Item = R>,
    ) -> DslResult<DslStatementResult<P>>
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

    pub fn insert_row_batch(&self, batch: RowBatch) -> DslResult<DslStatementResult<P>> {
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

    pub fn insert_batches(&self, batches: Vec<RecordBatch>) -> DslResult<DslStatementResult<P>> {
        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: Vec::new(),
            source: InsertSource::Batches(batches),
        };
        self.context.insert(plan)
    }

    pub fn insert_lazy(&self, frame: LazyFrame<P>) -> DslResult<DslStatementResult<P>> {
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
        let context = Arc::new(DslContext::new(pager));

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
        let context = Arc::new(DslContext::new(pager));

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
