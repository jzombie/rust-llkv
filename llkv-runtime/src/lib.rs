//! Query execution runtime for LLKV.
//!
//! This crate provides the runtime API (see [`RuntimeEngine`]) for executing SQL plans with full
//! transaction support. It coordinates between the transaction layer, storage layer,
//! and query executor to provide a complete database runtime.
//!
//! # Key Components
//!
//! - **[`RuntimeEngine`]**: Main execution engine for SQL operations
//! - **[`RuntimeSession`]**: Session-level interface with transaction management
//! - **[`TransactionContext`]**: Single-transaction execution context
//! - **Table Provider**: Integration with the query executor for table access
//!
//! # Transaction Support
//!
//! The runtime supports both:
//! - **Auto-commit**: Single-statement transactions (uses `TXN_ID_AUTO_COMMIT`)
//! - **Multi-statement**: Explicit BEGIN/COMMIT/ROLLBACK transactions
//!
//! # MVCC Integration
//!
//! All data modifications automatically include MVCC metadata:
//! - `row_id`: Unique row identifier
//! - `created_by`: Transaction ID that created the row
//! - `deleted_by`: Transaction ID that deleted the row (or `TXN_ID_NONE`)
//!
//! The runtime ensures these columns are injected and managed consistently.

#![forbid(unsafe_code)]

pub mod storage_namespace;

use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use rustc_hash::{FxHashMap, FxHashSet};

use arrow::array::{
    Array, ArrayRef, BooleanBuilder, Date32Builder, Float64Builder, Int64Builder, StringBuilder,
    UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{
    CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, GatherNullPolicy, IndexKind, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
use llkv_result::Error;
use llkv_storage::pager::{BoxedPager, MemPager, Pager};
use llkv_table::catalog::{FieldConstraints, FieldDefinition, TableCatalog};
use llkv_table::table::{RowIdFilter, ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::{FieldId, ROW_ID_FIELD_ID, RowId};
use llkv_table::{
    CatalogService, ConstraintColumnInfo, ConstraintService, CreateTableResult, ForeignKeyColumn,
    ForeignKeyTableInfo, InsertColumnConstraint, InsertMultiColumnUnique, InsertUniqueColumn,
    MetadataManager, MultiColumnUniqueEntryMeta, MultiColumnUniqueRegistration, SysCatalog,
    TableView, UniqueKey, build_composite_unique_key, canonical_table_name,
    constraints::ConstraintKind, ensure_multi_column_unique, ensure_single_column_unique,
};
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
    ColumnSpec, CreateIndexPlan, CreateTablePlan, CreateTableSource, DeletePlan, ForeignKeyAction,
    ForeignKeySpec, IndexColumnPlan, InsertPlan, InsertSource, IntoColumnSpec, NotNull, Nullable,
    OrderByPlan, OrderSortType, OrderTarget, PlanOperation, PlanStatement, PlanValue, SelectPlan,
    SelectProjection, UpdatePlan,
};

// Execution structures from llkv-executor
use llkv_executor::{ExecutorColumn, ExecutorMultiColumnUnique, ExecutorSchema, ExecutorTable};
pub use llkv_executor::{QueryExecutor, RowBatch, SelectExecution, TableProvider};

use crate::storage_namespace::{
    PersistentNamespace, StorageNamespace, StorageNamespaceRegistry, TemporaryNamespace,
};

// Import transaction structures from llkv-transaction for internal use.
pub use llkv_transaction::TransactionKind;
use llkv_transaction::{
    RowVersion, TXN_ID_AUTO_COMMIT, TXN_ID_NONE, TransactionContext, TransactionManager,
    TransactionResult, TxnId, TxnIdManager, mvcc::TransactionSnapshot,
};

// Internal low-level transaction session type (from llkv-transaction)
use llkv_transaction::TransactionSession;

// Note: RuntimeSession is the high-level wrapper that users should use instead of the lower-level TransactionSession API

/// Helper functions for MVCC column injection.
mod mvcc_columns {
    use super::*;
    use std::collections::HashMap;

    /// Build MVCC columns (row_id, created_by, deleted_by) for INSERT operations.
    ///
    /// Returns (row_id_array, created_by_array, deleted_by_array) and updates next_row_id.
    pub(crate) fn build_insert_mvcc_columns(
        row_count: usize,
        start_row_id: RowId,
        creator_txn_id: TxnId,
    ) -> (ArrayRef, ArrayRef, ArrayRef) {
        let mut row_builder = UInt64Builder::with_capacity(row_count);
        for offset in 0..row_count {
            row_builder.append_value(start_row_id + offset as u64);
        }

        let mut created_builder = UInt64Builder::with_capacity(row_count);
        let mut deleted_builder = UInt64Builder::with_capacity(row_count);
        for _ in 0..row_count {
            created_builder.append_value(creator_txn_id);
            deleted_builder.append_value(TXN_ID_NONE);
        }

        (
            Arc::new(row_builder.finish()) as ArrayRef,
            Arc::new(created_builder.finish()) as ArrayRef,
            Arc::new(deleted_builder.finish()) as ArrayRef,
        )
    }

    /// Build MVCC field definitions (row_id, created_by, deleted_by).
    ///
    /// Returns the three Field definitions that should be prepended to user columns.
    pub(crate) fn build_mvcc_fields() -> Vec<Field> {
        vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new(CREATED_BY_COLUMN_NAME, DataType::UInt64, false),
            Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false),
        ]
    }

    /// Build field with field_id metadata for a user column.
    pub(crate) fn build_field_with_metadata(
        name: &str,
        data_type: DataType,
        nullable: bool,
        field_id: FieldId,
    ) -> Field {
        let mut metadata = FxHashMap::with_capacity_and_hasher(1, Default::default());
        metadata.insert(
            llkv_table::constants::FIELD_ID_META_KEY.to_string(),
            field_id.to_string(),
        );
        Field::new(name, data_type, nullable)
            .with_metadata(metadata.into_iter().collect::<HashMap<String, String>>())
    }

    /// Build DELETE batch with row_id and deleted_by columns.
    ///
    /// This creates a minimal RecordBatch for marking rows as deleted.
    pub(crate) fn build_delete_batch(
        row_ids: Vec<RowId>,
        deleted_by_txn_id: TxnId,
    ) -> llkv_result::Result<RecordBatch> {
        let row_count = row_ids.len();

        let fields = vec![
            Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
            Field::new(DELETED_BY_COLUMN_NAME, DataType::UInt64, false),
        ];

        let arrays: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from(row_ids)),
            Arc::new(UInt64Array::from(vec![deleted_by_txn_id; row_count])),
        ];

        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(Error::Arrow)
    }
}

struct TableConstraintContext {
    schema_field_ids: Vec<FieldId>,
    column_constraints: Vec<InsertColumnConstraint>,
    unique_columns: Vec<InsertUniqueColumn>,
    multi_column_uniques: Vec<InsertMultiColumnUnique>,
    primary_key: Option<InsertMultiColumnUnique>,
}

/// Result of running a plan statement.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum RuntimeStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    CreateTable {
        table_name: String,
    },
    CreateIndex {
        table_name: String,
        index_name: Option<String>,
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

impl<P> fmt::Debug for RuntimeStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeStatementResult::CreateTable { table_name } => f
                .debug_struct("CreateTable")
                .field("table_name", table_name)
                .finish(),
            RuntimeStatementResult::CreateIndex {
                table_name,
                index_name,
            } => f
                .debug_struct("CreateIndex")
                .field("table_name", table_name)
                .field("index_name", index_name)
                .finish(),
            RuntimeStatementResult::NoOp => f.debug_struct("NoOp").finish(),
            RuntimeStatementResult::Insert {
                table_name,
                rows_inserted,
            } => f
                .debug_struct("Insert")
                .field("table_name", table_name)
                .field("rows_inserted", rows_inserted)
                .finish(),
            RuntimeStatementResult::Update {
                table_name,
                rows_updated,
            } => f
                .debug_struct("Update")
                .field("table_name", table_name)
                .field("rows_updated", rows_updated)
                .finish(),
            RuntimeStatementResult::Delete {
                table_name,
                rows_deleted,
            } => f
                .debug_struct("Delete")
                .field("table_name", table_name)
                .field("rows_deleted", rows_deleted)
                .finish(),
            RuntimeStatementResult::Select {
                table_name, schema, ..
            } => f
                .debug_struct("Select")
                .field("table_name", table_name)
                .field("schema", schema)
                .finish(),
            RuntimeStatementResult::Transaction { kind } => {
                f.debug_struct("Transaction").field("kind", kind).finish()
            }
        }
    }
}

impl<P> RuntimeStatementResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Convert a StatementResult from one pager type to another.
    /// Only works for non-SELECT results (CreateTable, Insert, Update, Delete, NoOp, Transaction).
    #[allow(dead_code)]
    pub(crate) fn convert_pager_type<Q>(self) -> Result<RuntimeStatementResult<Q>>
    where
        Q: Pager<Blob = EntryHandle> + Send + Sync,
    {
        match self {
            RuntimeStatementResult::CreateTable { table_name } => {
                Ok(RuntimeStatementResult::CreateTable { table_name })
            }
            RuntimeStatementResult::CreateIndex {
                table_name,
                index_name,
            } => Ok(RuntimeStatementResult::CreateIndex {
                table_name,
                index_name,
            }),
            RuntimeStatementResult::NoOp => Ok(RuntimeStatementResult::NoOp),
            RuntimeStatementResult::Insert {
                table_name,
                rows_inserted,
            } => Ok(RuntimeStatementResult::Insert {
                table_name,
                rows_inserted,
            }),
            RuntimeStatementResult::Update {
                table_name,
                rows_updated,
            } => Ok(RuntimeStatementResult::Update {
                table_name,
                rows_updated,
            }),
            RuntimeStatementResult::Delete {
                table_name,
                rows_deleted,
            } => Ok(RuntimeStatementResult::Delete {
                table_name,
                rows_deleted,
            }),
            RuntimeStatementResult::Transaction { kind } => {
                Ok(RuntimeStatementResult::Transaction { kind })
            }
            RuntimeStatementResult::Select { .. } => Err(Error::Internal(
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
        PlanStatement::CreateIndex(plan) => Some(&plan.table),
        PlanStatement::Insert(plan) => Some(&plan.table),
        PlanStatement::Update(plan) => Some(&plan.table),
        PlanStatement::Delete(plan) => Some(&plan.table),
        PlanStatement::Select(plan) => {
            // Return Some only for single-table queries
            if plan.tables.len() == 1 {
                Some(&plan.tables[0].table)
            } else {
                None
            }
        }
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
pub struct RuntimeContextWrapper<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    ctx: Arc<RuntimeContext<P>>,
    snapshot: RwLock<TransactionSnapshot>,
}

impl<P> RuntimeContextWrapper<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn new(ctx: Arc<RuntimeContext<P>>) -> Self {
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

    fn context(&self) -> &Arc<RuntimeContext<P>> {
        &self.ctx
    }

    fn ctx(&self) -> &RuntimeContext<P> {
        &self.ctx
    }
}

struct SessionNamespaces<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    persistent: Arc<PersistentNamespace<P>>,
    temporary: Option<Arc<TemporaryNamespace<BoxedPager>>>,
    registry: Arc<RwLock<StorageNamespaceRegistry>>,
}

impl<P> SessionNamespaces<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn new(base_context: Arc<RuntimeContext<P>>) -> Self {
        let persistent = Arc::new(PersistentNamespace::new(
            storage_namespace::PERSISTENT_NAMESPACE_ID.to_string(),
            Arc::clone(&base_context),
        ));

        let mut registry = StorageNamespaceRegistry::new(persistent.namespace_id().clone());
        registry.register_namespace(Arc::clone(&persistent), Vec::<String>::new(), false);

        let temporary = {
            let temp_pager = Arc::new(BoxedPager::from_arc(Arc::new(MemPager::default())));
            let temp_context = Arc::new(RuntimeContext::new(temp_pager));
            let namespace = Arc::new(TemporaryNamespace::new(
                storage_namespace::TEMPORARY_NAMESPACE_ID.to_string(),
                temp_context,
            ));
            registry.register_namespace(
                Arc::clone(&namespace),
                vec![storage_namespace::TEMPORARY_NAMESPACE_ID.to_string()],
                true,
            );
            namespace
        };

        Self {
            persistent,
            temporary: Some(temporary),
            registry: Arc::new(RwLock::new(registry)),
        }
    }

    fn persistent(&self) -> Arc<PersistentNamespace<P>> {
        Arc::clone(&self.persistent)
    }

    fn temporary(&self) -> Option<Arc<TemporaryNamespace<BoxedPager>>> {
        self.temporary.as_ref().map(Arc::clone)
    }

    fn registry(&self) -> Arc<RwLock<StorageNamespaceRegistry>> {
        Arc::clone(&self.registry)
    }
}

impl<P> Drop for SessionNamespaces<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn drop(&mut self) {
        if let Some(temp) = &self.temporary {
            temp.clear_tables();
        }
    }
}

/// A session for executing operations with optional transaction support.
///
/// This is a high-level wrapper around the transaction machinery that provides
/// a clean API for users. Operations can be executed directly or within a transaction.
pub struct RuntimeSession<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    // TODO: Allow generic pager type
    inner: TransactionSession<RuntimeContextWrapper<P>, RuntimeContextWrapper<MemPager>>,
    namespaces: Arc<SessionNamespaces<P>>,
}

impl<P> RuntimeSession<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Clone this session (reuses the same underlying TransactionSession).
    /// This is necessary to maintain transaction state across Engine clones.
    pub(crate) fn clone_session(&self) -> Self {
        Self {
            inner: self.inner.clone_session(),
            namespaces: self.namespaces.clone(),
        }
    }

    pub fn namespace_registry(&self) -> Arc<RwLock<StorageNamespaceRegistry>> {
        self.namespaces.registry()
    }

    fn resolve_namespace_for_table(&self, canonical: &str) -> storage_namespace::NamespaceId {
        self.namespace_registry()
            .read()
            .expect("namespace registry poisoned")
            .namespace_for_table(canonical)
    }

    fn namespace_for_select_plan(
        &self,
        plan: &SelectPlan,
    ) -> Option<storage_namespace::NamespaceId> {
        if plan.tables.len() != 1 {
            return None;
        }

        let qualified = plan.tables[0].qualified_name();
        let (_, canonical) = canonical_table_name(&qualified).ok()?;
        Some(self.resolve_namespace_for_table(&canonical))
    }

    fn select_from_temporary(&self, plan: SelectPlan) -> Result<RuntimeStatementResult<P>> {
        let temp_namespace = self
            .temporary_namespace()
            .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;

        let table_name = if plan.tables.len() == 1 {
            plan.tables[0].qualified_name()
        } else {
            String::new()
        };

        let execution = temp_namespace.context().execute_select(plan.clone())?;
        let schema = execution.schema();
        let batches = execution.collect()?;

        let combined = if batches.is_empty() {
            RecordBatch::new_empty(Arc::clone(&schema))
        } else if batches.len() == 1 {
            batches.into_iter().next().unwrap()
        } else {
            let refs: Vec<&RecordBatch> = batches.iter().collect();
            arrow::compute::concat_batches(&schema, refs)?
        };

        let execution =
            SelectExecution::from_batch(table_name.clone(), Arc::clone(&schema), combined);

        Ok(RuntimeStatementResult::Select {
            execution,
            table_name,
            schema,
        })
    }

    fn persistent_namespace(&self) -> Arc<PersistentNamespace<P>> {
        self.namespaces.persistent()
    }

    #[allow(dead_code)]
    fn temporary_namespace(&self) -> Option<Arc<TemporaryNamespace<BoxedPager>>> {
        self.namespaces.temporary()
    }

    /// Begin a transaction in this session.
    /// Creates an empty staging context for new tables created within the transaction.
    /// Existing tables are accessed via MVCC visibility filtering - NO data copying occurs.
    pub fn begin_transaction(&self) -> Result<RuntimeStatementResult<P>> {
        let staging_pager = Arc::new(MemPager::default());
        tracing::trace!(
            "BEGIN_TRANSACTION: Created staging pager at {:p}",
            &*staging_pager
        );
        let staging_ctx = Arc::new(RuntimeContext::new(staging_pager));

        // Staging context is EMPTY - used only for tables created within the transaction.
        // Existing tables are read from base context with MVCC visibility filtering.
        // No data copying occurs at BEGIN - this is pure MVCC.

        let staging_wrapper = Arc::new(RuntimeContextWrapper::new(staging_ctx));

        self.inner.begin_transaction(staging_wrapper)?;
        Ok(RuntimeStatementResult::Transaction {
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
    pub fn commit_transaction(&self) -> Result<RuntimeStatementResult<P>> {
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

        // Persist the next_txn_id to the catalog after a successful commit
        if matches!(kind, TransactionKind::Commit) {
            let ctx = base_ctx.ctx();
            let next_txn_id = ctx.txn_manager().current_next_txn_id();
            if let Err(e) = ctx.persist_next_txn_id(next_txn_id) {
                tracing::warn!("[COMMIT] Failed to persist next_txn_id: {}", e);
            }
        }

        // Return a StatementResult with the correct kind (Commit or Rollback)
        Ok(RuntimeStatementResult::Transaction { kind })
    }

    /// Rollback the current transaction, discarding all changes.
    pub fn rollback_transaction(&self) -> Result<RuntimeStatementResult<P>> {
        self.inner.rollback_transaction()?;
        let base_ctx = self.inner.context();
        let default_snapshot = base_ctx.ctx().default_snapshot();
        TransactionContext::set_snapshot(&**base_ctx, default_snapshot);
        Ok(RuntimeStatementResult::Transaction {
            kind: TransactionKind::Rollback,
        })
    }

    fn materialize_create_table_plan(&self, mut plan: CreateTablePlan) -> Result<CreateTablePlan> {
        if let Some(CreateTableSource::Select { plan: select_plan }) = plan.source.take() {
            let select_result = self.select(*select_plan)?;
            let (schema, batches) = match select_result {
                RuntimeStatementResult::Select {
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
    pub fn create_table_plan(&self, plan: CreateTablePlan) -> Result<RuntimeStatementResult<P>> {
        let mut plan = self.materialize_create_table_plan(plan)?;
        let namespace_id = plan
            .namespace
            .clone()
            .unwrap_or_else(|| storage_namespace::PERSISTENT_NAMESPACE_ID.to_string());
        plan.namespace = Some(namespace_id.clone());

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                temp_namespace.create_table(plan)?.convert_pager_type::<P>()
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                if self.has_active_transaction() {
                    let table_name = plan.name.clone();
                    match self
                        .inner
                        .execute_operation(PlanOperation::CreateTable(plan))
                    {
                        Ok(_) => Ok(RuntimeStatementResult::CreateTable { table_name }),
                        Err(e) => {
                            // If an error occurs during a transaction, abort it
                            self.abort_transaction();
                            Err(e)
                        }
                    }
                } else {
                    self.persistent_namespace().create_table(plan)
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }

    pub fn drop_table(&self, name: &str, if_exists: bool) -> Result<()> {
        let (_, canonical_table) = canonical_table_name(name)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                temp_namespace.drop_table(name, if_exists)
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                self.persistent_namespace().drop_table(name, if_exists)
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }
    /// Create an index (auto-commit only for now).
    pub fn create_index(&self, plan: CreateIndexPlan) -> Result<RuntimeStatementResult<P>> {
        let (_, canonical_table) = canonical_table_name(&plan.table)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                temp_namespace.create_index(plan)?.convert_pager_type::<P>()
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                if self.has_active_transaction() {
                    return Err(Error::InvalidArgumentError(
                        "CREATE INDEX is not supported inside an active transaction".into(),
                    ));
                }

                self.persistent_namespace().create_index(plan)
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
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
                    RuntimeStatementResult::Select { execution, .. } => execution.into_rows()?,
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
    pub fn insert(&self, plan: InsertPlan) -> Result<RuntimeStatementResult<P>> {
        tracing::trace!("Session::insert called for table={}", plan.table);
        let (plan, rows_inserted) = self.normalize_insert_plan(plan)?;
        let table_name = plan.table.clone();
        let (_, canonical_table) = canonical_table_name(&plan.table)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                temp_namespace
                    .context()
                    .insert(plan)?
                    .convert_pager_type::<P>()?;
                Ok(RuntimeStatementResult::Insert {
                    rows_inserted,
                    table_name,
                })
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                if self.has_active_transaction() {
                    match self.inner.execute_operation(PlanOperation::Insert(plan)) {
                        Ok(_) => {
                            tracing::trace!("Session::insert succeeded for table={}", table_name);
                            Ok(RuntimeStatementResult::Insert {
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
                            if matches!(e, Error::ConstraintError(_)) {
                                tracing::trace!("Transaction is_aborted=true");
                                self.abort_transaction();
                            }
                            Err(e)
                        }
                    }
                } else {
                    let context = self.inner.context();
                    let default_snapshot = context.ctx().default_snapshot();
                    TransactionContext::set_snapshot(&**context, default_snapshot);
                    TransactionContext::insert(&**context, plan)?;
                    Ok(RuntimeStatementResult::Insert {
                        rows_inserted,
                        table_name,
                    })
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }

    /// Select rows (outside or inside transaction).
    pub fn select(&self, plan: SelectPlan) -> Result<RuntimeStatementResult<P>> {
        if let Some(namespace_id) = self.namespace_for_select_plan(&plan)
            && namespace_id == storage_namespace::TEMPORARY_NAMESPACE_ID
        {
            return self.select_from_temporary(plan);
        }

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

                    Ok(RuntimeStatementResult::Select {
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
            let table_name = if plan.tables.len() == 1 {
                plan.tables[0].qualified_name()
            } else {
                String::new()
            };
            let execution = TransactionContext::execute_select(&**context, plan)?;
            let schema = execution.schema();
            Ok(RuntimeStatementResult::Select {
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
            RuntimeStatementResult::Select { execution, .. } => Ok(execution.collect_rows()?.rows),
            other => Err(Error::Internal(format!(
                "expected Select result when reading table '{table}', got {:?}",
                other
            ))),
        }
    }

    /// Update rows (outside or inside transaction).
    pub fn update(&self, plan: UpdatePlan) -> Result<RuntimeStatementResult<P>> {
        let (_, canonical_table) = canonical_table_name(&plan.table)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                temp_namespace
                    .context()
                    .update(plan)?
                    .convert_pager_type::<P>()
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
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
                        } => Ok(RuntimeStatementResult::Update {
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
                        } => Ok(RuntimeStatementResult::Update {
                            rows_updated,
                            table_name,
                        }),
                        _ => Err(Error::Internal("expected Update result".into())),
                    }
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }

    /// Delete rows (outside or inside transaction).
    pub fn delete(&self, plan: DeletePlan) -> Result<RuntimeStatementResult<P>> {
        let (_, canonical_table) = canonical_table_name(&plan.table)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                temp_namespace
                    .context()
                    .delete(plan)?
                    .convert_pager_type::<P>()
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
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
                        TransactionResult::Delete { rows_deleted } => {
                            Ok(RuntimeStatementResult::Delete {
                                rows_deleted,
                                table_name,
                            })
                        }
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
                        TransactionResult::Delete { rows_deleted } => {
                            Ok(RuntimeStatementResult::Delete {
                                rows_deleted,
                                table_name,
                            })
                        }
                        _ => Err(Error::Internal("expected Delete result".into())),
                    }
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }
}

pub struct RuntimeEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    context: Arc<RuntimeContext<P>>,
    session: RuntimeSession<P>,
}

impl<P> Clone for RuntimeEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        // IMPORTANT: Reuse the same session to maintain transaction state!
        // Creating a new session would break multi-statement transactions.
        tracing::debug!("[ENGINE] RuntimeEngine::clone() called - reusing same session");
        Self {
            context: Arc::clone(&self.context),
            session: self.session.clone_session(),
        }
    }
}

impl<P> RuntimeEngine<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        let context = Arc::new(RuntimeContext::new(pager));
        Self::from_context(context)
    }

    pub fn from_context(context: Arc<RuntimeContext<P>>) -> Self {
        tracing::debug!("[ENGINE] RuntimeEngine::from_context - creating new session");
        let session = context.create_session();
        tracing::debug!("[ENGINE] RuntimeEngine::from_context - created session");
        Self { context, session }
    }

    pub fn context(&self) -> Arc<RuntimeContext<P>> {
        Arc::clone(&self.context)
    }

    pub fn session(&self) -> &RuntimeSession<P> {
        &self.session
    }

    pub fn execute_statement(&self, statement: PlanStatement) -> Result<RuntimeStatementResult<P>> {
        match statement {
            PlanStatement::BeginTransaction => self.session.begin_transaction(),
            PlanStatement::CommitTransaction => self.session.commit_transaction(),
            PlanStatement::RollbackTransaction => self.session.rollback_transaction(),
            PlanStatement::CreateTable(plan) => self.session.create_table_plan(plan),
            PlanStatement::CreateIndex(plan) => self.session.create_index(plan),
            PlanStatement::Insert(plan) => self.session.insert(plan),
            PlanStatement::Update(plan) => self.session.update(plan),
            PlanStatement::Delete(plan) => self.session.delete(plan),
            PlanStatement::Select(plan) => self.session.select(plan),
        }
    }

    pub fn execute_all<I>(&self, statements: I) -> Result<Vec<RuntimeStatementResult<P>>>
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
///
/// Important: "lazy loading" here refers to *table metadata only* (schema,
/// executor-side column descriptors, and a small next-row-id counter). We do
/// NOT eagerly load or materialize the table's row data into memory. All
/// row/column data remains on the ColumnStore and is streamed in chunks during
/// query execution. This keeps the memory footprint low even for very large
/// tables.
///
/// Typical resource usage:
/// - Metadata per table: ~100s of bytes to a few KB (schema + field ids)
/// - ExecutorTable struct: small (handles + counters)
/// - Actual table rows: streamed from disk in chunks (never fully resident)
pub struct RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pager: Arc<P>,
    tables: RwLock<FxHashMap<String, Arc<ExecutorTable<P>>>>,
    dropped_tables: RwLock<FxHashSet<String>>,
    metadata: Arc<MetadataManager<P>>,
    constraint_service: ConstraintService<P>,
    catalog_service: CatalogService<P>,
    // Centralized catalog for table/field name resolution
    catalog: Arc<TableCatalog>,
    // Shared column store for all tables in this context
    // This ensures catalog state is synchronized across all tables
    store: Arc<ColumnStore<P>>,
    // Transaction manager for session-based transactions
    transaction_manager:
        TransactionManager<RuntimeContextWrapper<P>, RuntimeContextWrapper<MemPager>>,
    txn_manager: Arc<TxnIdManager>,
    txn_tables_with_new_rows: RwLock<FxHashMap<TxnId, FxHashSet<String>>>,
}

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(pager: Arc<P>) -> Self {
        Self::new_with_catalog_inner(pager, None)
    }

    pub fn new_with_catalog(pager: Arc<P>, catalog: Arc<TableCatalog>) -> Self {
        Self::new_with_catalog_inner(pager, Some(catalog))
    }

    fn new_with_catalog_inner(pager: Arc<P>, shared_catalog: Option<Arc<TableCatalog>>) -> Self {
        tracing::trace!("RuntimeContext::new called, pager={:p}", &*pager);

        let store = ColumnStore::open(Arc::clone(&pager)).expect("failed to open ColumnStore");
        let catalog = SysCatalog::new(&store);

        let next_txn_id = match catalog.get_next_txn_id() {
            Ok(Some(id)) => {
                tracing::debug!("[CONTEXT] Loaded next_txn_id={} from catalog", id);
                id
            }
            Ok(None) => {
                tracing::debug!("[CONTEXT] No persisted next_txn_id found, starting from default");
                TXN_ID_AUTO_COMMIT + 1
            }
            Err(e) => {
                tracing::warn!("[CONTEXT] Failed to load next_txn_id: {}, using default", e);
                TXN_ID_AUTO_COMMIT + 1
            }
        };

        let last_committed = match catalog.get_last_committed_txn_id() {
            Ok(Some(id)) => {
                tracing::debug!("[CONTEXT] Loaded last_committed={} from catalog", id);
                id
            }
            Ok(None) => {
                tracing::debug!(
                    "[CONTEXT] No persisted last_committed found, starting from default"
                );
                TXN_ID_AUTO_COMMIT
            }
            Err(e) => {
                tracing::warn!(
                    "[CONTEXT] Failed to load last_committed: {}, using default",
                    e
                );
                TXN_ID_AUTO_COMMIT
            }
        };

        let store_arc = Arc::new(store);
        let metadata = Arc::new(MetadataManager::new(Arc::clone(&store_arc)));

        let loaded_tables = match metadata.all_table_metas() {
            Ok(metas) => {
                tracing::debug!("[CONTEXT] Loaded {} table(s) from catalog", metas.len());
                metas
            }
            Err(e) => {
                tracing::warn!(
                    "[CONTEXT] Failed to load table metas: {}, starting with empty registry",
                    e
                );
                Vec::new()
            }
        };

        let transaction_manager =
            TransactionManager::new_with_initial_state(next_txn_id, last_committed);
        let txn_manager = transaction_manager.txn_manager();

        // LAZY LOADING: Only load table metadata at first access. We intentionally
        // avoid loading any row/column data into memory here. The executor
        // performs streaming reads from the ColumnStore when a query runs, so
        // large tables are never fully materialized.
        //
        // Benefits of this approach:
        // - Instant database open (no upfront I/O for table data)
        // - Lower memory footprint (only metadata cached)
        // - Natural parallelism: if multiple threads request different tables
        //   concurrently, those tables will be loaded concurrently by the
        //   caller threads (no global preload required).
        //
        // Future Optimizations (if profiling shows need):
        // 1. Eager parallel preload of a short "hot" list of tables (rayon)
        // 2. Background preload of catalog entries after startup
        // 3. LRU-based eviction for extremely large deployments
        // 4. Cache compact representations of schemas to reduce per-table RAM
        //
        // Note: `loaded_tables` holds catalog metadata that helped us discover
        // which tables exist; we discard it here because metadata will be
        // fetched on-demand during lazy loads.
        tracing::debug!(
            "[CONTEXT] Initialized with lazy loading for {} table(s)",
            loaded_tables.len()
        );

        // Initialize catalog and populate with existing tables
        let (catalog, is_shared_catalog) = match shared_catalog {
            Some(existing) => (existing, true),
            None => (Arc::new(TableCatalog::new()), false),
        };
        for (_table_id, table_meta) in &loaded_tables {
            if let Some(ref table_name) = table_meta.name
                && let Err(e) = catalog.register_table(table_name.as_str())
            {
                match e {
                    Error::CatalogError(ref msg)
                        if is_shared_catalog && msg.contains("already exists") =>
                    {
                        tracing::debug!(
                            "[CONTEXT] Shared catalog already contains table '{}'",
                            table_name
                        );
                    }
                    other => {
                        tracing::warn!(
                            "[CONTEXT] Failed to register table '{}' in catalog: {}",
                            table_name,
                            other
                        );
                    }
                }
            }
        }
        tracing::debug!(
            "[CONTEXT] Catalog initialized with {} table(s)",
            catalog.table_count()
        );

        let constraint_service =
            ConstraintService::new(Arc::clone(&metadata), Arc::clone(&catalog));
        let catalog_service = CatalogService::new(
            Arc::clone(&metadata),
            Arc::clone(&catalog),
            Arc::clone(&store_arc),
        );

        Self {
            pager,
            tables: RwLock::new(FxHashMap::default()), // Start with empty table cache
            dropped_tables: RwLock::new(FxHashSet::default()),
            metadata,
            constraint_service,
            catalog_service,
            catalog,
            store: store_arc,
            transaction_manager,
            txn_manager,
            txn_tables_with_new_rows: RwLock::new(FxHashMap::default()),
        }
    }

    /// Return the transaction ID manager shared with sessions.
    pub fn txn_manager(&self) -> Arc<TxnIdManager> {
        Arc::clone(&self.txn_manager)
    }

    /// Persist the next_txn_id to the catalog.
    pub fn persist_next_txn_id(&self, next_txn_id: TxnId) -> Result<()> {
        let catalog = SysCatalog::new(&self.store);
        catalog.put_next_txn_id(next_txn_id)?;
        let last_committed = self.txn_manager.last_committed();
        catalog.put_last_committed_txn_id(last_committed)?;
        tracing::debug!(
            "[CONTEXT] Persisted next_txn_id={}, last_committed={}",
            next_txn_id,
            last_committed
        );
        Ok(())
    }

    fn build_executor_multi_column_uniques(
        table: &ExecutorTable<P>,
        stored: &[MultiColumnUniqueEntryMeta],
    ) -> Vec<ExecutorMultiColumnUnique> {
        let mut results = Vec::with_capacity(stored.len());

        'outer: for entry in stored {
            if entry.column_ids.is_empty() {
                continue;
            }

            let mut column_indices = Vec::with_capacity(entry.column_ids.len());
            for field_id in &entry.column_ids {
                if let Some((idx, _)) = table
                    .schema
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, col)| &col.field_id == field_id)
                {
                    column_indices.push(idx);
                } else {
                    tracing::warn!(
                        "[CATALOG] Skipping persisted multi-column UNIQUE {:?} for table_id={} missing field_id {}",
                        entry.index_name,
                        table.table.table_id(),
                        field_id
                    );
                    continue 'outer;
                }
            }

            results.push(ExecutorMultiColumnUnique {
                index_name: entry.index_name.clone(),
                column_indices,
            });
        }

        results
    }

    /// Construct the default snapshot for auto-commit operations.
    pub fn default_snapshot(&self) -> TransactionSnapshot {
        TransactionSnapshot {
            txn_id: TXN_ID_AUTO_COMMIT,
            snapshot_id: self.txn_manager.last_committed(),
        }
    }

    /// Get the table catalog for schema and table name management.
    pub fn table_catalog(&self) -> Arc<TableCatalog> {
        Arc::clone(&self.catalog)
    }

    /// Create a new session for transaction management.
    /// Each session can have its own independent transaction.
    pub fn create_session(self: &Arc<Self>) -> RuntimeSession<P> {
        tracing::debug!("[SESSION] RuntimeContext::create_session called");
        let namespaces = Arc::new(SessionNamespaces::new(Arc::clone(self)));
        let wrapper = RuntimeContextWrapper::new(Arc::clone(self));
        let inner = self.transaction_manager.create_session(Arc::new(wrapper));
        tracing::debug!(
            "[SESSION] Created TransactionSession with session_id (will be logged by transaction manager)"
        );
        RuntimeSession { inner, namespaces }
    }

    /// Get a handle to an existing table by name.
    pub fn table(self: &Arc<Self>, name: &str) -> Result<RuntimeTableHandle<P>> {
        RuntimeTableHandle::new(Arc::clone(self), name)
    }

    /// Check if there's an active transaction (checks if ANY session has a transaction).
    #[deprecated(note = "Use session-based transactions instead")]
    pub fn has_active_transaction(&self) -> bool {
        self.transaction_manager.has_active_transaction()
    }

    pub fn create_table<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
    ) -> Result<RuntimeTableHandle<P>>
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
    ) -> Result<RuntimeTableHandle<P>>
    where
        C: IntoColumnSpec,
        I: IntoIterator<Item = C>,
    {
        self.create_table_with_options(name, columns, true)
    }

    pub fn create_table_plan(&self, plan: CreateTablePlan) -> Result<RuntimeStatementResult<P>> {
        if plan.columns.is_empty() && plan.source.is_none() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE requires explicit columns or a source".into(),
            ));
        }

        let (display_name, canonical_name) = canonical_table_name(&plan.name)?;
        let CreateTablePlan {
            name: _,
            if_not_exists,
            or_replace,
            columns,
            source,
            namespace: _,
            foreign_keys,
        } = plan;

        tracing::trace!(
            "DEBUG create_table_plan: table='{}' if_not_exists={} columns={}",
            display_name,
            if_not_exists,
            columns.len()
        );
        for (idx, col) in columns.iter().enumerate() {
            tracing::trace!(
                "  plan column[{}]: name='{}' primary_key={}",
                idx,
                col.name,
                col.primary_key
            );
        }
        let (exists, is_dropped) = {
            let tables = self.tables.read().unwrap();
            let in_cache = tables.contains_key(&canonical_name);
            let is_dropped = self
                .dropped_tables
                .read()
                .unwrap()
                .contains(&canonical_name);
            // Table exists if it's in cache and NOT marked as dropped
            (in_cache && !is_dropped, is_dropped)
        };
        tracing::trace!(
            "DEBUG create_table_plan: exists={}, is_dropped={}",
            exists,
            is_dropped
        );

        // If table was dropped, remove it from cache before creating new one
        if is_dropped {
            self.remove_table_entry(&canonical_name);
            self.dropped_tables.write().unwrap().remove(&canonical_name);
        }

        if exists {
            if or_replace {
                tracing::trace!(
                    "DEBUG create_table_plan: table '{}' exists and or_replace=true, removing existing table before recreation",
                    display_name
                );
                self.remove_table_entry(&canonical_name);
            } else if if_not_exists {
                tracing::trace!(
                    "DEBUG create_table_plan: table '{}' exists and if_not_exists=true, returning early WITHOUT creating",
                    display_name
                );
                return Ok(RuntimeStatementResult::CreateTable {
                    table_name: display_name,
                });
            } else {
                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' already exists",
                    display_name
                )));
            }
        }

        match source {
            Some(CreateTableSource::Batches { schema, batches }) => self.create_table_from_batches(
                display_name,
                canonical_name,
                schema,
                batches,
                if_not_exists,
            ),
            Some(CreateTableSource::Select { .. }) => Err(Error::Internal(
                "CreateTableSource::Select should be materialized before reaching RuntimeContext::create_table_plan"
                    .into(),
            )),
            None => self.create_table_from_columns(
                display_name,
                canonical_name,
                columns,
                foreign_keys,
                if_not_exists,
            ),
        }
    }

    pub fn create_index(&self, plan: CreateIndexPlan) -> Result<RuntimeStatementResult<P>> {
        if plan.columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE INDEX requires at least one column".into(),
            ));
        }

        for column_plan in &plan.columns {
            if !column_plan.ascending || column_plan.nulls_first {
                return Err(Error::InvalidArgumentError(
                    "only ASC indexes with NULLS LAST are supported".into(),
                ));
            }
        }

        let index_name = plan.name.clone();
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        let table = self.lookup_table(&canonical_name)?;

        let mut column_indices = Vec::with_capacity(plan.columns.len());
        let mut field_ids = Vec::with_capacity(plan.columns.len());
        let mut column_names = Vec::with_capacity(plan.columns.len());
        let mut seen_column_indices = FxHashSet::default();

        for column_plan in &plan.columns {
            let normalized = column_plan.name.to_ascii_lowercase();
            let col_idx = table
                .schema
                .lookup
                .get(&normalized)
                .copied()
                .ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "column '{}' does not exist in table '{}'",
                        column_plan.name, display_name
                    ))
                })?;
            if !seen_column_indices.insert(col_idx) {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column '{}' in CREATE INDEX",
                    column_plan.name
                )));
            }

            let column = &table.schema.columns[col_idx];
            column_indices.push(col_idx);
            field_ids.push(column.field_id);
            column_names.push(column.name.clone());
        }

        if plan.columns.len() == 1 {
            let field_id = field_ids[0];
            let column_name = column_names[0].clone();
            let table_id = table.table.table_id();
            let existing_indexes = table.table.list_registered_indexes(field_id)?;

            if existing_indexes.contains(&IndexKind::Sort) {
                if plan.if_not_exists {
                    return Ok(RuntimeStatementResult::CreateIndex {
                        table_name: display_name,
                        index_name,
                    });
                }

                return Err(Error::CatalogError(format!(
                    "Index already exists on column '{}'",
                    column_name
                )));
            }

            if plan.unique {
                let snapshot = self.default_snapshot();
                let existing_values =
                    self.scan_column_values(table.as_ref(), field_id, snapshot)?;
                ensure_single_column_unique(&existing_values, &[], &column_name)?;
                if let Some(table_id) = self.catalog.table_id(&canonical_name)
                    && let Some(resolver) = self.catalog.field_resolver(table_id)
                {
                    resolver.set_field_unique(&column_name, true)?;
                }
            }

            self.metadata.register_sort_index(table_id, field_id)?;
            self.metadata.flush_table(table_id)?;

            if let Some(updated_table) =
                Self::rebuild_executor_table_with_unique(table.as_ref(), field_id)
            {
                self.tables
                    .write()
                    .unwrap()
                    .insert(canonical_name.clone(), Arc::clone(&updated_table));
            } else {
                self.remove_table_entry(&canonical_name);
            }

            drop(table);

            return Ok(RuntimeStatementResult::CreateIndex {
                table_name: display_name,
                index_name,
            });
        }

        if !plan.unique {
            return Err(Error::InvalidArgumentError(
                "multi-column CREATE INDEX currently supports UNIQUE indexes only".into(),
            ));
        }

        let table_id = table.table.table_id();

        let snapshot = self.default_snapshot();
        let existing_rows = self.scan_multi_column_values(table.as_ref(), &field_ids, snapshot)?;
        ensure_multi_column_unique(&existing_rows, &[], &column_names)?;

        let executor_entry = ExecutorMultiColumnUnique {
            index_name: index_name.clone(),
            column_indices: column_indices.clone(),
        };

        let registration =
            self.metadata
                .register_multi_column_unique(table_id, &field_ids, index_name.clone())?;

        match registration {
            MultiColumnUniqueRegistration::Created => {
                self.metadata.flush_table(table_id)?;
                table.add_multi_column_unique(executor_entry);
            }
            MultiColumnUniqueRegistration::AlreadyExists {
                index_name: existing,
            } => {
                if plan.if_not_exists {
                    drop(table);
                    return Ok(RuntimeStatementResult::CreateIndex {
                        table_name: display_name,
                        index_name: existing,
                    });
                }
                return Err(Error::CatalogError(format!(
                    "Index already exists on columns '{}'",
                    column_names.join(", ")
                )));
            }
        }

        Ok(RuntimeStatementResult::CreateIndex {
            table_name: display_name,
            index_name,
        })
    }

    pub fn table_names(self: &Arc<Self>) -> Vec<String> {
        // Use catalog for table names (single source of truth)
        self.catalog.table_names()
    }

    pub fn table_view(&self, canonical_name: &str) -> Result<TableView> {
        let table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let mut logical_fields = self.store.user_field_ids_for_table(table_id);
        logical_fields.sort_by_key(|lfid| lfid.field_id());
        let field_ids: Vec<FieldId> = logical_fields.iter().map(|lfid| lfid.field_id()).collect();

        self.metadata
            .table_view(&self.catalog, table_id, &field_ids)
            .map_err(Into::into)
    }

    fn filter_visible_row_ids(
        &self,
        table: &ExecutorTable<P>,
        row_ids: Vec<RowId>,
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<RowId>> {
        filter_row_ids_for_snapshot(table.table.as_ref(), row_ids, &self.txn_manager, snapshot)
    }

    pub fn create_table_builder(&self, name: &str) -> RuntimeCreateTableBuilder<'_, P> {
        RuntimeCreateTableBuilder {
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
                .with_unique(column.unique)
                .with_check(column.check_expr.clone())
            })
            .collect())
    }

    pub fn export_table_rows(self: &Arc<Self>, name: &str) -> Result<RowBatch> {
        let handle = RuntimeTableHandle::new(Arc::clone(self), name)?;
        handle.lazy()?.collect_rows()
    }

    fn execute_create_table(&self, plan: CreateTablePlan) -> Result<RuntimeStatementResult<P>> {
        self.create_table_plan(plan)
    }

    fn create_table_with_options<C, I>(
        self: &Arc<Self>,
        name: &str,
        columns: I,
        if_not_exists: bool,
    ) -> Result<RuntimeTableHandle<P>>
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
            RuntimeStatementResult::CreateTable { .. } => {
                RuntimeTableHandle::new(Arc::clone(self), name)
            }
            other => Err(Error::InvalidArgumentError(format!(
                "unexpected statement result {other:?} when creating table"
            ))),
        }
    }

    pub fn insert(&self, plan: InsertPlan) -> Result<RuntimeStatementResult<P>> {
        // For non-transactional inserts, use TXN_ID_AUTO_COMMIT directly
        // instead of creating a temporary transaction
        let snapshot = TransactionSnapshot {
            txn_id: TXN_ID_AUTO_COMMIT,
            snapshot_id: self.txn_manager.last_committed(),
        };
        self.insert_with_snapshot(plan, snapshot)
    }

    pub fn insert_with_snapshot(
        &self,
        plan: InsertPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
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
                canonical_name.clone(),
                rows,
                plan.columns,
                snapshot,
            ),
            InsertSource::Batches(batches) => self.insert_batches(
                table.as_ref(),
                display_name.clone(),
                canonical_name.clone(),
                batches,
                plan.columns,
                snapshot,
            ),
            InsertSource::Select { .. } => Err(Error::Internal(
                "InsertSource::Select should be materialized before reaching RuntimeContext::insert"
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

        if matches!(result, Err(Error::NotFound)) {
            panic!(
                "BUG: insert yielded Error::NotFound for table '{}'. \
                 This should never happen: insert should never return NotFound after successful table lookup. \
                 This indicates a logic error in the runtime.",
                display_name
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

        // Scan to get the column data without materializing full columns
        let table_id = table.table.table_id();

        let mut fields: Vec<Field> = Vec::with_capacity(table.schema.columns.len() + 1);
        let mut logical_fields: Vec<LogicalFieldId> =
            Vec::with_capacity(table.schema.columns.len());

        fields.push(Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false));

        for column in &table.schema.columns {
            let logical_field_id = LogicalFieldId::for_user(table_id, column.field_id);
            logical_fields.push(logical_field_id);
            let field = mvcc_columns::build_field_with_metadata(
                &column.name,
                column.data_type.clone(),
                column.nullable,
                column.field_id,
            );
            fields.push(field);
        }

        let schema = Arc::new(Schema::new(fields));

        if logical_fields.is_empty() {
            // Tables without user columns should still return row_id batches.
            let mut row_id_builder = UInt64Builder::with_capacity(visible_row_ids.len());
            for row_id in &visible_row_ids {
                row_id_builder.append_value(*row_id);
            }
            let arrays: Vec<ArrayRef> = vec![Arc::new(row_id_builder.finish()) as ArrayRef];
            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
            return Ok(vec![batch]);
        }

        let mut stream = table.table.stream_columns(
            Arc::from(logical_fields),
            visible_row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut batches = Vec::new();
        while let Some(chunk) = stream.next_batch()? {
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(chunk.batch().num_columns() + 1);

            let mut row_id_builder = UInt64Builder::with_capacity(chunk.len());
            for row_id in chunk.row_ids() {
                row_id_builder.append_value(*row_id);
            }
            arrays.push(Arc::new(row_id_builder.finish()) as ArrayRef);

            let chunk_batch = chunk.into_batch();
            for column_array in chunk_batch.columns() {
                arrays.push(column_array.clone());
            }

            let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
            batches.push(batch);
        }

        Ok(batches)
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

    pub fn update(&self, plan: UpdatePlan) -> Result<RuntimeStatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        match self.update_with_snapshot(plan, snapshot) {
            Ok(result) => {
                self.txn_manager.mark_committed(snapshot.txn_id);
                Ok(result)
            }
            Err(err) => {
                self.txn_manager.mark_aborted(snapshot.txn_id);
                Err(err)
            }
        }
    }

    pub fn update_with_snapshot(
        &self,
        plan: UpdatePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let UpdatePlan {
            table,
            assignments,
            filter,
        } = plan;
        let (display_name, canonical_name) = canonical_table_name(&table)?;
        let table = self.lookup_table(&canonical_name)?;
        if let Some(filter) = filter {
            self.update_filtered_rows(
                table.as_ref(),
                display_name,
                canonical_name,
                assignments,
                filter,
                snapshot,
            )
        } else {
            self.update_all_rows(
                table.as_ref(),
                display_name,
                canonical_name,
                assignments,
                snapshot,
            )
        }
    }

    pub fn delete(&self, plan: DeletePlan) -> Result<RuntimeStatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        match self.delete_with_snapshot(plan, snapshot) {
            Ok(result) => {
                self.txn_manager.mark_committed(snapshot.txn_id);
                Ok(result)
            }
            Err(err) => {
                self.txn_manager.mark_aborted(snapshot.txn_id);
                Err(err)
            }
        }
    }

    pub fn delete_with_snapshot(
        &self,
        plan: DeletePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let (display_name, canonical_name) = canonical_table_name(&plan.table)?;
        // Get table - will be checked against snapshot during actual deletion
        let table = match self.tables.read().unwrap().get(&canonical_name) {
            Some(t) => Arc::clone(t),
            None => return Err(Error::NotFound),
        };
        match plan.filter {
            Some(filter) => self.delete_filtered_rows(
                table.as_ref(),
                display_name,
                canonical_name.clone(),
                filter,
                snapshot,
            ),
            None => self.delete_all_rows(table.as_ref(), display_name, canonical_name, snapshot),
        }
    }

    pub fn table_handle(self: &Arc<Self>, name: &str) -> Result<RuntimeTableHandle<P>> {
        RuntimeTableHandle::new(Arc::clone(self), name)
    }

    pub fn execute_select(self: &Arc<Self>, plan: SelectPlan) -> Result<SelectExecution<P>> {
        let snapshot = self.default_snapshot();
        self.execute_select_with_snapshot(plan, snapshot)
    }

    pub fn execute_select_with_snapshot(
        self: &Arc<Self>,
        plan: SelectPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<SelectExecution<P>> {
        // Handle SELECT without FROM clause (e.g., SELECT 42, SELECT {'a': 1})
        if plan.tables.is_empty() {
            let provider: Arc<dyn TableProvider<P>> = Arc::new(ContextProvider {
                context: Arc::clone(self),
            });
            let executor = QueryExecutor::new(provider);
            // No row filter needed since there's no table
            return executor.execute_select_with_filter(plan, None);
        }

        // Resolve canonical names for all tables (don't verify existence - executor will handle it with snapshot)
        let mut canonical_tables = Vec::new();
        for table_ref in &plan.tables {
            let qualified = table_ref.qualified_name();
            let (_display, canonical) = canonical_table_name(&qualified)?;
            // Parse canonical back into schema.table
            let parts: Vec<&str> = canonical.split('.').collect();
            let canon_ref = if parts.len() >= 2 {
                llkv_plan::TableRef::new(parts[0], parts[1])
            } else {
                llkv_plan::TableRef::new("", &canonical)
            };
            canonical_tables.push(canon_ref);
        }

        let mut canonical_plan = plan.clone();
        canonical_plan.tables = canonical_tables;

        let provider: Arc<dyn TableProvider<P>> = Arc::new(ContextProvider {
            context: Arc::clone(self),
        });
        let executor = QueryExecutor::new(provider);

        // For single-table queries, apply MVCC filtering
        let row_filter: Option<Arc<dyn RowIdFilter<P>>> = if canonical_plan.tables.len() == 1 {
            Some(Arc::new(MvccRowIdFilter::new(
                Arc::clone(&self.txn_manager),
                snapshot,
            )))
        } else {
            // For multi-table queries, we don't apply MVCC filtering yet (TODO)
            None
        };

        executor.execute_select_with_filter(canonical_plan, row_filter)
    }

    fn create_table_from_columns(
        &self,
        display_name: String,
        canonical_name: String,
        columns: Vec<ColumnSpec>,
        foreign_keys: Vec<ForeignKeySpec>,
        if_not_exists: bool,
    ) -> Result<RuntimeStatementResult<P>> {
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

        // Avoid repeating catalog work if the table already exists.
        {
            let tables = self.tables.read().unwrap();
            if tables.contains_key(&canonical_name) {
                if if_not_exists {
                    return Ok(RuntimeStatementResult::CreateTable {
                        table_name: display_name,
                    });
                }

                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' already exists",
                    display_name
                )));
            }
        }

        let CreateTableResult {
            table_id,
            catalog_table_id,
            table,
            table_columns,
            column_lookup,
        } = self
            .catalog_service
            .create_table_from_columns(&display_name, &canonical_name, &columns)?;

        tracing::trace!(
            "=== TABLE '{}' CREATED WITH table_id={} pager={:p} ===",
            display_name,
            table_id,
            &*self.pager
        );

        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(table_columns.len());
        for (idx, column) in table_columns.iter().enumerate() {
            tracing::trace!(
                "DEBUG create_table_from_columns[{}]: name='{}' data_type={:?} nullable={} primary_key={} unique={}",
                idx,
                column.name,
                column.data_type,
                column.nullable,
                column.primary_key,
                column.unique
            );
            column_defs.push(ExecutorColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                unique: column.unique,
                field_id: column.field_id,
                check_expr: column.check_expr.clone(),
            });
        }

        let schema = Arc::new(ExecutorSchema {
            columns: column_defs.clone(),
            lookup: column_lookup,
        });
        let table_entry = Arc::new(ExecutorTable {
            table: Arc::clone(&table),
            schema,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

        let mut tables = self.tables.write().unwrap();
        if tables.contains_key(&canonical_name) {
            drop(tables);
            self.catalog.unregister_table(table_id);
            if if_not_exists {
                return Ok(RuntimeStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name.clone(), Arc::clone(&table_entry));
        drop(tables);

        if !foreign_keys.is_empty() {
            let referencing_columns: Vec<ForeignKeyColumn> = column_defs
                .iter()
                .map(|column| ForeignKeyColumn {
                    name: column.name.clone(),
                    data_type: column.data_type.clone(),
                    nullable: column.nullable,
                    primary_key: column.primary_key,
                    unique: column.unique,
                    field_id: column.field_id,
                })
                .collect();

            let referencing_table = ForeignKeyTableInfo {
                display_name: display_name.clone(),
                canonical_name: canonical_name.clone(),
                table_id: catalog_table_id,
                columns: referencing_columns,
            };

            let fk_result = self.metadata.validate_and_register_foreign_keys(
                &referencing_table,
                &foreign_keys,
                |table_name| {
                    let (display, canonical) = canonical_table_name(table_name)?;
                    let referenced_table = self.lookup_table(&canonical).map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "referenced table '{}' does not exist",
                            table_name
                        ))
                    })?;

                    let columns = referenced_table
                        .schema
                        .columns
                        .iter()
                        .map(|column| ForeignKeyColumn {
                            name: column.name.clone(),
                            data_type: column.data_type.clone(),
                            nullable: column.nullable,
                            primary_key: column.primary_key,
                            unique: column.unique,
                            field_id: column.field_id,
                        })
                        .collect();

                    Ok(ForeignKeyTableInfo {
                        display_name: display,
                        canonical_name: canonical,
                        table_id: referenced_table.table.table_id(),
                        columns,
                    })
                },
                current_time_micros(),
            );

            if let Err(err) = fk_result {
                let field_ids: Vec<FieldId> =
                    column_defs.iter().map(|column| column.field_id).collect();
                let _ = self
                    .catalog_service
                    .drop_table(&canonical_name, table_id, &field_ids);
                self.remove_table_entry(&canonical_name);
                return Err(err);
            }
        }

        Ok(RuntimeStatementResult::CreateTable {
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
    ) -> Result<RuntimeStatementResult<P>> {
        {
            let tables = self.tables.read().unwrap();
            if tables.contains_key(&canonical_name) {
                if if_not_exists {
                    return Ok(RuntimeStatementResult::CreateTable {
                        table_name: display_name,
                    });
                }
                return Err(Error::CatalogError(format!(
                    "Catalog Error: Table '{}' already exists",
                    display_name
                )));
            }
        }

        let CreateTableResult {
            table_id,
            catalog_table_id: _,
            table,
            table_columns,
            column_lookup,
        } = self.catalog_service.create_table_from_schema(
            &display_name,
            &canonical_name,
            &schema,
        )?;

        tracing::trace!(
            "=== CTAS table '{}' created with table_id={} pager={:p} ===",
            display_name,
            table_id,
            &*self.pager
        );

        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(table_columns.len());
        for column in &table_columns {
            column_defs.push(ExecutorColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone(),
                nullable: column.nullable,
                primary_key: column.primary_key,
                unique: column.unique,
                field_id: column.field_id,
                check_expr: column.check_expr.clone(),
            });
        }

        let schema_arc = Arc::new(ExecutorSchema {
            columns: column_defs.clone(),
            lookup: column_lookup,
        });
        let table_entry = Arc::new(ExecutorTable {
            table: Arc::clone(&table),
            schema: schema_arc,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

        let mut next_row_id: RowId = 0;
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

            // Build MVCC columns using helper
            let (row_id_array, created_by_array, deleted_by_array) =
                mvcc_columns::build_insert_mvcc_columns(row_count, start_row, creator_txn_id);

            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_defs.len() + 3);
            arrays.push(row_id_array);
            arrays.push(created_by_array);
            arrays.push(deleted_by_array);

            // Build schema fields
            let mut fields: Vec<Field> = Vec::with_capacity(column_defs.len() + 3);
            fields.extend(mvcc_columns::build_mvcc_fields());

            for (idx, column) in column_defs.iter().enumerate() {
                let field = mvcc_columns::build_field_with_metadata(
                    &column.name,
                    column.data_type.clone(),
                    column.nullable,
                    column.field_id,
                );
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
                return Ok(RuntimeStatementResult::CreateTable {
                    table_name: display_name,
                });
            }
            return Err(Error::CatalogError(format!(
                "Catalog Error: Table '{}' already exists",
                display_name
            )));
        }
        tables.insert(canonical_name.clone(), Arc::clone(&table_entry));
        drop(tables); // Release write lock before catalog operations

        Ok(RuntimeStatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn rebuild_executor_table_with_unique(
        table: &ExecutorTable<P>,
        field_id: FieldId,
    ) -> Option<Arc<ExecutorTable<P>>> {
        let mut columns = table.schema.columns.clone();
        let mut found = false;
        for column in &mut columns {
            if column.field_id == field_id {
                column.unique = true;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }

        let schema = Arc::new(ExecutorSchema {
            columns,
            lookup: table.schema.lookup.clone(),
        });

        let next_row_id = table.next_row_id.load(Ordering::SeqCst);
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        let uniques = table.multi_column_uniques();

        Some(Arc::new(ExecutorTable {
            table: Arc::clone(&table.table),
            schema,
            next_row_id: AtomicU64::new(next_row_id),
            total_rows: AtomicU64::new(total_rows),
            multi_column_uniques: RwLock::new(uniques),
        }))
    }

    fn record_table_with_new_rows(&self, txn_id: TxnId, canonical_name: String) {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return;
        }

        let mut guard = self.txn_tables_with_new_rows.write().unwrap();
        guard.entry(txn_id).or_default().insert(canonical_name);
    }

    fn collect_rows_created_by_txn(
        &self,
        table: &ExecutorTable<P>,
        txn_id: TxnId,
    ) -> Result<Vec<Vec<PlanValue>>> {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return Ok(Vec::new());
        }

        if table.schema.columns.is_empty() {
            return Ok(Vec::new());
        }

        let Some(first_field_id) = table.schema.first_field_id() else {
            return Ok(Vec::new());
        };
        let filter_expr = full_table_scan_filter(first_field_id);

        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let mut logical_fields: Vec<LogicalFieldId> =
            Vec::with_capacity(table.schema.columns.len() + 2);
        logical_fields.push(LogicalFieldId::for_mvcc_created_by(table_id));
        logical_fields.push(LogicalFieldId::for_mvcc_deleted_by(table_id));
        for column in &table.schema.columns {
            logical_fields.push(LogicalFieldId::for_user(table_id, column.field_id));
        }

        let logical_fields: Arc<[LogicalFieldId]> = logical_fields.into();
        let mut stream = table.table.stream_columns(
            Arc::clone(&logical_fields),
            row_ids,
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut rows = Vec::new();
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() < table.schema.columns.len() + 2 {
                continue;
            }

            let created_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("missing created_by column in MVCC data".into()))?;
            let deleted_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::Internal("missing deleted_by column in MVCC data".into()))?;

            for row_idx in 0..batch.num_rows() {
                let created_by = if created_col.is_null(row_idx) {
                    TXN_ID_AUTO_COMMIT
                } else {
                    created_col.value(row_idx)
                };
                if created_by != txn_id {
                    continue;
                }

                let deleted_by = if deleted_col.is_null(row_idx) {
                    TXN_ID_NONE
                } else {
                    deleted_col.value(row_idx)
                };
                if deleted_by != TXN_ID_NONE {
                    continue;
                }

                let mut row_values = Vec::with_capacity(table.schema.columns.len());
                for col_idx in 0..table.schema.columns.len() {
                    let array = batch.column(col_idx + 2);
                    let value = llkv_plan::plan_value_from_array(array, row_idx)?;
                    row_values.push(value);
                }
                rows.push(row_values);
            }
        }

        Ok(rows)
    }

    fn validate_primary_keys_for_commit(&self, txn_id: TxnId) -> Result<()> {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return Ok(());
        }

        let tables = {
            let mut guard = self.txn_tables_with_new_rows.write().unwrap();
            guard.remove(&txn_id)
        };

        let Some(tables) = tables else {
            return Ok(());
        };

        if tables.is_empty() {
            return Ok(());
        }

        let snapshot = TransactionSnapshot {
            txn_id: TXN_ID_AUTO_COMMIT,
            snapshot_id: self.txn_manager.last_committed(),
        };

        for table_name in tables {
            let table = self.lookup_table(&table_name)?;
            let constraint_ctx = self.build_table_constraint_context(table.as_ref())?;
            let Some(primary_key) = constraint_ctx.primary_key.as_ref() else {
                continue;
            };

            let new_rows = self.collect_rows_created_by_txn(table.as_ref(), txn_id)?;
            if new_rows.is_empty() {
                continue;
            }

            let column_order: Vec<usize> = (0..table.schema.columns.len()).collect();
            self.constraint_service.validate_primary_key_rows(
                &constraint_ctx.schema_field_ids,
                primary_key,
                &column_order,
                &new_rows,
                |field_ids| self.scan_multi_column_values(table.as_ref(), field_ids, snapshot),
            )?;
        }

        Ok(())
    }

    fn clear_transaction_state(&self, txn_id: TxnId) {
        if txn_id == TXN_ID_AUTO_COMMIT {
            return;
        }

        let mut guard = self.txn_tables_with_new_rows.write().unwrap();
        guard.remove(&txn_id);
    }

    fn coerce_plan_value_for_column(
        &self,
        value: PlanValue,
        column: &ExecutorColumn,
    ) -> Result<PlanValue> {
        match value {
            PlanValue::Null => Ok(PlanValue::Null),
            PlanValue::Integer(v) => match &column.data_type {
                DataType::Int64 => Ok(PlanValue::Integer(v)),
                DataType::Float64 => Ok(PlanValue::Float(v as f64)),
                DataType::Boolean => Ok(PlanValue::Integer(if v != 0 { 1 } else { 0 })),
                DataType::Utf8 => Ok(PlanValue::String(v.to_string())),
                DataType::Date32 => {
                    let casted = i32::try_from(v).map_err(|_| {
                        Error::InvalidArgumentError(format!(
                            "integer literal out of range for DATE column '{}'",
                            column.name
                        ))
                    })?;
                    Ok(PlanValue::Integer(casted as i64))
                }
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign integer to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::Integer(v)),
            },
            PlanValue::Float(v) => match &column.data_type {
                DataType::Int64 => Ok(PlanValue::Integer(v as i64)),
                DataType::Float64 => Ok(PlanValue::Float(v)),
                DataType::Boolean => Ok(PlanValue::Integer(if v != 0.0 { 1 } else { 0 })),
                DataType::Utf8 => Ok(PlanValue::String(v.to_string())),
                DataType::Date32 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign floating-point value to DATE column '{}'",
                    column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign floating-point value to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::Float(v)),
            },
            PlanValue::String(s) => match &column.data_type {
                DataType::Boolean => {
                    let normalized = s.trim().to_ascii_lowercase();
                    match normalized.as_str() {
                        "true" | "t" | "1" => Ok(PlanValue::Integer(1)),
                        "false" | "f" | "0" => Ok(PlanValue::Integer(0)),
                        _ => Err(Error::InvalidArgumentError(format!(
                            "cannot assign string '{}' to BOOLEAN column '{}'",
                            s, column.name
                        ))),
                    }
                }
                DataType::Utf8 => Ok(PlanValue::String(s)),
                DataType::Date32 => {
                    let days = parse_date32_literal(&s)?;
                    Ok(PlanValue::Integer(days as i64))
                }
                DataType::Int64 | DataType::Float64 => Err(Error::InvalidArgumentError(format!(
                    "cannot assign string '{}' to numeric column '{}'",
                    s, column.name
                ))),
                DataType::Struct(_) => Err(Error::InvalidArgumentError(format!(
                    "cannot assign string to STRUCT column '{}'",
                    column.name
                ))),
                _ => Ok(PlanValue::String(s)),
            },
            PlanValue::Struct(map) => match &column.data_type {
                DataType::Struct(_) => Ok(PlanValue::Struct(map)),
                _ => Err(Error::InvalidArgumentError(format!(
                    "cannot assign struct value to column '{}'",
                    column.name
                ))),
            },
        }
    }

    // TODO: Make streamable; don't buffer all values in memory at once
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
            table.table.as_ref(),
            row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Gather the column values for visible rows
        let logical_field_id = LogicalFieldId::for_user(table_id, field_id);
        let row_count = row_ids.len();
        let mut stream = match table.table.stream_columns(
            vec![logical_field_id],
            row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        // TODO: Don't buffer all values; make this streamable
        let mut values = Vec::with_capacity(row_count);
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }
            let array = batch.column(0);
            for row_idx in 0..batch.num_rows() {
                if let Ok(value) = llkv_plan::plan_value_from_array(array, row_idx) {
                    values.push(value);
                }
            }
        }

        Ok(values)
    }

    // TODO: Make streamable; don't buffer all values in memory at once
    fn scan_multi_column_values(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<Vec<PlanValue>>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        use llkv_expr::{Expr, Filter, Operator};
        use std::ops::Bound;

        let match_all_filter = Filter {
            field_id: field_ids[0],
            op: Operator::Range {
                lower: Bound::Unbounded,
                upper: Bound::Unbounded,
            },
        };
        let filter_expr = Expr::Pred(match_all_filter);

        let row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let logical_field_ids: Vec<_> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let total_rows = row_ids.len();
        let mut stream = match table.table.stream_columns(
            logical_field_ids,
            row_ids,
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); total_rows];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            if batch.num_columns() == 0 {
                continue;
            }

            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < rows.len(),
                        "stream chunk produced out-of-bounds row index"
                    );
                    if let Some(row) = rows.get_mut(target_index) {
                        match llkv_plan::plan_value_from_array(array, local_idx) {
                            Ok(value) => row.push(value),
                            Err(_) => row.push(PlanValue::Null),
                        }
                    }
                }
            }
        }

        Ok(rows)
    }

    fn collect_row_values_for_ids(
        &self,
        table: &ExecutorTable<P>,
        row_ids: &[RowId],
        field_ids: &[FieldId],
    ) -> Result<Vec<Vec<PlanValue>>> {
        if row_ids.is_empty() || field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let logical_field_ids: Vec<LogicalFieldId> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let mut stream = match table.table.stream_columns(
            logical_field_ids.clone(),
            row_ids.to_vec(),
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); row_ids.len()];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    if let Some(row) = rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }

        Ok(rows)
    }

    fn collect_visible_child_rows(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        snapshot: TransactionSnapshot,
    ) -> Result<Vec<(RowId, Vec<PlanValue>)>> {
        if field_ids.is_empty() {
            return Ok(Vec::new());
        }

        let anchor_field = field_ids[0];
        let filter_expr = full_table_scan_filter(anchor_field);
        let raw_row_ids = match table.table.filter_row_ids(&filter_expr) {
            Ok(ids) => ids,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let visible_row_ids = filter_row_ids_for_snapshot(
            table.table.as_ref(),
            raw_row_ids,
            &self.txn_manager,
            snapshot,
        )?;

        if visible_row_ids.is_empty() {
            return Ok(Vec::new());
        }

        let table_id = table.table.table_id();
        let logical_field_ids: Vec<LogicalFieldId> = field_ids
            .iter()
            .map(|&fid| LogicalFieldId::for_user(table_id, fid))
            .collect();

        let mut stream = match table.table.stream_columns(
            logical_field_ids.clone(),
            visible_row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        ) {
            Ok(stream) => stream,
            Err(Error::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut rows = vec![Vec::with_capacity(field_ids.len()); visible_row_ids.len()];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    if let Some(row) = rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }

        Ok(visible_row_ids.into_iter().zip(rows).collect())
    }

    fn build_table_constraint_context(
        &self,
        table: &ExecutorTable<P>,
    ) -> Result<TableConstraintContext> {
        let schema_field_ids: Vec<FieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| column.field_id)
            .collect();

        let column_constraints: Vec<InsertColumnConstraint> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .map(|(idx, column)| InsertColumnConstraint {
                schema_index: idx,
                column: ConstraintColumnInfo {
                    name: column.name.clone(),
                    field_id: column.field_id,
                    data_type: column.data_type.clone(),
                    nullable: column.nullable,
                    check_expr: column.check_expr.clone(),
                },
            })
            .collect();

        let unique_columns: Vec<InsertUniqueColumn> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.unique && !column.primary_key)
            .map(|(idx, column)| InsertUniqueColumn {
                schema_index: idx,
                field_id: column.field_id,
                name: column.name.clone(),
            })
            .collect();

        let mut multi_column_uniques: Vec<InsertMultiColumnUnique> = Vec::new();
        for constraint in table.multi_column_uniques() {
            if constraint.column_indices.is_empty() {
                continue;
            }

            let mut schema_indices = Vec::with_capacity(constraint.column_indices.len());
            let mut field_ids = Vec::with_capacity(constraint.column_indices.len());
            let mut column_names = Vec::with_capacity(constraint.column_indices.len());
            for &col_idx in &constraint.column_indices {
                let column = table.schema.columns.get(col_idx).ok_or_else(|| {
                    Error::Internal(format!(
                        "multi-column UNIQUE constraint references invalid column index {}",
                        col_idx
                    ))
                })?;
                schema_indices.push(col_idx);
                field_ids.push(column.field_id);
                column_names.push(column.name.clone());
            }

            multi_column_uniques.push(InsertMultiColumnUnique {
                schema_indices,
                field_ids,
                column_names,
            });
        }

        let primary_indices: Vec<usize> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.primary_key)
            .map(|(idx, _)| idx)
            .collect();

        let primary_key = if primary_indices.is_empty() {
            None
        } else {
            let mut field_ids = Vec::with_capacity(primary_indices.len());
            let mut column_names = Vec::with_capacity(primary_indices.len());
            for &idx in &primary_indices {
                let column = table.schema.columns.get(idx).ok_or_else(|| {
                    Error::Internal(format!(
                        "primary key references invalid column index {}",
                        idx
                    ))
                })?;
                field_ids.push(column.field_id);
                column_names.push(column.name.clone());
            }

            Some(InsertMultiColumnUnique {
                schema_indices: primary_indices.clone(),
                field_ids,
                column_names,
            })
        };

        Ok(TableConstraintContext {
            schema_field_ids,
            column_constraints,
            unique_columns,
            multi_column_uniques,
            primary_key,
        })
    }

    fn insert_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        mut rows: Vec<Vec<PlanValue>>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
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

        for row in rows.iter_mut() {
            for (position, value) in row.iter_mut().enumerate() {
                let schema_index = column_order
                    .get(position)
                    .copied()
                    .ok_or_else(|| Error::Internal("invalid INSERT column index mapping".into()))?;
                let column = table.schema.columns.get(schema_index).ok_or_else(|| {
                    Error::Internal(format!(
                        "INSERT column index {} out of bounds for table '{}'",
                        schema_index, display_name
                    ))
                })?;
                let normalized = normalize_insert_value_for_column(column, value.clone())?;
                *value = normalized;
            }
        }

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] Validating constraints for {} row(s) before insert",
                rows.len()
            );
            for (i, row) in rows.iter().enumerate() {
                tracing::trace!("[KEYS]   row[{}]: {:?}", i, row);
            }
        }

        self.constraint_service.validate_insert_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &constraint_ctx.unique_columns,
            &constraint_ctx.multi_column_uniques,
            primary_key_spec,
            &column_order,
            &rows,
            |field_id| self.scan_column_values(table, field_id, snapshot),
            |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
        )?;

        self.check_foreign_keys_on_insert(table, &display_name, &rows, &column_order, snapshot)?;

        let row_count = rows.len();
        let mut column_values: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(row_count); table.schema.columns.len()];
        for row in rows {
            for (idx, value) in row.into_iter().enumerate() {
                let dest_index = column_order[idx];
                column_values[dest_index].push(value);
            }
        }

        let start_row = table.next_row_id.load(Ordering::SeqCst);

        // Build MVCC columns using helper
        let (row_id_array, created_by_array, deleted_by_array) =
            mvcc_columns::build_insert_mvcc_columns(row_count, start_row, snapshot.txn_id);

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_values.len() + 3);
        arrays.push(row_id_array);
        arrays.push(created_by_array);
        arrays.push(deleted_by_array);

        let mut fields: Vec<Field> = Vec::with_capacity(column_values.len() + 3);
        fields.extend(mvcc_columns::build_mvcc_fields());

        for (column, values) in table.schema.columns.iter().zip(column_values.into_iter()) {
            let array = build_array_for_column(&column.data_type, &values)?;
            let field = mvcc_columns::build_field_with_metadata(
                &column.name,
                column.data_type.clone(),
                column.nullable,
                column.field_id,
            );
            arrays.push(array);
            fields.push(field);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        tracing::trace!(
            table_name = %display_name,
            store_ptr = ?std::ptr::addr_of!(*table.table.store()),
            "About to call table.append"
        );
        table.table.append(&batch)?;
        table
            .next_row_id
            .store(start_row + row_count as u64, Ordering::SeqCst);
        table
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        self.record_table_with_new_rows(snapshot.txn_id, canonical_name);

        Ok(RuntimeStatementResult::Insert {
            table_name: display_name,
            rows_inserted: row_count,
        })
    }

    fn insert_batches(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        batches: Vec<RecordBatch>,
        columns: Vec<String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        if batches.is_empty() {
            return Ok(RuntimeStatementResult::Insert {
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

            match self.insert_rows(
                table,
                display_name.clone(),
                canonical_name.clone(),
                rows,
                columns.clone(),
                snapshot,
            )? {
                RuntimeStatementResult::Insert { rows_inserted, .. } => {
                    total_rows_inserted += rows_inserted;
                }
                _ => unreachable!("insert_rows must return Insert result"),
            }
        }

        Ok(RuntimeStatementResult::Insert {
            table_name: display_name,
            rows_inserted: total_rows_inserted,
        })
    }

    fn update_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        assignments: Vec<ColumnAssignment>,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
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

        let mut seen_columns: FxHashSet<String> =
            FxHashSet::with_capacity_and_hasher(assignments.len(), Default::default());
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
            return Ok(RuntimeStatementResult::Update {
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

        let mut stream = table.table.stream_columns(
            logical_fields.clone(),
            row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < new_rows.len(),
                        "column stream produced out-of-range row index"
                    );
                    if let Some(row) = new_rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }
        debug_assert!(
            new_rows
                .iter()
                .all(|row| row.len() == table.schema.columns.len())
        );

        tracing::trace!(
            table = %display_name,
            row_count,
            rows = ?new_rows,
            "update_filtered_rows captured source rows"
        );

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();
        let mut original_primary_key_keys: Vec<Option<UniqueKey>> = Vec::new();
        if let Some(pk) = primary_key_spec {
            original_primary_key_keys.reserve(row_count);
            for row in &new_rows {
                let mut values = Vec::with_capacity(pk.schema_indices.len());
                for &idx in &pk.schema_indices {
                    let value = row.get(idx).cloned().unwrap_or(PlanValue::Null);
                    values.push(value);
                }
                let key = build_composite_unique_key(&values, &pk.column_names)?;
                original_primary_key_keys.push(key);
            }
        }

        let column_positions: FxHashMap<FieldId, usize> = FxHashMap::from_iter(
            table
                .schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, column)| (column.field_id, idx)),
        );

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
                    let coerced = self.coerce_plan_value_for_column(new_value, &column)?;
                    row[column_index] = coerced;
                }
            }
        }

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_order = resolve_insert_columns(&column_names, table.schema.as_ref())?;
        self.constraint_service.validate_row_level_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &column_order,
            &new_rows,
        )?;

        if let Some(pk) = primary_key_spec {
            self.constraint_service.validate_update_primary_keys(
                &constraint_ctx.schema_field_ids,
                pk,
                &column_order,
                &new_rows,
                &original_primary_key_keys,
                |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
            )?;
        }

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            canonical_name.clone(),
            row_ids.clone(),
            snapshot,
            false,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            canonical_name,
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn update_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        assignments: Vec<ColumnAssignment>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
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
            return Ok(RuntimeStatementResult::Update {
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

        let mut seen_columns: FxHashSet<String> =
            FxHashSet::with_capacity_and_hasher(assignments.len(), Default::default());
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
            return Ok(RuntimeStatementResult::Update {
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

        let mut stream = table.table.stream_columns(
            logical_fields.clone(),
            row_ids.clone(),
            GatherNullPolicy::IncludeNulls,
        )?;

        let mut new_rows: Vec<Vec<PlanValue>> =
            vec![Vec::with_capacity(table.schema.columns.len()); row_count];
        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let base = chunk.row_offset();
            let local_len = batch.num_rows();
            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                for local_idx in 0..local_len {
                    let target_index = base + local_idx;
                    debug_assert!(
                        target_index < new_rows.len(),
                        "column stream produced out-of-range row index"
                    );
                    if let Some(row) = new_rows.get_mut(target_index) {
                        let value = llkv_plan::plan_value_from_array(array, local_idx)?;
                        row.push(value);
                    }
                }
            }
        }
        debug_assert!(
            new_rows
                .iter()
                .all(|row| row.len() == table.schema.columns.len())
        );

        let constraint_ctx = self.build_table_constraint_context(table)?;
        let primary_key_spec = constraint_ctx.primary_key.as_ref();
        let mut original_primary_key_keys: Vec<Option<UniqueKey>> = Vec::new();
        if let Some(pk) = primary_key_spec {
            original_primary_key_keys.reserve(row_count);
            for row in &new_rows {
                let mut values = Vec::with_capacity(pk.schema_indices.len());
                for &idx in &pk.schema_indices {
                    let value = row.get(idx).cloned().unwrap_or(PlanValue::Null);
                    values.push(value);
                }
                let key = build_composite_unique_key(&values, &pk.column_names)?;
                original_primary_key_keys.push(key);
            }
        }

        let column_positions: FxHashMap<FieldId, usize> = FxHashMap::from_iter(
            table
                .schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, column)| (column.field_id, idx)),
        );

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
                    let coerced = self.coerce_plan_value_for_column(new_value, &column)?;
                    row[column_index] = coerced;
                }
            }
        }

        let column_names: Vec<String> = table
            .schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let column_order = resolve_insert_columns(&column_names, table.schema.as_ref())?;
        self.constraint_service.validate_row_level_constraints(
            &constraint_ctx.schema_field_ids,
            &constraint_ctx.column_constraints,
            &column_order,
            &new_rows,
        )?;

        if let Some(pk) = primary_key_spec {
            self.constraint_service.validate_update_primary_keys(
                &constraint_ctx.schema_field_ids,
                pk,
                &column_order,
                &new_rows,
                &original_primary_key_keys,
                |field_ids| self.scan_multi_column_values(table, field_ids, snapshot),
            )?;
        }

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            canonical_name.clone(),
            row_ids.clone(),
            snapshot,
            false,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
            canonical_name,
            new_rows,
            column_names,
            snapshot,
        )?;

        Ok(RuntimeStatementResult::Update {
            table_name: display_name,
            rows_updated: row_count,
        })
    }

    fn delete_filtered_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let schema = table.schema.as_ref();
        let filter_expr = translate_predicate(filter, schema)?;
        let row_ids = table.table.filter_row_ids(&filter_expr)?;
        let row_ids = self.filter_visible_row_ids(table, row_ids, snapshot)?;
        tracing::trace!(
            table = %display_name,
            rows = row_ids.len(),
            "delete_filtered_rows collected row ids"
        );
        self.apply_delete(table, display_name, canonical_name, row_ids, snapshot, true)
    }

    fn delete_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
        let total_rows = table.total_rows.load(Ordering::SeqCst);
        if total_rows == 0 {
            return Ok(RuntimeStatementResult::Delete {
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
        self.apply_delete(table, display_name, canonical_name, row_ids, snapshot, true)
    }

    fn apply_delete(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        canonical_name: String,
        row_ids: Vec<RowId>,
        snapshot: TransactionSnapshot,
        enforce_foreign_keys: bool,
    ) -> Result<RuntimeStatementResult<P>> {
        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        if enforce_foreign_keys {
            self.check_foreign_keys_on_delete(
                table,
                &display_name,
                &canonical_name,
                &row_ids,
                snapshot,
            )?;
        }

        self.detect_delete_conflicts(table, &display_name, &row_ids, snapshot)?;

        let removed = row_ids.len();

        // Build DELETE batch using helper
        let batch = mvcc_columns::build_delete_batch(row_ids.clone(), snapshot.txn_id)?;
        table.table.append(&batch)?;

        let removed_u64 = u64::try_from(removed)
            .map_err(|_| Error::InvalidArgumentError("row count exceeds supported range".into()))?;
        table.total_rows.fetch_sub(removed_u64, Ordering::SeqCst);

        Ok(RuntimeStatementResult::Delete {
            table_name: display_name,
            rows_deleted: removed,
        })
    }

    fn check_foreign_keys_on_delete(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        _canonical_name: &str,
        row_ids: &[RowId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() {
            return Ok(());
        }

        self.constraint_service.validate_delete_foreign_keys(
            table.table.table_id(),
            row_ids,
            |request| {
                self.collect_row_values_for_ids(
                    table,
                    request.referenced_row_ids,
                    request.referenced_field_ids,
                )
            },
            |request| {
                let child_table = self.lookup_table(request.referencing_table_canonical)?;
                self.collect_visible_child_rows(
                    child_table.as_ref(),
                    request.referencing_field_ids,
                    snapshot,
                )
            },
        )
    }

    fn check_foreign_keys_on_insert(
        &self,
        table: &ExecutorTable<P>,
        _display_name: &str,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        let schema_field_ids: Vec<FieldId> = table
            .schema
            .columns
            .iter()
            .map(|column| column.field_id)
            .collect();

        self.constraint_service.validate_insert_foreign_keys(
            table.table.table_id(),
            &schema_field_ids,
            column_order,
            rows,
            |request| {
                let parent_table = self.lookup_table(request.referenced_table_canonical)?;
                self.scan_multi_column_values(
                    parent_table.as_ref(),
                    request.referenced_field_ids,
                    snapshot,
                )
            },
        )
    }

    fn detect_delete_conflicts(
        &self,
        table: &ExecutorTable<P>,
        display_name: &str,
        row_ids: &[RowId],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        if row_ids.is_empty() {
            return Ok(());
        }

        let table_id = table.table.table_id();
        let deleted_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);
        let logical_fields: Arc<[LogicalFieldId]> = Arc::from([deleted_lfid]);

        if let Err(err) = table
            .table
            .store()
            .prepare_gather_context(logical_fields.as_ref())
        {
            match err {
                Error::NotFound => return Ok(()),
                other => return Err(other),
            }
        }

        let mut stream = table.table.stream_columns(
            Arc::clone(&logical_fields),
            row_ids.to_vec(),
            GatherNullPolicy::IncludeNulls,
        )?;

        while let Some(chunk) = stream.next_batch()? {
            let batch = chunk.batch();
            let window = chunk.row_ids();
            let deleted_column = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::Internal(
                        "failed to read MVCC deleted_by column for conflict detection".into(),
                    )
                })?;

            for (idx, row_id) in window.iter().enumerate() {
                let deleted_by = if deleted_column.is_null(idx) {
                    TXN_ID_NONE
                } else {
                    deleted_column.value(idx)
                };

                if deleted_by == TXN_ID_NONE || deleted_by == snapshot.txn_id {
                    continue;
                }

                let status = self.txn_manager.status(deleted_by);
                if !status.is_active() {
                    continue;
                }

                tracing::debug!(
                    "[MVCC] delete conflict: table='{}' row_id={} deleted_by={} status={:?} current_txn={}",
                    display_name,
                    row_id,
                    deleted_by,
                    status,
                    snapshot.txn_id
                );

                return Err(Error::TransactionContextError(format!(
                    "transaction conflict on table '{}' for row {}: row locked by transaction {} ({:?})",
                    display_name, row_id, deleted_by, status
                )));
            }
        }

        Ok(())
    }

    fn collect_update_rows(
        &self,
        table: &ExecutorTable<P>,
        filter_expr: &LlkvExpr<'static, FieldId>,
        expressions: &[ScalarExpr<FieldId>],
        snapshot: TransactionSnapshot,
    ) -> Result<(Vec<RowId>, Vec<Vec<PlanValue>>)> {
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

    pub fn lookup_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        // Fast path: check if table is already loaded
        {
            let tables = self.tables.read().unwrap();
            if let Some(table) = tables.get(canonical_name) {
                // Check if table has been dropped
                if self.dropped_tables.read().unwrap().contains(canonical_name) {
                    // Table was dropped - treat as not found
                    return Err(Error::NotFound);
                }
                tracing::trace!(
                    "=== LOOKUP_TABLE '{}' (cached) table_id={} columns={} context_pager={:p} ===",
                    canonical_name,
                    table.table.table_id(),
                    table.schema.columns.len(),
                    &*self.pager
                );
                return Ok(Arc::clone(table));
            }
        } // Release read lock

        // Slow path: load table from catalog (happens once per table)
        tracing::debug!(
            "[LAZY_LOAD] Loading table '{}' from catalog",
            canonical_name
        );

        // Check catalog first for table existence
        let catalog_table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let table_id = catalog_table_id;
        let table = Table::new_with_store(table_id, Arc::clone(&self.store))?;
        let store = table.store();
        let mut logical_fields = store.user_field_ids_for_table(table_id);
        logical_fields.sort_by_key(|lfid| lfid.field_id());
        let field_ids: Vec<FieldId> = logical_fields.iter().map(|lfid| lfid.field_id()).collect();
        let table_view = self
            .metadata
            .table_view(&self.catalog, table_id, &field_ids)?;
        let table_meta = table_view.table_meta.clone().ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;
        let column_metas = table_view.column_metas;
        let constraint_records = table_view.constraint_records;
        let multi_column_uniques = table_view.multi_column_uniques;
        let catalog_field_resolver = self.catalog.field_resolver(catalog_table_id);
        let mut metadata_primary_keys: FxHashSet<FieldId> = FxHashSet::default();
        let mut metadata_unique_fields: FxHashSet<FieldId> = FxHashSet::default();
        let mut has_primary_key_records = false;
        let mut has_single_unique_records = false;

        for record in constraint_records
            .iter()
            .filter(|record| record.is_active())
        {
            match &record.kind {
                ConstraintKind::PrimaryKey(pk) => {
                    has_primary_key_records = true;
                    for field_id in &pk.field_ids {
                        metadata_primary_keys.insert(*field_id);
                        metadata_unique_fields.insert(*field_id);
                    }
                }
                ConstraintKind::Unique(unique) => {
                    if unique.field_ids.len() == 1 {
                        has_single_unique_records = true;
                        metadata_unique_fields.insert(unique.field_ids[0]);
                    }
                }
                _ => {}
            }
        }

        // Build ExecutorSchema from metadata manager snapshots
        let mut executor_columns = Vec::new();
        let mut lookup = FxHashMap::with_capacity_and_hasher(field_ids.len(), Default::default());

        for (idx, lfid) in logical_fields.iter().enumerate() {
            let field_id = lfid.field_id();
            let normalized_index = executor_columns.len();

            let column_name = column_metas
                .get(idx)
                .and_then(|meta| meta.as_ref())
                .and_then(|meta| meta.name.clone())
                .unwrap_or_else(|| format!("col_{}", field_id));

            let normalized = column_name.to_ascii_lowercase();
            lookup.insert(normalized, normalized_index);

            let fallback_constraints: FieldConstraints = catalog_field_resolver
                .as_ref()
                .and_then(|resolver| resolver.field_constraints_by_name(&column_name))
                .unwrap_or_default();

            let metadata_primary = metadata_primary_keys.contains(&field_id);
            let primary_key = if has_primary_key_records {
                metadata_primary
            } else {
                fallback_constraints.primary_key
            };

            let metadata_unique = metadata_primary || metadata_unique_fields.contains(&field_id);
            let unique = if has_primary_key_records || has_single_unique_records {
                metadata_unique
            } else {
                fallback_constraints.primary_key || fallback_constraints.unique
            };

            let data_type = store.data_type(*lfid)?;
            let nullable = !primary_key;

            executor_columns.push(ExecutorColumn {
                name: column_name,
                data_type,
                nullable,
                primary_key,
                unique,
                field_id,
                check_expr: fallback_constraints.check_expr.clone(),
            });
        }

        let exec_schema = Arc::new(ExecutorSchema {
            columns: executor_columns,
            lookup,
        });

        // Find the maximum row_id in the table to set next_row_id correctly
        let max_row_id = {
            use arrow::array::UInt64Array;
            use llkv_column_map::store::rowid_fid;
            use llkv_column_map::store::scan::{
                PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
                PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
            };

            struct MaxRowIdVisitor {
                max: RowId,
            }

            impl PrimitiveVisitor for MaxRowIdVisitor {
                fn u64_chunk(&mut self, values: &UInt64Array) {
                    for i in 0..values.len() {
                        let val = values.value(i);
                        if val > self.max {
                            self.max = val;
                        }
                    }
                }
            }

            impl PrimitiveWithRowIdsVisitor for MaxRowIdVisitor {}
            impl PrimitiveSortedVisitor for MaxRowIdVisitor {}
            impl PrimitiveSortedWithRowIdsVisitor for MaxRowIdVisitor {}

            // Scan the row_id column for any user field in this table
            let row_id_field = rowid_fid(LogicalFieldId::for_user(table_meta.table_id, 1));
            let mut visitor = MaxRowIdVisitor { max: 0 };

            match ScanBuilder::new(table.store(), row_id_field)
                .options(ScanOptions::default())
                .run(&mut visitor)
            {
                Ok(_) => visitor.max,
                Err(llkv_result::Error::NotFound) => 0,
                Err(e) => {
                    tracing::warn!(
                        "[LAZY_LOAD] Failed to scan max row_id for table '{}': {}",
                        canonical_name,
                        e
                    );
                    0
                }
            }
        };

        let next_row_id = if max_row_id > 0 {
            max_row_id.saturating_add(1)
        } else {
            0
        };

        // Get the actual persisted row count from table metadata
        // This is an O(1) catalog lookup that reads ColumnDescriptor.total_row_count
        // Fallback to 0 for truly empty tables
        let total_rows = table.total_rows().unwrap_or(0);

        let executor_table = Arc::new(ExecutorTable {
            table: Arc::new(table),
            schema: exec_schema,
            next_row_id: AtomicU64::new(next_row_id),
            total_rows: AtomicU64::new(total_rows),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

        if !multi_column_uniques.is_empty() {
            let executor_uniques =
                Self::build_executor_multi_column_uniques(&executor_table, &multi_column_uniques);
            executor_table.set_multi_column_uniques(executor_uniques);
        }

        // Cache the loaded table
        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(canonical_name.to_string(), Arc::clone(&executor_table));
        }

        // Register fields in catalog (may already be registered from RuntimeContext::new())
        if let Some(field_resolver) = self.catalog.field_resolver(catalog_table_id) {
            for col in &executor_table.schema.columns {
                let definition = FieldDefinition::new(&col.name)
                    .with_primary_key(col.primary_key)
                    .with_unique(col.unique)
                    .with_check_expr(col.check_expr.clone());
                let _ = field_resolver.register_field(definition); // Ignore "already exists" errors
            }
            tracing::debug!(
                "[CATALOG] Registered {} field(s) for lazy-loaded table '{}'",
                executor_table.schema.columns.len(),
                canonical_name
            );
        }

        tracing::debug!(
            "[LAZY_LOAD] Loaded table '{}' (id={}) with {} columns, next_row_id={}",
            canonical_name,
            table_meta.table_id,
            field_ids.len(),
            next_row_id
        );

        Ok(executor_table)
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
        let (table_id, column_field_ids) = {
            let tables = self.tables.read().unwrap();
            let Some(entry) = tables.get(&canonical_name) else {
                if if_exists {
                    return Ok(());
                } else {
                    return Err(Error::CatalogError(format!(
                        "Catalog Error: Table '{}' does not exist",
                        display_name
                    )));
                }
            };

            let field_ids = entry
                .schema
                .columns
                .iter()
                .map(|col| col.field_id)
                .collect::<Vec<_>>();
            (entry.table.table_id(), field_ids)
        };

        let referencing = self.constraint_service.referencing_foreign_keys(table_id)?;

        for detail in referencing {
            if detail.referencing_table_canonical == canonical_name {
                continue;
            }

            if self.is_table_marked_dropped(&detail.referencing_table_canonical) {
                continue;
            }

            let constraint_label = detail.constraint_name.as_deref().unwrap_or("FOREIGN KEY");
            return Err(Error::ConstraintError(format!(
                "Cannot drop table '{}' because it is referenced by foreign key constraint '{}' on table '{}'",
                display_name, constraint_label, detail.referencing_table_display
            )));
        }

        self.catalog_service
            .drop_table(&canonical_name, table_id, &column_field_ids)?;
        tracing::debug!(
            "[CATALOG] Unregistered table '{}' (table_id={}) from catalog",
            canonical_name,
            table_id
        );

        self.dropped_tables
            .write()
            .unwrap()
            .insert(canonical_name.clone());
        Ok(())
    }

    pub fn is_table_marked_dropped(&self, canonical_name: &str) -> bool {
        self.dropped_tables.read().unwrap().contains(canonical_name)
    }
}

// Implement TransactionContext for ContextWrapper to enable llkv-transaction integration
impl<P> TransactionContext for RuntimeContextWrapper<P>
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
        RuntimeContext::table_column_specs(self.context(), table_name)
    }

    fn export_table_rows(
        &self,
        table_name: &str,
    ) -> llkv_result::Result<llkv_transaction::RowBatch> {
        let batch = RuntimeContext::export_table_rows(self.context(), table_name)?;
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
        RuntimeContext::get_batches_with_row_ids_with_snapshot(
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
        RuntimeContext::execute_select_with_snapshot(self.context(), plan, self.snapshot())
    }

    fn create_table_plan(
        &self,
        plan: CreateTablePlan,
    ) -> llkv_result::Result<TransactionResult<P>> {
        let result = RuntimeContext::create_table_plan(self.context(), plan)?;
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
            RuntimeContext::insert_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn update(&self, plan: UpdatePlan) -> llkv_result::Result<TransactionResult<P>> {
        let snapshot = self.current_snapshot();
        let result = if snapshot.txn_id == TXN_ID_AUTO_COMMIT {
            self.ctx().update(plan)?
        } else {
            RuntimeContext::update_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn delete(&self, plan: DeletePlan) -> llkv_result::Result<TransactionResult<P>> {
        let snapshot = self.current_snapshot();
        let result = if snapshot.txn_id == TXN_ID_AUTO_COMMIT {
            self.ctx().delete(plan)?
        } else {
            RuntimeContext::delete_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn create_index(&self, plan: CreateIndexPlan) -> llkv_result::Result<TransactionResult<P>> {
        let result = RuntimeContext::create_index(self.context(), plan)?;
        Ok(convert_statement_result(result))
    }

    fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> llkv_result::Result<usize> {
        RuntimeContext::append_batches_with_row_ids(self.context(), table_name, batches)
    }

    fn table_names(&self) -> Vec<String> {
        RuntimeContext::table_names(self.context())
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

    fn catalog_snapshot(&self) -> llkv_table::catalog::TableCatalogSnapshot {
        let ctx = self.context();
        ctx.catalog.snapshot()
    }

    fn validate_commit_constraints(&self, txn_id: TxnId) -> llkv_result::Result<()> {
        self.ctx.validate_primary_keys_for_commit(txn_id)
    }

    fn clear_transaction_state(&self, txn_id: TxnId) {
        self.ctx.clear_transaction_state(txn_id);
    }
}

// Helper to convert StatementResult between types
fn convert_statement_result<P>(result: RuntimeStatementResult<P>) -> TransactionResult<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    use llkv_transaction::TransactionResult as TxResult;
    match result {
        RuntimeStatementResult::CreateTable { table_name } => TxResult::CreateTable { table_name },
        RuntimeStatementResult::CreateIndex {
            table_name,
            index_name,
        } => TxResult::CreateIndex {
            table_name,
            index_name,
        },
        RuntimeStatementResult::Insert { rows_inserted, .. } => TxResult::Insert { rows_inserted },
        RuntimeStatementResult::Update { rows_updated, .. } => TxResult::Update {
            rows_matched: rows_updated,
            rows_updated,
        },
        RuntimeStatementResult::Delete { rows_deleted, .. } => TxResult::Delete { rows_deleted },
        RuntimeStatementResult::Transaction { kind } => TxResult::Transaction { kind },
        _ => panic!("unsupported StatementResult conversion"),
    }
}

fn filter_row_ids_for_snapshot<P>(
    table: &Table<P>,
    row_ids: Vec<RowId>,
    txn_manager: &TxnIdManager,
    snapshot: TransactionSnapshot,
) -> Result<Vec<RowId>>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    tracing::debug!(
        "[FILTER_ROWS] Filtering {} row IDs for snapshot txn_id={}, snapshot_id={}",
        row_ids.len(),
        snapshot.txn_id,
        snapshot.snapshot_id
    );

    if row_ids.is_empty() {
        return Ok(row_ids);
    }

    let table_id = table.table_id();
    let created_lfid = LogicalFieldId::for_mvcc_created_by(table_id);
    let deleted_lfid = LogicalFieldId::for_mvcc_deleted_by(table_id);
    let logical_fields: Arc<[LogicalFieldId]> = Arc::from([created_lfid, deleted_lfid]);

    if let Err(err) = table
        .store()
        .prepare_gather_context(logical_fields.as_ref())
    {
        match err {
            Error::NotFound => {
                tracing::trace!(
                    "[FILTER_ROWS] MVCC columns not found for table_id={}, treating all rows as visible",
                    table_id
                );
                return Ok(row_ids);
            }
            other => {
                tracing::error!(
                    "[FILTER_ROWS] Failed to prepare gather context: {:?}",
                    other
                );
                return Err(other);
            }
        }
    }

    let total_rows = row_ids.len();
    let mut stream = match table.stream_columns(
        Arc::clone(&logical_fields),
        row_ids,
        GatherNullPolicy::IncludeNulls,
    ) {
        Ok(stream) => stream,
        Err(err) => {
            tracing::error!("[FILTER_ROWS] stream_columns error: {:?}", err);
            return Err(err);
        }
    };

    let mut visible = Vec::with_capacity(total_rows);

    while let Some(chunk) = stream.next_batch()? {
        let batch = chunk.batch();
        let window = chunk.row_ids();

        if batch.num_columns() < 2 {
            tracing::debug!(
                "[FILTER_ROWS] version_batch has < 2 columns for table_id={}, returning window rows unfiltered",
                table_id
            );
            visible.extend_from_slice(window);
            continue;
        }

        let created_column = batch.column(0).as_any().downcast_ref::<UInt64Array>();
        let deleted_column = batch.column(1).as_any().downcast_ref::<UInt64Array>();

        if created_column.is_none() || deleted_column.is_none() {
            tracing::debug!(
                "[FILTER_ROWS] Failed to downcast MVCC columns for table_id={}, returning window rows unfiltered",
                table_id
            );
            visible.extend_from_slice(window);
            continue;
        }

        let created_column = created_column.unwrap();
        let deleted_column = deleted_column.unwrap();

        for (idx, row_id) in window.iter().enumerate() {
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
    }

    tracing::debug!(
        "[FILTER_ROWS] Filtered from {} to {} visible rows",
        total_rows,
        visible.len()
    );
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
    fn filter(&self, table: &Table<P>, row_ids: Vec<RowId>) -> Result<Vec<RowId>> {
        tracing::trace!(
            "[MVCC_FILTER] filter() called with row_ids {:?}, snapshot txn={}, snapshot_id={}",
            row_ids,
            self.snapshot.txn_id,
            self.snapshot.snapshot_id
        );
        let result = filter_row_ids_for_snapshot(table, row_ids, &self.txn_manager, self.snapshot);
        if let Ok(ref visible) = result {
            tracing::trace!(
                "[MVCC_FILTER] filter() returning visible row_ids: {:?}",
                visible
            );
        }
        result
    }
}

// Wrapper to implement TableProvider for Context
struct ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<RuntimeContext<P>>,
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
pub struct RuntimeLazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<RuntimeContext<P>>,
    plan: SelectPlan,
}

impl<P> RuntimeLazyFrame<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn scan(context: Arc<RuntimeContext<P>>, table: &str) -> Result<Self> {
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
        let index = schema.lookup.get(&normalized).ok_or_else(|| {
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{}'",
                column
            ))
        })?;
        resolved.push(*index);
    }
    Ok(resolved)
}

fn normalize_insert_value_for_column(
    column: &ExecutorColumn,
    value: PlanValue,
) -> Result<PlanValue> {
    match (&column.data_type, value) {
        (_, PlanValue::Null) => Ok(PlanValue::Null),
        (DataType::Int64, PlanValue::Integer(v)) => Ok(PlanValue::Integer(v)),
        (DataType::Int64, PlanValue::Float(v)) => Ok(PlanValue::Integer(v as i64)),
        (DataType::Int64, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into INT column '{}'",
            column.name
        ))),
        (DataType::Boolean, PlanValue::Integer(v)) => {
            Ok(PlanValue::Integer(if v != 0 { 1 } else { 0 }))
        }
        (DataType::Boolean, PlanValue::Float(v)) => {
            Ok(PlanValue::Integer(if v != 0.0 { 1 } else { 0 }))
        }
        (DataType::Boolean, PlanValue::String(s)) => {
            let normalized = s.trim().to_ascii_lowercase();
            let value = match normalized.as_str() {
                "true" | "t" | "1" => 1,
                "false" | "f" | "0" => 0,
                _ => {
                    return Err(Error::InvalidArgumentError(format!(
                        "cannot insert string '{}' into BOOLEAN column '{}'",
                        s, column.name
                    )));
                }
            };
            Ok(PlanValue::Integer(value))
        }
        (DataType::Boolean, PlanValue::Struct(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert struct into BOOLEAN column '{}'",
            column.name
        ))),
        (DataType::Float64, PlanValue::Integer(v)) => Ok(PlanValue::Float(v as f64)),
        (DataType::Float64, PlanValue::Float(v)) => Ok(PlanValue::Float(v)),
        (DataType::Float64, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DOUBLE column '{}'",
            column.name
        ))),
        (DataType::Utf8, PlanValue::Integer(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::Float(v)) => Ok(PlanValue::String(v.to_string())),
        (DataType::Utf8, PlanValue::String(s)) => Ok(PlanValue::String(s)),
        (DataType::Utf8, PlanValue::Struct(_)) => Err(Error::InvalidArgumentError(format!(
            "cannot insert struct into STRING column '{}'",
            column.name
        ))),
        (DataType::Date32, PlanValue::Integer(days)) => {
            let casted = i32::try_from(days).map_err(|_| {
                Error::InvalidArgumentError(format!(
                    "integer literal out of range for DATE column '{}'",
                    column.name
                ))
            })?;
            Ok(PlanValue::Integer(casted as i64))
        }
        (DataType::Date32, PlanValue::String(text)) => {
            let days = parse_date32_literal(&text)?;
            Ok(PlanValue::Integer(days as i64))
        }
        (DataType::Date32, other) => Err(Error::InvalidArgumentError(format!(
            "cannot insert {other:?} into DATE column '{}'",
            column.name
        ))),
        (DataType::Struct(_), PlanValue::Struct(map)) => Ok(PlanValue::Struct(map)),
        (DataType::Struct(_), other) => Err(Error::InvalidArgumentError(format!(
            "expected struct value for struct column '{}', got {other:?}",
            column.name
        ))),
        (other_type, other_value) => Err(Error::InvalidArgumentError(format!(
            "unsupported Arrow data type {:?} for INSERT value {:?} in column '{}'",
            other_type, other_value, column.name
        ))),
    }
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
                    PlanValue::String(_) | PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-integer into INT column".into(),
                        ));
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(values.len());
            for value in values {
                match value {
                    PlanValue::Null => builder.append_null(),
                    PlanValue::Integer(v) => builder.append_value(*v != 0),
                    PlanValue::Float(v) => builder.append_value(*v != 0.0),
                    PlanValue::String(s) => {
                        let normalized = s.trim().to_ascii_lowercase();
                        match normalized.as_str() {
                            "true" | "t" | "1" => builder.append_value(true),
                            "false" | "f" | "0" => builder.append_value(false),
                            _ => {
                                return Err(Error::InvalidArgumentError(format!(
                                    "cannot insert string '{}' into BOOLEAN column",
                                    s
                                )));
                            }
                        }
                    }
                    PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert struct into BOOLEAN column".into(),
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
                    PlanValue::String(_) | PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-numeric into DOUBLE column".into(),
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
                    PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert struct into STRING column".into(),
                        ));
                    }
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
                    PlanValue::Float(_) | PlanValue::Struct(_) => {
                        return Err(Error::InvalidArgumentError(
                            "cannot insert non-date value into DATE column".into(),
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
        DataType::Struct(fields) => {
            use arrow::array::StructArray;
            let mut field_arrays: Vec<(FieldRef, ArrayRef)> = Vec::with_capacity(fields.len());

            for field in fields.iter() {
                let field_name = field.name();
                let field_type = field.data_type();
                let mut field_values = Vec::with_capacity(values.len());

                for value in values {
                    match value {
                        PlanValue::Null => field_values.push(PlanValue::Null),
                        PlanValue::Struct(map) => {
                            let field_value =
                                map.get(field_name).cloned().unwrap_or(PlanValue::Null);
                            field_values.push(field_value);
                        }
                        _ => {
                            return Err(Error::InvalidArgumentError(format!(
                                "expected struct value for struct column, got {:?}",
                                value
                            )));
                        }
                    }
                }

                let field_array = build_array_for_column(field_type, &field_values)?;
                field_arrays.push((Arc::clone(field), field_array));
            }

            Ok(Arc::new(StructArray::from(field_arrays)))
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
            Error::InvalidArgumentError(format!(
                "Binder Error: does not have a column named '{name}'"
            ))
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
        ScalarExpr::GetField { base, field_name } => {
            let base_expr = translate_scalar(base, schema)?;
            Ok(ScalarExpr::GetField {
                base: Box::new(base_expr),
                field_name: field_name.clone(),
            })
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
            PlanValue::Null | PlanValue::String(_) | PlanValue::Struct(_) => Err(
                Error::InvalidArgumentError("cannot negate non-numeric literal".into()),
            ),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => plan_value_from_sql_expr(expr),
        SqlExpr::Nested(inner) => plan_value_from_sql_expr(inner),
        SqlExpr::Dictionary(fields) => {
            let mut map = std::collections::HashMap::new();
            for field in fields {
                let key = field.key.value.clone();
                let value = plan_value_from_sql_expr(&field.value)?;
                map.insert(key, value);
            }
            Ok(PlanValue::Struct(map))
        }
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
pub struct RuntimeRangeSelectRows {
    rows: Vec<Vec<PlanValue>>,
}

impl RuntimeRangeSelectRows {
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
pub struct RuntimeRangeSpec {
    start: i64,
    #[allow(dead_code)] // Used for validation, computed into row_count
    end: i64,
    row_count: usize,
    column_name_lower: String,
    table_alias_lower: Option<String>,
}

impl RuntimeRangeSpec {
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

pub fn extract_rows_from_range(select: &Select) -> Result<Option<RuntimeRangeSelectRows>> {
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

    Ok(Some(RuntimeRangeSelectRows { rows }))
}

fn build_range_projection_expr(expr: &SqlExpr, spec: &RuntimeRangeSpec) -> Result<RangeProjection> {
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

fn parse_range_spec(select: &Select) -> Result<Option<RuntimeRangeSpec>> {
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
) -> Result<Option<RuntimeRangeSpec>> {
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

    Ok(Some(RuntimeRangeSpec {
        start,
        end,
        row_count,
        column_name_lower,
        table_alias_lower,
    }))
}

pub struct RuntimeCreateTableBuilder<'ctx, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    ctx: &'ctx RuntimeContext<P>,
    plan: CreateTablePlan,
}

impl<'ctx, P> RuntimeCreateTableBuilder<'ctx, P>
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

    pub fn finish(self) -> Result<RuntimeStatementResult<P>> {
        self.ctx.execute_create_table(self.plan)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeRow {
    values: Vec<(String, PlanValue)>,
}

impl RuntimeRow {
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

pub fn row() -> RuntimeRow {
    RuntimeRow::new()
}

#[doc(hidden)]
pub enum RuntimeInsertRowKind {
    Named {
        columns: Vec<String>,
        values: Vec<PlanValue>,
    },
    Positional(Vec<PlanValue>),
}

pub trait IntoInsertRow {
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind>;
}

impl IntoInsertRow for RuntimeRow {
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
        let row = self;
        if row.values.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        let columns = row.columns();
        let values = row.values_for_columns(&columns)?;
        Ok(RuntimeInsertRowKind::Named { columns, values })
    }
}

// Remove the generic impl for `&T` which caused unconditional-recursion
// and noop-clone clippy warnings. Callers can pass owned values or use
// the provided tuple/array/Vec implementations.

impl<T> IntoInsertRow for Vec<T>
where
    T: Into<PlanValue>,
{
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
        if self.is_empty() {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(RuntimeInsertRowKind::Positional(
            self.into_iter().map(Into::into).collect(),
        ))
    }
}

impl<T, const N: usize> IntoInsertRow for [T; N]
where
    T: Into<PlanValue>,
{
    fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
        if N == 0 {
            return Err(Error::InvalidArgumentError(
                "insert requires at least one column".into(),
            ));
        }
        Ok(RuntimeInsertRowKind::Positional(
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
            fn into_insert_row(self) -> Result<RuntimeInsertRowKind> {
                let ($($value,)+) = self;
                Ok(RuntimeInsertRowKind::Positional(vec![$($value.into(),)+]))
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

pub struct RuntimeTableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    context: Arc<RuntimeContext<P>>,
    display_name: String,
    _canonical_name: String,
}

impl<P> RuntimeTableHandle<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub fn new(context: Arc<RuntimeContext<P>>, name: &str) -> Result<Self> {
        let (display_name, canonical_name) = canonical_table_name(name)?;
        context.lookup_table(&canonical_name)?;
        Ok(Self {
            context,
            display_name,
            _canonical_name: canonical_name,
        })
    }

    pub fn lazy(&self) -> Result<RuntimeLazyFrame<P>> {
        RuntimeLazyFrame::scan(Arc::clone(&self.context), &self.display_name)
    }

    pub fn insert_rows<R>(
        &self,
        rows: impl IntoIterator<Item = R>,
    ) -> Result<RuntimeStatementResult<P>>
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
                RuntimeInsertRowKind::Named { columns, values } => {
                    if let Some(existing) = &mode {
                        if !matches!(existing, InsertMode::Named) {
                            return Err(Error::InvalidArgumentError(
                                "cannot mix positional and named insert rows".into(),
                            ));
                        }
                    } else {
                        mode = Some(InsertMode::Named);
                        let mut seen =
                            FxHashSet::with_capacity_and_hasher(columns.len(), Default::default());
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
                RuntimeInsertRowKind::Positional(values) => {
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

    pub fn insert_row_batch(&self, batch: RowBatch) -> Result<RuntimeStatementResult<P>> {
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

    pub fn insert_batches(&self, batches: Vec<RecordBatch>) -> Result<RuntimeStatementResult<P>> {
        let plan = InsertPlan {
            table: self.display_name.clone(),
            columns: Vec::new(),
            source: InsertSource::Batches(batches),
        };
        self.context.insert(plan)
    }

    pub fn insert_lazy(&self, frame: RuntimeLazyFrame<P>) -> Result<RuntimeStatementResult<P>> {
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
        let context = Arc::new(RuntimeContext::new(pager));

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
        let context = Arc::new(RuntimeContext::new(pager));

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
