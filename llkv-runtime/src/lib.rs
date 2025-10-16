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
    Array, ArrayRef, Date32Builder, Float64Builder, Int64Builder, StringBuilder, UInt64Array,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ColumnStore;
use llkv_column_map::store::{
    CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, GatherNullPolicy, IndexKind, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_expr::expr::{Expr as LlkvExpr, Filter, Operator, ScalarExpr};
// Literal is not used at top-level; keep it out to avoid unused import warnings.
use llkv_result::Error;
use llkv_storage::pager::{BoxedPager, MemPager, Pager};
use llkv_table::catalog::{FieldConstraints, FieldDefinition, TableCatalog};
use llkv_table::table::{RowIdFilter, ScanProjection, ScanStreamOptions, Table};
use llkv_table::types::{FieldId, ROW_ID_FIELD_ID, RowId, TableId};
use llkv_table::{CATALOG_TABLE_ID, ColMeta, MultiColumnUniqueEntryMeta, SysCatalog, TableMeta};
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
    ColumnSpec, CreateIndexPlan, CreateTablePlan, CreateTableSource, DeletePlan, IndexColumnPlan,
    InsertPlan, InsertSource, IntoColumnSpec, NotNull, Nullable, OrderByPlan, OrderSortType,
    OrderTarget, PlanOperation, PlanStatement, PlanValue, SelectPlan, SelectProjection, UpdatePlan,
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
///
/// These functions consolidate the repeated logic for adding MVCC metadata columns
/// (row_id, created_by, deleted_by) to RecordBatches.
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

#[derive(Hash, Eq, PartialEq, Debug)]
enum UniqueKey {
    Int(i64),
    Float(u64),
    Str(String),
    Composite(Vec<UniqueKey>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StoredMultiColumnUnique {
    index_name: Option<String>,
    field_ids: Vec<FieldId>,
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
        if let Some(namespace_id) = self.namespace_for_select_plan(&plan) {
            if namespace_id == storage_namespace::TEMPORARY_NAMESPACE_ID {
                return self.select_from_temporary(plan);
            }
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
    multi_column_uniques: RwLock<FxHashMap<String, Vec<StoredMultiColumnUnique>>>,
    // Centralized catalog for table/field name resolution
    catalog: Arc<TableCatalog>,
    // Transaction manager for session-based transactions
    transaction_manager:
        TransactionManager<RuntimeContextWrapper<P>, RuntimeContextWrapper<MemPager>>,
    txn_manager: Arc<TxnIdManager>,
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

        // Load transaction state and table registry from catalog if it exists
        let (next_txn_id, last_committed, loaded_tables, persisted_unique_metas) =
            match ColumnStore::open(Arc::clone(&pager)) {
                Ok(store) => {
                    let catalog = SysCatalog::new(&store);
                    let next_txn_id = match catalog.get_next_txn_id() {
                        Ok(Some(id)) => {
                            tracing::debug!("[CONTEXT] Loaded next_txn_id={} from catalog", id);
                            id
                        }
                        Ok(None) => {
                            tracing::debug!(
                                "[CONTEXT] No persisted next_txn_id found, starting from default"
                            );
                            TXN_ID_AUTO_COMMIT + 1
                        }
                        Err(e) => {
                            tracing::warn!(
                                "[CONTEXT] Failed to load next_txn_id: {}, using default",
                                e
                            );
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

                    // Load table registry from catalog
                    let loaded_tables = match catalog.all_table_metas() {
                        Ok(metas) => {
                            tracing::debug!(
                                "[CONTEXT] Loaded {} table(s) from catalog",
                                metas.len()
                            );
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

                    let persisted_unique_metas = match catalog.all_multi_column_unique_metas() {
                        Ok(entries) => entries,
                        Err(e) => {
                            tracing::warn!(
                                "[CONTEXT] Failed to load multi-column unique metas: {}, starting with empty set",
                                e
                            );
                            Vec::new()
                        }
                    };

                    (
                        next_txn_id,
                        last_committed,
                        loaded_tables,
                        persisted_unique_metas,
                    )
                }
                Err(e) => {
                    tracing::warn!(
                        "[CONTEXT] Failed to open ColumnStore: {}, using default state",
                        e
                    );
                    (
                        TXN_ID_AUTO_COMMIT + 1,
                        TXN_ID_AUTO_COMMIT,
                        Vec::new(),
                        Vec::new(),
                    )
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

        let mut canonical_by_id: FxHashMap<TableId, String> = FxHashMap::default();
        for (table_id, table_meta) in &loaded_tables {
            if let Some(ref table_name) = table_meta.name {
                canonical_by_id.insert(*table_id, table_name.to_ascii_lowercase());
            }
        }

        let mut persisted_multi: FxHashMap<String, Vec<StoredMultiColumnUnique>> =
            FxHashMap::default();
        for meta in persisted_unique_metas {
            if meta.uniques.is_empty() {
                continue;
            }
            let Some(canonical_name) = canonical_by_id.get(&meta.table_id) else {
                tracing::debug!(
                    "[CONTEXT] Skipping persisted multi-column UNIQUE metadata for unknown table_id={}",
                    meta.table_id
                );
                continue;
            };

            let mut stored_entries = Vec::with_capacity(meta.uniques.len());
            for entry in meta.uniques {
                if entry.column_ids.is_empty() {
                    continue;
                }
                stored_entries.push(StoredMultiColumnUnique {
                    index_name: entry.index_name,
                    field_ids: entry.column_ids,
                });
            }

            if !stored_entries.is_empty() {
                persisted_multi.insert(canonical_name.clone(), stored_entries);
            }
        }

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

        Self {
            pager,
            tables: RwLock::new(FxHashMap::default()), // Start with empty table cache
            dropped_tables: RwLock::new(FxHashSet::default()),
            multi_column_uniques: RwLock::new(persisted_multi),
            catalog,
            transaction_manager,
            txn_manager,
        }
    }

    /// Return the transaction ID manager shared with sessions.
    pub fn txn_manager(&self) -> Arc<TxnIdManager> {
        Arc::clone(&self.txn_manager)
    }

    /// Persist the next_txn_id to the catalog.
    pub fn persist_next_txn_id(&self, next_txn_id: TxnId) -> Result<()> {
        let store = ColumnStore::open(Arc::clone(&self.pager))?;
        let catalog = SysCatalog::new(&store);
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

    fn persist_multi_column_uniques(
        &self,
        table_id: TableId,
        entries: &[StoredMultiColumnUnique],
    ) -> Result<()> {
        let store = ColumnStore::open(Arc::clone(&self.pager))?;
        let catalog = SysCatalog::new(&store);
        let metas: Vec<MultiColumnUniqueEntryMeta> = entries
            .iter()
            .map(|entry| MultiColumnUniqueEntryMeta {
                index_name: entry.index_name.clone(),
                column_ids: entry.field_ids.clone(),
            })
            .collect();
        catalog.put_multi_column_uniques(table_id, &metas)?;
        Ok(())
    }

    fn build_executor_multi_column_uniques(
        table: &ExecutorTable<P>,
        stored: &[StoredMultiColumnUnique],
    ) -> Vec<ExecutorMultiColumnUnique> {
        let mut results = Vec::with_capacity(stored.len());

        'outer: for entry in stored {
            if entry.field_ids.is_empty() {
                continue;
            }

            let mut column_indices = Vec::with_capacity(entry.field_ids.len());
            for field_id in &entry.field_ids {
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
            self.multi_column_uniques
                .write()
                .unwrap()
                .remove(&canonical_name);
        }

        if exists {
            if plan.or_replace {
                tracing::trace!(
                    "DEBUG create_table_plan: table '{}' exists and or_replace=true, removing existing table before recreation",
                    display_name
                );
                self.remove_table_entry(&canonical_name);
                self.multi_column_uniques
                    .write()
                    .unwrap()
                    .remove(&canonical_name);
            } else if plan.if_not_exists {
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

        match plan.source {
            Some(CreateTableSource::Batches { schema, batches }) => self.create_table_from_batches(
                display_name,
                canonical_name,
                schema,
                batches,
                plan.if_not_exists,
            ),
            Some(CreateTableSource::Select { .. }) => Err(Error::Internal(
                "CreateTableSource::Select should be materialized before reaching RuntimeContext::create_table_plan"
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
            let logical_field_id = LogicalFieldId::for_user(table_id, field_id);
            let store = ColumnStore::open(Arc::clone(&self.pager))?;
            let existing_indexes = match store.list_persisted_indexes(logical_field_id) {
                Ok(kinds) => kinds,
                Err(Error::NotFound) => Vec::new(),
                Err(err) => return Err(err),
            };

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
                self.ensure_existing_rows_unique(table.as_ref(), field_id, &column_name)?;
                if let Some(table_id) = self.catalog.table_id(&canonical_name)
                    && let Some(resolver) = self.catalog.field_resolver(table_id)
                {
                    resolver.set_field_unique(&column_name, true)?;
                }
            }

            store.register_index(logical_field_id, IndexKind::Sort)?;

            if let Some(updated_table) = Self::rebuild_executor_table_with_unique(
                table.as_ref(),
                field_id,
            ) {
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

        if let Some(existing) = self
            .multi_column_uniques
            .read()
            .unwrap()
            .get(&canonical_name)
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|entry| entry.field_ids == field_ids)
                    .cloned()
            })
        {
            if plan.if_not_exists {
                drop(table);
                return Ok(RuntimeStatementResult::CreateIndex {
                    table_name: display_name,
                    index_name: existing.index_name.clone(),
                });
            }

            return Err(Error::CatalogError(format!(
                "Index already exists on columns '{}'",
                column_names.join(", ")
            )));
        }

        self.ensure_existing_rows_unique_multi(table.as_ref(), &field_ids, &column_names)?;

        let executor_entry = ExecutorMultiColumnUnique {
            index_name: index_name.clone(),
            column_indices: column_indices.clone(),
        };

        let mut entries_to_persist = {
            let guard = self.multi_column_uniques.read().unwrap();
            guard.get(&canonical_name).cloned().unwrap_or_default()
        };

        if entries_to_persist
            .iter()
            .any(|existing| existing.field_ids == field_ids)
        {
            // Race condition guard: another thread inserted the same index after the initial check.
            if plan.if_not_exists {
                drop(table);
                return Ok(RuntimeStatementResult::CreateIndex {
                    table_name: display_name,
                    index_name: index_name.clone(),
                });
            }
            return Err(Error::CatalogError(format!(
                "Index already exists on columns '{}'",
                column_names.join(", ")
            )));
        }

        let stored_entry = StoredMultiColumnUnique {
            index_name: index_name.clone(),
            field_ids: field_ids.clone(),
        };
        entries_to_persist.push(stored_entry.clone());

        self.persist_multi_column_uniques(table.table.table_id(), &entries_to_persist)?;

        {
            let mut guard = self.multi_column_uniques.write().unwrap();
            guard.insert(canonical_name.clone(), entries_to_persist);
        }

        table.add_multi_column_unique(executor_entry);

        Ok(RuntimeStatementResult::CreateIndex {
            table_name: display_name,
            index_name,
        })
    }

    pub fn table_names(self: &Arc<Self>) -> Vec<String> {
        // Use catalog for table names (single source of truth)
        self.catalog.table_names()
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
        let result = self.update_with_snapshot(plan, snapshot)?;
        self.txn_manager.mark_committed(snapshot.txn_id);
        Ok(result)
    }

    pub fn update_with_snapshot(
        &self,
        plan: UpdatePlan,
        snapshot: TransactionSnapshot,
    ) -> Result<RuntimeStatementResult<P>> {
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

    pub fn delete(&self, plan: DeletePlan) -> Result<RuntimeStatementResult<P>> {
        let snapshot = self.txn_manager.begin_transaction();
        let result = self.delete_with_snapshot(plan, snapshot)?;
        self.txn_manager.mark_committed(snapshot.txn_id);
        Ok(result)
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
                filter,
                snapshot,
                snapshot.txn_id,
            ),
            None => self.delete_all_rows(table.as_ref(), display_name, snapshot, snapshot.txn_id),
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

        self.multi_column_uniques
            .write()
            .unwrap()
            .remove(&canonical_name);

        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(columns.len());
        let mut lookup = FxHashMap::with_capacity_and_hasher(columns.len(), Default::default());
        for (idx, column) in columns.iter().enumerate() {
            let normalized = column.name.to_ascii_lowercase();
            if lookup.insert(normalized.clone(), idx).is_some() {
                return Err(Error::InvalidArgumentError(format!(
                    "duplicate column name '{}' in table '{}'",
                    column.name, display_name
                )));
            }
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
                field_id: (idx + 1) as FieldId,
                check_expr: column.check_expr.clone(),
            });
            let pushed = column_defs.last().unwrap();
            tracing::trace!(
                "DEBUG create_table_from_columns[{}]: pushed ExecutorColumn name='{}' primary_key={} unique={}",
                idx,
                pushed.name,
                pushed.primary_key,
                pushed.unique
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

        let store = ColumnStore::open(Arc::clone(&self.pager))?;
        for column in &column_defs {
            let logical_field_id = LogicalFieldId::for_user(table_id, column.field_id);
            store.ensure_column_registered(logical_field_id, &column.data_type)?;
        }

        let schema = Arc::new(ExecutorSchema {
            columns: column_defs.clone(), // Clone for catalog registration below
            lookup,
        });
        let table_entry = Arc::new(ExecutorTable {
            table: Arc::new(table),
            schema,
            next_row_id: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
            multi_column_uniques: RwLock::new(Vec::new()),
        });

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
        tables.insert(canonical_name.clone(), table_entry);
        drop(tables); // Release write lock before catalog operations

        // Register table in catalog
        let registered_table_id = self.catalog.register_table(display_name.as_str())?;
        tracing::debug!(
            "[CATALOG] Registered table '{}' with catalog_id={}",
            display_name,
            registered_table_id
        );

        // Register fields in catalog
        if let Some(field_resolver) = self.catalog.field_resolver(registered_table_id) {
            for column in &column_defs {
                let definition = FieldDefinition::new(&column.name)
                    .with_primary_key(column.primary_key)
                    .with_unique(column.unique)
                    .with_check_expr(column.check_expr.clone());
                if let Err(e) = field_resolver.register_field(definition) {
                    tracing::warn!(
                        "[CATALOG] Failed to register field '{}' in table '{}': {}",
                        column.name,
                        display_name,
                        e
                    );
                }
            }
            tracing::debug!(
                "[CATALOG] Registered {} field(s) for table '{}'",
                column_defs.len(),
                display_name
            );
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
        if schema.fields().is_empty() {
            return Err(Error::InvalidArgumentError(
                "CREATE TABLE AS SELECT requires at least one column".into(),
            ));
        }
        self.multi_column_uniques
            .write()
            .unwrap()
            .remove(&canonical_name);
        let mut column_defs: Vec<ExecutorColumn> = Vec::with_capacity(schema.fields().len());
        let mut lookup =
            FxHashMap::with_capacity_and_hasher(schema.fields().len(), Default::default());
        for (idx, field) in schema.fields().iter().enumerate() {
            let data_type = match field.data_type() {
                DataType::Int64
                | DataType::Float64
                | DataType::Utf8
                | DataType::Date32
                | DataType::Struct(_) => field.data_type().clone(),
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
                unique: false,      // CTAS does not preserve UNIQUE constraints
                field_id: (idx + 1) as FieldId,
                check_expr: None, // CTAS does not preserve CHECK constraints
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
        tracing::trace!(
            "=== INSERTING TABLE '{}' INTO TABLES MAP (pager={:p}) ===",
            canonical_name,
            &*self.pager
        );
        for (idx, col) in table_entry.schema.columns.iter().enumerate() {
            tracing::trace!(
                "  inserting column[{}]: name='{}' primary_key={} unique={}",
                idx,
                col.name,
                col.primary_key,
                col.unique
            );
        }
        tables.insert(canonical_name.clone(), table_entry);
        drop(tables); // Release write lock before catalog operations

        // Register table in catalog
        let registered_table_id = self.catalog.register_table(display_name.as_str())?;
        tracing::debug!(
            "[CATALOG] Registered table '{}' (CTAS) with catalog_id={}",
            display_name,
            registered_table_id
        );

        // Register fields in catalog
        if let Some(field_resolver) = self.catalog.field_resolver(registered_table_id) {
            for column in &column_defs {
                let definition = FieldDefinition::new(&column.name)
                    .with_primary_key(column.primary_key)
                    .with_unique(column.unique)
                    .with_check_expr(column.check_expr.clone());
                if let Err(e) = field_resolver.register_field(definition) {
                    tracing::warn!(
                        "[CATALOG] Failed to register field '{}' in table '{}': {}",
                        column.name,
                        display_name,
                        e
                    );
                }
            }
            tracing::debug!(
                "[CATALOG] Registered {} field(s) for table '{}' (CTAS)",
                column_defs.len(),
                display_name
            );
        }

        Ok(RuntimeStatementResult::CreateTable {
            table_name: display_name,
        })
    }

    fn check_check_constraints(
        &self,
        table: &ExecutorTable<P>,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
    ) -> Result<()> {
        use sqlparser::dialect::GenericDialect;
        use sqlparser::parser::Parser;

        // Find columns with CHECK constraints
        let check_columns: Vec<(usize, &ExecutorColumn, &str)> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, col)| {
                col.check_expr
                    .as_ref()
                    .map(|expr| (idx, col, expr.as_str()))
            })
            .collect();

        if check_columns.is_empty() {
            return Ok(());
        }

        let dialect = GenericDialect {};

        // Check each row against all CHECK constraints
        for (row_idx, row) in rows.iter().enumerate() {
            for (col_idx, column, check_expr_str) in &check_columns {
                // Parse the CHECK expression
                let sql = format!("SELECT {}", check_expr_str);
                let ast = Parser::parse_sql(&dialect, &sql).map_err(|e| {
                    Error::InvalidArgumentError(format!(
                        "Failed to parse CHECK expression '{}': {}",
                        check_expr_str, e
                    ))
                })?;

                let stmt = ast.into_iter().next().ok_or_else(|| {
                    Error::InvalidArgumentError(format!(
                        "CHECK expression '{}' resulted in empty AST",
                        check_expr_str
                    ))
                })?;

                let select = match stmt {
                    sqlparser::ast::Statement::Query(q) => q,
                    _ => {
                        return Err(Error::InvalidArgumentError(format!(
                            "CHECK expression '{}' did not parse as SELECT",
                            check_expr_str
                        )));
                    }
                };

                // Extract the expression from SELECT
                let body = match *select.body {
                    sqlparser::ast::SetExpr::Select(s) => s,
                    _ => {
                        return Err(Error::InvalidArgumentError(format!(
                            "CHECK expression '{}' is not a simple SELECT",
                            check_expr_str
                        )));
                    }
                };

                if body.projection.len() != 1 {
                    return Err(Error::InvalidArgumentError(format!(
                        "CHECK expression '{}' must have exactly one projection",
                        check_expr_str
                    )));
                }

                let expr = match &body.projection[0] {
                    sqlparser::ast::SelectItem::UnnamedExpr(e)
                    | sqlparser::ast::SelectItem::ExprWithAlias { expr: e, .. } => e,
                    _ => {
                        return Err(Error::InvalidArgumentError(format!(
                            "CHECK expression '{}' projection is not a simple expression",
                            check_expr_str
                        )));
                    }
                };

                // Evaluate the expression against this row
                let result = Self::evaluate_check_expression(
                    expr,
                    row,
                    column_order,
                    table,
                    *col_idx,
                    row_idx,
                )?;

                if !result {
                    return Err(Error::ConstraintError(format!(
                        "CHECK constraint failed for column '{}': {}",
                        column.name, check_expr_str
                    )));
                }
            }
        }

        Ok(())
    }

    fn check_not_null_constraints(
        &self,
        table: &ExecutorTable<P>,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
    ) -> Result<()> {
        let not_null_columns: Vec<(usize, &ExecutorColumn)> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| !column.nullable)
            .collect();

        if not_null_columns.is_empty() {
            return Ok(());
        }

        for (col_idx, column) in not_null_columns {
            let insert_pos = column_order
                .iter()
                .position(|&dest_idx| dest_idx == col_idx)
                .ok_or_else(|| {
                    Error::ConstraintError(format!(
                        "NOT NULL column '{}' missing from INSERT/UPDATE",
                        column.name
                    ))
                })?;

            for row in rows {
                if matches!(row.get(insert_pos), Some(PlanValue::Null)) {
                    return Err(Error::ConstraintError(format!(
                        "NOT NULL constraint failed for column '{}'",
                        column.name
                    )));
                }
            }
        }

        Ok(())
    }

    fn evaluate_check_expression(
        expr: &sqlparser::ast::Expr,
        row: &[PlanValue],
        column_order: &[usize],
        table: &ExecutorTable<P>,
        _check_column_idx: usize,
        _row_idx: usize,
    ) -> Result<bool> {
        use sqlparser::ast::Expr as SqlExpr;

        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                let left_val = Self::evaluate_check_expr_value(left, row, column_order, table)?;
                let right_val = Self::evaluate_check_expr_value(right, row, column_order, table)?;

                use sqlparser::ast::BinaryOperator;
                match op {
                    BinaryOperator::Eq => {
                        // NULL = anything is UNKNOWN, which doesn't violate CHECK
                        if matches!(left_val, PlanValue::Null)
                            || matches!(right_val, PlanValue::Null)
                        {
                            return Ok(true); // Unknown = pass CHECK
                        }
                        Ok(left_val == right_val)
                    }
                    BinaryOperator::NotEq => {
                        // NULL != anything is UNKNOWN, which doesn't violate CHECK
                        if matches!(left_val, PlanValue::Null)
                            || matches!(right_val, PlanValue::Null)
                        {
                            return Ok(true); // Unknown = pass CHECK
                        }
                        Ok(left_val != right_val)
                    }
                    BinaryOperator::Lt => {
                        // NULL < anything is UNKNOWN, which doesn't violate CHECK
                        if matches!(left_val, PlanValue::Null)
                            || matches!(right_val, PlanValue::Null)
                        {
                            return Ok(true); // Unknown = pass CHECK
                        }
                        match (left_val, right_val) {
                            (PlanValue::Integer(l), PlanValue::Integer(r)) => Ok(l < r),
                            (PlanValue::Float(l), PlanValue::Float(r)) => Ok(l < r),
                            _ => Err(Error::InvalidArgumentError(
                                "CHECK constraint < operator requires numeric values".into(),
                            )),
                        }
                    }
                    BinaryOperator::LtEq => {
                        // NULL <= anything is UNKNOWN, which doesn't violate CHECK
                        if matches!(left_val, PlanValue::Null)
                            || matches!(right_val, PlanValue::Null)
                        {
                            return Ok(true); // Unknown = pass CHECK
                        }
                        match (left_val, right_val) {
                            (PlanValue::Integer(l), PlanValue::Integer(r)) => Ok(l <= r),
                            (PlanValue::Float(l), PlanValue::Float(r)) => Ok(l <= r),
                            _ => Err(Error::InvalidArgumentError(
                                "CHECK constraint <= operator requires numeric values".into(),
                            )),
                        }
                    }
                    BinaryOperator::Gt => {
                        // NULL > anything is UNKNOWN, which doesn't violate CHECK
                        if matches!(left_val, PlanValue::Null)
                            || matches!(right_val, PlanValue::Null)
                        {
                            return Ok(true); // Unknown = pass CHECK
                        }
                        match (left_val, right_val) {
                            (PlanValue::Integer(l), PlanValue::Integer(r)) => Ok(l > r),
                            (PlanValue::Float(l), PlanValue::Float(r)) => Ok(l > r),
                            _ => Err(Error::InvalidArgumentError(
                                "CHECK constraint > operator requires numeric values".into(),
                            )),
                        }
                    }
                    BinaryOperator::GtEq => {
                        // NULL >= anything is UNKNOWN, which doesn't violate CHECK
                        if matches!(left_val, PlanValue::Null)
                            || matches!(right_val, PlanValue::Null)
                        {
                            return Ok(true); // Unknown = pass CHECK
                        }
                        match (left_val, right_val) {
                            (PlanValue::Integer(l), PlanValue::Integer(r)) => Ok(l >= r),
                            (PlanValue::Float(l), PlanValue::Float(r)) => Ok(l >= r),
                            _ => Err(Error::InvalidArgumentError(
                                "CHECK constraint >= operator requires numeric values".into(),
                            )),
                        }
                    }
                    _ => Err(Error::InvalidArgumentError(format!(
                        "Unsupported operator in CHECK constraint: {:?}",
                        op
                    ))),
                }
            }
            SqlExpr::IsNull(inner) => {
                let value = Self::evaluate_check_expr_value(inner, row, column_order, table)?;
                Ok(matches!(value, PlanValue::Null))
            }
            SqlExpr::IsNotNull(inner) => {
                let value = Self::evaluate_check_expr_value(inner, row, column_order, table)?;
                Ok(!matches!(value, PlanValue::Null))
            }
            _ => Err(Error::InvalidArgumentError(format!(
                "Unsupported expression in CHECK constraint: {:?}",
                expr
            ))),
        }
    }

    fn evaluate_check_expr_value(
        expr: &sqlparser::ast::Expr,
        row: &[PlanValue],
        column_order: &[usize],
        table: &ExecutorTable<P>,
    ) -> Result<PlanValue> {
        use sqlparser::ast::Expr as SqlExpr;

        match expr {
            SqlExpr::BinaryOp { left, op, right } => {
                // Handle arithmetic operations in CHECK expressions (e.g., i + j)
                let left_val = Self::evaluate_check_expr_value(left, row, column_order, table)?;
                let right_val = Self::evaluate_check_expr_value(right, row, column_order, table)?;

                use sqlparser::ast::BinaryOperator;
                match op {
                    BinaryOperator::Plus => match (left_val, right_val) {
                        (PlanValue::Null, _) | (_, PlanValue::Null) => Ok(PlanValue::Null),
                        (PlanValue::Integer(l), PlanValue::Integer(r)) => {
                            Ok(PlanValue::Integer(l + r))
                        }
                        (PlanValue::Float(l), PlanValue::Float(r)) => Ok(PlanValue::Float(l + r)),
                        (PlanValue::Integer(l), PlanValue::Float(r)) => {
                            Ok(PlanValue::Float(l as f64 + r))
                        }
                        (PlanValue::Float(l), PlanValue::Integer(r)) => {
                            Ok(PlanValue::Float(l + r as f64))
                        }
                        _ => Err(Error::InvalidArgumentError(
                            "CHECK constraint + operator requires numeric values".into(),
                        )),
                    },
                    BinaryOperator::Minus => match (left_val, right_val) {
                        (PlanValue::Null, _) | (_, PlanValue::Null) => Ok(PlanValue::Null),
                        (PlanValue::Integer(l), PlanValue::Integer(r)) => {
                            Ok(PlanValue::Integer(l - r))
                        }
                        (PlanValue::Float(l), PlanValue::Float(r)) => Ok(PlanValue::Float(l - r)),
                        (PlanValue::Integer(l), PlanValue::Float(r)) => {
                            Ok(PlanValue::Float(l as f64 - r))
                        }
                        (PlanValue::Float(l), PlanValue::Integer(r)) => {
                            Ok(PlanValue::Float(l - r as f64))
                        }
                        _ => Err(Error::InvalidArgumentError(
                            "CHECK constraint - operator requires numeric values".into(),
                        )),
                    },
                    BinaryOperator::Multiply => match (left_val, right_val) {
                        (PlanValue::Null, _) | (_, PlanValue::Null) => Ok(PlanValue::Null),
                        (PlanValue::Integer(l), PlanValue::Integer(r)) => {
                            Ok(PlanValue::Integer(l * r))
                        }
                        (PlanValue::Float(l), PlanValue::Float(r)) => Ok(PlanValue::Float(l * r)),
                        (PlanValue::Integer(l), PlanValue::Float(r)) => {
                            Ok(PlanValue::Float(l as f64 * r))
                        }
                        (PlanValue::Float(l), PlanValue::Integer(r)) => {
                            Ok(PlanValue::Float(l * r as f64))
                        }
                        _ => Err(Error::InvalidArgumentError(
                            "CHECK constraint * operator requires numeric values".into(),
                        )),
                    },
                    BinaryOperator::Divide => match (left_val, right_val) {
                        (PlanValue::Null, _) | (_, PlanValue::Null) => Ok(PlanValue::Null),
                        (PlanValue::Integer(l), PlanValue::Integer(r)) => {
                            if r == 0 {
                                Err(Error::InvalidArgumentError(
                                    "Division by zero in CHECK constraint".into(),
                                ))
                            } else {
                                Ok(PlanValue::Integer(l / r))
                            }
                        }
                        (PlanValue::Float(l), PlanValue::Float(r)) => {
                            if r == 0.0 {
                                Err(Error::InvalidArgumentError(
                                    "Division by zero in CHECK constraint".into(),
                                ))
                            } else {
                                Ok(PlanValue::Float(l / r))
                            }
                        }
                        (PlanValue::Integer(l), PlanValue::Float(r)) => {
                            if r == 0.0 {
                                Err(Error::InvalidArgumentError(
                                    "Division by zero in CHECK constraint".into(),
                                ))
                            } else {
                                Ok(PlanValue::Float(l as f64 / r))
                            }
                        }
                        (PlanValue::Float(l), PlanValue::Integer(r)) => {
                            if r == 0 {
                                Err(Error::InvalidArgumentError(
                                    "Division by zero in CHECK constraint".into(),
                                ))
                            } else {
                                Ok(PlanValue::Float(l / r as f64))
                            }
                        }
                        _ => Err(Error::InvalidArgumentError(
                            "CHECK constraint / operator requires numeric values".into(),
                        )),
                    },
                    _ => Err(Error::InvalidArgumentError(format!(
                        "Unsupported binary operator in CHECK constraint value expression: {:?}",
                        op
                    ))),
                }
            }
            SqlExpr::Identifier(ident) => {
                // Simple column reference
                let column_name = &ident.value;
                let col_idx = table
                    .schema
                    .columns
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(column_name))
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "Unknown column '{}' in CHECK constraint",
                            column_name
                        ))
                    })?;

                // Find value in row
                let insert_pos = column_order
                    .iter()
                    .position(|&dest_idx| dest_idx == col_idx)
                    .ok_or_else(|| {
                        Error::InvalidArgumentError(format!(
                            "Column '{}' not provided in INSERT",
                            column_name
                        ))
                    })?;

                Ok(row[insert_pos].clone())
            }
            SqlExpr::CompoundIdentifier(idents) => {
                // Struct field access like t.t or table.t.t
                if idents.len() == 2 {
                    // col.field format
                    let column_name = &idents[0].value;
                    let field_name = &idents[1].value;

                    let col_idx = table
                        .schema
                        .columns
                        .iter()
                        .position(|c| c.name.eq_ignore_ascii_case(column_name))
                        .ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "Unknown column '{}' in CHECK constraint",
                                column_name
                            ))
                        })?;

                    let insert_pos = column_order
                        .iter()
                        .position(|&dest_idx| dest_idx == col_idx)
                        .ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "Column '{}' not provided in INSERT",
                                column_name
                            ))
                        })?;

                    // Extract struct field
                    match &row[insert_pos] {
                        PlanValue::Struct(fields) => fields
                            .iter()
                            .find(|(name, _)| name.eq_ignore_ascii_case(field_name))
                            .map(|(_, val)| val.clone())
                            .ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "Struct field '{}' not found in column '{}'",
                                    field_name, column_name
                                ))
                            }),
                        _ => Err(Error::InvalidArgumentError(format!(
                            "Column '{}' is not a struct, cannot access field '{}'",
                            column_name, field_name
                        ))),
                    }
                } else if idents.len() == 3 {
                    // table.col.field format - ignore table part
                    let column_name = &idents[1].value;
                    let field_name = &idents[2].value;

                    let col_idx = table
                        .schema
                        .columns
                        .iter()
                        .position(|c| c.name.eq_ignore_ascii_case(column_name))
                        .ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "Unknown column '{}' in CHECK constraint",
                                column_name
                            ))
                        })?;

                    let insert_pos = column_order
                        .iter()
                        .position(|&dest_idx| dest_idx == col_idx)
                        .ok_or_else(|| {
                            Error::InvalidArgumentError(format!(
                                "Column '{}' not provided in INSERT",
                                column_name
                            ))
                        })?;

                    match &row[insert_pos] {
                        PlanValue::Struct(fields) => fields
                            .iter()
                            .find(|(name, _)| name.eq_ignore_ascii_case(field_name))
                            .map(|(_, val)| val.clone())
                            .ok_or_else(|| {
                                Error::InvalidArgumentError(format!(
                                    "Struct field '{}' not found in column '{}'",
                                    field_name, column_name
                                ))
                            }),
                        _ => Err(Error::InvalidArgumentError(format!(
                            "Column '{}' is not a struct, cannot access field '{}'",
                            column_name, field_name
                        ))),
                    }
                } else {
                    Err(Error::InvalidArgumentError(format!(
                        "Unsupported compound identifier in CHECK constraint: {} parts",
                        idents.len()
                    )))
                }
            }
            SqlExpr::Value(val_with_span) => {
                // Numeric literal
                match &val_with_span.value {
                    sqlparser::ast::Value::Number(n, _) => {
                        if let Ok(i) = n.parse::<i64>() {
                            Ok(PlanValue::Integer(i))
                        } else if let Ok(f) = n.parse::<f64>() {
                            Ok(PlanValue::Float(f))
                        } else {
                            Err(Error::InvalidArgumentError(format!(
                                "Invalid number in CHECK constraint: {}",
                                n
                            )))
                        }
                    }
                    sqlparser::ast::Value::SingleQuotedString(s)
                    | sqlparser::ast::Value::DoubleQuotedString(s) => {
                        Ok(PlanValue::String(s.clone()))
                    }
                    sqlparser::ast::Value::Null => Ok(PlanValue::Null),
                    _ => Err(Error::InvalidArgumentError(format!(
                        "Unsupported value type in CHECK constraint: {:?}",
                        val_with_span.value
                    ))),
                }
            }
            _ => Err(Error::InvalidArgumentError(format!(
                "Unsupported expression type in CHECK constraint: {:?}",
                expr
            ))),
        }
    }

    fn ensure_existing_rows_unique(
        &self,
        table: &ExecutorTable<P>,
        field_id: FieldId,
        column_name: &str,
    ) -> Result<()> {
        let snapshot = self.default_snapshot();
        let values = self.scan_column_values(table, field_id, snapshot)?;

        // TODO: This is inefficient for large datasets; consider a more efficient approach
        let mut seen: FxHashSet<UniqueKey> = FxHashSet::default();

        for value in values {
            let Some(key) = Self::unique_key_component(&value, column_name)? else {
                continue;
            };

            if !seen.insert(key) {
                return Err(Error::ConstraintError(format!(
                    "constraint violation on column '{}'",
                    column_name
                )));
            }
        }

        Ok(())
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

    fn ensure_existing_rows_unique_multi(
        &self,
        table: &ExecutorTable<P>,
        field_ids: &[FieldId],
        column_names: &[String],
    ) -> Result<()> {
        if field_ids.is_empty() {
            return Ok(());
        }

        let snapshot = self.default_snapshot();
        let rows = self.scan_multi_column_values(table, field_ids, snapshot)?;
        let mut seen: FxHashSet<UniqueKey> = FxHashSet::default();

        for values in rows {
            if values.len() != field_ids.len() {
                continue;
            }

            if let Some(key) = Self::build_composite_unique_key(&values, column_names)?
                && !seen.insert(key)
            {
                return Err(Error::ConstraintError(format!(
                    "constraint violation on columns '{}'",
                    column_names.join(", ")
                )));
            }
        }

        Ok(())
    }

    fn check_unique_constraints(
        &self,
        table: &ExecutorTable<P>,
        rows: &[Vec<PlanValue>],
        column_order: &[usize],
        snapshot: TransactionSnapshot,
    ) -> Result<()> {
        let unique_columns: Vec<(usize, &ExecutorColumn)> = table
            .schema
            .columns
            .iter()
            .enumerate()
            .filter(|(_, col)| col.unique && !col.primary_key)
            .collect();

        for (col_idx, column) in unique_columns {
            let existing_values = self.scan_column_values(table, column.field_id, snapshot)?;
            let mut new_values: Vec<PlanValue> = Vec::new();

            for row in rows {
                let Some(insert_position) = column_order.iter().position(|&dest| dest == col_idx)
                else {
                    continue;
                };
                let value = row[insert_position].clone();

                if matches!(value, PlanValue::Null) {
                    continue;
                }

                if existing_values.contains(&value) || new_values.contains(&value) {
                    return Err(Error::ConstraintError(format!(
                        "constraint violation on column '{}'",
                        column.name
                    )));
                }

                new_values.push(value);
            }
        }

        let multi_uniques = table.multi_column_uniques();
        for constraint in multi_uniques {
            if constraint.column_indices.is_empty() {
                continue;
            }

            let mut constraint_column_names = Vec::with_capacity(constraint.column_indices.len());
            let mut field_ids = Vec::with_capacity(constraint.column_indices.len());
            for &col_idx in &constraint.column_indices {
                let Some(column) = table.schema.columns.get(col_idx) else {
                    return Err(Error::Internal(format!(
                        "multi-column UNIQUE constraint references invalid column index {}",
                        col_idx
                    )));
                };
                constraint_column_names.push(column.name.clone());
                field_ids.push(column.field_id);
            }

            let existing_rows = self.scan_multi_column_values(table, &field_ids, snapshot)?;
            let mut existing_keys: FxHashSet<UniqueKey> = FxHashSet::default();
            for row_values in existing_rows {
                if let Some(key) =
                    Self::build_composite_unique_key(&row_values, &constraint_column_names)?
                {
                    existing_keys.insert(key);
                }
            }

            let mut new_keys: FxHashSet<UniqueKey> = FxHashSet::default();
            for row in rows {
                let mut values_for_constraint = Vec::with_capacity(constraint.column_indices.len());
                for &col_idx in &constraint.column_indices {
                    if let Some(pos) = column_order.iter().position(|&dest| dest == col_idx) {
                        values_for_constraint.push(row[pos].clone());
                    } else {
                        values_for_constraint.push(PlanValue::Null);
                    }
                }

                if let Some(key) = Self::build_composite_unique_key(
                    &values_for_constraint,
                    &constraint_column_names,
                )? && (existing_keys.contains(&key) || !new_keys.insert(key))
                {
                    return Err(Error::ConstraintError(format!(
                        "constraint violation on columns '{}'",
                        constraint_column_names.join(", ")
                    )));
                }
            }
        }

        Ok(())
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
            let mut new_values: Vec<PlanValue> = Vec::new();

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
                    if existing_values.contains(new_value) || new_values.contains(new_value) {
                        return Err(Error::ConstraintError(format!(
                            "constraint violation on column '{}'",
                            column.name
                        )));
                    }

                    new_values.push(new_value.clone());
                }
            }
        }

        Ok(())
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

    fn unique_key_component(value: &PlanValue, column_name: &str) -> Result<Option<UniqueKey>> {
        match value {
            PlanValue::Null => Ok(None),
            PlanValue::Integer(v) => Ok(Some(UniqueKey::Int(*v))),
            PlanValue::Float(v) => Ok(Some(UniqueKey::Float(v.to_bits()))),
            PlanValue::String(s) => Ok(Some(UniqueKey::Str(s.clone()))),
            PlanValue::Struct(_) => Err(Error::InvalidArgumentError(format!(
                "UNIQUE index is not supported on struct column '{}'",
                column_name
            ))),
        }
    }

    fn build_composite_unique_key(
        values: &[PlanValue],
        column_names: &[String],
    ) -> Result<Option<UniqueKey>> {
        if values.is_empty() {
            return Ok(None);
        }

        let mut components = Vec::with_capacity(values.len());
        for (value, column_name) in values.iter().zip(column_names) {
            match Self::unique_key_component(value, column_name)? {
                Some(component) => components.push(component),
                None => return Ok(None),
            }
        }

        Ok(Some(UniqueKey::Composite(components)))
    }

    fn insert_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        rows: Vec<Vec<PlanValue>>,
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

        self.check_not_null_constraints(table, &rows, &column_order)?;
        // Check CHECK constraints
        self.check_check_constraints(table, &rows, &column_order)?;
        // Check UNIQUE constraints
        self.check_unique_constraints(table, &rows, &column_order, snapshot)?;

        if display_name == "keys" {
            tracing::trace!(
                "[KEYS] Checking PRIMARY KEY constraints - {} rows to insert",
                rows.len()
            );
            for (i, row) in rows.iter().enumerate() {
                tracing::trace!("[KEYS]   row[{}]: {:?}", i, row);
            }
        }

        let constraint_result =
            self.check_primary_key_constraints(table, &rows, &column_order, snapshot);

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
        table.table.append(&batch)?;
        table
            .next_row_id
            .store(start_row + row_count as u64, Ordering::SeqCst);
        table
            .total_rows
            .fetch_add(row_count as u64, Ordering::SeqCst);

        Ok(RuntimeStatementResult::Insert {
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

            match self.insert_rows(table, display_name.clone(), rows, columns.clone(), snapshot)? {
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
                    row[column_index] = new_value;
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
        self.check_not_null_constraints(table, &new_rows, &column_order)?;
        self.check_check_constraints(table, &new_rows, &column_order)?;

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            row_ids.clone(),
            snapshot.txn_id,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
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
                    row[column_index] = new_value;
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
        self.check_not_null_constraints(table, &new_rows, &column_order)?;
        self.check_check_constraints(table, &new_rows, &column_order)?;

        let _ = self.apply_delete(
            table,
            display_name.clone(),
            row_ids.clone(),
            snapshot.txn_id,
        )?;

        let _ = self.insert_rows(
            table,
            display_name.clone(),
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
        filter: LlkvExpr<'static, String>,
        snapshot: TransactionSnapshot,
        txn_id: TxnId,
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
        self.apply_delete(table, display_name, row_ids, txn_id)
    }

    fn delete_all_rows(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        snapshot: TransactionSnapshot,
        txn_id: TxnId,
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
        self.apply_delete(table, display_name, row_ids, txn_id)
    }

    fn apply_delete(
        &self,
        table: &ExecutorTable<P>,
        display_name: String,
        row_ids: Vec<RowId>,
        txn_id: TxnId,
    ) -> Result<RuntimeStatementResult<P>> {
        if row_ids.is_empty() {
            return Ok(RuntimeStatementResult::Delete {
                table_name: display_name,
                rows_deleted: 0,
            });
        }

        let removed = row_ids.len();

        // Build DELETE batch using helper
        let batch = mvcc_columns::build_delete_batch(row_ids.clone(), txn_id)?;
        table.table.append(&batch)?;

        let removed_u64 = u64::try_from(removed)
            .map_err(|_| Error::InvalidArgumentError("row count exceeds supported range".into()))?;
        table.total_rows.fetch_sub(removed_u64, Ordering::SeqCst);

        Ok(RuntimeStatementResult::Delete {
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
        let _catalog_table_id = self.catalog.table_id(canonical_name).ok_or_else(|| {
            Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
        })?;

        let store = ColumnStore::open(Arc::clone(&self.pager))?;
        let catalog = SysCatalog::new(&store);

        // Find the table metadata in the catalog
        let all_metas = catalog.all_table_metas()?;
        let (table_id, _meta) = all_metas
            .iter()
            .find(|(_, meta)| {
                meta.name
                    .as_ref()
                    .map(|n| n.to_ascii_lowercase() == canonical_name)
                    .unwrap_or(false)
            })
            .ok_or_else(|| {
                Error::InvalidArgumentError(format!("unknown table '{}'", canonical_name))
            })?;

        // Open the table and build ExecutorTable
        let table = Table::new(*table_id, Arc::clone(&self.pager))?;
        let schema = table.schema()?;
        let catalog_field_resolver = self.catalog.field_resolver(_catalog_table_id);

        // Build ExecutorSchema from Arrow schema (skip row_id field at index 0)
        let mut executor_columns = Vec::new();
        let mut lookup =
            FxHashMap::with_capacity_and_hasher(schema.fields().len(), Default::default());

        for (idx, field) in schema.fields().iter().enumerate().skip(1) {
            // Get field_id from metadata
            let field_id = field
                .metadata()
                .get(llkv_table::constants::FIELD_ID_META_KEY)
                .and_then(|s| s.parse::<FieldId>().ok())
                .unwrap_or(idx as FieldId);

            let normalized = field.name().to_ascii_lowercase();
            let col_idx = executor_columns.len();
            lookup.insert(normalized, col_idx);

            let constraints: FieldConstraints = catalog_field_resolver
                .as_ref()
                .and_then(|resolver| resolver.field_constraints_by_name(field.name()))
                .unwrap_or_default();

            executor_columns.push(ExecutorColumn {
                name: field.name().to_string(),
                data_type: field.data_type().clone(),
                nullable: field.is_nullable(),
                primary_key: constraints.primary_key,
                unique: constraints.primary_key || constraints.unique,
                field_id,
                check_expr: constraints.check_expr.clone(),
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
            let row_id_field = rowid_fid(LogicalFieldId::for_user(*table_id, 1));
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

        if let Some(stored) = self
            .multi_column_uniques
            .read()
            .unwrap()
            .get(canonical_name)
            .cloned()
        {
            let executor_uniques =
                Self::build_executor_multi_column_uniques(&executor_table, &stored);
            executor_table.set_multi_column_uniques(executor_uniques);
        }

        // Cache the loaded table
        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(canonical_name.to_string(), Arc::clone(&executor_table));
        }

        // Register fields in catalog (may already be registered from RuntimeContext::new())
        if let Some(field_resolver) = self.catalog.field_resolver(_catalog_table_id) {
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
            table_id,
            schema.fields().len() - 1,
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

        // Don't remove from tables cache - keep it for transactions with earlier snapshots
        // self.remove_table_entry(&canonical_name);

        // Unregister from catalog
        self.catalog.unregister_table(&canonical_name);
        tracing::debug!(
            "[CATALOG] Unregistered table '{}' from catalog",
            canonical_name
        );

        self.dropped_tables
            .write()
            .unwrap()
            .insert(canonical_name.clone());
        self.multi_column_uniques
            .write()
            .unwrap()
            .remove(&canonical_name);
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

        // Skip any reserved table IDs
        while llkv_table::reserved::is_reserved_table_id(next) {
            next = next.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        let mut following = next
            .checked_add(1)
            .ok_or_else(|| Error::InvalidArgumentError("exhausted available table ids".into()))?;

        // Skip any reserved table IDs for the next allocation
        while llkv_table::reserved::is_reserved_table_id(following) {
            following = following.checked_add(1).ok_or_else(|| {
                Error::InvalidArgumentError("exhausted available table ids".into())
            })?;
        }

        catalog.put_next_table_id(following)?;
        Ok(next)
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
