// TODO: Implement a common trait (similar to CataglogDdl) for runtime sessions and llkv-transaction sessions

use std::sync::{Arc, RwLock};

use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use llkv_storage::pager::{BoxedPager, MemPager, Pager};
use llkv_table::canonical_table_name;
use simd_r_drive_entry_handle::EntryHandle;

use crate::storage_namespace::{
    self, PersistentNamespace, StorageNamespace, StorageNamespaceRegistry, TemporaryNamespace,
};
use crate::{
    AlterTablePlan, CatalogDdl, ColumnSpec, CreateIndexPlan, CreateTablePlan, CreateTableSource,
    DeletePlan, DropIndexPlan, DropTablePlan, InsertPlan, InsertSource, PlanOperation, PlanValue,
    RenameTablePlan, RuntimeContext, RuntimeStatementResult, RuntimeTransactionContext,
    SelectExecution, SelectPlan, SelectProjection, TransactionContext, TransactionKind,
    TransactionResult, TransactionSession, UpdatePlan,
};

pub(crate) struct SessionNamespaces<P>
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
    pub(crate) fn new(base_context: Arc<RuntimeContext<P>>) -> Self {
        let persistent = Arc::new(PersistentNamespace::new(
            storage_namespace::PERSISTENT_NAMESPACE_ID.to_string(),
            Arc::clone(&base_context),
        ));

        let mut registry = StorageNamespaceRegistry::new(
            StorageNamespace::namespace_id(persistent.as_ref()).clone(),
        );
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

    pub(crate) fn persistent(&self) -> Arc<PersistentNamespace<P>> {
        Arc::clone(&self.persistent)
    }

    pub(crate) fn temporary(&self) -> Option<Arc<TemporaryNamespace<BoxedPager>>> {
        self.temporary.as_ref().map(Arc::clone)
    }

    pub(crate) fn registry(&self) -> Arc<RwLock<StorageNamespaceRegistry>> {
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
    // TODO: Allow generic pager type for the secondary pager context
    // NOTE: Sessions always embed a `MemPager` for temporary namespaces; extend the
    // wrapper when pluggable temp storage is supported.
    inner: TransactionSession<RuntimeTransactionContext<P>, RuntimeTransactionContext<MemPager>>,
    namespaces: Arc<SessionNamespaces<P>>,
}

impl<P> RuntimeSession<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    pub(crate) fn from_parts(
        inner: TransactionSession<
            RuntimeTransactionContext<P>,
            RuntimeTransactionContext<MemPager>,
        >,
        namespaces: Arc<SessionNamespaces<P>>,
    ) -> Self {
        Self { inner, namespaces }
    }

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

        let temp_context = temp_namespace.context();
        let snapshot = temp_context.default_snapshot();
        let execution = temp_context.execute_select(plan.clone(), snapshot)?;
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

    fn base_transaction_context(&self) -> Arc<RuntimeTransactionContext<P>> {
        Arc::clone(self.inner.context())
    }

    fn with_autocommit_transaction_context<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&RuntimeTransactionContext<P>) -> Result<T>,
    {
        let context = self.base_transaction_context();
        let default_snapshot = context.ctx().default_snapshot();
        TransactionContext::set_snapshot(&*context, default_snapshot);
        f(context.as_ref())
    }

    fn run_autocommit_insert(&self, plan: InsertPlan) -> Result<TransactionResult<P>> {
        self.with_autocommit_transaction_context(|ctx| TransactionContext::insert(ctx, plan))
    }

    fn run_autocommit_update(&self, plan: UpdatePlan) -> Result<TransactionResult<P>> {
        self.with_autocommit_transaction_context(|ctx| TransactionContext::update(ctx, plan))
    }

    fn run_autocommit_delete(&self, plan: DeletePlan) -> Result<TransactionResult<P>> {
        self.with_autocommit_transaction_context(|ctx| TransactionContext::delete(ctx, plan))
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

        let staging_wrapper = Arc::new(RuntimeTransactionContext::new(staging_ctx));

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

    /// Check if a table was created in the current active transaction.
    pub fn is_table_created_in_transaction(&self, table_name: &str) -> bool {
        self.inner.is_table_created_in_transaction(table_name)
    }

    /// Get column specifications for a table created in the current transaction.
    /// Returns `None` if there's no active transaction or the table wasn't created in it.
    pub fn table_column_specs_from_transaction(&self, table_name: &str) -> Option<Vec<ColumnSpec>> {
        self.inner.table_column_specs_from_transaction(table_name)
    }

    /// Get tables that reference the given table via foreign keys created in the current transaction.
    /// Returns an empty vector if there's no active transaction or no transactional FKs reference this table.
    pub fn tables_referencing_in_transaction(&self, referenced_table: &str) -> Vec<String> {
        self.inner
            .tables_referencing_in_transaction(referenced_table)
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
                    TransactionContext::apply_create_table_plan(&**self.inner.context(), plan)?;
                }
                PlanOperation::DropTable(plan) => {
                    TransactionContext::drop_table(&**self.inner.context(), plan)?;
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

    fn materialize_ctas_plan(&self, mut plan: CreateTablePlan) -> Result<CreateTablePlan> {
        // Only materialize if source is a SELECT query
        // If source is already Batches, leave it alone
        if matches!(plan.source, Some(CreateTableSource::Select { .. })) {
            if let Some(CreateTableSource::Select { plan: select_plan }) = plan.source.take() {
                let select_result = self.execute_select_plan(*select_plan)?;
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
        }
        Ok(plan)
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
                let select_result = self.execute_select_plan(*select_plan)?;
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
    pub fn execute_insert_plan(&self, plan: InsertPlan) -> Result<RuntimeStatementResult<P>> {
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
                let temp_context = temp_namespace.context();
                let temp_tx_context = RuntimeTransactionContext::new(temp_context);
                match TransactionContext::insert(&temp_tx_context, plan)? {
                    TransactionResult::Insert { .. } => {}
                    _ => {
                        return Err(Error::Internal(
                            "unexpected transaction result for temporary INSERT".into(),
                        ));
                    }
                }
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
                    let result = self.run_autocommit_insert(plan)?;
                    if !matches!(result, TransactionResult::Insert { .. }) {
                        return Err(Error::Internal(
                            "unexpected transaction result for INSERT operation".into(),
                        ));
                    }
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
    pub fn execute_select_plan(&self, plan: SelectPlan) -> Result<RuntimeStatementResult<P>> {
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
            let table_name = if plan.tables.len() == 1 {
                plan.tables[0].qualified_name()
            } else {
                String::new()
            };
            let execution = self.with_autocommit_transaction_context(|ctx| {
                TransactionContext::execute_select(ctx, plan)
            })?;
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
        match self.execute_select_plan(plan)? {
            RuntimeStatementResult::Select { execution, .. } => Ok(execution.collect_rows()?.rows),
            other => Err(Error::Internal(format!(
                "expected Select result when reading table '{}', got {:?}",
                table, other
            ))),
        }
    }

    pub fn execute_update_plan(&self, plan: UpdatePlan) -> Result<RuntimeStatementResult<P>> {
        let (_, canonical_table) = canonical_table_name(&plan.table)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                let temp_context = temp_namespace.context();
                let table_name = plan.table.clone();
                let temp_tx_context = RuntimeTransactionContext::new(temp_context);
                match TransactionContext::update(&temp_tx_context, plan)? {
                    TransactionResult::Update { rows_updated, .. } => {
                        Ok(RuntimeStatementResult::Update {
                            rows_updated,
                            table_name,
                        })
                    }
                    _ => Err(Error::Internal(
                        "unexpected transaction result for temporary UPDATE".into(),
                    )),
                }
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
                    let table_name = plan.table.clone();
                    let result = self.run_autocommit_update(plan)?;
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

    pub fn execute_delete_plan(&self, plan: DeletePlan) -> Result<RuntimeStatementResult<P>> {
        let (_, canonical_table) = canonical_table_name(&plan.table)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                let temp_context = temp_namespace.context();
                let table_name = plan.table.clone();
                let temp_tx_context = RuntimeTransactionContext::new(temp_context);
                match TransactionContext::delete(&temp_tx_context, plan)? {
                    TransactionResult::Delete { rows_deleted } => {
                        Ok(RuntimeStatementResult::Delete {
                            rows_deleted,
                            table_name,
                        })
                    }
                    _ => Err(Error::Internal(
                        "unexpected transaction result for temporary DELETE".into(),
                    )),
                }
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
                    let table_name = plan.table.clone();
                    let result = self.run_autocommit_delete(plan)?;
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

/// Implement [`CatalogDdl`] directly on the session so callers must import the trait
/// to perform schema mutations. This keeps all runtime DDL entry points aligned with
/// the shared contract used by contexts and storage namespaces.
impl<P> CatalogDdl for RuntimeSession<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type CreateTableOutput = RuntimeStatementResult<P>;
    type DropTableOutput = RuntimeStatementResult<P>;
    type RenameTableOutput = ();
    type AlterTableOutput = RuntimeStatementResult<P>;
    type CreateIndexOutput = RuntimeStatementResult<P>;
    type DropIndexOutput = RuntimeStatementResult<P>;

    fn create_table(&self, plan: CreateTablePlan) -> Result<Self::CreateTableOutput> {
        let target_namespace = plan
            .namespace
            .clone()
            .unwrap_or_else(|| storage_namespace::PERSISTENT_NAMESPACE_ID.to_string())
            .to_ascii_lowercase();

        let plan = self.materialize_ctas_plan(plan)?;

        match target_namespace.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                let result = temp_namespace.create_table(plan)?;
                result.convert_pager_type::<P>()
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                if self.has_active_transaction() {
                    match self
                        .inner
                        .execute_operation(PlanOperation::CreateTable(plan))
                    {
                        Ok(TransactionResult::CreateTable { table_name }) => {
                            Ok(RuntimeStatementResult::CreateTable { table_name })
                        }
                        Ok(TransactionResult::NoOp) => Ok(RuntimeStatementResult::NoOp),
                        Ok(_) => Err(Error::Internal(
                            "expected CreateTable result during transactional CREATE TABLE".into(),
                        )),
                        Err(err) => {
                            self.abort_transaction();
                            Err(err)
                        }
                    }
                } else {
                    if self.inner.has_table_locked_by_other_session(&plan.name) {
                        return Err(Error::TransactionContextError(format!(
                            "table '{}' is locked by another active transaction",
                            plan.name
                        )));
                    }
                    self.persistent_namespace().create_table(plan)
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }

    fn drop_table(&self, plan: DropTablePlan) -> Result<Self::DropTableOutput> {
        let (_, canonical_table) = canonical_table_name(&plan.name)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                temp_namespace.drop_table(plan)?;
                Ok(RuntimeStatementResult::NoOp)
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                if self.has_active_transaction() {
                    let referencing_tables = self.tables_referencing_in_transaction(&plan.name);
                    if !referencing_tables.is_empty() {
                        let referencing_table = &referencing_tables[0];
                        self.abort_transaction();
                        return Err(Error::CatalogError(format!(
                            "Catalog Error: Could not drop the table because this table is main key table of the table \"{}\".",
                            referencing_table
                        )));
                    }

                    match self
                        .inner
                        .execute_operation(PlanOperation::DropTable(plan.clone()))
                    {
                        Ok(TransactionResult::NoOp) => Ok(RuntimeStatementResult::NoOp),
                        Ok(_) => Err(Error::Internal(
                            "expected NoOp result for DROP TABLE during transactional execution"
                                .into(),
                        )),
                        Err(err) => {
                            self.abort_transaction();
                            Err(err)
                        }
                    }
                } else {
                    if self.inner.has_table_locked_by_other_session(&plan.name) {
                        return Err(Error::TransactionContextError(format!(
                            "table '{}' is locked by another active transaction",
                            plan.name
                        )));
                    }
                    self.persistent_namespace().drop_table(plan)?;
                    Ok(RuntimeStatementResult::NoOp)
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }

    fn rename_table(&self, plan: RenameTablePlan) -> Result<Self::RenameTableOutput> {
        if self.has_active_transaction() {
            return Err(Error::InvalidArgumentError(
                "ALTER TABLE RENAME is not supported inside an active transaction".into(),
            ));
        }

        let (_, canonical_table) = canonical_table_name(&plan.current_name)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;
                match temp_namespace.rename_table(plan.clone()) {
                    Ok(()) => Ok(()),
                    Err(err) if plan.if_exists && super::is_table_missing_error(&err) => Ok(()),
                    Err(err) => Err(err),
                }
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                match self.persistent_namespace().rename_table(plan.clone()) {
                    Ok(()) => Ok(()),
                    Err(err) if plan.if_exists && super::is_table_missing_error(&err) => Ok(()),
                    Err(err) => Err(err),
                }
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }

    fn alter_table(&self, plan: AlterTablePlan) -> Result<Self::AlterTableOutput> {
        let (_, canonical_table) = canonical_table_name(&plan.table_name)?;
        let namespace_id = self.resolve_namespace_for_table(&canonical_table);

        match namespace_id.as_str() {
            storage_namespace::TEMPORARY_NAMESPACE_ID => {
                let temp_namespace = self
                    .temporary_namespace()
                    .ok_or_else(|| Error::Internal("temporary namespace unavailable".into()))?;

                let context = temp_namespace.context();
                let catalog_service = &context.catalog_service;
                let view = match catalog_service.table_view(&canonical_table) {
                    Ok(view) => view,
                    Err(err) if plan.if_exists && super::is_table_missing_error(&err) => {
                        return Ok(RuntimeStatementResult::NoOp);
                    }
                    Err(err) => return Err(err),
                };
                let table_id = view
                    .table_meta
                    .as_ref()
                    .ok_or_else(|| Error::Internal("table metadata missing".into()))?
                    .table_id;

                super::validate_alter_table_operation(
                    &plan.operation,
                    &view,
                    table_id,
                    catalog_service,
                )?;

                temp_namespace.alter_table(plan)?.convert_pager_type::<P>()
            }
            storage_namespace::PERSISTENT_NAMESPACE_ID => {
                let persistent = self.persistent_namespace();
                let context = persistent.context();
                let catalog_service = &context.catalog_service;
                let view = match catalog_service.table_view(&canonical_table) {
                    Ok(view) => view,
                    Err(err) if plan.if_exists && super::is_table_missing_error(&err) => {
                        return Ok(RuntimeStatementResult::NoOp);
                    }
                    Err(err) => return Err(err),
                };
                let table_id = view
                    .table_meta
                    .as_ref()
                    .ok_or_else(|| Error::Internal("table metadata missing".into()))?
                    .table_id;

                super::validate_alter_table_operation(
                    &plan.operation,
                    &view,
                    table_id,
                    catalog_service,
                )?;

                persistent.alter_table(plan)
            }
            other => Err(Error::InvalidArgumentError(format!(
                "Unknown storage namespace '{}'",
                other
            ))),
        }
    }

    fn create_index(&self, plan: CreateIndexPlan) -> Result<Self::CreateIndexOutput> {
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

    fn drop_index(&self, plan: DropIndexPlan) -> Result<Self::DropIndexOutput> {
        if self.has_active_transaction() {
            return Err(Error::InvalidArgumentError(
                "DROP INDEX is not supported inside an active transaction".into(),
            ));
        }

        let mut dropped = false;

        match self.persistent_namespace().drop_index(plan.clone()) {
            Ok(Some(_)) => {
                dropped = true;
            }
            Ok(None) => {}
            Err(err) => {
                if !super::is_index_not_found_error(&err) {
                    return Err(err);
                }
            }
        }

        if !dropped {
            if let Some(temp_namespace) = self.temporary_namespace() {
                match temp_namespace.drop_index(plan.clone()) {
                    Ok(Some(_)) => {
                        dropped = true;
                    }
                    Ok(None) => {}
                    Err(err) => {
                        if !super::is_index_not_found_error(&err) {
                            return Err(err);
                        }
                    }
                }
            }
        }

        if dropped || plan.if_exists {
            Ok(RuntimeStatementResult::NoOp)
        } else {
            Err(Error::CatalogError(format!(
                "Index '{}' does not exist",
                plan.name
            )))
        }
    }
}
