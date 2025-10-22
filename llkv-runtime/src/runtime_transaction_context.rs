use std::sync::{Arc, RwLock};

use arrow::record_batch::RecordBatch;
use llkv_expr::expr::Expr as LlkvExpr;
use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_table::{CatalogDdl, SingleColumnIndexDescriptor, TableId};
use llkv_transaction::{
    TransactionContext, TransactionResult, TransactionSnapshot, TxnId, TXN_ID_AUTO_COMMIT,
};
use simd_r_drive_entry_handle::EntryHandle;

use crate::{
    AlterTablePlan, ColumnSpec, CreateIndexPlan, CreateTablePlan, DeletePlan, DropIndexPlan,
    DropTablePlan, InsertPlan, RenameTablePlan, RuntimeContext, RuntimeStatementResult,
    SelectExecution, SelectPlan, UpdatePlan,
};

/// Transaction-scoped façade over [`RuntimeContext`].
///
/// This type implements [`TransactionContext`] so the runtime can participate in
/// the `llkv-transaction` engine without exposing transactional methods on the
/// shared [`RuntimeContext`] itself. Each instance keeps track of the active
/// snapshot and delegates operations to the underlying context, applying MVCC
/// filtering and conversion of statement results into transaction results.
pub struct RuntimeTransactionContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    ctx: Arc<RuntimeContext<P>>,
    snapshot: RwLock<TransactionSnapshot>,
}

impl<P> RuntimeTransactionContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(ctx: Arc<RuntimeContext<P>>) -> Self {
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

    pub(crate) fn context(&self) -> &Arc<RuntimeContext<P>> {
        &self.ctx
    }

    pub(crate) fn ctx(&self) -> &RuntimeContext<P> {
        &self.ctx
    }
}

impl<P> CatalogDdl for RuntimeTransactionContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type CreateTableOutput = TransactionResult<P>;
    type DropTableOutput = ();
    type RenameTableOutput = ();
    type AlterTableOutput = TransactionResult<P>;
    type CreateIndexOutput = TransactionResult<P>;
    type DropIndexOutput = Option<SingleColumnIndexDescriptor>;

    fn create_table(&self, plan: CreateTablePlan) -> LlkvResult<Self::CreateTableOutput> {
        let ctx = self.context();
        let result = CatalogDdl::create_table(ctx.as_ref(), plan)?;
        Ok(convert_statement_result(result))
    }

    fn drop_table(&self, plan: DropTablePlan) -> LlkvResult<Self::DropTableOutput> {
        CatalogDdl::drop_table(self.ctx.as_ref(), plan)
    }

    fn rename_table(&self, plan: RenameTablePlan) -> LlkvResult<Self::RenameTableOutput> {
        CatalogDdl::rename_table(self.ctx.as_ref(), plan)
    }

    fn alter_table(&self, plan: AlterTablePlan) -> LlkvResult<Self::AlterTableOutput> {
        let ctx = self.context();
        let result = CatalogDdl::alter_table(ctx.as_ref(), plan)?;
        Ok(convert_statement_result(result))
    }

    fn create_index(&self, plan: CreateIndexPlan) -> LlkvResult<Self::CreateIndexOutput> {
        let ctx = self.context();
        let result = CatalogDdl::create_index(ctx.as_ref(), plan)?;
        Ok(convert_statement_result(result))
    }

    fn drop_index(&self, plan: DropIndexPlan) -> LlkvResult<Self::DropIndexOutput> {
        CatalogDdl::drop_index(self.ctx.as_ref(), plan)
    }
}

// Implement TransactionContext to integrate with llkv-transaction.
impl<P> TransactionContext for RuntimeTransactionContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    type Pager = P;
    type Snapshot = llkv_table::catalog::TableCatalogSnapshot;

    fn set_snapshot(&self, snapshot: TransactionSnapshot) {
        self.update_snapshot(snapshot);
    }

    fn snapshot(&self) -> TransactionSnapshot {
        self.current_snapshot()
    }

    fn table_column_specs(&self, table_name: &str) -> LlkvResult<Vec<ColumnSpec>> {
        RuntimeContext::table_column_specs(self.context(), table_name)
    }

    fn export_table_rows(&self, table_name: &str) -> LlkvResult<llkv_transaction::RowBatch> {
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
    ) -> LlkvResult<Vec<RecordBatch>> {
        RuntimeContext::get_batches_with_row_ids_with_snapshot(
            self.context(),
            table_name,
            filter,
            self.snapshot(),
        )
    }

    fn execute_select(&self, plan: SelectPlan) -> LlkvResult<SelectExecution<Self::Pager>> {
        RuntimeContext::execute_select_with_snapshot(self.context(), plan, self.snapshot())
    }

    fn apply_create_table_plan(
        &self,
        plan: CreateTablePlan,
    ) -> LlkvResult<TransactionResult<P>> {
        let ctx = self.context();
        let result = CatalogDdl::create_table(ctx.as_ref(), plan)?;
        Ok(convert_statement_result(result))
    }

    fn drop_table(&self, plan: DropTablePlan) -> LlkvResult<()> {
        CatalogDdl::drop_table(self.ctx.as_ref(), plan)
    }

    fn insert(&self, plan: InsertPlan) -> LlkvResult<TransactionResult<P>> {
        tracing::trace!(
            "[TX_RUNTIME] TransactionContext::insert plan.table='{}', context_pager={:p}",
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

    fn update(&self, plan: UpdatePlan) -> LlkvResult<TransactionResult<P>> {
        let snapshot = self.current_snapshot();
        let result = if snapshot.txn_id == TXN_ID_AUTO_COMMIT {
            self.ctx().update(plan)?
        } else {
            RuntimeContext::update_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn delete(&self, plan: DeletePlan) -> LlkvResult<TransactionResult<P>> {
        let snapshot = self.current_snapshot();
        let result = if snapshot.txn_id == TXN_ID_AUTO_COMMIT {
            self.ctx().delete(plan)?
        } else {
            RuntimeContext::delete_with_snapshot(self.context(), plan, snapshot)?
        };
        Ok(convert_statement_result(result))
    }

    fn create_index(&self, plan: CreateIndexPlan) -> LlkvResult<TransactionResult<P>> {
        let ctx = self.context();
        let result = CatalogDdl::create_index(ctx.as_ref(), plan)?;
        Ok(convert_statement_result(result))
    }

    fn append_batches_with_row_ids(
        &self,
        table_name: &str,
        batches: Vec<RecordBatch>,
    ) -> LlkvResult<usize> {
        RuntimeContext::append_batches_with_row_ids(self.context(), table_name, batches)
    }

    fn table_names(&self) -> Vec<String> {
        RuntimeContext::table_names(self.context())
    }

    fn table_id(&self, table_name: &str) -> LlkvResult<TableId> {
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

    fn catalog_snapshot(&self) -> Self::Snapshot {
        let ctx = self.context();
        ctx.catalog.snapshot()
    }

    fn validate_commit_constraints(&self, txn_id: TxnId) -> LlkvResult<()> {
        self.ctx.validate_primary_keys_for_commit(txn_id)
    }

    fn clear_transaction_state(&self, txn_id: TxnId) {
        self.ctx.clear_transaction_state(txn_id);
    }
}

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
