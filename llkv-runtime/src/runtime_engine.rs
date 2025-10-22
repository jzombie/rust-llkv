use std::sync::Arc;

use llkv_result::Result;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::{CatalogDdl, PlanStatement, RuntimeContext, RuntimeSession, RuntimeStatementResult};

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
            PlanStatement::CreateTable(plan) => CatalogDdl::create_table(&self.session, plan),
            PlanStatement::DropTable(plan) => CatalogDdl::drop_table(&self.session, plan),
            PlanStatement::DropIndex(plan) => CatalogDdl::drop_index(&self.session, plan),
            PlanStatement::AlterTable(plan) => CatalogDdl::alter_table(&self.session, plan),
            PlanStatement::CreateIndex(plan) => CatalogDdl::create_index(&self.session, plan),
            PlanStatement::Insert(plan) => self.session.execute_insert_plan(plan),
            PlanStatement::Update(plan) => self.session.execute_update_plan(plan),
            PlanStatement::Delete(plan) => self.session.execute_delete_plan(plan),
            PlanStatement::Select(plan) => self.session.execute_select_plan(plan),
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
