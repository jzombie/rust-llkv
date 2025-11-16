use std::sync::Arc;

use llkv_result::Result;
use llkv_storage::pager::{BoxedPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

use crate::{CatalogDdl, PlanStatement, RuntimeContext, RuntimeSession, RuntimeStatementResult};

type StatementResult = RuntimeStatementResult<BoxedPager>;

pub struct RuntimeEngine {
    context: Arc<RuntimeContext<BoxedPager>>,
    session: RuntimeSession,
}

impl Clone for RuntimeEngine {
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

impl RuntimeEngine {
    pub fn new<P>(pager: Arc<P>) -> Self
    where
        P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
    {
        let boxed = Arc::new(BoxedPager::from_arc(pager));
        let context = Arc::new(RuntimeContext::new(boxed));
        Self::from_context(context)
    }

    pub fn from_boxed_pager(pager: Arc<BoxedPager>) -> Self {
        let context = Arc::new(RuntimeContext::new(pager));
        Self::from_context(context)
    }

    pub fn from_context(context: Arc<RuntimeContext<BoxedPager>>) -> Self {
        tracing::debug!("[ENGINE] RuntimeEngine::from_context - creating new session");
        let session = context.create_session();
        tracing::debug!("[ENGINE] RuntimeEngine::from_context - created session");
        Self { context, session }
    }

    pub fn context(&self) -> Arc<RuntimeContext<BoxedPager>> {
        Arc::clone(&self.context)
    }

    pub fn session(&self) -> &RuntimeSession {
        &self.session
    }

    /// Rebuilds the information schema tables strictly in-memory.
    ///
    /// This forwards to [`RuntimeSession::refresh_information_schema`], which
    /// materializes the system tables from catalog metadata and issues CTAS commands
    /// without touching the primary pager heap or any user tables.
    pub fn refresh_information_schema(&self) -> Result<()> {
        self.session.refresh_information_schema()
    }

    pub fn execute_statement(&self, statement: PlanStatement) -> Result<StatementResult> {
        match statement {
            PlanStatement::BeginTransaction => self.session.begin_transaction(),
            PlanStatement::CommitTransaction => self.session.commit_transaction(),
            PlanStatement::RollbackTransaction => self.session.rollback_transaction(),
            PlanStatement::CreateTable(plan) => CatalogDdl::create_table(&self.session, plan),
            PlanStatement::DropTable(plan) => CatalogDdl::drop_table(&self.session, plan),
            PlanStatement::CreateView(plan) => {
                CatalogDdl::create_view(&self.session, plan)?;
                Ok(RuntimeStatementResult::NoOp)
            }
            PlanStatement::DropView(plan) => {
                CatalogDdl::drop_view(&self.session, plan)?;
                Ok(RuntimeStatementResult::NoOp)
            }
            PlanStatement::DropIndex(plan) => CatalogDdl::drop_index(&self.session, plan),
            PlanStatement::Reindex(plan) => self.context().reindex_index(plan),
            PlanStatement::AlterTable(plan) => CatalogDdl::alter_table(&self.session, plan),
            PlanStatement::CreateIndex(plan) => CatalogDdl::create_index(&self.session, plan),
            PlanStatement::Insert(plan) => self.session.execute_insert_plan(plan),
            PlanStatement::Update(plan) => self.session.execute_update_plan(plan),
            PlanStatement::Delete(plan) => self.session.execute_delete_plan(plan),
            PlanStatement::Truncate(plan) => self.session.execute_truncate_plan(plan),
            PlanStatement::Select(plan) => self.session.execute_select_plan(*plan),
        }
    }

    pub fn execute_all<I>(&self, statements: I) -> Result<Vec<StatementResult>>
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
