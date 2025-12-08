//! SELECT execution support for `RuntimeContext`.
//!
//! This module hosts the high-level entry point that wires the runtime context
//! to the query executor, ensuring MVCC visibility is enforced via row filters.

use crate::SelectExecution;
use llkv_executor::{ExecutorTableProvider, QueryExecutor};
use llkv_plan::SelectPlan;
use llkv_result::Result;
use llkv_storage::pager::Pager;
use llkv_table::table::RowIdFilter;
use llkv_transaction::{MvccRowIdFilter, TransactionSnapshot};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

use super::{RuntimeContext, provider::ContextProvider};

impl<P> RuntimeContext<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync + std::fmt::Debug + 'static,
{
    /// Execute a SELECT plan while enforcing MVCC visibility rules.
    ///
    /// For information_schema tables, MVCC filtering is skipped since they contain
    /// metadata created in a separate context and should always be fully visible.
    pub(crate) fn execute_select(
        self: &Arc<Self>,
        plan: SelectPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<SelectExecution<P>> {
        let provider: Arc<dyn ExecutorTableProvider<P>> = Arc::new(ContextProvider {
            context: Arc::clone(self),
        });

        let executor = QueryExecutor::with_default_planner(provider);

        // Check if any table in the query is from information_schema
        let is_information_schema = plan.tables.iter().any(|table_ref| {
            table_ref
                .qualified_name()
                .starts_with("information_schema.")
        });

        // Skip MVCC filtering for information_schema tables
        let row_filter = if is_information_schema {
            None
        } else {
            Some(Arc::new(MvccRowIdFilter::new(
                Arc::clone(&self.txn_manager),
                snapshot,
            )) as Arc<dyn RowIdFilter<P>>)
        };

        executor.execute_select_with_row_filter(plan, row_filter)
    }
}
