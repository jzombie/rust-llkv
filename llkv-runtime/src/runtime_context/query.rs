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
    P: Pager<Blob = EntryHandle> + Send + Sync + 'static,
{
    /// Execute a SELECT plan while enforcing MVCC visibility rules.
    pub(crate) fn execute_select(
        self: &Arc<Self>,
        plan: SelectPlan,
        snapshot: TransactionSnapshot,
    ) -> Result<SelectExecution<P>> {
        let provider: Arc<dyn ExecutorTableProvider<P>> = Arc::new(ContextProvider {
            context: Arc::clone(self),
        });

        let executor = QueryExecutor::new(provider);
        let row_filter: Arc<dyn RowIdFilter<P>> = Arc::new(MvccRowIdFilter::new(
            Arc::clone(&self.txn_manager),
            snapshot,
        ));

        executor.execute_select_with_filter(plan, Some(row_filter))
    }
}
