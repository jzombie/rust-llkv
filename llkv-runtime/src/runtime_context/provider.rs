//! ContextProvider implementation for TableProvider trait

use crate::runtime_context::RuntimeContext;
use llkv_executor::{ExecutorTable, ExecutorTableProvider};
use llkv_result::Result;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Internal provider struct that wraps RuntimeContext for use with llkv_executor.
pub(crate) struct ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) context: Arc<RuntimeContext<P>>,
}

impl<P> ExecutorTableProvider<P> for ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        self.context.lookup_table(canonical_name)
    }

    fn batch_approximate_row_counts(&self, tables: &[&str]) -> Result<Vec<Option<usize>>> {
        self.context.batch_lookup_row_counts(tables)
    }
}
