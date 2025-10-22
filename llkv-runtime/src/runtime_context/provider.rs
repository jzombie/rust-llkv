//! ContextProvider implementation for TableProvider trait

use std::sync::Arc;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use llkv_executor::{ExecutorTable, TableProvider};
use llkv_result::Result;
use crate::runtime_context::RuntimeContext;

/// Internal provider struct that wraps RuntimeContext for use with llkv_executor.
pub(crate) struct ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) context: Arc<RuntimeContext<P>>,
}

impl<P> TableProvider<P> for ContextProvider<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> Result<Arc<ExecutorTable<P>>> {
        self.context.lookup_table(canonical_name)
    }
}
