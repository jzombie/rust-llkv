use std::sync::Arc;

use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;

use crate::{ExecutorResult, ExecutorTable};

/// Supplies executor tables by canonical name.
pub trait ExecutorTableProvider<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn get_table(&self, canonical_name: &str) -> ExecutorResult<Arc<ExecutorTable<P>>>;
}
