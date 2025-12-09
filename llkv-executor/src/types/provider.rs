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

    /// Get approximate row counts for multiple tables in a batch.
    fn batch_approximate_row_counts(&self, tables: &[&str]) -> ExecutorResult<Vec<Option<usize>>> {
        let mut out = Vec::with_capacity(tables.len());
        for name in tables {
            let count = self.get_table(name).ok().and_then(|t| t.table.total_rows().ok().map(|n| n as usize));
            out.push(count);
        }
        Ok(out)
    }
}
