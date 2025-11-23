//! TableProvider trait for executor table access.

use crate::ExecutorResult;
use crate::types::ExecutorTable;
use llkv_storage::pager::Pager;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

/// Trait for providing table access to the executor.
///
/// Implementations of this trait are responsible for resolving canonical table names
/// to `ExecutorTable` instances that can be used for query execution.
pub trait ExecutorTableProvider<P>: Send + Sync
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    /// Retrieve a table by its canonical name.
    ///
    /// # Arguments
    ///
    /// * `canonical_name` - The canonical (normalized) name of the table
    ///
    /// # Returns
    ///
    /// An `Arc` to the `ExecutorTable` if found, or an error if the table doesn't exist
    /// or cannot be accessed.
    fn get_table(&self, canonical_name: &str) -> ExecutorResult<Arc<ExecutorTable<P>>>;
}
