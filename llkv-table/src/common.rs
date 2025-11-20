//! Shared constants and types for LLKV table providers.

pub use llkv_storage::constants::{
    CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, FIELD_ID_META_KEY, ROW_ID_COLUMN_NAME,
};

use llkv_result::Result;

/// Trait for receiving notifications about table events.
pub trait TableEventListener: Send + Sync {
    /// Called when new rows are appended to a table.
    fn on_rows_appended(&self, row_ids: &[u64]) -> Result<()>;
}

/// Default batch size for scanning.
pub const DEFAULT_SCAN_BATCH_SIZE: usize = 1024;
