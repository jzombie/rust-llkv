//! Shared constants and types for LLKV table providers.

pub use llkv_storage::constants::{
    CREATED_BY_COLUMN_NAME, DELETED_BY_COLUMN_NAME, FIELD_ID_META_KEY, ROW_ID_COLUMN_NAME,
};

/// Default batch size for scanning.
pub const DEFAULT_SCAN_BATCH_SIZE: usize = 1024;
