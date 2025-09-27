//! Legacy `TableError` module.
//!
//! Table-layer code now uses the unified [`llkv_result::Error`] type for all
//! failures. This stub remains temporarily to keep downstream references from
//! breaking during refactors. Remove once the crate graph no longer mentions
//! `llkv_result::table_error`.
