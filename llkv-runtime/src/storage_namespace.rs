//! Legacy storage namespace module retained for compatibility.
//!
//! The runtime namespace abstractions now live in `runtime_storage_namespace`. This stub re-exports
//! the new module so downstream code that still references `llkv_runtime::storage_namespace` can
//! transition gradually. No new code should depend on this module name.

#[allow(deprecated)]
pub use crate::runtime_storage_namespace::*;
