//! Core type definitions for the columnar storage engine.
//!
//! These identifiers are defined in `llkv-types` so they can be shared without
//! depending on the storage layer. The column-map crate re-exports them for
//! backwards compatibility.

pub use llkv_types::ids::*;
