//! Re-export gather utilities so existing `llkv_table::gather` imports continue
//! to work while the implementation lives in `llkv-column-map`.

pub use llkv_column_map::gather::*;
