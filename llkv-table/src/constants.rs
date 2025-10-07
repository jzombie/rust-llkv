#![forbid(unsafe_code)]

/// Default max rows per streamed batch. Tune as needed.
pub const STREAM_BATCH_ROWS: usize = 65_536;

// Re-export the canonical metadata key from the column-map store so other
// crates can refer to `llkv_table::constants::FIELD_ID_META_KEY` and get the
// authoritative value.
pub use llkv_column_map::store::FIELD_ID_META_KEY;
