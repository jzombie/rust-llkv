#![forbid(unsafe_code)]

/// Default max rows per streamed batch. Tune as needed.
pub const STREAM_BATCH_ROWS: usize = 65_536;
