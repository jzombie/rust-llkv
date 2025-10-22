#![forbid(unsafe_code)]

/// Default max rows per streamed batch. Tune as needed.
pub const STREAM_BATCH_ROWS: usize = 65_536;

/// Number of constraint records to scan in a single batch when iterating over constraints.
///
/// This value represents a trade-off between memory usage and I/O batching performance:
/// - Larger values: Improve scan throughput by reducing the number of I/O operations,
///   but increase memory usage as more records are held in memory simultaneously.
/// - Smaller values: Reduce memory footprint, but may increase the total number of
///   I/O operations required to scan all constraints.
///
/// The value of 256 was chosen empirically to provide good performance for typical
/// workloads while keeping memory usage reasonable. For tables with many constraints
/// (hundreds or thousands), this batching significantly reduces I/O overhead compared
/// to reading one record at a time.
pub const CONSTRAINT_SCAN_CHUNK_SIZE: usize = 256;

pub use llkv_column_map::store::FIELD_ID_META_KEY;
