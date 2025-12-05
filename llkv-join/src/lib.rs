//! High-level join planning API that wraps hash join execution.
//!
//! This crate exposes shared types (`JoinKey`, `JoinType`, `JoinOptions`) used by the
//! planner and runtime to negotiate join configuration. Execution currently routes
//! through the hash join implementation in [`hash_join_rowid_stream`], with a placeholder for
//! alternate algorithms when they land.
#![forbid(unsafe_code)]

mod hash_join;

use llkv_result::{Error, Result as LlkvResult};
use llkv_storage::pager::Pager;
use llkv_table::table::Table;
use llkv_table::types::FieldId;
use simd_r_drive_entry_handle::EntryHandle;
use std::fmt;

pub use hash_join::hash_join_rowid_stream;

/// Type of join to perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoinType {
    /// Emit only matching row pairs.
    Inner,
    /// Emit all left rows; unmatched left rows have NULL right columns.
    Left,
    /// Emit all right rows; unmatched right rows have NULL left columns.
    Right,
    /// Emit all rows from both sides; unmatched rows have NULLs.
    Full,
    /// Emit left rows that have at least one match (no right columns).
    Semi,
    /// Emit left rows that have no match (no right columns).
    Anti,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JoinType::Inner => write!(f, "INNER"),
            JoinType::Left => write!(f, "LEFT"),
            JoinType::Right => write!(f, "RIGHT"),
            JoinType::Full => write!(f, "FULL"),
            JoinType::Semi => write!(f, "SEMI"),
            JoinType::Anti => write!(f, "ANTI"),
        }
    }
}

/// Join key pair describing which columns to equate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoinKey {
    /// Field ID from the left table.
    pub left_field: FieldId,
    /// Field ID from the right table.
    pub right_field: FieldId,
    /// If true, NULL == NULL for this key (SQL-style NULL-safe equality).
    /// If false, NULL != NULL (Arrow default).
    pub null_equals_null: bool,
}

impl JoinKey {
    /// Create a join key with standard Arrow null semantics (NULL != NULL).
    pub fn new(left_field: FieldId, right_field: FieldId) -> Self {
        Self {
            left_field,
            right_field,
            null_equals_null: false,
        }
    }

    /// Create a join key with SQL-style NULL-safe equality (NULL == NULL).
    pub fn null_safe(left_field: FieldId, right_field: FieldId) -> Self {
        Self {
            left_field,
            right_field,
            null_equals_null: true,
        }
    }
}

/// Algorithm to use for join execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum JoinAlgorithm {
    /// Hash join: build hash table on one side, probe with other.
    /// O(N+M) complexity - suitable for production workloads.
    /// Default and recommended for all equality joins.
    #[default]
    Hash,
    /// Sort-merge join: sort both sides, then merge.
    /// Good for pre-sorted inputs or when memory is constrained.
    /// Not yet implemented.
    SortMerge,
}

impl fmt::Display for JoinAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JoinAlgorithm::Hash => write!(f, "Hash"),
            JoinAlgorithm::SortMerge => write!(f, "SortMerge"),
        }
    }
}

/// Options controlling join execution.
#[derive(Clone, Debug)]
pub struct JoinOptions {
    /// Type of join to perform.
    pub join_type: JoinType,
    /// Algorithm to use. Planner may override this based on table sizes.
    pub algorithm: JoinAlgorithm,
    /// Target number of probe rows per output `RecordBatch`.
    /// Larger batches reduce per-batch overhead (fewer Arrow gathers) at the
    /// cost of increased peak memory; smaller batches improve latency.
    pub batch_size: usize,
    /// Memory limit in bytes for hash table (hash join only).
    /// When exceeded, algorithm will partition and spill to disk.
    pub memory_limit_bytes: Option<usize>,
    /// Concurrency hint: number of threads for parallel partitions.
    pub concurrency: usize,
}

/// Reference to a row in a specific batch (batch index, row index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JoinRowRef {
    pub batch: usize,
    pub row: usize,
}

/// Batch of join output row references with access to source batches for zero-copy projection.
pub struct JoinIndexBatch<'a> {
    pub left_rows: Vec<JoinRowRef>,
    pub right_rows: Vec<Option<JoinRowRef>>, // empty for SEMI/ANTI
    pub left_batches: &'a [arrow::record_batch::RecordBatch],
    pub right_batches: &'a [arrow::record_batch::RecordBatch],
}

impl Default for JoinOptions {
    fn default() -> Self {
        Self {
            join_type: JoinType::Inner,
            algorithm: JoinAlgorithm::Hash,
            batch_size: 8192,
            memory_limit_bytes: None,
            concurrency: 1,
        }
    }
}

impl JoinOptions {
    /// Create options for an inner join with default settings.
    pub fn inner() -> Self {
        Self {
            join_type: JoinType::Inner,
            ..Default::default()
        }
    }

    /// Create options for a left outer join with default settings.
    pub fn left() -> Self {
        Self {
            join_type: JoinType::Left,
            ..Default::default()
        }
    }

    /// Create options for a right outer join with default settings.
    pub fn right() -> Self {
        Self {
            join_type: JoinType::Right,
            ..Default::default()
        }
    }

    /// Create options for a full outer join with default settings.
    pub fn full() -> Self {
        Self {
            join_type: JoinType::Full,
            ..Default::default()
        }
    }

    /// Create options for a semi join with default settings.
    pub fn semi() -> Self {
        Self {
            join_type: JoinType::Semi,
            ..Default::default()
        }
    }

    /// Create options for an anti join with default settings.
    pub fn anti() -> Self {
        Self {
            join_type: JoinType::Anti,
            ..Default::default()
        }
    }

    /// Set the join algorithm.
    pub fn with_algorithm(mut self, algorithm: JoinAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the output batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the memory limit for hash joins.
    pub fn with_memory_limit(mut self, limit_bytes: usize) -> Self {
        self.memory_limit_bytes = Some(limit_bytes);
        self
    }

    /// Set the concurrency level.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }
}

// TODO: Build out more fully or remove
// NOTE: Validation presently only asserts that zero keys implies a Cartesian
// join. Extend this once the planner provides richer metadata about key
// compatibility (equality types, null semantics, etc.).
/// Validate join keys before execution.
///
/// Note: Empty keys = cross product (Cartesian product).
pub fn validate_join_keys(_keys: &[JoinKey]) -> LlkvResult<()> {
    // Empty keys is valid for cross product
    Ok(())
}

/// Validate join options before execution.
pub fn validate_join_options(options: &JoinOptions) -> LlkvResult<()> {
    if options.batch_size == 0 {
        return Err(Error::InvalidArgumentError(
            "join batch_size must be > 0".to_string(),
        ));
    }
    if options.concurrency == 0 {
        return Err(Error::InvalidArgumentError(
            "join concurrency must be > 0".to_string(),
        ));
    }
    Ok(())
}

/// Extension trait emitting join results as row-id pairs (zero-copy friendly).
pub trait TableJoinRowIdExt<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn join_rowid_stream<F>(
        &self,
        right: &Table<P>,
        keys: &[JoinKey],
        options: &JoinOptions,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(JoinIndexBatch<'_>);
}

impl<P> TableJoinRowIdExt<P> for Table<P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    fn join_rowid_stream<F>(
        &self,
        right: &Table<P>,
        keys: &[JoinKey],
        options: &JoinOptions,
        on_batch: F,
    ) -> LlkvResult<()>
    where
        F: FnMut(JoinIndexBatch<'_>),
    {
        validate_join_keys(keys)?;
        validate_join_options(options)?;

        match options.algorithm {
            JoinAlgorithm::Hash => {
                hash_join::hash_join_rowid_stream(self, right, keys, options, on_batch)
            }
            JoinAlgorithm::SortMerge => Err(Error::Internal(
                "Sort-merge join not yet implemented; use JoinAlgorithm::Hash".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_key_constructors() {
        let key = JoinKey::new(10, 20);
        assert_eq!(key.left_field, 10);
        assert_eq!(key.right_field, 20);
        assert!(!key.null_equals_null);

        let key_null_safe = JoinKey::null_safe(10, 20);
        assert!(key_null_safe.null_equals_null);
    }

    #[test]
    fn test_join_options_builders() {
        let inner = JoinOptions::inner();
        assert_eq!(inner.join_type, JoinType::Inner);

        let left = JoinOptions::left()
            .with_algorithm(JoinAlgorithm::Hash)
            .with_batch_size(1024)
            .with_memory_limit(1_000_000)
            .with_concurrency(4);
        assert_eq!(left.join_type, JoinType::Left);
        assert_eq!(left.algorithm, JoinAlgorithm::Hash);
        assert_eq!(left.batch_size, 1024);
        assert_eq!(left.memory_limit_bytes, Some(1_000_000));
        assert_eq!(left.concurrency, 4);
    }

    #[test]
    fn test_validate_join_keys() {
        // Empty keys are valid (cross product)
        let empty: Vec<JoinKey> = vec![];
        assert!(validate_join_keys(&empty).is_ok());

        let keys = vec![JoinKey::new(1, 2)];
        assert!(validate_join_keys(&keys).is_ok());
    }

    #[test]
    fn test_validate_join_options() {
        let bad_batch = JoinOptions {
            batch_size: 0,
            ..Default::default()
        };
        assert!(validate_join_options(&bad_batch).is_err());

        let bad_concurrency = JoinOptions {
            concurrency: 0,
            ..Default::default()
        };
        assert!(validate_join_options(&bad_concurrency).is_err());

        let good = JoinOptions::default();
        assert!(validate_join_options(&good).is_ok());
    }

    #[test]
    fn test_join_type_display() {
        assert_eq!(JoinType::Inner.to_string(), "INNER");
        assert_eq!(JoinType::Left.to_string(), "LEFT");
        assert_eq!(JoinType::Right.to_string(), "RIGHT");
        assert_eq!(JoinType::Full.to_string(), "FULL");
        assert_eq!(JoinType::Semi.to_string(), "SEMI");
        assert_eq!(JoinType::Anti.to_string(), "ANTI");
    }

    #[test]
    fn test_join_algorithm_display() {
        assert_eq!(JoinAlgorithm::Hash.to_string(), "Hash");
        assert_eq!(JoinAlgorithm::SortMerge.to_string(), "SortMerge");
    }
}
