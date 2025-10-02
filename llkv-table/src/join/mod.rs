//! Table join operations.
//!
//! This module provides join primitives for combining two tables based on
//! equality predicates over one or more columns. The implementation is
//! designed to support both immediate execution and integration with query
//! planners and SQL frontends.
//!
//! ## Design Philosophy
//!
//! - **Streaming-first**: Joins yield batches incrementally rather than
//!   materializing entire results, keeping memory usage bounded.
//! - **Frontend-agnostic**: Types and APIs are independent of SQL/DDL syntax;
//!   planners can map logical join nodes to these physical operators.
//! - **Extensible algorithms**: The `JoinAlgorithm` enum allows choosing
//!   between nested-loop (correctness-focused) and hash join (performance)
//!   without changing the call site.
//!
//! ## Join Types
//!
//! - **Inner**: Emit only matching row pairs.
//! - **Left**: Emit all left rows; unmatched left rows have NULL right columns.
//! - **Right**: Emit all right rows; unmatched right rows have NULL left columns.
//! - **Full**: Emit all rows from both sides, NULLs for unmatched.
//! - **Semi**: Emit left rows that have at least one match (no right columns).
//! - **Anti**: Emit left rows that have no match (no right columns).
//!
//! ## Null Semantics
//!
//! By default, NULL != NULL for join equality (Arrow semantics). Set
//! `JoinKey::null_equals_null` to true for SQL-style NULL-safe equality.
//!
//! ## Example
//!
//! ```ignore
//! use llkv_table::join::{JoinType, JoinOptions, JoinKey};
//! use llkv_table::Table;
//!
//! let left = Table::new(1, pager.clone())?;
//! let right = Table::new(2, pager.clone())?;
//!
//! let keys = vec![JoinKey {
//!     left_field: 10,
//!     right_field: 20,
//!     null_equals_null: false,
//! }];
//!
//! let options = JoinOptions {
//!     join_type: JoinType::Inner,
//!     ..Default::default()
//! };
//!
//! left.join_stream(&right, &keys, &options, |batch| {
//!     // Process each output batch
//!     println!("Batch: {} rows", batch.num_rows());
//! })?;
//! ```

pub(crate) mod hash_join;

use crate::types::FieldId;
use llkv_result::{Error, Result as LlkvResult};
use std::fmt;

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
    /// Batch size for output RecordBatches.
    pub batch_size: usize,
    /// Memory limit in bytes for hash table (hash join only).
    /// When exceeded, algorithm will partition and spill to disk.
    pub memory_limit_bytes: Option<usize>,
    /// Concurrency hint: number of threads for parallel partitions.
    pub concurrency: usize,
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

/// Validate join keys before execution.
pub(crate) fn validate_join_keys(keys: &[JoinKey]) -> LlkvResult<()> {
    if keys.is_empty() {
        return Err(Error::InvalidArgumentError(
            "join requires at least one key pair".to_string(),
        ));
    }
    Ok(())
}

/// Validate join options before execution.
pub(crate) fn validate_join_options(options: &JoinOptions) -> LlkvResult<()> {
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
        let empty: Vec<JoinKey> = vec![];
        assert!(validate_join_keys(&empty).is_err());

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
