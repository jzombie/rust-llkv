//! Core type definitions for the Parquet store.

use serde::{Deserialize, Serialize};

/// Unique identifier for a table.
///
/// This is a simple wrapper around `u64` to provide type safety and prevent
/// mixing up table IDs with other numeric values like physical keys or row IDs.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    bitcode::Encode,
    bitcode::Decode,
)]
pub struct TableId(pub u64);

impl From<u64> for TableId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<TableId> for u64 {
    fn from(id: TableId) -> Self {
        id.0
    }
}

/// Unique identifier for a Parquet file within the pager.
///
/// This is an alias for the pager's physical key type. Each Parquet file
/// is stored as a single blob in the pager and referenced by its `FileId`.
pub type FileId = llkv_storage::types::PhysicalKey;

/// Statistics for a column within a Parquet file.
///
/// These are extracted from Parquet metadata and used for query pruning.
#[derive(Debug, Clone, Serialize, Deserialize, bitcode::Encode, bitcode::Decode)]
pub struct ColumnStats {
    pub min: Option<Vec<u8>>,
    pub max: Option<Vec<u8>>,
    pub null_count: u64,
    pub distinct_count: Option<u64>,
}
