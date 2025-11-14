use super::config::ColumnStoreConfig;
use super::constants::TARGET_CHUNK_BYTES;
use crate::types::RowId;

/// Heuristics that guide callers when sizing write batches for the column store.
///
/// The values are derived from the store's ingest configuration so higher layers can
/// adapt without duplicating storage-level constants. Callers should treat these
/// numbers as soft targets: exceeding the maximum batch rows will be clamped, but
/// smaller batches are always accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnStoreWriteHints {
    /// Target chunk size used when splitting incoming arrays.
    pub target_chunk_bytes: usize,
    /// Preferred number of rows to buffer per insert before flushing.
    pub recommended_insert_batch_rows: usize,
    /// Hard ceiling for literal INSERT batches before storage splits them eagerly.
    pub max_insert_batch_rows: usize,
    /// Fallback slice size for exotic variable-width arrays lacking offset metadata.
    pub varwidth_fallback_rows_per_slice: usize,
}

impl ColumnStoreWriteHints {
    pub(crate) fn from_config(cfg: &ColumnStoreConfig) -> Self {
        let row_id_width = std::mem::size_of::<RowId>().max(1);
        let recommended_rows = (TARGET_CHUNK_BYTES / row_id_width).max(1);
        let max_rows = recommended_rows.saturating_mul(4).max(recommended_rows);
        Self {
            target_chunk_bytes: TARGET_CHUNK_BYTES,
            recommended_insert_batch_rows: recommended_rows,
            max_insert_batch_rows: max_rows,
            varwidth_fallback_rows_per_slice: cfg.varwidth_fallback_rows_per_slice,
        }
    }

    /// Clamp a requested batch size to the store's supported envelope.
    pub fn clamp_insert_batch_rows(&self, requested_rows: usize) -> usize {
        match requested_rows {
            0 => 0,
            _ => requested_rows.min(self.max_insert_batch_rows),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_uses_target_chunk_bytes() {
        let cfg = ColumnStoreConfig {
            varwidth_fallback_rows_per_slice: 2048,
        };
        let hints = ColumnStoreWriteHints::from_config(&cfg);
        let expected_rows = TARGET_CHUNK_BYTES / std::mem::size_of::<RowId>();
        assert_eq!(hints.target_chunk_bytes, TARGET_CHUNK_BYTES);
        assert_eq!(hints.varwidth_fallback_rows_per_slice, 2048);
        assert_eq!(hints.recommended_insert_batch_rows, expected_rows);
        assert!(hints.max_insert_batch_rows >= hints.recommended_insert_batch_rows);
    }

    #[test]
    fn clamp_insert_batch_rows_caps_large_values() {
        let cfg = ColumnStoreConfig {
            varwidth_fallback_rows_per_slice: 512,
        };
        let hints = ColumnStoreWriteHints::from_config(&cfg);
        assert_eq!(hints.clamp_insert_batch_rows(0), 0);
        let max = hints.max_insert_batch_rows;
        assert_eq!(hints.clamp_insert_batch_rows(max * 10), max);
        assert_eq!(hints.clamp_insert_batch_rows(max - 1), max - 1);
    }
}
