//! Parquet file writing utilities.

use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Minimum batch size threshold (rows).
/// Batches smaller than this should be merged together to improve storage efficiency.
pub const MIN_BATCH_SIZE: usize = 1024;

/// Target batch size (rows).
/// Batches are merged up to this size, and oversized batches are split to this size.
/// Matches the default row group size for optimal Parquet performance.
pub const TARGET_BATCH_SIZE: usize = 8192;

/// Configuration for Parquet file writing.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Compression algorithm to use.
    pub compression: Compression,
    /// Maximum number of rows per row group.
    pub max_row_group_size: usize,
    /// Enable statistics collection.
    pub enable_statistics: bool,
    /// Enable dictionary encoding.
    pub enable_dictionary: bool,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            compression: Compression::SNAPPY, // Fast compression
            max_row_group_size: 8192,
            enable_statistics: true,
            enable_dictionary: true,
        }
    }
}

impl WriterConfig {
    /// Create configuration with no compression (fastest writes).
    pub fn uncompressed() -> Self {
        Self {
            compression: Compression::UNCOMPRESSED,
            ..Default::default()
        }
    }

    /// Create configuration with Snappy compression (balanced).
    pub fn snappy() -> Self {
        Self {
            compression: Compression::SNAPPY,
            ..Default::default()
        }
    }

    /// Create configuration with Zstd compression (best compression).
    pub fn zstd() -> Self {
        Self {
            compression: Compression::ZSTD(parquet::basic::ZstdLevel::default()),
            ..Default::default()
        }
    }

    /// Create configuration with LZ4 compression (fast).
    pub fn lz4() -> Self {
        Self {
            compression: Compression::LZ4,
            ..Default::default()
        }
    }

    /// Create configuration with Gzip compression (good compatibility).
    pub fn gzip() -> Self {
        Self {
            compression: Compression::GZIP(parquet::basic::GzipLevel::default()),
            ..Default::default()
        }
    }

    /// Set compression algorithm.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set maximum row group size.
    pub fn with_max_row_group_size(mut self, size: usize) -> Self {
        self.max_row_group_size = size;
        self
    }

    /// Enable or disable statistics collection.
    pub fn with_statistics(mut self, enable: bool) -> Self {
        self.enable_statistics = enable;
        self
    }

    /// Enable or disable dictionary encoding.
    pub fn with_dictionary(mut self, enable: bool) -> Self {
        self.enable_dictionary = enable;
        self
    }

    /// Convert to Parquet WriterProperties.
    pub(crate) fn to_writer_properties(&self) -> WriterProperties {
        let mut builder = WriterProperties::builder()
            .set_compression(self.compression)
            .set_max_row_group_size(self.max_row_group_size);

        if self.enable_statistics {
            builder =
                builder.set_statistics_enabled(parquet::file::properties::EnabledStatistics::Page);
        }

        if self.enable_dictionary {
            builder = builder.set_dictionary_enabled(true);
        }

        builder.build()
    }
}

/// Write a RecordBatch to an in-memory Parquet file with default configuration.
///
/// Returns the serialized Parquet bytes ready to store in the pager.
pub fn write_parquet_to_memory(batch: &RecordBatch) -> Result<Vec<u8>> {
    write_parquet_to_memory_with_config(batch, &WriterConfig::default())
}

/// Write a RecordBatch to an in-memory Parquet file with custom configuration.
///
/// Returns the serialized Parquet bytes ready to store in the pager.
pub fn write_parquet_to_memory_with_config(
    batch: &RecordBatch,
    config: &WriterConfig,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();

    let props = config.to_writer_properties();

    // Use synchronous ArrowWriter for simplicity
    let mut writer = ArrowWriter::try_new(&mut buffer, batch.schema(), Some(props))
        .map_err(|e| Error::Internal(format!("failed to create Parquet writer: {}", e)))?;

    writer
        .write(batch)
        .map_err(|e| Error::Internal(format!("failed to write RecordBatch to Parquet: {}", e)))?;

    writer
        .close()
        .map_err(|e| Error::Internal(format!("failed to close Parquet writer: {}", e)))?;

    Ok(buffer)
}

/// Merge small batches together and split large batches to achieve TARGET_BATCH_SIZE.
///
/// This function optimizes batch sizes for storage efficiency:
/// - Small batches (< MIN_BATCH_SIZE) are merged together
/// - Large batches (> TARGET_BATCH_SIZE) are split into chunks
/// - Batches near TARGET_BATCH_SIZE are left as-is
///
/// All batches must have the same schema.
pub fn optimize_batch_sizes(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let schema = batches[0].schema();

    // Verify all batches have the same schema
    for batch in &batches[1..] {
        if batch.schema() != schema {
            return Err(Error::InvalidArgumentError(
                "All batches must have the same schema".into(),
            ));
        }
    }

    let mut optimized = Vec::new();
    let mut accumulator: Option<RecordBatch> = None;

    for batch in batches {
        let mut current_batch = batch;

        // If we have an accumulator, try to merge with it
        if let Some(acc) = accumulator.take() {
            let combined_rows = acc.num_rows() + current_batch.num_rows();

            // If combining would still be under or near target, merge them
            if combined_rows <= TARGET_BATCH_SIZE {
                current_batch = arrow::compute::concat_batches(&schema, &[acc, current_batch])
                    .map_err(|e| Error::Arrow(e))?;
            } else {
                // Accumulator is full enough, flush it
                // If accumulator is below MIN_BATCH_SIZE but would exceed TARGET when combined,
                // still flush it (trade-off: small batch vs oversized batch)
                if acc.num_rows() >= MIN_BATCH_SIZE || combined_rows > TARGET_BATCH_SIZE * 2 {
                    optimized.push(acc);
                } else {
                    // Merge anyway, will split below if needed
                    current_batch = arrow::compute::concat_batches(&schema, &[acc, current_batch])
                        .map_err(|e| Error::Arrow(e))?;
                }
            }
        }

        // Split oversized batches
        if current_batch.num_rows() > TARGET_BATCH_SIZE {
            let num_chunks = (current_batch.num_rows() + TARGET_BATCH_SIZE - 1) / TARGET_BATCH_SIZE;
            let chunk_size = (current_batch.num_rows() + num_chunks - 1) / num_chunks;

            let mut offset = 0;
            while offset < current_batch.num_rows() {
                let length = chunk_size.min(current_batch.num_rows() - offset);
                let chunk = current_batch.slice(offset, length);

                if chunk.num_rows() < MIN_BATCH_SIZE && offset + length < current_batch.num_rows() {
                    // This chunk is too small and there's more data, accumulate it
                    accumulator = Some(chunk);
                } else {
                    optimized.push(chunk);
                }

                offset += length;
            }
        } else if current_batch.num_rows() <= MIN_BATCH_SIZE {
            // Batch is too small (or exactly at minimum), accumulate for merging
            accumulator = Some(current_batch);
        } else {
            // Batch is an acceptable size, keep it
            optimized.push(current_batch);
        }
    }

    // Flush any remaining accumulator
    if let Some(acc) = accumulator {
        optimized.push(acc);
    }

    Ok(optimized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_write_parquet_to_memory() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let bytes = write_parquet_to_memory(&batch).unwrap();

        // Verify Parquet magic number
        assert_eq!(&bytes[0..4], b"PAR1");
        assert!(bytes.len() > 100); // Should have some content
    }

    #[test]
    fn test_writer_config_presets() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![1, 2, 3]))]).unwrap();

        // Test configs that should work without optional features
        let uncompressed =
            write_parquet_to_memory_with_config(&batch, &WriterConfig::uncompressed()).unwrap();
        let snappy = write_parquet_to_memory_with_config(&batch, &WriterConfig::snappy()).unwrap();

        // All should be valid Parquet
        assert_eq!(&uncompressed[0..4], b"PAR1");
        assert_eq!(&snappy[0..4], b"PAR1");

        println!(
            "Sizes: uncompressed={}, snappy={}",
            uncompressed.len(),
            snappy.len()
        );
    }

    #[test]
    fn test_custom_config() {
        let schema = Arc::new(Schema::new(vec![Field::new("data", DataType::Utf8, false)]));

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["test"; 100]))])
                .unwrap();

        // Use compression that works without optional features
        let config = WriterConfig::default()
            .with_compression(Compression::SNAPPY)
            .with_max_row_group_size(50)
            .with_statistics(true);

        let bytes = write_parquet_to_memory_with_config(&batch, &config).unwrap();
        assert_eq!(&bytes[0..4], b"PAR1");
    }

    #[test]
    fn test_optimize_batch_sizes_empty() {
        let result = optimize_batch_sizes(Vec::new()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_optimize_batch_sizes_merge_small() {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));

        // Create 3 small batches (512 rows each, total 1536)
        let batches: Vec<RecordBatch> = (0..3)
            .map(|i| {
                let array = UInt64Array::from_iter_values(i * 512..(i + 1) * 512);
                RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap()
            })
            .collect();

        let optimized = optimize_batch_sizes(batches).unwrap();

        // Should merge into 1 batch of 1536 rows (below TARGET_BATCH_SIZE)
        assert_eq!(optimized.len(), 1);
        assert_eq!(optimized[0].num_rows(), 1536);
    }

    #[test]
    fn test_optimize_batch_sizes_split_large() {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));

        // Create 1 large batch (20000 rows)
        let array = UInt64Array::from_iter_values(0..20000);
        let large_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();

        let optimized = optimize_batch_sizes(vec![large_batch]).unwrap();

        // Should split into multiple batches
        assert!(optimized.len() >= 2);

        // Each batch should be reasonably sized
        for batch in &optimized {
            assert!(
                batch.num_rows() >= MIN_BATCH_SIZE
                    || batch.num_rows() == optimized.last().unwrap().num_rows()
            );
        }

        // Total rows should be preserved
        let total_rows: usize = optimized.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 20000);
    }

    #[test]
    fn test_optimize_batch_sizes_mixed() {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));

        let mut batches = Vec::new();

        // Small batch (512 rows)
        batches.push(
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(UInt64Array::from_iter_values(0..512))],
            )
            .unwrap(),
        );

        // Medium batch (5000 rows) - acceptable size
        batches.push(
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(UInt64Array::from_iter_values(512..5512))],
            )
            .unwrap(),
        );

        // Large batch (15000 rows) - should be split
        batches.push(
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(UInt64Array::from_iter_values(5512..20512))],
            )
            .unwrap(),
        );

        let optimized = optimize_batch_sizes(batches).unwrap();

        // Should have multiple batches
        assert!(optimized.len() >= 2);

        // Total rows preserved
        let total_rows: usize = optimized.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 20512);
    }
}
