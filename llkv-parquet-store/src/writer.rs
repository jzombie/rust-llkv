//! Parquet file writing utilities.

use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

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
}
