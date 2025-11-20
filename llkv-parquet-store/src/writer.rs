//! Parquet file writing utilities.

use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Write a RecordBatch to an in-memory Parquet file.
///
/// Returns the serialized Parquet bytes ready to store in the pager.
pub fn write_parquet_to_memory(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();

    // Configure Parquet writer properties
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY) // Fast compression
        .set_max_row_group_size(8192) // Reasonable row group size
        .build();

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
}
