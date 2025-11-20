//! Parquet file reading utilities.

use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use llkv_result::{Error, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Read RecordBatches from Parquet bytes.
///
/// Optionally applies column projection. Accepts `bytes::Bytes` for zero-copy
/// operation when backed by mmap.
pub fn read_parquet_from_memory(
    bytes: Bytes,
    projection: Option<Vec<usize>>,
) -> Result<Vec<RecordBatch>> {
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| Error::Internal(format!("failed to create Parquet reader: {}", e)))?;

    if let Some(proj) = projection {
        let schema_desc = builder.parquet_schema();
        let mask = parquet::arrow::ProjectionMask::leaves(schema_desc, proj);
        builder = builder.with_projection(mask);
    }

    let reader = builder
        .build()
        .map_err(|e| Error::Internal(format!("failed to build Parquet reader: {}", e)))?;

    let mut batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| Error::Internal(format!("failed to read Parquet batch: {}", e)))?;
        batches.push(batch);
    }

    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::write_parquet_to_memory;
    use arrow::array::{StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_read_parquet_from_memory() {
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
        let batches = read_parquet_from_memory(Bytes::from(bytes), None).unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[0].num_columns(), 2);
    }

    #[test]
    fn test_read_with_projection() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(UInt64Array::from(vec![10, 20])),
            ],
        )
        .unwrap();

        let bytes = write_parquet_to_memory(&batch).unwrap();

        // Project only columns 0 and 2
        let batches = read_parquet_from_memory(Bytes::from(bytes), Some(vec![0, 2])).unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 2);
    }
}
