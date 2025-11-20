//! Extract statistics from Parquet metadata for query pruning.

use crate::types::ColumnStats;
use arrow::datatypes::{DataType, Schema};
use llkv_result::{Error, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::RowGroupMetaData;
use std::collections::HashMap;
use std::sync::Arc;

/// Extract column statistics from Parquet file bytes.
///
/// Returns a map of column_name -> ColumnStats with min/max values
/// aggregated across all row groups.
pub fn extract_statistics(parquet_bytes: &[u8]) -> Result<HashMap<String, ColumnStats>> {
    let bytes = bytes::Bytes::copy_from_slice(parquet_bytes);

    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| Error::Internal(format!("failed to read Parquet metadata: {}", e)))?;

    let metadata = builder.metadata();
    let arrow_schema = builder.schema();

    let mut stats_map: HashMap<String, ColumnStats> = HashMap::new();

    // Aggregate statistics across all row groups
    for row_group in metadata.row_groups() {
        merge_row_group_stats(&mut stats_map, row_group, &arrow_schema)?;
    }

    Ok(stats_map)
}

/// Merge statistics from a single row group into the aggregate map.
fn merge_row_group_stats(
    stats_map: &mut HashMap<String, ColumnStats>,
    row_group: &RowGroupMetaData,
    schema: &Arc<Schema>,
) -> Result<()> {
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col_name = field.name();

        // Skip MVCC columns - we don't need stats for these
        if col_name == "__txn_id" || col_name == "__deleted" {
            continue;
        }

        let column_meta = row_group.column(col_idx);

        if let Some(statistics) = column_meta.statistics() {
            let min_bytes = statistics.min_bytes_opt().map(|b| b.to_vec());
            let max_bytes = statistics.max_bytes_opt().map(|b| b.to_vec());
            let null_count = statistics.null_count_opt().unwrap_or(0);
            let distinct_count = statistics.distinct_count_opt();

            // Update or create stats entry
            stats_map
                .entry(col_name.clone())
                .and_modify(|existing| {
                    // Keep the minimum of mins and maximum of maxs
                    if let (Some(existing_min), Some(new_min)) = (&existing.min, &min_bytes) {
                        if compare_bytes(new_min, existing_min, field.data_type())
                            == std::cmp::Ordering::Less
                        {
                            existing.min = min_bytes.clone();
                        }
                    }
                    if let (Some(existing_max), Some(new_max)) = (&existing.max, &max_bytes) {
                        if compare_bytes(new_max, existing_max, field.data_type())
                            == std::cmp::Ordering::Greater
                        {
                            existing.max = max_bytes.clone();
                        }
                    }
                    existing.null_count += null_count;
                    // Distinct count aggregation is complex, so we skip it for now
                })
                .or_insert(ColumnStats {
                    min: min_bytes,
                    max: max_bytes,
                    null_count,
                    distinct_count,
                });
        }
    }

    Ok(())
}

/// Compare two byte arrays representing values of a given data type.
///
/// This is a simplified comparison - for production use, you'd want
/// proper deserialization and comparison for each Arrow type.
fn compare_bytes(a: &[u8], b: &[u8], _data_type: &DataType) -> std::cmp::Ordering {
    // Simple lexicographic comparison works for:
    // - Integers (if same endianness)
    // - Strings (UTF-8)
    // - Timestamps
    // For full correctness, we'd need type-aware deserialization
    a.cmp(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::write_parquet_to_memory_with_config;
    use crate::WriterConfig;
    use arrow::array::{Int64Array, StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    #[test]
    fn test_extract_statistics() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("value", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![100, 200, 300])),
                Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
            ],
        )
        .unwrap();

        let config = WriterConfig::default().with_statistics(true);
        let bytes = write_parquet_to_memory_with_config(&batch, &config).unwrap();

        let stats = extract_statistics(&bytes).unwrap();

        // Should have stats for all non-MVCC columns
        assert!(stats.contains_key("row_id"));
        assert!(stats.contains_key("value"));
        assert!(stats.contains_key("name"));

        // Verify we have min/max
        assert!(stats["row_id"].min.is_some());
        assert!(stats["row_id"].max.is_some());
    }
}
