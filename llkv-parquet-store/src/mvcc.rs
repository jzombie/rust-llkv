//! MVCC transaction visibility filtering.

use arrow::array::{Array, BooleanArray, UInt64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use llkv_result::{Error, Result};
use std::sync::Arc;

/// Apply MVCC visibility filtering to a RecordBatch.
///
/// Returns only rows visible to the given transaction ID:
/// - `created_by_txn <= txn_id`
/// - `deleted_by_txn IS NULL OR deleted_by_txn > txn_id`
pub fn apply_mvcc_filter(batch: RecordBatch, txn_id: u64) -> Result<RecordBatch> {
    let created_by = batch
        .column_by_name("created_by_txn")
        .ok_or_else(|| Error::InvalidArgumentError("missing created_by_txn column".into()))?;

    let deleted_by = batch
        .column_by_name("deleted_by_txn")
        .ok_or_else(|| Error::InvalidArgumentError("missing deleted_by_txn column".into()))?;

    if !matches!(created_by.data_type(), DataType::UInt64) {
        return Err(Error::InvalidArgumentError(
            "created_by_txn must be UInt64".into(),
        ));
    }

    if !matches!(deleted_by.data_type(), DataType::UInt64) {
        return Err(Error::InvalidArgumentError(
            "deleted_by_txn must be UInt64".into(),
        ));
    }

    let created_arr = created_by.as_any().downcast_ref::<UInt64Array>().unwrap();
    let deleted_arr = deleted_by.as_any().downcast_ref::<UInt64Array>().unwrap();

    // Build filter: created_by <= txn_id AND (deleted_by IS NULL OR deleted_by > txn_id)
    let filter = BooleanArray::from_iter((0..batch.num_rows()).map(|i| {
        let created = created_arr.value(i);
        let visible_created = created <= txn_id;

        let visible_deleted = if deleted_arr.is_null(i) {
            true
        } else {
            deleted_arr.value(i) > txn_id
        };

        Some(visible_created && visible_deleted)
    }));

    // Apply filter
    arrow::compute::filter_record_batch(&batch, &filter)
        .map_err(|e| Error::Internal(format!("failed to apply MVCC filter: {}", e)))
}

/// Deduplicate rows with the same row_id, keeping only the latest version.
///
/// For each row_id, keeps the row with the highest created_by_txn value.
/// This implements Last-Write-Wins (LWW) semantics.
pub fn deduplicate_by_row_id(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(batches);
    }

    use arrow::array::UInt64Array;
    use arrow::compute;
    use rustc_hash::FxHashMap;

    // Collect all rows with their metadata
    let mut row_versions: FxHashMap<u64, (usize, usize, u64)> = FxHashMap::default();

    for (batch_idx, batch) in batches.iter().enumerate() {
        let row_id_col = batch
            .column_by_name("row_id")
            .ok_or_else(|| Error::InvalidArgumentError("missing row_id column".into()))?;
        let created_by_col = batch
            .column_by_name("created_by_txn")
            .ok_or_else(|| Error::InvalidArgumentError("missing created_by_txn column".into()))?;

        let row_ids = row_id_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::InvalidArgumentError("row_id must be UInt64".into()))?;
        let created_by = created_by_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::InvalidArgumentError("created_by_txn must be UInt64".into()))?;

        for row_idx in 0..batch.num_rows() {
            let row_id = row_ids.value(row_idx);
            let txn_id = created_by.value(row_idx);

            // Keep track of the latest version (highest txn_id) for each row_id
            row_versions
                .entry(row_id)
                .and_modify(|(b_idx, r_idx, existing_txn)| {
                    if txn_id > *existing_txn {
                        *b_idx = batch_idx;
                        *r_idx = row_idx;
                        *existing_txn = txn_id;
                    }
                })
                .or_insert((batch_idx, row_idx, txn_id));
        }
    }

    // Build filtered batches keeping only the latest version of each row
    let mut result = Vec::new();

    for (batch_idx, batch) in batches.into_iter().enumerate() {
        let row_id_col = batch.column_by_name("row_id").unwrap();
        let created_by_col = batch.column_by_name("created_by_txn").unwrap();

        let row_ids = row_id_col.as_any().downcast_ref::<UInt64Array>().unwrap();
        let created_by = created_by_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // Build boolean mask for rows to keep
        let mut keep_mask = Vec::with_capacity(batch.num_rows());

        for row_idx in 0..batch.num_rows() {
            let row_id = row_ids.value(row_idx);
            let txn_id = created_by.value(row_idx);

            // Keep this row if it's the latest version
            let is_latest = row_versions
                .get(&row_id)
                .map(|(b_idx, r_idx, latest_txn)| {
                    *b_idx == batch_idx && *r_idx == row_idx && *latest_txn == txn_id
                })
                .unwrap_or(false);

            keep_mask.push(is_latest);
        }

        // Filter the batch
        let filter = BooleanArray::from(keep_mask);
        let filtered = compute::filter_record_batch(&batch, &filter)
            .map_err(|e| Error::Internal(format!("failed to deduplicate: {}", e)))?;

        if filtered.num_rows() > 0 {
            result.push(filtered);
        }
    }

    Ok(result)
}

/// Add MVCC columns to a RecordBatch.
///
/// Adds `created_by_txn` (current transaction) and `deleted_by_txn` (NULL).
pub fn add_mvcc_columns(batch: RecordBatch, txn_id: u64) -> Result<RecordBatch> {
    let num_rows = batch.num_rows();

    // Create MVCC arrays
    let created_by = Arc::new(UInt64Array::from(vec![txn_id; num_rows]));
    let deleted_by = Arc::new(UInt64Array::from(vec![None; num_rows]));

    // Combine with existing columns
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(arrow::datatypes::Field::new(
        "created_by_txn",
        DataType::UInt64,
        false,
    )));
    fields.push(Arc::new(arrow::datatypes::Field::new(
        "deleted_by_txn",
        DataType::UInt64,
        true,
    )));

    let mut columns = batch.columns().to_vec();
    columns.push(created_by);
    columns.push(deleted_by);

    let new_schema = Arc::new(arrow::datatypes::Schema::new(fields));

    RecordBatch::try_new(new_schema, columns)
        .map_err(|e| Error::Internal(format!("failed to add MVCC columns: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn test_apply_mvcc_filter() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("created_by_txn", DataType::UInt64, false),
            Field::new("deleted_by_txn", DataType::UInt64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
                Arc::new(UInt64Array::from(vec![1, 2, 3, 4])), // created_by
                Arc::new(UInt64Array::from(vec![None, Some(3), Some(5), None])), // deleted_by
            ],
        )
        .unwrap();

        // View at txn 3
        let filtered = apply_mvcc_filter(batch, 3).unwrap();

        // Should see rows:
        // - Row 1 (id=1, created=1, deleted=None): visible
        // - Row 2 (id=2, created=2, deleted=3): NOT visible (deleted at txn 3, not > 3)
        // - Row 3 (id=3, created=3, deleted=5): visible
        // - Row 4 (id=4, created=4): NOT visible (created after txn 3)
        assert_eq!(filtered.num_rows(), 2);

        let ids = filtered
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(ids.value(0), 1);
        assert_eq!(ids.value(1), 3);
    }

    #[test]
    fn test_add_mvcc_columns() {
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

        let with_mvcc = add_mvcc_columns(batch, 42).unwrap();

        assert_eq!(with_mvcc.num_columns(), 4);
        assert_eq!(with_mvcc.schema().field(2).name(), "created_by_txn");
        assert_eq!(with_mvcc.schema().field(3).name(), "deleted_by_txn");

        let created = with_mvcc
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(created.value(0), 42);
    }

    #[test]
    fn test_lww_deduplication() {
        use arrow::array::StringArray;

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            Arc::new(Field::new("row_id", DataType::UInt64, false)),
            Arc::new(Field::new("name", DataType::Utf8, false)),
            Arc::new(Field::new("created_by_txn", DataType::UInt64, false)),
            Arc::new(Field::new("deleted_by_txn", DataType::UInt64, true)),
        ]));

        // Batch 1: rows [1,2,3] at txn 1
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice_v1", "Bob_v1", "Charlie_v1"])),
                Arc::new(UInt64Array::from(vec![1, 1, 1])),
                Arc::new(UInt64Array::from(vec![None, None, None])),
            ],
        )
        .unwrap();

        // Batch 2: rows [2,3,4] at txn 2 (updates 2 & 3)
        let batch2 = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![2, 3, 4])),
                Arc::new(StringArray::from(vec!["Bob_v2", "Charlie_v2", "Dave"])),
                Arc::new(UInt64Array::from(vec![2, 2, 2])),
                Arc::new(UInt64Array::from(vec![None, None, None])),
            ],
        )
        .unwrap();

        let result = deduplicate_by_row_id(vec![batch1, batch2]).unwrap();
        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 4);

        // Verify LWW: row 1 kept v1, rows 2&3 updated to v2, row 4 new
        let mut names = Vec::new();
        for batch in &result {
            let name_col = batch
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..name_col.len() {
                names.push(name_col.value(i).to_string());
            }
        }
        names.sort();
        assert_eq!(names, vec!["Alice_v1", "Bob_v2", "Charlie_v2", "Dave"]);
    }
}
