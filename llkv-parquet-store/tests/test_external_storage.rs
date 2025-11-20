//! Test external storage for vector embeddings.

use arrow::array::{Array, BinaryArray, FixedSizeListArray, Float32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_parquet_store::{
    add_mvcc_columns, ParquetStore, EXTERNAL_STORAGE_KEY, EXTERNAL_STORAGE_VALUE,
};
use llkv_result::Result;
use llkv_storage::pager::{MemPager, Pager};
use std::collections::HashMap;
use std::sync::Arc;

const VECTOR_DIM: usize = 128;

fn create_external_vector_schema() -> Arc<Schema> {
    let mut metadata = HashMap::new();
    metadata.insert(
        EXTERNAL_STORAGE_KEY.to_string(),
        EXTERNAL_STORAGE_VALUE.to_string(),
    );

    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                VECTOR_DIM as i32,
            ),
            false,
        )
        .with_metadata(metadata),
    ]))
}

fn generate_vector_batch(schema: Arc<Schema>, start_id: u64, num_vectors: usize) -> RecordBatch {
    let ids: Vec<u64> = (start_id..start_id + num_vectors as u64).collect();
    let id_array = UInt64Array::from(ids);

    let total_values = num_vectors * VECTOR_DIM;
    let values: Vec<f32> = (0..total_values).map(|i| i as f32).collect();
    let values_array = Float32Array::from(values);

    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let embedding_array =
        FixedSizeListArray::new(field, VECTOR_DIM as i32, Arc::new(values_array), None);

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(embedding_array)]).unwrap()
}

#[test]
fn test_external_storage_write_and_read() -> Result<()> {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(pager)?;

    let schema = create_external_vector_schema();
    let table_id = store.create_table("vectors_external", schema.clone())?;

    // Write a batch
    let batch = generate_vector_batch(schema.clone(), 0, 100);
    let batch_with_mvcc = add_mvcc_columns(batch, 1)?;
    store.append_many(table_id, vec![batch_with_mvcc])?;

    // Read back
    let batches: Vec<_> = store
        .scan(table_id, &[], None, None)?
        .collect::<Result<Vec<_>>>()?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 100);

    // Verify embedding column
    let embedding_col = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("Expected FixedSizeListArray");

    assert_eq!(embedding_col.len(), 100);

    // Check first vector values
    let values = embedding_col
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("Expected Float32Array");

    let first_vec: Vec<f32> = (0..VECTOR_DIM).map(|i| values.value(i)).collect();
    let expected: Vec<f32> = (0..VECTOR_DIM).map(|i| i as f32).collect();

    assert_eq!(first_vec, expected, "First vector values don't match");

    Ok(())
}

#[test]
fn test_external_storage_multiple_batches() -> Result<()> {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(pager)?;

    let schema = create_external_vector_schema();
    let table_id = store.create_table("vectors_external", schema.clone())?;

    // Write multiple batches
    let mut batches = Vec::new();
    for i in 0..5 {
        let batch = generate_vector_batch(schema.clone(), i * 50, 50);
        let batch_with_mvcc = add_mvcc_columns(batch, 1)?;
        batches.push(batch_with_mvcc);
    }
    store.append_many(table_id, batches)?;

    // Read back
    let read_batches: Vec<_> = store
        .scan(table_id, &[], None, None)?
        .collect::<Result<Vec<_>>>()?;

    let total_rows: usize = read_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 250);

    Ok(())
}

#[test]
fn test_external_storage_gc() -> Result<()> {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(Arc::clone(&pager))?;

    let schema = create_external_vector_schema();
    let table_id = store.create_table("vectors_external", schema.clone())?;

    // Write some vectors
    let batch = generate_vector_batch(schema.clone(), 0, 10);
    let batch_with_mvcc = add_mvcc_columns(batch, 1)?;
    store.append_many(table_id, vec![batch_with_mvcc])?;

    // Count keys before GC (should include catalog, parquet file, and external blobs)
    let all_keys_before = pager.enumerate_keys()?;
    assert!(
        all_keys_before.len() > 11,
        "Expected at least catalog + file + 10 blobs"
    );

    // Collect reachable keys - should include external blob keys
    let reachable = store.collect_reachable_keys()?;
    assert!(
        reachable.len() > 11,
        "Reachable should include external blobs"
    );

    // Run GC - should free nothing since everything is reachable
    let freed = store.garbage_collect()?;
    assert_eq!(freed, 0, "No keys should be freed");

    // Drop the table - frees Parquet file immediately but NOT external blobs
    store.drop_table("vectors_external")?;

    // Run GC again - should free only the external blobs (Parquet file already freed by drop_table)
    let freed = store.garbage_collect()?;
    assert_eq!(
        freed, 10,
        "Should free exactly 10 external blobs, got {}",
        freed
    );

    // Only catalog key should remain
    let all_keys_after = pager.enumerate_keys()?;
    assert_eq!(all_keys_after.len(), 1, "Only catalog key should remain");
    assert_eq!(all_keys_after[0], 0, "Remaining key should be catalog (0)");

    Ok(())
}

#[test]
fn test_external_storage_lww_deduplication() -> Result<()> {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(pager)?;

    // Simple schema: row_id + binary value (stored externally)
    let mut metadata = HashMap::new();
    metadata.insert(
        EXTERNAL_STORAGE_KEY.to_string(),
        EXTERNAL_STORAGE_VALUE.to_string(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("value", DataType::Binary, false).with_metadata(metadata),
    ]));

    let table_id = store.create_table("lww_test", schema.clone())?;

    // Insert 10k rows (0..10_000) in batches of 1k
    // Each value is just the row_id encoded as bytes
    let batch_size = 1_000;
    for batch_start in (0u64..10_000).step_by(batch_size) {
        let ids: Vec<u64> = (batch_start..batch_start + batch_size as u64).collect();
        let values: Vec<Vec<u8>> = ids.iter().map(|id| id.to_le_bytes().to_vec()).collect();
        let values_ref: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(BinaryArray::from_vec(values_ref)),
            ],
        )?;
        store.append_many(table_id, vec![add_mvcc_columns(batch, 1)?])?;
    }

    // Update row_id 42 -> value [99, 99, ...]
    let val42 = vec![99u8; 8];
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![42])),
            Arc::new(BinaryArray::from_vec(vec![val42.as_slice()])),
        ],
    )?;
    store.append_many(table_id, vec![add_mvcc_columns(batch, 2)?])?;

    // Update row_id 5_000 -> value [88, 88, ...]
    let val5k = vec![88u8; 8];
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![5_000])),
            Arc::new(BinaryArray::from_vec(vec![val5k.as_slice()])),
        ],
    )?;
    store.append_many(table_id, vec![add_mvcc_columns(batch, 3)?])?;

    // Update row_id 9_999 -> value [77, 77, ...]
    let val9k = vec![77u8; 8];
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![9_999])),
            Arc::new(BinaryArray::from_vec(vec![val9k.as_slice()])),
        ],
    )?;
    store.append_many(table_id, vec![add_mvcc_columns(batch, 4)?])?;

    // Scan and verify
    let batches: Vec<_> = store
        .scan(table_id, &[], None, None)?
        .collect::<Result<Vec<_>>>()?;

    // Should still have exactly 10k rows (no duplicates)
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 10_000, "Should have exactly 10k unique rows");

    // Collect all rows into a map
    let mut row_map = std::collections::HashMap::new();
    for batch in &batches {
        let row_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            row_map.insert(row_ids.value(i), values.value(i).to_vec());
        }
    }

    assert_eq!(row_map.len(), 10_000, "Should have 10k unique row_ids");

    // Verify updated values
    assert_eq!(
        row_map[&42],
        vec![99; 8],
        "row_id 42 should have updated value"
    );
    assert_eq!(
        row_map[&5_000],
        vec![88; 8],
        "row_id 5_000 should have updated value"
    );
    assert_eq!(
        row_map[&9_999],
        vec![77; 8],
        "row_id 9_999 should have updated value"
    );

    // Verify non-updated values remain original
    assert_eq!(
        row_map[&0],
        0u64.to_le_bytes().to_vec(),
        "row_id 0 should be unchanged"
    );
    assert_eq!(
        row_map[&100],
        100u64.to_le_bytes().to_vec(),
        "row_id 100 should be unchanged"
    );
    assert_eq!(
        row_map[&9_998],
        9_998u64.to_le_bytes().to_vec(),
        "row_id 9_998 should be unchanged"
    );

    Ok(())
}

#[test]
fn test_external_storage_lww_mixed_row_ids() -> Result<()> {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(pager)?;

    let schema = create_external_vector_schema();
    let table_id = store.create_table("vectors_lww_mixed", schema.clone())?;

    let field = Arc::new(Field::new("item", DataType::Float32, false));

    // Write row_id=1,2,3 at txn_id=1
    let ids1 = vec![1u64, 2, 3];
    let values1: Vec<f32> = (0..3 * VECTOR_DIM).map(|i| i as f32).collect();
    let values_array1 = Float32Array::from(values1);
    let embedding_array1 = FixedSizeListArray::new(
        field.clone(),
        VECTOR_DIM as i32,
        Arc::new(values_array1),
        None,
    );
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(ids1)),
            Arc::new(embedding_array1),
        ],
    )
    .unwrap();
    let batch1_with_mvcc = add_mvcc_columns(batch1, 1)?;
    store.append_many(table_id, vec![batch1_with_mvcc])?;

    // Update ONLY row_id=2 at txn_id=2 with new values [5000, 5001, ...]
    let ids2 = vec![2u64];
    let values2: Vec<f32> = (5000..5000 + VECTOR_DIM).map(|i| i as f32).collect();
    let values_array2 = Float32Array::from(values2);
    let embedding_array2 = FixedSizeListArray::new(
        field.clone(),
        VECTOR_DIM as i32,
        Arc::new(values_array2),
        None,
    );
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(ids2)),
            Arc::new(embedding_array2),
        ],
    )
    .unwrap();
    let batch2_with_mvcc = add_mvcc_columns(batch2, 2)?;
    store.append_many(table_id, vec![batch2_with_mvcc])?;

    // Read back - should get exactly 3 rows (1, 2, 3) with row_id=2 having updated values
    let batches: Vec<_> = store
        .scan(table_id, &[], None, None)?
        .collect::<Result<Vec<_>>>()?;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3, "Should have 3 rows after LWW");

    // Collect all row_ids
    let mut all_row_ids = Vec::new();
    for batch in &batches {
        let row_id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        for i in 0..row_id_col.len() {
            all_row_ids.push(row_id_col.value(i));
        }
    }
    all_row_ids.sort();
    assert_eq!(
        all_row_ids,
        vec![1, 2, 3],
        "Should have unique row_ids 1, 2, 3"
    );

    // Find row_id=2 and verify it has the updated embedding
    for batch in &batches {
        let row_id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let embedding_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();

        for i in 0..row_id_col.len() {
            if row_id_col.value(i) == 2 {
                // Verify this is the updated version (starting with 5000)
                let values = embedding_col
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap();
                let start_idx = i * VECTOR_DIM;
                let vec: Vec<f32> = (start_idx..start_idx + VECTOR_DIM)
                    .map(|j| values.value(j))
                    .collect();
                let expected: Vec<f32> = (5000..5000 + VECTOR_DIM).map(|v| v as f32).collect();
                assert_eq!(
                    vec, expected,
                    "row_id=2 should have updated values starting with 5000"
                );
                return Ok(());
            }
        }
    }

    panic!("row_id=2 not found in results");
}
