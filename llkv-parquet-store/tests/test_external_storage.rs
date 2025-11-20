//! Test external storage for vector embeddings.

use arrow::array::{Array, FixedSizeListArray, Float32Array, UInt64Array};
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
