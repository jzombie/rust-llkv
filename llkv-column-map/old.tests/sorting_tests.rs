use arrow::array::{Array, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::{ColumnStore, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::rng;
use rand::seq::SliceRandom;

use std::collections::HashMap;
use std::sync::Arc;

/// Test helper to create a standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

#[test]
fn test_large_sort_u64() {
    const NUM_ROWS: usize = 1_000_000;
    const NUM_BATCHES: usize = 10;
    const BATCH_SIZE: usize = NUM_ROWS / NUM_BATCHES;
    let field_id = fid(301);

    // --- 1. Setup ---
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let mut metadata = HashMap::new();
    metadata.insert("field_id".to_string(), u64::from(field_id).to_string());

    let data_field = Field::new("data", DataType::UInt64, false).with_metadata(metadata);
    let row_id_field = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let schema = Arc::new(Schema::new(vec![row_id_field, data_field]));

    // --- 2. Generate and ingest unsorted data ---
    println!("Generating and ingesting {} shuffled u64 rows...", NUM_ROWS);
    let mut data: Vec<u64> = (0..NUM_ROWS as u64).collect();
    data.shuffle(&mut rng());

    for i in 0..NUM_BATCHES {
        let start = i * BATCH_SIZE;
        let end = start + BATCH_SIZE;

        // row_id must be present; keep unique to avoid LWW deletes.
        let rid_vec: Vec<u64> = (start as u64..end as u64).collect();
        let rid_array = Arc::new(UInt64Array::from(rid_vec));

        let batch_data = &data[start..end];
        let array = Arc::new(UInt64Array::from(batch_data.to_vec()));

        let batch = RecordBatch::try_new(schema.clone(), vec![rid_array, array]).unwrap();
        store.append(&batch).unwrap();
    }
    println!("Ingestion complete.");

    // --- 3. Create the sort index ---
    println!("Creating sort index for u64 column...");
    store.create_sort_index(field_id).unwrap();
    println!("Sort index created.");

    // --- 4. Scan in sorted order and verify ---
    println!("Scanning u64 column in sorted order...");
    let mut merge = store.scan_sorted(field_id).unwrap();

    let mut collected_results = Vec::with_capacity(NUM_ROWS);
    while let Some((arr_dyn, start, len)) = merge.next_run() {
        let a = arr_dyn.as_any().downcast_ref::<UInt64Array>().unwrap();
        let end = start + len;
        for i in start..end { collected_results.push(a.value(i)); }
    }
    println!("Scan complete.");

    // --- 5. Final assertion ---
    let expected_sorted_data: Vec<u64> = (0..NUM_ROWS as u64).collect();
    assert_eq!(collected_results.len(), NUM_ROWS);
    assert_eq!(collected_results, expected_sorted_data);
    println!("U64 Verification successful!");
}

#[test]
fn test_large_sort_i32() {
    const NUM_ROWS: usize = 500_000;
    const NUM_BATCHES: usize = 5;
    const BATCH_SIZE: usize = NUM_ROWS / NUM_BATCHES;
    let field_id = fid(302);

    // --- 1. Setup ---
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let mut metadata = HashMap::new();
    metadata.insert("field_id".to_string(), u64::from(field_id).to_string());

    let data_field = Field::new("data", DataType::Int32, false).with_metadata(metadata);
    let row_id_field = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let schema = Arc::new(Schema::new(vec![row_id_field, data_field]));

    // --- 2. Generate and ingest unsorted data (with negatives) ---
    println!("Generating and ingesting {} shuffled i32 rows...", NUM_ROWS);
    let mut data: Vec<i32> = (0..NUM_ROWS as i32)
        .map(|i| i - (NUM_ROWS as i32 / 2))
        .collect();
    data.shuffle(&mut rng());

    for i in 0..NUM_BATCHES {
        let start = i * BATCH_SIZE;
        let end = start + BATCH_SIZE;

        // row_id must be present; keep unique to avoid LWW deletes.
        let rid_vec: Vec<u64> = (start as u64..end as u64).collect();
        let rid_array = Arc::new(UInt64Array::from(rid_vec));

        let batch_data = &data[start..end];
        let array = Arc::new(Int32Array::from(batch_data.to_vec()));

        let batch = RecordBatch::try_new(schema.clone(), vec![rid_array, array]).unwrap();
        store.append(&batch).unwrap();
    }
    println!("Ingestion complete.");

    // --- 3. Create the sort index ---
    println!("Creating sort index for i32 column...");
    store.create_sort_index(field_id).unwrap();
    println!("Sort index created.");

    // --- 4. Scan in sorted order and verify ---
    println!("Scanning i32 column in sorted order...");
    let mut merge = store.scan_sorted(field_id).unwrap();

    let mut collected_results = Vec::with_capacity(NUM_ROWS);
    while let Some((arr_dyn, start, len)) = merge.next_run() {
        let a = arr_dyn.as_any().downcast_ref::<Int32Array>().unwrap();
        let end = start + len;
        for i in start..end { collected_results.push(a.value(i)); }
    }
    println!("Scan complete.");

    // --- 5. Final assertion ---
    let mut expected_sorted_data: Vec<i32> = (0..NUM_ROWS as i32)
        .map(|i| i - (NUM_ROWS as i32 / 2))
        .collect();
    expected_sorted_data.sort();

    assert_eq!(collected_results.len(), NUM_ROWS);
    assert_eq!(collected_results, expected_sorted_data);
    println!("I32 Verification successful!");
}
