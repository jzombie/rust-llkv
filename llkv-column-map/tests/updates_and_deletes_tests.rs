use arrow::array::{Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper function to scan all chunks of a column and collect results into a Vec<String>.
fn scan_and_collect_all(store: &ColumnStore<MemPager>, field_id: u64) -> Vec<String> {
    store
        .scan(field_id)
        .unwrap()
        .flat_map(|res| {
            let array = res.unwrap();
            let str_array = array.as_any().downcast_ref::<StringArray>().unwrap();
            str_array
                .into_iter()
                .map(|opt| opt.unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn test_deletes_and_updates() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = 201;

    // --- 1. Initial Ingestion ---
    let mut metadata = HashMap::new();
    metadata.insert("field_id".to_string(), field_id.to_string());
    let field = Field::new("data", DataType::Utf8, false).with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));

    let initial_array = Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
    let initial_batch = RecordBatch::try_new(schema.clone(), vec![initial_array]).unwrap();
    store.append(&initial_batch).unwrap();

    // --- 2. Verify Initial State ---
    let result1 = scan_and_collect_all(&store, field_id);
    assert_eq!(result1, vec!["a", "b", "c", "d", "e"]);
    println!("State after initial append: {:?}", result1);

    // --- 3. Perform Deletes ---
    // Delete logical rows 1 ("b") and 3 ("d").
    let mut rows_to_delete = RoaringBitmap::new();
    rows_to_delete.insert(1);
    rows_to_delete.insert(3);
    store.delete_rows(field_id, &rows_to_delete).unwrap();

    // --- 4. Verify State After Deletes ---
    let result2 = scan_and_collect_all(&store, field_id);
    assert_eq!(result2, vec!["a", "c", "e"]);
    println!("State after deleting 'b' and 'd': {:?}", result2);

    // --- 5. Perform an "Update" (by appending more data) ---
    // In this model, an update is just appending a new version. Here we'll just
    // append new, distinct data to show multiple chunks are handled correctly.
    let update_array = Arc::new(StringArray::from(vec!["f", "g"]));
    let update_batch = RecordBatch::try_new(schema.clone(), vec![update_array]).unwrap();
    store.append(&update_batch).unwrap();

    // --- 6. Verify Final State ---
    // The scan should now return the filtered first chunk and the complete second chunk.
    let final_result = scan_and_collect_all(&store, field_id);
    assert_eq!(final_result, vec!["a", "c", "e", "f", "g"]);
    println!("Final state after appending 'f', 'g': {:?}", final_result);
}
