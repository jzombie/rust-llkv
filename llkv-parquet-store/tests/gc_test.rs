//! Integration tests for garbage collection.

use arrow::array::{StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_parquet_store::ParquetStore;
use llkv_storage::pager::{MemPager, Pager};
use std::sync::Arc;

#[test]
fn test_gc_empty_store() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(Arc::clone(&pager)).unwrap();

    // Empty store should have no garbage
    let freed = store.garbage_collect().unwrap();
    assert_eq!(freed, 0);

    // Verify catalog key exists
    let all_keys = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys, vec![0]); // Just catalog
}

#[test]
fn test_gc_after_table_creation() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(Arc::clone(&pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("value", DataType::Utf8, false),
    ]));

    // Create table (doesn't allocate any file keys yet)
    store.create_table("test", schema.clone()).unwrap();

    // No garbage yet
    let freed = store.garbage_collect().unwrap();
    assert_eq!(freed, 0);
}

#[test]
fn test_gc_after_appends() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(Arc::clone(&pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let table_id = store.create_table("users", schema.clone()).unwrap();

    // Append 3 batches
    for i in 0..3 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![i * 10, i * 10 + 1])),
                Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            ],
        )
        .unwrap();

        let batch_with_mvcc = llkv_parquet_store::add_mvcc_columns(batch, 1).unwrap();
        store.append_many(table_id, vec![batch_with_mvcc]).unwrap();
    }

    // All keys are reachable, no garbage
    let freed = store.garbage_collect().unwrap();
    assert_eq!(freed, 0);

    // Verify we have catalog + 3 files
    let all_keys = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys.len(), 4);
}

#[test]
fn test_gc_after_table_drop() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(Arc::clone(&pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("value", DataType::UInt64, false),
    ]));

    let table_id = store.create_table("temp_table", schema.clone()).unwrap();

    // Append 2 batches
    for i in 0..2 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![i])),
                Arc::new(UInt64Array::from(vec![i * 100])),
            ],
        )
        .unwrap();

        let batch_with_mvcc = llkv_parquet_store::add_mvcc_columns(batch, 1).unwrap();
        store.append_many(table_id, vec![batch_with_mvcc]).unwrap();
    }

    // Before drop: catalog + 2 files
    let all_keys_before = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys_before.len(), 3);

    // Drop the table (this calls free_many internally)
    store.drop_table("temp_table").unwrap();

    // After drop: only catalog remains (drop_table already freed the keys)
    let all_keys_after = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys_after.len(), 1);
    assert_eq!(all_keys_after, vec![0]);

    // No additional garbage to collect
    let freed = store.garbage_collect().unwrap();
    assert_eq!(freed, 0);
}

#[test]
fn test_gc_multiple_tables() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(Arc::clone(&pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("x", DataType::UInt64, false),
    ]));

    // Create table1 with 2 files
    let t1 = store.create_table("table1", schema.clone()).unwrap();
    for i in 0..2 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![i])),
                Arc::new(UInt64Array::from(vec![i])),
            ],
        )
        .unwrap();
        let batch_with_mvcc = llkv_parquet_store::add_mvcc_columns(batch, 1).unwrap();
        store.append_many(t1, vec![batch_with_mvcc]).unwrap();
    }

    // Create table2 with 1 file
    let t2 = store.create_table("table2", schema.clone()).unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![100])),
            Arc::new(UInt64Array::from(vec![200])),
        ],
    )
    .unwrap();
    let batch_with_mvcc = llkv_parquet_store::add_mvcc_columns(batch, 1).unwrap();
    store.append_many(t2, vec![batch_with_mvcc]).unwrap();

    // Should have catalog + 3 files, no garbage
    let freed = store.garbage_collect().unwrap();
    assert_eq!(freed, 0);

    let all_keys = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys.len(), 4);

    // Drop table1
    store.drop_table("table1").unwrap();

    // Now only catalog + 1 file (from table2)
    let all_keys = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys.len(), 2);

    // No garbage since drop_table freed them
    let freed = store.garbage_collect().unwrap();
    assert_eq!(freed, 0);
}

#[test]
fn test_gc_simulated_leak() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(Arc::clone(&pager)).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::Utf8, false),
    ]));

    let table_id = store.create_table("test", schema.clone()).unwrap();

    // Append a file normally
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    )
    .unwrap();
    let batch_with_mvcc = llkv_parquet_store::add_mvcc_columns(batch, 1).unwrap();
    store.append_many(table_id, vec![batch_with_mvcc]).unwrap();

    // Simulate a leaked key by manually writing to the pager
    // (e.g., from a crashed transaction that allocated but never cataloged)
    use llkv_storage::pager::{BatchPut, Pager};
    let leaked_key = pager.alloc_many(1).unwrap()[0];
    pager
        .batch_put(&[BatchPut::Raw {
            key: leaked_key,
            bytes: b"leaked data".to_vec(),
        }])
        .unwrap();

    // Now we have a leaked key
    let all_keys = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys.len(), 3); // catalog + 1 real file + 1 leaked

    // Garbage collect should find and free it
    let freed = store.garbage_collect().unwrap();
    assert_eq!(freed, 1);

    // Now only reachable keys remain
    let all_keys = pager.enumerate_keys().unwrap();
    assert_eq!(all_keys.len(), 2); // catalog + 1 real file
}
