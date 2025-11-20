//! Basic integration tests for ParquetStore.

use arrow::array::{StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_parquet_store::{add_mvcc_columns, ParquetStore};
use llkv_result::Result;
use llkv_storage::pager::MemPager;
use std::sync::Arc;

#[test]
fn test_end_to_end_workflow() {
    // Create store
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(pager).unwrap();

    // Define schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::UInt64, false),
    ]));

    // Create table
    let table_id = store.create_table("people", schema.clone()).unwrap();

    // Insert data
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            Arc::new(UInt64Array::from(vec![30, 25, 35])),
        ],
    )
    .unwrap();

    let batch_with_mvcc = add_mvcc_columns(batch, 1).unwrap();
    store.append_many(table_id, vec![batch_with_mvcc]).unwrap();

    // Read back
    let results: Vec<_> = store
        .scan(table_id, &[], None, None)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].num_rows(), 3);
}

#[test]
fn test_multiple_appends() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(pager).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("value", DataType::UInt64, false),
    ]));

    let table_id = store.create_table("data", schema.clone()).unwrap();

    // Append first batch
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1, 2])),
            Arc::new(UInt64Array::from(vec![10, 20])),
        ],
    )
    .unwrap();

    let batch1_mvcc = add_mvcc_columns(batch1, 1).unwrap();
    store.append_many(table_id, vec![batch1_mvcc]).unwrap();

    // Append second batch
    let batch2 = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(vec![3, 4])),
            Arc::new(UInt64Array::from(vec![30, 40])),
        ],
    )
    .unwrap();

    let batch2_mvcc = add_mvcc_columns(batch2, 2).unwrap();
    store.append_many(table_id, vec![batch2_mvcc]).unwrap();

    // Read all data
    let results: Vec<_> = store
        .scan(table_id, &[], None, None)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();

    // Should have 2 batches (one per append)
    assert_eq!(results.len(), 2);

    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 4);
}

#[test]
fn test_persistence_across_reopens() {
    let pager = Arc::new(MemPager::new());

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::Utf8, false),
    ]));

    let table_id = {
        let store = ParquetStore::open(Arc::clone(&pager)).unwrap();
        let tid = store.create_table("persistent", schema.clone()).unwrap();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1])),
                Arc::new(StringArray::from(vec!["test"])),
            ],
        )
        .unwrap();

        let batch_mvcc = add_mvcc_columns(batch, 1).unwrap();
        store.append_many(tid, vec![batch_mvcc]).unwrap();

        tid
    };

    // Reopen store with same pager
    let store2 = ParquetStore::open(pager).unwrap();

    // Verify table exists
    let tables = store2.list_tables();
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0], "persistent");

    // Verify data
    let results: Vec<_> = store2
        .scan(table_id, &[], None, None)
        .unwrap()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].num_rows(), 1);
}
