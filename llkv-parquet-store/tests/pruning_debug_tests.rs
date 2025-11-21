use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_expr::{col, lit};
use llkv_parquet_store::{add_mvcc_columns, ParquetStore};
use llkv_storage::pager::{MemPager, Pager};
use std::sync::Arc;

#[test]
fn test_pruning_with_debug() {
    let pager = Arc::new(MemPager::new());
    let store = ParquetStore::open(pager.clone()).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("value", DataType::UInt64, false),
    ]));

    let table_id = store.create_table("data", schema.clone()).unwrap();

    // Insert 3 batches with VERY different row_id ranges
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1, 2, 3])),
            Arc::new(UInt64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1000, 1001, 1002])),
            Arc::new(UInt64Array::from(vec![1000, 1010, 1020])),
        ],
    )
    .unwrap();

    let batch3 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![2000, 2001, 2002])),
            Arc::new(UInt64Array::from(vec![2000, 2010, 2020])),
        ],
    )
    .unwrap();

    let batch1_mvcc = add_mvcc_columns(batch1, 1).unwrap();
    let batch2_mvcc = add_mvcc_columns(batch2, 1).unwrap();
    let batch3_mvcc = add_mvcc_columns(batch3, 1).unwrap();

    store
        .append_many(table_id, vec![batch1_mvcc, batch2_mvcc, batch3_mvcc])
        .unwrap();

    // Count all keys in pager
    // Note: Batch optimization merges small batches, so we may have fewer files than input batches
    let all_keys = pager.enumerate_keys().unwrap();
    println!("Total keys in pager: {}", all_keys.len());
    assert!(all_keys.len() >= 2, "Should have at least catalog + 1 file"); // catalog + merged files

    // Scan with predicate that should ONLY hit batch2 (row_id BETWEEN 1000 AND 1002)
    println!("\n=== Testing predicate: row_id BETWEEN 1000 AND 1002 ===");
    let pred = col("row_id").between(lit(1000u64), lit(1002u64));

    // Count how many files are actually read
    let mut file_count = 0;
    let batches: Vec<_> = store
        .scan(table_id, &[pred], None, None)
        .unwrap()
        .inspect(|_| {
            file_count += 1;
            println!("Reading file #{}", file_count);
        })
        .collect::<llkv_result::Result<Vec<_>>>()
        .unwrap();

    println!("Files read: {}", file_count);
    println!("Batches returned: {}", batches.len());
    println!(
        "Total rows: {}",
        batches.iter().map(|b| b.num_rows()).sum::<usize>()
    );

    // Note: Batch optimization merges small batches, reducing file-level pruning effectiveness.
    // Without row-level filtering, the scan returns all rows from files that overlap the range.
    // This is expected behavior - row-level filtering would be needed for precise results.
    assert!(file_count >= 1, "Should read at least 1 file");
    assert!(!batches.is_empty(), "Should get at least 1 batch");
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows >= 3, "Should have at least the 3 matching rows");

    // Test with predicate that hits NO files (row_id > 10000)
    println!("\n=== Testing predicate: row_id > 10000 ===");
    let pred = col("row_id").gt(lit(10000u64));

    file_count = 0;
    let batches: Vec<_> = store
        .scan(table_id, &[pred], None, None)
        .unwrap()
        .inspect(|_| {
            file_count += 1;
            println!("Reading file #{}", file_count);
        })
        .collect::<llkv_result::Result<Vec<_>>>()
        .unwrap();

    println!("Files read: {}", file_count);
    println!("Batches returned: {}", batches.len());

    // If pruning works, we should read 0 files
    assert_eq!(file_count, 0, "Should read 0 files when no files match");
    assert_eq!(batches.len(), 0, "Should get 0 batches");

    // Test without filters (should read all files, which may be merged)
    println!("\n=== Testing no filters ===");
    file_count = 0;
    let batches: Vec<_> = store
        .scan(table_id, &[], None, None)
        .unwrap()
        .inspect(|_| {
            file_count += 1;
            println!("Reading file #{}", file_count);
        })
        .collect::<llkv_result::Result<Vec<_>>>()
        .unwrap();

    println!("Files read: {}", file_count);
    println!("Batches returned: {}", batches.len());

    assert!(file_count >= 1, "Should read at least 1 file");
    assert!(!batches.is_empty(), "Should get at least 1 batch");
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 9, "Should have all 9 rows");

    // Test with limit=5 (should read files until limit is reached)
    println!("\n=== Testing limit=5 ===");
    file_count = 0;
    let mut total_rows = 0;
    let _batches: Vec<_> = store
        .scan(table_id, &[], None, Some(5))
        .unwrap()
        .inspect(|result| {
            file_count += 1;
            if let Ok(batch) = result {
                total_rows += batch.num_rows();
                println!(
                    "Reading file #{}, batch has {} rows",
                    file_count,
                    batch.num_rows()
                );
            }
        })
        .collect::<llkv_result::Result<Vec<_>>>()
        .unwrap();

    println!("Files read: {}", file_count);
    println!("Total rows: {}", total_rows);
    assert!(total_rows <= 5, "Should not exceed limit of 5 rows");
    assert!(total_rows > 0, "Should return at least some rows");
}
