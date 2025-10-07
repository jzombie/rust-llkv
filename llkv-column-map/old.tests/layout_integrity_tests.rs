use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, RowId};
use roaring::RoaringTreemap;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to build a simple schema with "row_id" and one UInt64 data field.
fn u64_schema_with_fid(fid: LogicalFieldId) -> Arc<Schema> {
    let mut md = HashMap::new();
    md.insert(
        crate::FIELD_ID_META_KEY.to_string(),
        u64::from(fid).to_string(),
    );
    let data_field = Field::new("data", DataType::UInt64, false).with_metadata(md);
    let row_id_field = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![row_id_field, data_field]))
}

#[test]
fn test_layout_integrity_under_churn() {
    // This test verifies that the store's internal layout accounting
    // remains consistent after large appends, LWW updates, and deletes
    // that cause chunk rewrites and descriptor page chain modifications.
    // --- 1. Setup ---
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = LogicalFieldId::for_user_table_0(901);
    let schema = u64_schema_with_fid(field_id);

    // --- 2. Initial Large Ingestion ---
    // Ingest a batch large enough to be split into multiple chunks,
    // which will also likely require multiple descriptor pages.
    // 200k u64 values = ~1.6MB, should create 2 chunks.
    const INITIAL_ROWS: usize = 200_000;
    let initial_rids: Vec<u64> = (0..INITIAL_ROWS as u64).collect();
    let initial_vals: Vec<u64> = (0..INITIAL_ROWS as u64).map(|i| i * 10).collect();
    let r0 = Arc::new(UInt64Array::from(initial_rids));
    let v0 = Arc::new(UInt64Array::from(initial_vals));
    let b0 = RecordBatch::try_new(schema.clone(), vec![r0, v0]).unwrap();
    store.append(&b0).unwrap();

    println!("--- After Initial Ingestion ---");
    // VERIFY 1: The store must be consistent after the initial multi-chunk append.
    store
        .verify_integrity()
        .expect("Store must be consistent after large append");
    let stats1 = store.get_layout_stats().unwrap();
    let col_stats1 = &stats1[0];
    assert_eq!(col_stats1.total_rows, INITIAL_ROWS as u64);
    assert!(
        col_stats1.total_chunks > 1,
        "Append should have created multiple chunks"
    );
    println!(
        "OK: Store is consistent with {} rows in {} chunks.",
        col_stats1.total_rows, col_stats1.total_chunks
    );

    // --- 3. LWW Updates ---
    // Update 10 rows by reusing their row_ids. These rows will be spread
    // across the existing chunks, forcing in-place rewrites.
    let update_rids: Vec<u64> = (0..10u64).map(|i| i * 1000).collect(); // e.g., row_id 0, 1000, 2000...
    let update_vals: Vec<u64> = (0..10u64).map(|i| 9_000_000 + i).collect();
    let r1 = Arc::new(UInt64Array::from(update_rids));
    let v1 = Arc::new(UInt64Array::from(update_vals));
    let b1 = RecordBatch::try_new(schema.clone(), vec![r1, v1]).unwrap();
    store.append(&b1).unwrap();

    println!("\n--- After LWW Updates ---");
    // VERIFY 2: The store must remain consistent after LWW updates.
    // The total row count should not have changed.
    store
        .verify_integrity()
        .expect("Store must be consistent after LWW updates");
    let stats2 = store.get_layout_stats().unwrap();
    let col_stats2 = &stats2[0];
    assert_eq!(
        col_stats2.total_rows, INITIAL_ROWS as u64,
        "LWW update should not change total row count"
    );
    println!(
        "OK: Store is consistent. Row count is unchanged at {}.",
        col_stats2.total_rows
    );

    // --- 4. Deletes ---
    // Delete 5 rows by their global index.
    let mut to_delete = RoaringTreemap::new();
    to_delete.insert(5);
    to_delete.insert(50);
    to_delete.insert(500);
    to_delete.insert(5000);
    to_delete.insert(50000);
    let deletes: Vec<RowId> = to_delete.iter().collect();
    store.delete_rows(&[field_id], &deletes).unwrap();

    println!("\n--- After Deletes ---");
    // VERIFY 3: The store must be consistent after deletes.
    // The total row count should be reduced by the number of deleted rows.
    store
        .verify_integrity()
        .expect("Store must be consistent after deletes");
    let stats3 = store.get_layout_stats().unwrap();
    let col_stats3 = &stats3[0];
    assert_eq!(
        col_stats3.total_rows,
        (INITIAL_ROWS - 5) as u64,
        "Row count should decrease by 5 after deletes"
    );
    println!(
        "OK: Store is consistent. Row count correctly reduced to {}.",
        col_stats3.total_rows
    );
}
