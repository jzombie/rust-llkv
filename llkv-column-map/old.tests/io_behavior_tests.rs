use arrow::array::{Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::storage::pager::{InstrumentedPager, MemPager, Pager};
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, Namespace};
use roaring::RoaringTreemap;
use simd_r_drive_entry_handle::EntryHandle;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Test helper to create a standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

/// Helper to build a simple schema with "row_id" and one UInt64 data field.
fn u64_schema_with_fid(fid: LogicalFieldId) -> Arc<Schema> {
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(fid).to_string());
    let data_field = Field::new("data", DataType::UInt64, false).with_metadata(md);
    let row_id_field = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    Arc::new(Schema::new(vec![row_id_field, data_field]))
}

/// Generic helper to scan a u64 field into a Vec<u64> for any Pager type.
fn scan_u64<P: Pager<Blob = EntryHandle>>(store: &ColumnStore<P>, fid: LogicalFieldId) -> Vec<u64> {
    let mut out = Vec::new();
    let it = store.scan(fid).unwrap();
    for arr_res in it {
        let arr = arr_res.unwrap();
        let a = arr
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("field must be UInt64");
        for i in 0..a.len() {
            out.push(a.value(i));
        }
    }
    out
}

#[test]
fn test_instrumented_paging_io_behavior() {
    // This test verifies the exact physical I/O counts during a series of
    // storage operations to catch regressions in storage behavior.
    // --- 1. Setup ---
    // Wrap the MemPager to track I/O operations.
    let (pager, stats) = InstrumentedPager::new(MemPager::new());
    let store = ColumnStore::open(Arc::new(pager)).unwrap();
    let field_id = fid(950);
    let schema = u64_schema_with_fid(field_id);

    // --- Phase 2: Initial Append ---
    // This first append creates new pages for the catalog, descriptors, and data.
    let rid0 = Arc::new(UInt64Array::from(vec![0, 1, 2]));
    let v0 = Arc::new(UInt64Array::from(vec![10, 20, 30]));
    let b0 = RecordBatch::try_new(schema.clone(), vec![rid0, v0]).unwrap();
    store.append(&b0).unwrap();

    // Check I/O stats after the first append.
    let puts_after_append = stats.physical_puts.load(Ordering::Relaxed);
    let gets_after_append = stats.physical_gets.load(Ordering::Relaxed);
    println!(
        "Stats after initial append: gets={}, puts={}",
        gets_after_append, puts_after_append
    );
    assert!(
        puts_after_append > 0,
        "Should have physical puts after first append"
    );
    assert!(
        gets_after_append > 0,
        "Should have gets to check for catalog/descriptors"
    );
    assert_eq!(stats.physical_frees.load(Ordering::Relaxed), 0);

    // --- Phase 3: LWW Update ---
    // This append triggers a Last-Writer-Wins update. It reads the existing
    // chunk metadata and the chunks themselves to perform an in-place rewrite.
    let rid1 = Arc::new(UInt64Array::from(vec![1u64])); // Overwrite row_id 1
    let v1 = Arc::new(UInt64Array::from(vec![200u64]));
    let b1 = RecordBatch::try_new(schema.clone(), vec![rid1, v1]).unwrap();
    store.append(&b1).unwrap();

    let puts_after_lww = stats.physical_puts.load(Ordering::Relaxed);
    let gets_after_lww = stats.physical_gets.load(Ordering::Relaxed);
    println!(
        "Stats after LWW update: gets={}, puts={}",
        gets_after_lww, puts_after_lww
    );
    assert!(
        puts_after_lww > puts_after_append,
        "LWW update should add more puts"
    );
    assert!(
        gets_after_lww > gets_after_append,
        "LWW update requires gets to find old data"
    );
    assert_eq!(
        stats.physical_frees.load(Ordering::Relaxed),
        0,
        "In-place rewrite should not free pages"
    );

    // --- Phase 4: Compacting Delete ---
    // Delete a row. This triggers an in-place rewrite and then a compaction,
    // which *will* free the old, now-obsolete data pages.
    let mut to_delete = RoaringTreemap::new();
    to_delete.insert(0); // Delete the row with global index 0 (value 10)
    store.delete_rows(field_id, &to_delete).unwrap();

    let puts_after_delete = stats.physical_puts.load(Ordering::Relaxed);
    let gets_after_delete = stats.physical_gets.load(Ordering::Relaxed);
    let frees_after_delete = stats.physical_frees.load(Ordering::Relaxed);
    println!(
        "Stats after delete+compact: gets={}, puts={}, frees={}",
        gets_after_delete, puts_after_delete, frees_after_delete
    );
    // Assert that the delete/compaction process involved reads, writes (for new chunks), and frees (for old chunks).
    assert!(puts_after_delete > puts_after_lww);
    assert!(gets_after_delete > gets_after_lww);
    assert!(
        frees_after_delete > 0,
        "Compaction after delete should free obsolete pages"
    );
    println!("\nFinal IO Stats: {:?}", stats);
}

#[test]
fn test_exact_io_counts_for_simple_append() {
    // This is a "white box" test that verifies the exact number of physical
    // I/O operations for a simple, single-chunk append. This is useful for
    // catching subtle performance regressions but is more brittle to
    // implementation changes than relational checks.
    let (pager, stats) = InstrumentedPager::new(MemPager::new());
    let store = ColumnStore::open(Arc::new(pager)).unwrap();
    let field_id = fid(101);
    let schema = u64_schema_with_fid(field_id);

    // --- The Operation ---
    // Append a single batch with 3 rows. This will create one data chunk
    // and one corresponding row_id chunk.
    let rids = Arc::new(UInt64Array::from(vec![0, 1, 2]));
    let vals = Arc::new(UInt64Array::from(vec![10, 20, 30]));
    let batch = RecordBatch::try_new(schema, vec![rids, vals]).unwrap();
    store.append(&batch).unwrap();

    // --- Verification ---

    // Expected GETS: 3
    assert_eq!(
        stats.physical_gets.load(Ordering::Relaxed),
        3,
        "Expected exactly 3 physical gets for the first append to a new store"
    );
    // Expected PUTS: 7
    assert_eq!(
        stats.physical_puts.load(Ordering::Relaxed),
        7,
        "Expected exactly 7 physical puts"
    );
    // Expected ALLOCS: 6
    assert_eq!(
        stats.physical_allocs.load(Ordering::Relaxed),
        6,
        "Expected exactly 6 physical allocs"
    );
    // Expected FREES: 0
    assert_eq!(stats.physical_frees.load(Ordering::Relaxed), 0);
}

#[test]
fn test_large_scale_churn_io() {
    // --- 1. Setup ---
    const NUM_ROWS: u64 = 1_000_000;
    const NUM_DELETES: u64 = 10_000;
    const NUM_UPDATES: u64 = 100_000;

    // We append in smaller batches to create multiple chunks. This ensures
    // that the delete operation will make some chunks small enough to trigger
    // the compaction logic that frees pages.
    const NUM_INITIAL_BATCHES: u64 = 20;
    const BATCH_SIZE: u64 = NUM_ROWS / NUM_INITIAL_BATCHES;

    let (pager, stats) = InstrumentedPager::new(MemPager::new());
    let store = ColumnStore::open(Arc::new(pager)).unwrap();
    let field_id = fid(1001);
    let schema = u64_schema_with_fid(field_id);

    // --- 2. Phase 1: Bulk Upsert (Insert) of 1M Entries in Batches ---
    println!(
        "\n--- Phase 1: Appending 1,000,000 rows in {} batches ---",
        NUM_INITIAL_BATCHES
    );
    for i in 0..NUM_INITIAL_BATCHES {
        let start = i * BATCH_SIZE;
        let end = start + BATCH_SIZE;
        let rids = Arc::new(UInt64Array::from((start..end).collect::<Vec<_>>()));
        let vals = Arc::new(UInt64Array::from(
            (start..end).map(|i| i * 10).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![rids, vals]).unwrap();
        store.append(&batch).unwrap();
    }

    let gets_after_insert = stats.get_batches.load(Ordering::Relaxed);
    let puts_after_insert = stats.put_batches.load(Ordering::Relaxed);
    let allocs_after_insert = stats.alloc_batches.load(Ordering::Relaxed);
    println!(
        "Stats after 1M insert: {} get batches, {} put batches, {} alloc batches",
        gets_after_insert, puts_after_insert, allocs_after_insert
    );
    assert_eq!(stats.free_batches.load(Ordering::Relaxed), 0);

    // --- 3. Phase 2: Delete 10,000 Entries ---
    println!("\n--- Phase 2: Deleting 10,000 rows ---");
    let mut to_delete = RoaringTreemap::new();
    // Delete the first 10,000 even-numbered global row indexes.
    for i in 0..(NUM_DELETES * 2) {
        if i % 2 == 0 {
            to_delete.insert(i);
        }
    }
    store.delete_rows(field_id, &to_delete).unwrap();

    let gets_after_delete = stats.get_batches.load(Ordering::Relaxed);
    let puts_after_delete = stats.put_batches.load(Ordering::Relaxed);
    let allocs_after_delete = stats.alloc_batches.load(Ordering::Relaxed);
    let frees_after_delete = stats.free_batches.load(Ordering::Relaxed);
    println!(
        "Stats after 10k delete: {} get batches, {} put batches, {} alloc batches, {} free batches",
        gets_after_delete, puts_after_delete, allocs_after_delete, frees_after_delete
    );
    assert!(gets_after_delete > gets_after_insert);
    assert!(puts_after_delete > puts_after_insert);
    assert!(
        frees_after_delete > 0,
        "Deletes should trigger compaction and free pages"
    );
    assert_eq!(
        scan_u64(&store, field_id).len(),
        (NUM_ROWS - NUM_DELETES as u64) as usize
    );

    // --- 4. Phase 3: Bulk Upsert (Update) of 100,000 Entries ---
    println!("\n--- Phase 3: Updating 100,000 rows ---");
    // Update row_ids in a range that was not deleted
    let update_start_row_id = (NUM_DELETES * 2) as u64; // Start updates after the deleted range
    let rids_update = Arc::new(UInt64Array::from(
        (0..NUM_UPDATES)
            .map(|i| update_start_row_id + i)
            .collect::<Vec<_>>(),
    ));
    let vals_update = Arc::new(UInt64Array::from(
        (0..NUM_UPDATES).map(|i| 9_000_000 + i).collect::<Vec<_>>(),
    ));
    let update_batch = RecordBatch::try_new(schema, vec![rids_update, vals_update]).unwrap();
    store.append(&update_batch).unwrap();

    let gets_after_update = stats.get_batches.load(Ordering::Relaxed);
    let puts_after_update = stats.put_batches.load(Ordering::Relaxed);
    let allocs_after_update = stats.alloc_batches.load(Ordering::Relaxed);
    let frees_after_update = stats.free_batches.load(Ordering::Relaxed);
    println!(
        "Stats after 100k update: {} get batches, {} put batches, {} alloc batches, {} free batches",
        gets_after_update, puts_after_update, allocs_after_update, frees_after_update
    );
    assert!(gets_after_update > gets_after_delete);
    assert!(puts_after_update > puts_after_delete);
    assert_eq!(frees_after_update, frees_after_delete);

    // The number of rows should remain the same after an update of existing keys
    assert_eq!(
        scan_u64(&store, field_id).len(),
        (NUM_ROWS - NUM_DELETES as u64) as usize
    );
    println!("\nTest Complete. Final IO Stats: {:?}", stats);
}
