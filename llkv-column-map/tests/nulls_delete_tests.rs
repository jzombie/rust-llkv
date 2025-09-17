//! Verify that appending nulls performs LWW deletes at scale.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, Namespace};

/// Helper to build a LogicalFieldId for user data.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

#[test]
fn nulls_are_lww_deletes_u64_large() {
    // Scale: 1,000,000 rows across 10 batches.
    const N_ROWS: usize = 1_000_000;
    const N_BATCHES: usize = 10;
    const BATCH: usize = N_ROWS / N_BATCHES;

    let field_id = fid(710);

    // Build store.
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Schema: row_id u64 (non-null) + data u64 (nullable) with field_id tag.
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let row_id_f = Field::new("row_id", DataType::UInt64, false);
    let data_f = Field::new("data", DataType::UInt64, true).with_metadata(md);
    let schema = Arc::new(Schema::new(vec![row_id_f, data_f]));

    // Initial append: rows 0..N_ROWS-1 with values equal to row_id.
    // Ingest across multiple batches.
    for b in 0..N_BATCHES {
        let start = b * BATCH;
        let end = start + BATCH;

        let rids: Vec<u64> = (start as u64..end as u64).collect();
        let vals: Vec<u64> = rids.clone();

        let rid_arr = Arc::new(UInt64Array::from(rids));
        let val_arr = Arc::new(UInt64Array::from(vals));

        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
        store.append(&batch).unwrap();
    }

    // Second append: delete every 10th row by appending null for that row_id.
    // This creates ~100,000 LWW deletes in a single batch.
    let del_rids: Vec<u64> = (0..N_ROWS as u64).step_by(10).collect();
    let del_vals = UInt64Array::from(vec![None; del_rids.len()]);

    let del_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(del_rids.clone())),
            Arc::new(del_vals),
        ],
    )
    .unwrap();
    store.append(&del_batch).unwrap();

    // Scan unsorted and verify:
    // - No value is a multiple of 10 (all deleted)
    // - Count equals N_ROWS - N_ROWS/10
    // - Sum matches expected sum after deletions
    let mut count: u64 = 0;
    let mut sum: u128 = 0;

    let it = store.scan(field_id).unwrap();
    for arr_res in it {
        let arr = arr_res.unwrap();
        let u = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
        for i in 0..u.len() {
            let v = u.value(i);
            // Every 10th row was deleted via null upsert.
            assert!(v % 10 != 0, "found deleted value still present: {}", v);
            count += 1;
            sum += v as u128;
        }
    }

    let expected_count: u64 = (N_ROWS as u64) - (N_ROWS as u64 / 10);
    assert_eq!(count, expected_count);

    // Sum(0..N-1) = N*(N-1)/2
    // Deleted are 0,10,20,... => 10*Sum(0..M-1) with M = N/10
    let n = N_ROWS as u128;
    let m = (N_ROWS as u128) / 10;
    let sum_all = n * (n - 1) / 2;
    let sum_deleted = 10u128 * (m * (m - 1) / 2);
    let expected_sum = sum_all - sum_deleted;

    assert_eq!(sum, expected_sum);
}
