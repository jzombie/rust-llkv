// These tests validate automatic LWW-by-row_id plus explicit deletes.
// Requirements enforced by ColumnStore::append:
// - Every RecordBatch must include a non-nullable UInt64 "row_id" column.
// - LWW applies per field (logical column). The same row_id values can
//   appear in different fields without cross-field conflicts.
//
// Notes:
// - We keep row_id first in the schema and the batch values vector.
// - For LWW, reusing a row_id in a later append will tombstone the prior
//   version for that field automatically before the append completes.
//
// Implementation note for current store:
// - The store now performs in-place chunk rewrites for LWW and deletes,
//   not tombstones. Tests still describe "tombstone" behavior but the
//   observable results (only latest value survives) are the same.
use arrow::array::{Array, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::LogicalFieldId;

use roaring::RoaringTreemap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Helper: build a schema with "row_id" inserted at column 0.
fn schema_with_row_id(mut fields: Vec<Field>) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    fields.insert(0, rid);
    Arc::new(Schema::new(fields))
}

/// Helper: scan a u64 field into a Vec<u64> (order is append order unless
/// sorted; we just collect).
fn scan_u64(store: &ColumnStore<MemPager>, fid: LogicalFieldId) -> Vec<u64> {
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

/// Helper: scan an i32 field into a Vec<i32>.
fn scan_i32(store: &ColumnStore<MemPager>, fid: LogicalFieldId) -> Vec<i32> {
    let mut out = Vec::new();
    let it = store.scan(fid).unwrap();
    for arr_res in it {
        let arr = arr_res.unwrap();
        let a = arr
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("field must be Int32");
        for i in 0..a.len() {
            out.push(a.value(i));
        }
    }
    out
}

/// Helper: find the current global row index of a specific u64 value in
/// a field by scanning in append order. Returns None if not found.
/// This is used because LWW rewrites can shift global indexes.
fn find_index_of_u64(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    needle: u64,
) -> Option<u64> {
    let mut idx: u64 = 0;
    let it = store.scan(fid).unwrap();
    for arr_res in it {
        let arr = arr_res.ok()?;
        let a = arr.as_any().downcast_ref::<UInt64Array>()?;
        for i in 0..a.len() {
            if a.value(i) == needle {
                return Some(idx);
            }
            idx += 1;
        }
    }
    None
}

/// Test 1: LWW update replaces prior values that share the same row_id,
/// for a single UInt64 field.
#[test]
fn test_lww_updates_single_field_u64() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let field_id = LogicalFieldId::for_user_table_0(700);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);

    let schema = schema_with_row_id(vec![data_f]);
    // Initial 6 rows: row_id = 0..5, values are distinct.
    let rid0 = Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4, 5]));
    let v0 = Arc::new(UInt64Array::from(vec![10, 20, 30, 40, 50, 60]));
    let b0 = RecordBatch::try_new(schema.clone(), vec![rid0, v0]).unwrap();
    store.append(&b0).unwrap();

    // Update two rows by reusing row_id 1 and 3 (LWW).
    let rid1 = Arc::new(UInt64Array::from(vec![1u64, 3u64]));
    let v1 = Arc::new(UInt64Array::from(vec![200u64, 400u64]));
    let b1 = RecordBatch::try_new(schema.clone(), vec![rid1, v1]).unwrap();
    store.append(&b1).unwrap();

    // Update one more row_id 5.
    let rid2 = Arc::new(UInt64Array::from(vec![5u64]));
    let v2 = Arc::new(UInt64Array::from(vec![6000u64]));
    let b2 = RecordBatch::try_new(schema.clone(), vec![rid2, v2]).unwrap();
    store.append(&b2).unwrap();

    // Collect all current values for fid=700 and verify winners.
    let mut got = scan_u64(&store, field_id);
    got.sort_unstable();

    // Expected after LWW:
    // row_id:0->10, 1->200, 2->30, 3->400, 4->50, 5->6000
    let mut expect = vec![10, 200, 30, 400, 50, 6000];
    expect.sort_unstable();

    assert_eq!(got.len(), expect.len());
    assert_eq!(got, expect);
}

/// Test 2: LWW is per-field. Reusing the same row_id in another field
/// does not affect it.
#[test]
fn test_lww_per_field_isolation() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Two fields with different logical ids.
    let fida = LogicalFieldId::for_user_table_0(800);
    let fidb = LogicalFieldId::for_user_table_0(801);

    let mut mda = HashMap::new();
    mda.insert("field_id".to_string(), u64::from(fida).to_string());
    let fa = Field::new("a_data", DataType::UInt64, false).with_metadata(mda);

    let mut mdb = HashMap::new();
    mdb.insert("field_id".to_string(), u64::from(fidb).to_string());
    let fb = Field::new("b_data", DataType::Int32, false).with_metadata(mdb);

    let schema_ab = schema_with_row_id(vec![fa.clone(), fb.clone()]);
    // Ingest one batch with both fields populated for row_id 0..4.
    let r0 = Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4]));
    let a0 = Arc::new(UInt64Array::from(vec![10, 20, 30, 40, 50]));
    let b0 = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
    let init = RecordBatch::try_new(schema_ab.clone(), vec![r0, a0, b0]).unwrap();
    store.append(&init).unwrap();

    // Update field A only (row_id 1 and 3).
    // Use a schema that includes only row_id and field A, so the number
    // of arrays matches the number of fields.
    let mut mda2 = HashMap::new();
    mda2.insert("field_id".to_string(), u64::from(fida).to_string());
    let fa2 = Field::new("a_data", DataType::UInt64, false).with_metadata(mda2);
    let schema_a_only = schema_with_row_id(vec![fa2]);

    let r1 = Arc::new(UInt64Array::from(vec![1u64, 3u64]));
    let a1 = Arc::new(UInt64Array::from(vec![200u64, 400u64]));
    let upd_a = RecordBatch::try_new(schema_a_only.clone(), vec![r1, a1]).unwrap();
    store.append(&upd_a).unwrap();

    // Verify field A reflects LWW.
    let mut a_vals = scan_u64(&store, fida);
    a_vals.sort_unstable();
    let mut a_expect = vec![10, 200, 30, 400, 50];
    a_expect.sort_unstable();
    assert_eq!(a_vals, a_expect);

    // Verify field B is unaffected.
    let mut b_vals = scan_i32(&store, fidb);
    b_vals.sort_unstable();
    let mut b_expect = vec![1, 2, 3, 4, 5];
    b_expect.sort_unstable();
    assert_eq!(b_vals, b_expect);
}

/// Test 3: Explicit deletes by global row index co-exist with LWW.
/// We choose sizes so the LWW tombstone ratio stays below the default
/// compaction threshold (0.30), keeping global indices stable.
#[test]
fn test_deletes_and_updates() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let field_id = LogicalFieldId::for_user_table_0(820);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);

    let schema = schema_with_row_id(vec![data_f]);
    // Initial 10 rows: row_id 0..9, values 10x index.
    let rid0 = Arc::new(UInt64Array::from((0u64..10u64).collect::<Vec<u64>>()));
    let vals0 = Arc::new(UInt64Array::from(
        (0u64..10u64).map(|i| i * 10).collect::<Vec<u64>>(),
    ));
    let b0 = RecordBatch::try_new(schema.clone(), vec![rid0, vals0]).unwrap();
    store.append(&b0).unwrap();

    // LWW update 2 of 10 rows (ratio = 0.2 < 0.30).
    let rid1 = Arc::new(UInt64Array::from(vec![1u64, 3u64]));
    let vals1 = Arc::new(UInt64Array::from(vec![111u64, 333u64]));
    let b1 = RecordBatch::try_new(schema.clone(), vec![rid1, vals1]).unwrap();
    store.append(&b1).unwrap();

    // Now explicitly delete the row whose VALUE is 40. Because LWW can
    // shift global indexes, resolve its current global index first.
    let idx_40 =
        find_index_of_u64(&store, field_id, 40).expect("value 40 should exist before delete");

    let mut bm = RoaringTreemap::new();
    bm.insert(idx_40);
    store.delete_rows(field_id, &bm).unwrap();

    // Collect and verify: we should have 9 rows now.
    let got = scan_u64(&store, field_id);
    assert_eq!(got.len(), 9);

    // Build the expected set by row_id:
    // 0->0, 1->111(LWW), 2->20, 3->333(LWW), 4->(deleted),
    // 5->50, 6->60, 7->70, 8->80, 9->90
    let mut exp_map = BTreeMap::new();
    exp_map.insert(0u64, 0u64);
    exp_map.insert(1, 111);
    exp_map.insert(2, 20);
    exp_map.insert(3, 333);
    // 4 deleted
    exp_map.insert(5, 50);
    exp_map.insert(6, 60);
    exp_map.insert(7, 70);
    exp_map.insert(8, 80);
    exp_map.insert(9, 90);

    // Check membership ignoring order.
    let mut expected: Vec<u64> = exp_map.values().copied().collect();
    expected.sort_unstable();
    let mut got_sorted = got.clone();
    got_sorted.sort_unstable();

    assert_eq!(got_sorted, expected);
}
