mod common;

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::debug::ColumnStoreDebug;
use llkv_column_map::types::{LogicalFieldId, Namespace};

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn seed_small_store() -> ColumnStore<MemPager> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Two small columns, deterministic values
    let fid_u64 = fid(10);
    let mut md1 = HashMap::new();
    md1.insert("field_id".to_string(), u64::from(fid_u64).to_string());
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md1),
    ]));
    let rid: Vec<u64> = (0..100u64).collect();
    let vals: Vec<u64> = (0..100u64).map(|x| x ^ 0xA5A5_A5A5_A5A5_A5A5).collect();
    let rb = RecordBatch::try_new(
        schema1,
        vec![
            Arc::new(UInt64Array::from(rid)),
            Arc::new(UInt64Array::from(vals)),
        ],
    )
    .unwrap();
    store.append(&rb).unwrap();

    let fid_i32 = fid(20);
    let mut md2 = HashMap::new();
    md2.insert("field_id".to_string(), u64::from(fid_i32).to_string());
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::Int32, false).with_metadata(md2),
    ]));
    let rid2: Vec<u64> = (0..50u64).collect();
    let vals2: Vec<i32> = (0..50i32).map(|x| x.wrapping_mul(3) - 7).collect();
    let rb2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(UInt64Array::from(rid2)),
            Arc::new(Int32Array::from(vals2)),
        ],
    )
    .unwrap();
    store.append(&rb2).unwrap();

    // Build sort index for u64 to include perm nodes in DOT
    store.create_sort_index(fid_u64).unwrap();

    store
}

#[test]
fn snapshot_storage_dot_canonical() {
    let store = seed_small_store();
    let colors: HashMap<_, _> = HashMap::new();
    let dot = store.render_storage_as_dot(&colors);
    let canon = common::canonicalize_dot(&dot);
    common::assert_matches_golden(&canon, "tests/snapshots/storage_small.dot");
}

#[test]
fn snapshot_storage_table_ascii() {
    let store = seed_small_store();
    let table = store.render_storage_as_formatted_string();
    common::assert_matches_golden(&table, "tests/snapshots/storage_small_table.txt");
}
