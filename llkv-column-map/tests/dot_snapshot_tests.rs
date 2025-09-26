mod common;

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{BinaryBuilder, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::debug::ColumnStoreDebug;
use llkv_column_map::store::{ColumnStore, IndexKind, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::{LogicalFieldId, Namespace, TableId};
use llkv_storage::pager::MemPager;

fn fid(table_id: TableId, field_id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(table_id)
        .with_field_id(field_id)
}

fn seed_small_store() -> ColumnStore<MemPager> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Table 0: two small columns, deterministic values
    let fid_u64 = fid(0, 10);
    let mut md1 = HashMap::new();
    md1.insert("field_id".to_string(), u64::from(fid_u64).to_string());
    let schema1 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
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

    let fid_i32 = fid(0, 20);
    let mut md2 = HashMap::new();
    md2.insert("field_id".to_string(), u64::from(fid_i32).to_string());
    let schema2 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
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

    // Also add a small Binary column to table 0
    let fid_bin = fid(0, 30);
    let mut md3 = HashMap::new();
    md3.insert("field_id".to_string(), u64::from(fid_bin).to_string());
    let schema3 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::Binary, false).with_metadata(md3),
    ]));
    let rid3: Vec<u64> = (0..25u64).collect();
    let mut bb = BinaryBuilder::new();
    for r in 0..25u64 {
        let len = 5 + ((r as usize * 3) % 8);
        let byte = (0x40u8).wrapping_add((r % 26) as u8);
        bb.append_value(vec![byte; len]);
    }
    let rb3 = RecordBatch::try_new(
        schema3,
        vec![Arc::new(UInt64Array::from(rid3)), Arc::new(bb.finish())],
    )
    .unwrap();
    store.append(&rb3).unwrap();

    // Table 1: one u64 and one i32 column (smaller)
    let fid1_u64 = fid(1, 10);
    let mut md4 = HashMap::new();
    md4.insert("field_id".to_string(), u64::from(fid1_u64).to_string());
    let schema4 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md4),
    ]));
    let rid4: Vec<u64> = (0..30u64).collect();
    let vals4: Vec<u64> = (0..30u64).map(|x| x * 2 + 1).collect();
    let rb4 = RecordBatch::try_new(
        schema4,
        vec![
            Arc::new(UInt64Array::from(rid4)),
            Arc::new(UInt64Array::from(vals4)),
        ],
    )
    .unwrap();
    store.append(&rb4).unwrap();

    let fid1_i32 = fid(1, 20);
    let mut md5 = HashMap::new();
    md5.insert("field_id".to_string(), u64::from(fid1_i32).to_string());
    let schema5 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::Int32, false).with_metadata(md5),
    ]));
    let rid5: Vec<u64> = (0..15u64).collect();
    let vals5: Vec<i32> = (0..15i32).map(|x| x * x - 3).collect();
    let rb5 = RecordBatch::try_new(
        schema5,
        vec![
            Arc::new(UInt64Array::from(rid5)),
            Arc::new(Int32Array::from(vals5)),
        ],
    )
    .unwrap();
    store.append(&rb5).unwrap();

    // Table 2: tiny single-column table
    let fid2_u64 = fid(2, 1);
    let mut md6 = HashMap::new();
    md6.insert("field_id".to_string(), u64::from(fid2_u64).to_string());
    let schema6 = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md6),
    ]));
    let rid6: Vec<u64> = (0..10u64).collect();
    let vals6: Vec<u64> = (100..110u64).collect();
    let rb6 = RecordBatch::try_new(
        schema6,
        vec![
            Arc::new(UInt64Array::from(rid6)),
            Arc::new(UInt64Array::from(vals6)),
        ],
    )
    .unwrap();
    store.append(&rb6).unwrap();

    // Build sort index for some u64 columns to include perm nodes in DOT
    store.register_index(fid_u64, IndexKind::Sort).unwrap();
    store.register_index(fid1_u64, IndexKind::Sort).unwrap();

    store
}

#[cfg_attr(
    miri,
    ignore = "Miri lacks support for the mmap/mprotect calls used by MemPager"
)]
#[test]
fn snapshot_storage_dot_canonical() {
    let store = seed_small_store();
    let colors: HashMap<_, _> = HashMap::new();
    let dot = store.render_storage_as_dot(&colors);
    let canon = common::canonicalize_dot(&dot);
    common::assert_matches_golden(&canon, "tests/snapshots/storage_small.dot");
}

#[cfg_attr(
    miri,
    ignore = "Miri lacks support for the mmap/mprotect calls used by MemPager"
)]
#[test]
fn snapshot_storage_table_ascii() {
    let store = seed_small_store();
    let table = store.render_storage_as_formatted_string();
    common::assert_matches_golden(&table, "tests/snapshots/storage_small_table.txt");
}
