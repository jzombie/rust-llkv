use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::{ColumnStore, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

fn fid_user(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

#[test]
fn presence_index_cross_column_queries() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Column A schema
    let fid_a = fid_user(101);
    let mut md_a = HashMap::new();
    md_a.insert("field_id".to_string(), u64::from(fid_a).to_string());
    let schema_a = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_a),
    ]));

    // Column B schema
    let fid_b = fid_user(102);
    let mut md_b = HashMap::new();
    md_b.insert("field_id".to_string(), u64::from(fid_b).to_string());
    let schema_b = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_b),
    ]));

    // Append to A: [0, 1000) and [2000, 3000)
    let a_r0: Vec<u64> = (0..1000).collect();
    let a_v0: Vec<u64> = a_r0.iter().map(|x| x * 10).collect();
    let a_b0 = RecordBatch::try_new(
        schema_a.clone(),
        vec![
            Arc::new(UInt64Array::from(a_r0)),
            Arc::new(UInt64Array::from(a_v0)),
        ],
    )
    .unwrap();
    store.append(&a_b0).unwrap();

    let a_r1: Vec<u64> = (2000..3000).collect();
    let a_v1: Vec<u64> = a_r1.iter().map(|x| x * 10).collect();
    let a_b1 = RecordBatch::try_new(
        schema_a,
        vec![
            Arc::new(UInt64Array::from(a_r1)),
            Arc::new(UInt64Array::from(a_v1)),
        ],
    )
    .unwrap();
    store.append(&a_b1).unwrap();

    // Append to B: [500, 1500) and [4000, 4500)
    let b_r0: Vec<u64> = (500..1500).collect();
    let b_v0: Vec<u64> = b_r0.iter().map(|x| x * 7).collect();
    let b_b0 = RecordBatch::try_new(
        schema_b.clone(),
        vec![
            Arc::new(UInt64Array::from(b_r0)),
            Arc::new(UInt64Array::from(b_v0)),
        ],
    )
    .unwrap();
    store.append(&b_b0).unwrap();

    let b_r1: Vec<u64> = (4000..4500).collect();
    let b_v1: Vec<u64> = b_r1.iter().map(|x| x * 7).collect();
    let b_b1 = RecordBatch::try_new(
        schema_b,
        vec![
            Arc::new(UInt64Array::from(b_r1)),
            Arc::new(UInt64Array::from(b_v1)),
        ],
    )
    .unwrap();
    store.append(&b_b1).unwrap();

    // Presence checks: A contains its ranges, not B-only ranges
    assert!(store.has_row_id(fid_a, 10).unwrap());
    assert!(store.has_row_id(fid_a, 2500).unwrap());
    assert!(!store.has_row_id(fid_a, 1250).unwrap());
    assert!(!store.has_row_id(fid_a, 4300).unwrap());

    // Presence checks: B contains its ranges, not A-only ranges
    assert!(store.has_row_id(fid_b, 1250).unwrap());
    assert!(store.has_row_id(fid_b, 4300).unwrap());
    assert!(!store.has_row_id(fid_b, 250).unwrap());
    assert!(!store.has_row_id(fid_b, 2750).unwrap());
}
