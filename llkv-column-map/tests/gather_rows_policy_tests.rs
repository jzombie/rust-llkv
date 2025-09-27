use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::{ColumnStore, GatherNullPolicy, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::LogicalFieldId;
use llkv_storage::pager::MemPager;

#[test]
fn drop_nulls_policy_removes_null_rows() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager)).expect("store");

    let fid = LogicalFieldId::for_user(42, 7);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(fid).to_string());

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("col", DataType::Int32, true).with_metadata(md),
    ]));

    let row_ids = Arc::new(UInt64Array::from(vec![1, 2, 3]));
    let values = Arc::new(Int32Array::from(vec![Some(10), None, Some(30)]));
    let batch = RecordBatch::try_new(schema, vec![row_ids, values]).expect("batch");
    store.append(&batch).expect("append");

    let raw = store
        .gather_rows(&[fid], &[1, 2, 3, 4], GatherNullPolicy::IncludeNulls)
        .expect("include nulls");
    let raw = raw
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("int32 array");
    assert_eq!(raw.len(), 4);
    assert!(raw.is_null(1));
    assert!(raw.is_null(3));

    let dropped = store
        .gather_rows(&[fid], &[1, 2, 3, 4], GatherNullPolicy::DropNulls)
        .expect("drop nulls");
    let dropped = dropped
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("int32 array");
    assert_eq!(dropped.len(), 2);
    assert_eq!(dropped.values(), &[10, 30]);
}
