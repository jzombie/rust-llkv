use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::{
    ColumnStore, FIELD_ID_META_KEY, GatherNullPolicy, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_storage::pager::MemPager;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    // Use the SAME field_id that SQL uses for the first table's first column
    let fid = LogicalFieldId::for_user(1, 1);
    eprintln!(
        "Testing LWW with field_id for_user(1, 1) = {}",
        u64::from(fid)
    );
    let mut md = HashMap::new();
    md.insert(FIELD_ID_META_KEY.to_string(), u64::from(fid).to_string());

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("x", DataType::Int64, true).with_metadata(md),
    ]));

    // Initial insert: rowid 1 -> x=100
    println!("Step 1: Insert rowid=1 with x=100");
    let row_ids1 = Arc::new(UInt64Array::from(vec![1]));
    let values1 = Arc::new(Int64Array::from(vec![Some(100)]));
    let batch1 = RecordBatch::try_new(Arc::clone(&schema), vec![row_ids1, values1])?;
    store.append(&batch1)?;

    let result = store.gather_rows(&[fid], &[1], GatherNullPolicy::IncludeNulls)?;
    let arr = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    println!("After step 1, rowid=1 has x={:?}", arr.value(0));

    // LWW update: rowid 1 -> x=200
    println!("\nStep 2: Update rowid=1 to x=200 (LWW)");
    let row_ids2 = Arc::new(UInt64Array::from(vec![1]));
    let values2 = Arc::new(Int64Array::from(vec![Some(200)]));
    let batch2 = RecordBatch::try_new(Arc::clone(&schema), vec![row_ids2, values2])?;
    store.append(&batch2)?;

    let result = store.gather_rows(&[fid], &[1], GatherNullPolicy::IncludeNulls)?;
    let arr = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    println!("After step 2, rowid=1 has x={:?}", arr.value(0));

    // Another LWW update: rowid 1 -> x=300
    println!("\nStep 3: Update rowid=1 to x=300 (LWW)");
    let row_ids3 = Arc::new(UInt64Array::from(vec![1]));
    let values3 = Arc::new(Int64Array::from(vec![Some(300)]));
    let batch3 = RecordBatch::try_new(Arc::clone(&schema), vec![row_ids3, values3])?;
    store.append(&batch3)?;

    let result = store.gather_rows(&[fid], &[1], GatherNullPolicy::IncludeNulls)?;
    let arr = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    println!("After step 3, rowid=1 has x={:?}", arr.value(0));

    // Test with multiple rows - one starting as NULL
    println!("\nStep 4: Insert rowids 1,2,3 with x=10,20,NULL");
    let row_ids4 = Arc::new(UInt64Array::from(vec![1, 2, 3]));
    let values4 = Arc::new(Int64Array::from(vec![Some(10), Some(20), None]));
    let batch4 = RecordBatch::try_new(Arc::clone(&schema), vec![row_ids4, values4])?;
    store.append(&batch4)?;

    let result = store.gather_rows(&[fid], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    let arr = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    println!("After step 4:");
    for i in 0..3 {
        if arr.is_null(i) {
            println!("  rowid={} has x=NULL", i + 1);
        } else {
            println!("  rowid={} has x={}", i + 1, arr.value(i));
        }
    }

    // Update the NULL value to a non-NULL value
    println!("\nStep 5: UPDATE rowid 3 from NULL to 999");
    let row_ids5 = Arc::new(UInt64Array::from(vec![3]));
    let values5 = Arc::new(Int64Array::from(vec![Some(999)]));
    let batch5 = RecordBatch::try_new(Arc::clone(&schema), vec![row_ids5, values5])?;
    store.append(&batch5)?;

    let result = store.gather_rows(&[fid], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    let arr = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    println!("After step 5:");
    for i in 0..3 {
        if arr.is_null(i) {
            println!("  rowid={} has x=NULL", i + 1);
        } else {
            println!("  rowid={} has x={}", i + 1, arr.value(i));
        }
    }

    Ok(())
}
