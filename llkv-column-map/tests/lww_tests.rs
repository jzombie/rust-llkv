use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::{
    ColumnStore, FIELD_ID_META_KEY, GatherNullPolicy, ROW_ID_COLUMN_NAME,
};
use llkv_column_map::types::LogicalFieldId;
use llkv_storage::pager::MemPager;

#[test]
fn test_lww() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    // Use the SAME field_id that SQL uses for the first table's first column
    let fid = LogicalFieldId::for_user(1, 1);
    let mut md = HashMap::new();
    md.insert(FIELD_ID_META_KEY.to_string(), u64::from(fid).to_string());

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("x", DataType::Int64, true).with_metadata(md),
    ]));

    // Initial insert: rowid 1 -> x=100
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
    assert_eq!(arr.value(0), 100);

    // LWW update: rowid 1 -> x=200
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
    assert_eq!(arr.value(0), 200);

    // Another LWW update: rowid 1 -> x=300
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
    assert_eq!(arr.value(0), 300);

    // Test with multiple rows - one starting as NULL
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
    assert_eq!(arr.value(0), 10);
    assert_eq!(arr.value(1), 20);
    assert!(arr.is_null(2));

    // Update the NULL value to a non-NULL value
    let row_ids5 = Arc::new(UInt64Array::from(vec![3]));
    let values5 = Arc::new(Int64Array::from(vec![Some(999)]));
    let batch5 = RecordBatch::try_new(Arc::clone(&schema), vec![row_ids5, values5])?;
    store.append(&batch5)?;

    let result = store.gather_rows(&[fid], &[3], GatherNullPolicy::IncludeNulls)?;
    let arr = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(arr.value(0), 999);

    Ok(())
}

#[test]
fn test_sequential_lww() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    // Use field_id 4294967297 (table_id=1, field_id=1) like SQL tables do
    let field_id = LogicalFieldId::for_user(1, 1);

    let mut md = HashMap::new();
    md.insert(
        FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("x", DataType::Int64, true).with_metadata(md),
    ]));

    // Step 1: Insert [1, 0, NULL] for row_ids [1, 2, 3]
    {
        let rowid_array: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let value_array: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(0), None]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid_array, value_array])?;

        store.append(&batch)?;
    }

    // Verify initial state
    let batch = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(arr.value(0), 1);
    assert_eq!(arr.value(1), 0);
    assert!(arr.is_null(2));

    // Step 2: Update row 1 to value 2 (simulating WHERE clause update)
    {
        let rowid_array: ArrayRef = Arc::new(UInt64Array::from(vec![1]));
        let value_array: ArrayRef = Arc::new(Int64Array::from(vec![2]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid_array, value_array])?;

        store.append(&batch)?;
    }

    // Verify after first update
    let batch = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(arr.value(0), 2);
    assert_eq!(arr.value(1), 0);
    assert!(arr.is_null(2));

    // Step 3: Update all 3 rows to value 99 (simulating no-WHERE update)
    {
        let rowid_array: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let value_array: ArrayRef = Arc::new(Int64Array::from(vec![99, 99, 99]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid_array, value_array])?;

        store.append(&batch)?;
    }

    // Verify final state
    let batch = store.gather_rows(&[field_id], &[1, 2, 3], GatherNullPolicy::IncludeNulls)?;
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(arr.value(0), 99);
    assert_eq!(arr.value(1), 99);
    assert_eq!(arr.value(2), 99);

    // Gather one at a time to see if there's a difference
    for rid in [1, 2, 3] {
        let batch = store.gather_rows(&[field_id], &[rid], GatherNullPolicy::IncludeNulls)?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 99);
    }

    Ok(())
}

#[test]
fn test_null_to_value_multicolumn() -> Result<(), Box<dyn std::error::Error>> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    let field_id_x = LogicalFieldId::for_user(1, 1);
    let field_id_y = LogicalFieldId::for_user(1, 2);

    let mut md_x = HashMap::new();
    md_x.insert(
        FIELD_ID_META_KEY.to_string(),
        u64::from(field_id_x).to_string(),
    );

    let mut md_y = HashMap::new();
    md_y.insert(
        FIELD_ID_META_KEY.to_string(),
        u64::from(field_id_y).to_string(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("x", DataType::Int64, true).with_metadata(md_x),
        Field::new("y", DataType::Utf8, true).with_metadata(md_y),
    ]));

    // Step 1: INSERT 3 rows like SQL does
    {
        let rowid = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let x = Arc::new(Int64Array::from(vec![Some(1), Some(0), None]));
        let y = Arc::new(StringArray::from(vec![
            Some("true"),
            Some("false"),
            Some("NULL"),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        store.append(&batch)?;
    }

    // Step 2: UPDATE row 1 several times (like WHERE clause updates)
    {
        let rowid = Arc::new(UInt64Array::from(vec![1]));
        let x = Arc::new(Int64Array::from(vec![2]));
        let y = Arc::new(StringArray::from(vec![Some("true")]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        store.append(&batch)?;
    }

    {
        let rowid = Arc::new(UInt64Array::from(vec![1]));
        let x = Arc::new(Int64Array::from(vec![2]));
        let y = Arc::new(StringArray::from(vec![Some("unknown")]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        store.append(&batch)?;
    }

    let batch = store.gather_rows(
        &[field_id_x, field_id_y],
        &[1, 2, 3],
        GatherNullPolicy::IncludeNulls,
    )?;
    let x_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let y_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(x_arr.value(0), 2);
    assert_eq!(y_arr.value(0), "unknown");

    // Step 3: UPDATE all 3 rows x=99 (keeping current y values)
    {
        let rowid = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let x = Arc::new(Int64Array::from(vec![99, 99, 99]));
        let y = Arc::new(StringArray::from(vec![
            Some("unknown"),
            Some("false"),
            Some("NULL"),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![rowid, x, y])?;
        store.append(&batch)?;
    }

    let batch = store.gather_rows(
        &[field_id_x, field_id_y],
        &[1, 2, 3],
        GatherNullPolicy::IncludeNulls,
    )?;
    let x_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let y_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(x_arr.value(0), 99);
    assert_eq!(x_arr.value(1), 99);
    assert_eq!(x_arr.value(2), 99);

    assert_eq!(y_arr.value(0), "unknown");
    assert_eq!(y_arr.value(1), "false");
    assert_eq!(y_arr.value(2), "NULL");

    Ok(())
}
