use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_result::Result;
use llkv_storage::pager::MemPager;

fn logical_fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn schema_for_field(field_id: LogicalFieldId, name: &str, dtype: DataType) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data = Field::new(name, dtype, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data]))
}

fn seed_store() -> Result<(ColumnStore<MemPager>, LogicalFieldId, LogicalFieldId)> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    let rid: Vec<u64> = (0..10u64).collect();
    let rid_arr: ArrayRef = Arc::new(UInt64Array::from(rid.clone()));

    let fid_a = logical_fid(1);
    let schema_a = schema_for_field(fid_a, "col_a", DataType::UInt64);
    let vals_a: Vec<u64> = rid.iter().map(|v| v * 2).collect();
    let arr_a: ArrayRef = Arc::new(UInt64Array::from(vals_a));
    let batch_a = RecordBatch::try_new(schema_a, vec![rid_arr.clone(), arr_a])?;
    store.append(&batch_a)?;

    let fid_b = logical_fid(2);
    let schema_b = schema_for_field(fid_b, "col_b", DataType::Int32);
    let vals_b: Vec<i32> = rid.iter().map(|v| (*v as i32) - 3).collect();
    let arr_b: ArrayRef = Arc::new(Int32Array::from(vals_b));
    let batch_b = RecordBatch::try_new(schema_b, vec![rid_arr, arr_b])?;
    store.append(&batch_b)?;

    Ok((store, fid_a, fid_b))
}

#[test]
fn gather_rows_multi_matches_single_columns() -> Result<()> {
    let (store, fid_a, fid_b) = seed_store()?;
    let row_ids = vec![0, 3, 5, 9];

    let multi = store.gather_rows_multi(&[fid_a, fid_b], &row_ids)?;
    assert_eq!(multi.len(), 2);

    let single_a = store.gather_rows(fid_a, &row_ids)?;
    let single_b = store.gather_rows(fid_b, &row_ids)?;

    let multi_a = multi[0].as_any().downcast_ref::<UInt64Array>().unwrap();
    let expected_a = single_a.as_any().downcast_ref::<UInt64Array>().unwrap();
    assert_eq!(multi_a.values(), expected_a.values());

    let multi_b = multi[1].as_any().downcast_ref::<Int32Array>().unwrap();
    let expected_b = single_b.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(multi_b.values(), expected_b.values());

    Ok(())
}

#[test]
fn gather_rows_multi_rejects_duplicate_row_ids() -> Result<()> {
    let (store, fid_a, _) = seed_store()?;
    let err = store
        .gather_rows_multi(&[fid_a], &[1, 1])
        .expect_err("duplicate row_ids should error");
    assert!(matches!(err, llkv_result::Error::Internal(_)));
    Ok(())
}

#[test]
fn gather_rows_multi_empty_row_ids() -> Result<()> {
    let (store, fid_a, fid_b) = seed_store()?;
    let result = store.gather_rows_multi(&[fid_a, fid_b], &[])?;
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 0);
    assert_eq!(result[1].len(), 0);
    Ok(())
}
