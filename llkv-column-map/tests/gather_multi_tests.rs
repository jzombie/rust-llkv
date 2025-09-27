use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use llkv_column_map::ROW_ID_COLUMN_NAME;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::LogicalFieldId;
use llkv_result::Result;
use llkv_storage::pager::MemPager;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

fn schema_for_field(field_id: LogicalFieldId, name: &str, dtype: DataType) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data = Field::new(name, dtype, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data]))
}

fn schema_for_nullable_field(field_id: LogicalFieldId, name: &str, dtype: DataType) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data = Field::new(name, dtype, true).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data]))
}

fn seed_store() -> Result<(ColumnStore<MemPager>, LogicalFieldId, LogicalFieldId)> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    let rid: Vec<u64> = (0..10u64).collect();
    let rid_arr: ArrayRef = Arc::new(UInt64Array::from(rid.clone()));

    let fid_a = LogicalFieldId::for_user_table_0(1);
    let schema_a = schema_for_field(fid_a, "col_a", DataType::UInt64);
    let vals_a: Vec<u64> = rid.iter().map(|v| v * 2).collect();
    let arr_a: ArrayRef = Arc::new(UInt64Array::from(vals_a));
    let batch_a = RecordBatch::try_new(schema_a, vec![rid_arr.clone(), arr_a])?;
    store.append(&batch_a)?;

    let fid_b = LogicalFieldId::for_user_table_0(2);
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

    let multi = store.gather_rows_multi(&[fid_a, fid_b], &row_ids, false)?;
    assert_eq!(multi.num_columns(), 2);

    let single_a = store.gather_rows(fid_a, &row_ids, false)?;
    let single_b = store.gather_rows(fid_b, &row_ids, false)?;

    let multi_a = multi
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let expected_a = single_a.as_any().downcast_ref::<UInt64Array>().unwrap();
    assert_eq!(multi_a.values(), expected_a.values());

    let multi_b = multi
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let expected_b = single_b.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(multi_b.values(), expected_b.values());

    Ok(())
}

#[test]
fn gather_rows_multi_rejects_duplicate_row_ids() -> Result<()> {
    let (store, fid_a, _) = seed_store()?;
    let err = store
        .gather_rows_multi(&[fid_a], &[1, 1], false)
        .expect_err("duplicate row_ids should error");
    assert!(matches!(err, llkv_result::Error::Internal(_)));
    Ok(())
}

#[test]
fn gather_rows_multi_empty_row_ids() -> Result<()> {
    let (store, fid_a, fid_b) = seed_store()?;
    let result = store.gather_rows_multi(&[fid_a, fid_b], &[], false)?;
    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.column(0).len(), 0);
    assert_eq!(result.column(1).len(), 0);
    Ok(())
}

#[test]
fn gather_rows_multi_with_nulls() -> Result<()> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    let rid: Vec<u64> = (0..12u64).collect();
    let rid_arr: ArrayRef = Arc::new(UInt64Array::from(rid.clone()));

    let fid_dense = LogicalFieldId::for_user_table_0(10);
    let schema_dense = schema_for_field(fid_dense, "dense", DataType::UInt64);
    let dense_vals: Vec<u64> = rid.iter().map(|v| v * 7).collect();
    let dense_arr: ArrayRef = Arc::new(UInt64Array::from(dense_vals));
    let dense_batch = RecordBatch::try_new(schema_dense, vec![rid_arr.clone(), dense_arr.clone()])?;
    store.append(&dense_batch)?;

    let fid_sparse = LogicalFieldId::for_user_table_0(11);
    let schema_sparse = schema_for_nullable_field(fid_sparse, "sparse", DataType::Int32);
    let sparse_rids: Vec<u64> = rid.iter().copied().filter(|v| v % 3 != 1).collect();
    let sparse_values: Vec<i32> = sparse_rids.iter().map(|v| (*v as i32) - 5).collect();
    let sparse_rid_arr: ArrayRef = Arc::new(UInt64Array::from(sparse_rids.clone()));
    let sparse_val_arr: ArrayRef = Arc::new(Int32Array::from(sparse_values));
    let sparse_batch = RecordBatch::try_new(schema_sparse, vec![sparse_rid_arr, sparse_val_arr])?;
    store.append(&sparse_batch)?;

    let gather_ids: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    let multi = store.gather_rows_multi(&[fid_dense, fid_sparse], &gather_ids, true)?;
    assert_eq!(multi.num_columns(), 2);

    let dense_arr = multi
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("dense downcast");
    for (idx, row_id) in gather_ids.iter().enumerate() {
        assert_eq!(dense_arr.value(idx), row_id * 7);
    }

    let sparse_arr = multi
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("sparse downcast");
    assert_eq!(sparse_arr.len(), gather_ids.len());
    for (idx, row_id) in gather_ids.iter().enumerate() {
        if row_id % 3 == 1 {
            assert!(sparse_arr.is_null(idx));
        } else {
            let expected = (*row_id as i32) - 5;
            assert_eq!(sparse_arr.value(idx), expected);
        }
    }

    let err = store
        .gather_rows_multi(&[fid_sparse], &gather_ids, false)
        .expect_err("missing rows should error without null support");
    assert!(matches!(err, llkv_result::Error::Internal(_)));

    Ok(())
}

#[test]
fn gather_rows_multi_shuffled_with_nulls_preserves_alignment() -> Result<()> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager))?;

    let all_rids: Vec<u64> = (0..12u64).collect();
    let rid_arr: ArrayRef = Arc::new(UInt64Array::from(all_rids.clone()));

    let fid_dense = LogicalFieldId::for_user_table_0(20);
    let schema_dense = schema_for_field(fid_dense, "dense_shuffle", DataType::UInt64);
    let dense_vals: Vec<u64> = all_rids.iter().map(|rid| rid * 11).collect();
    let dense_arr: ArrayRef = Arc::new(UInt64Array::from(dense_vals.clone()));
    let dense_batch = RecordBatch::try_new(schema_dense, vec![rid_arr.clone(), dense_arr.clone()])?;
    store.append(&dense_batch)?;

    let fid_sparse = LogicalFieldId::for_user_table_0(21);
    let schema_sparse = schema_for_nullable_field(fid_sparse, "sparse_shuffle", DataType::Int32);
    let sparse_present_rids: Vec<u64> = all_rids
        .iter()
        .copied()
        .filter(|rid| rid % 4 != 2)
        .collect();
    let sparse_values: Vec<i32> = sparse_present_rids
        .iter()
        .map(|rid| (*rid as i32) * -3)
        .collect();
    let sparse_rid_arr: ArrayRef = Arc::new(UInt64Array::from(sparse_present_rids.clone()));
    let sparse_val_arr: ArrayRef = Arc::new(Int32Array::from(sparse_values.clone()));
    let sparse_batch = RecordBatch::try_new(schema_sparse, vec![sparse_rid_arr, sparse_val_arr])?;
    store.append(&sparse_batch)?;

    let mut shuffled_ids = all_rids.clone();
    let mut rng = StdRng::seed_from_u64(0x05A1_71E5_FCE3_73B4_u64);
    shuffled_ids.shuffle(&mut rng);
    assert_eq!(shuffled_ids.len(), all_rids.len());

    let multi = store.gather_rows_multi(&[fid_dense, fid_sparse], &shuffled_ids, true)?;
    assert_eq!(multi.num_columns(), 2);
    assert_eq!(multi.num_rows(), shuffled_ids.len());

    let multi_dense = multi
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("dense downcast");
    let multi_sparse = multi
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("sparse downcast");

    assert_eq!(multi_dense.len(), shuffled_ids.len());
    assert_eq!(multi_sparse.len(), shuffled_ids.len());

    let sparse_lookup: HashMap<u64, i32> = sparse_present_rids
        .iter()
        .copied()
        .zip(sparse_values.iter().copied())
        .collect();

    for (idx, row_id) in shuffled_ids.iter().enumerate() {
        let expected_dense = row_id * 11;
        assert_eq!(multi_dense.value(idx), expected_dense);

        if let Some(expected_sparse) = sparse_lookup.get(row_id) {
            assert!(!multi_sparse.is_null(idx));
            assert_eq!(multi_sparse.value(idx), *expected_sparse);
        } else {
            assert!(multi_sparse.is_null(idx));
        }
    }

    let single_dense = store.gather_rows(fid_dense, &shuffled_ids, false)?;
    let single_sparse = store.gather_rows(fid_sparse, &shuffled_ids, true)?;

    let single_dense = single_dense
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("single dense downcast");
    let single_sparse = single_sparse
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("single sparse downcast");

    assert_eq!(single_dense.values(), multi_dense.values());
    assert_eq!(single_sparse.len(), multi_sparse.len());
    for idx in 0..multi_sparse.len() {
        assert_eq!(single_sparse.is_null(idx), multi_sparse.is_null(idx));
        if !multi_sparse.is_null(idx) {
            assert_eq!(single_sparse.value(idx), multi_sparse.value(idx));
        }
    }

    Ok(())
}
