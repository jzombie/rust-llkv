use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn schema_with_row_id(field_id: LogicalFieldId) -> Arc<Schema> {
    let rid = Field::new("row_id", DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data_f]))
}

fn seed_store_shuffled(
    n_rows: usize,
    batches: usize,
) -> (ColumnStore<MemPager>, LogicalFieldId, Vec<usize>) {
    assert!(n_rows % batches == 0);

    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(4242);
    let schema = schema_with_row_id(field_id);

    // Build a single global permutation of 0..N-1 and its inverse mapping value->row_id.
    let mut vals: Vec<usize> = (0..n_rows).collect();
    let mut rng = StdRng::seed_from_u64(0xA1B2_C3D4_E5F6_0708);
    vals.as_mut_slice().shuffle(&mut rng);

    let mut inv: Vec<usize> = vec![0; n_rows];
    for (i, &v) in vals.iter().enumerate() {
        inv[v] = i;
    }

    let batch_size = n_rows / batches;
    for b in 0..batches {
        let start = b * batch_size;
        let end = start + batch_size;

        // row_id is the global row index; values are the shuffled permutation slice.
        let rid: Vec<u64> = (start as u64..end as u64).collect();
        let data_slice: Vec<u64> = vals[start..end].iter().map(|&x| x as u64).collect();

        let rid_arr = Arc::new(UInt64Array::from(rid));
        let val_arr = Arc::new(UInt64Array::from(data_slice));
        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
        store.append(&batch).unwrap();
    }

    (store, field_id, inv)
}

#[test]
fn unsorted_scan_with_row_ids_is_correct() {
    const N: usize = 200_000;
    const BATCHES: usize = 8;

    let (store, field_id, inv) = seed_store_shuffled(N, BATCHES);

    // Verify by checking value->row_id mapping per element against the known inverse perm.
    let rid_fid = field_id.with_namespace(Namespace::RowIdShadow);
    let mut seen = 0usize;
    for res in store.scan_with_row_ids(field_id, rid_fid).unwrap() {
        let (vals_any, rids_arr) = res.unwrap();
        let vals = vals_any
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("values must be UInt64");
        assert_eq!(vals.len(), rids_arr.len());
        for i in 0..vals.len() {
            let v = vals.value(i) as usize;
            let rid = rids_arr.value(i) as usize;
            assert_eq!(rid, inv[v]);
        }
        seen += vals.len();
    }
    assert_eq!(seen, N);
}

#[test]
fn sorted_scan_with_row_ids_is_correct() {
    const N: usize = 200_000;
    const BATCHES: usize = 8;

    let (store, field_id, inv) = seed_store_shuffled(N, BATCHES);
    store.create_sort_index(field_id).unwrap();

    let rid_fid = field_id.with_namespace(Namespace::RowIdShadow);
    let mut count = 0usize;
    let mut prev: Option<u64> = None;

    let mut m = store.scan_sorted_with_row_ids(field_id, rid_fid).unwrap();
    while let Some((vals_dyn, rids, start, len)) = m.next_run() {
        let vals = vals_dyn.as_any().downcast_ref::<UInt64Array>().unwrap();
        let end = start + len;
        for i in start..end {
            let v = vals.value(i);
            if let Some(p) = prev { assert!(v >= p, "values must be non-decreasing in sorted scan"); }
            prev = Some(v);
            let rid = rids.value(i) as usize;
            assert_eq!(rid, inv[v as usize]);
        }
        count += len;
    }

    assert_eq!(count, N);
}
