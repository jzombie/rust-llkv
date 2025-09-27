use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::ColumnStore;
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn schema_with_row_id(field_id: LogicalFieldId) -> Arc<Schema> {
    let rid = Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false);
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
    let field_id = LogicalFieldId::for_user_table_0(4242);
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
    struct Check<'a> { inv: &'a [usize], seen: usize }
    impl llkv_column_map::store::iter::PrimitiveWithRowIdsVisitor for Check<'_> {
        fn u64_chunk(&mut self, vals: &UInt64Array, rids: &UInt64Array) {
            assert_eq!(vals.len(), rids.len());
            for i in 0..vals.len() {
                let v = vals.value(i) as usize;
                let rid = rids.value(i) as usize;
                assert_eq!(rid, self.inv[v]);
            }
            self.seen += vals.len();
        }
        fn i32_chunk(&mut self, _vals: &Int32Array, _rids: &UInt64Array) {}
    }
    let mut chk = Check { inv: &inv, seen: 0 };
    store.scan_with_row_ids_visit(field_id, rid_fid, &mut chk).unwrap();
    seen = chk.seen;
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

    struct SortedCheck<'a> { inv: &'a [usize], prev: Option<u64>, count: usize }
    impl llkv_column_map::store::iter::PrimitiveSortedWithRowIdsVisitor for SortedCheck<'_> {
        fn u64_run_with_rids(&mut self, vals: &UInt64Array, rids: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            for i in s..e {
                let v = vals.value(i);
                if let Some(p) = self.prev { assert!(v >= p); }
                self.prev = Some(v);
                let rid = rids.value(i) as usize;
                assert_eq!(rid, self.inv[v as usize]);
            }
            self.count += l;
        }
        fn i32_run_with_rids(&mut self, _v: &Int32Array, _r: &UInt64Array, _s: usize, _l: usize) {}
    }
    let mut sc = SortedCheck { inv: &inv, prev: None, count: 0 };
    store.scan_sorted_with_row_ids_visit(field_id, rid_fid, &mut sc).unwrap();
    count = sc.count;

    assert_eq!(count, N);
}
