use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::{self, ColumnStore};
use llkv_column_map::types::{LogicalFieldId, Namespace};

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn schema_with_row_id(field: Field) -> Arc<Schema> {
    let rid = Field::new("row_id", DataType::UInt64, false);
    Arc::new(Schema::new(vec![rid, field]))
}

#[test]
fn visitor_unsorted_and_sorted_u64() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(6001);

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    let schema = schema_with_row_id(data_f);

    // Two chunks to exercise coalescing
    for b in 0..2u64 {
        let start = b * 5;
        let rid: Vec<u64> = (start..start + 5).collect();
        let vals: Vec<u64> = vec![5 - b, 4 - b, 3 - b, 2 - b, 1 - b]; // unsorted across chunks
        let rid_arr = Arc::new(UInt64Array::from(rid));
        let val_arr = Arc::new(UInt64Array::from(vals));
        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
        store.append(&batch).unwrap();
    }

    // Unsorted visit: sum values
    struct SumVisitor(u128);
    impl store::iter::PrimitiveVisitor for SumVisitor {
        fn u64_chunk(&mut self, a: &UInt64Array) {
            for i in 0..a.len() {
                self.0 += a.value(i) as u128;
            }
        }
    }
    let mut sv = SumVisitor(0);
    store.scan_visit(field_id, &mut sv).unwrap();
    assert!(sv.0 > 0);

    // Sorted visit: ensure non-decreasing
    store.create_sort_index(field_id).unwrap();
    struct SortedCheck(Option<u64>);
    impl store::iter::PrimitiveSortedVisitor for SortedCheck {
        fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            for i in s..e {
                let v = a.value(i);
                if let Some(p) = self.0 { assert!(v >= p); }
                self.0 = Some(v);
            }
        }
    }
    let mut sc = SortedCheck(None);
    store.scan_sorted_visit(field_id, &mut sc).unwrap();
}

#[test]
fn visitor_unsorted_with_row_ids_i32() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(6002);

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::Int32, false).with_metadata(md);
    let schema = schema_with_row_id(data_f);

    let rid: Vec<u64> = (0..10u64).collect();
    let vals: Vec<i32> = (0..10i32).rev().collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(Int32Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr.clone(), val_arr]).unwrap();
    store.append(&batch).unwrap();

    let rid_fid = field_id.with_namespace(Namespace::RowIdShadow);

    struct Checker { sum: i128, sum_r: u128 }
    impl store::iter::PrimitiveWithRowIdsVisitor for Checker {
        fn i32_chunk(&mut self, vals: &Int32Array, rids: &UInt64Array) {
            for i in 0..vals.len() {
                self.sum += vals.value(i) as i128;
                self.sum_r += rids.value(i) as u128;
            }
        }
    }
    let mut ck = Checker { sum: 0, sum_r: 0 };
    store.scan_with_row_ids_visit(field_id, rid_fid, &mut ck).unwrap();
    assert!(ck.sum != 0 || ck.sum_r != 0);
}

