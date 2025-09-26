use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::ColumnStore;
use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

fn fid_user(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

#[test]
fn unsorted_scan_works_without_index_u64() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let fid = fid_user(11);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(fid).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));

    let n = 100_000usize;
    let rid: Vec<u64> = (0..n as u64).collect();
    let vals: Vec<u64> = (0..n as u64).rev().collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();

    struct SumU64<'a> {
        acc: &'a std::cell::Cell<u128>,
    }
    impl<'a> PrimitiveVisitor for SumU64<'a> {
        fn u64_chunk(&mut self, a: &UInt64Array) {
            let mut s = 0u128;
            for i in 0..a.len() {
                s += a.value(i) as u128;
            }
            self.acc.set(self.acc.get() + s);
        }
    }
    impl<'a> PrimitiveSortedVisitor for SumU64<'a> {}
    impl<'a> PrimitiveWithRowIdsVisitor for SumU64<'a> {}
    impl<'a> PrimitiveSortedWithRowIdsVisitor for SumU64<'a> {}

    let acc = std::cell::Cell::new(0u128);
    let mut v = SumU64 { acc: &acc };
    // No index created; unsorted scan should still work
    store
        .scan(
            fid,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: false,

                limit: None,
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut v,
        )
        .unwrap();
    assert!(acc.get() > 0);
}

#[test]
fn unsorted_with_row_ids_works_without_index() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let fid = fid_user(12);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(fid).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::Int32, false).with_metadata(md),
    ]));

    let n = 10_000i32;
    let rid: Vec<u64> = (0..n as u64).collect();
    let vals: Vec<i32> = (0..n).rev().collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(Int32Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();

    struct Count<'a> {
        cnt: &'a std::cell::Cell<usize>,
    }
    impl<'a> PrimitiveWithRowIdsVisitor for Count<'a> {
        fn i32_chunk_with_rids(&mut self, _v: &Int32Array, r: &UInt64Array) {
            self.cnt.set(self.cnt.get() + r.len());
        }
    }
    impl<'a> PrimitiveVisitor for Count<'a> {}
    impl<'a> PrimitiveSortedVisitor for Count<'a> {}
    impl<'a> PrimitiveSortedWithRowIdsVisitor for Count<'a> {}

    let cnt = std::cell::Cell::new(0usize);
    let mut v = Count { cnt: &cnt };
    store
        .scan(
            fid,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: true,

                limit: None,
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut v,
        )
        .unwrap();
    assert_eq!(cnt.get(), 10_000usize);
}

#[test]
fn sorted_scan_without_index_returns_error() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let fid = fid_user(13);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(fid).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));
    let rid: Vec<u64> = (0..1000u64).collect();
    let vals: Vec<u64> = (0..1000u64).rev().collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();

    struct Noop;
    impl PrimitiveSortedVisitor for Noop {}
    impl PrimitiveVisitor for Noop {}
    impl PrimitiveWithRowIdsVisitor for Noop {}
    impl PrimitiveSortedWithRowIdsVisitor for Noop {}

    let mut v = Noop;
    let res = store.scan(
        fid,
        ScanOptions {
            sorted: true,
            reverse: false,
            with_row_ids: false,

            limit: None,
            offset: 0,
            include_nulls: false,
            nulls_first: false,
            anchor_row_id_field: None,
        },
        &mut v,
    );
    assert!(res.is_err());
}
