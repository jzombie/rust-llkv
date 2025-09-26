use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{PrimitiveSortedVisitor, PrimitiveVisitor, ScanOptions};
use llkv_column_map::store::{ColumnStore, IndexKind};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

fn fid_user(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

fn make_schema_u64(field_id: LogicalFieldId) -> Arc<Schema> {
    let rid = Field::new("row_id", DataType::UInt64, false);
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    Arc::new(Schema::new(vec![rid, data_f]))
}

#[test]
fn pagination_unsorted_u64() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let field_id = fid_user(42);
    let schema = make_schema_u64(field_id);
    let n = 1000usize;
    // Use a distinct order to validate unsorted semantics (append order)
    let rid: Vec<u64> = (0..n as u64).collect();
    let vals: Vec<u64> = (0..n as u64).map(|x| x * 2).collect(); // 0,2,4,...
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals.clone()));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();

    // Collector for unsorted chunks
    struct CollectU64<'a> {
        out: &'a RefCell<Vec<u64>>,
    }
    impl<'a> llkv_column_map::store::scan::PrimitiveSortedVisitor for CollectU64<'a> {}
    impl<'a> llkv_column_map::store::scan::PrimitiveWithRowIdsVisitor for CollectU64<'a> {}
    impl<'a> llkv_column_map::store::scan::PrimitiveSortedWithRowIdsVisitor for CollectU64<'a> {}
    impl<'a> PrimitiveVisitor for CollectU64<'a> {
        fn u64_chunk(&mut self, a: &UInt64Array) {
            let mut v = self.out.borrow_mut();
            for i in 0..a.len() {
                v.push(a.value(i));
            }
        }
    }

    // Unbounded (baseline)
    let all: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectU64 { out: &all };
    store
        .scan(
            field_id,
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
            &mut vis,
        )
        .unwrap();
    let all = all.into_inner();
    assert_eq!(all.len(), n);
    assert_eq!(all, vals);

    // Limit only
    let k = 25usize;
    let take_only: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectU64 { out: &take_only };
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: false,

                limit: Some(k),
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut vis,
        )
        .unwrap();
    let take_only = take_only.into_inner();
    assert_eq!(take_only, vals[..k].to_vec());

    // Offset only
    let off = 975usize;
    let tail: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectU64 { out: &tail };
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: false,

                limit: None,
                offset: off,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut vis,
        )
        .unwrap();
    let tail = tail.into_inner();
    assert_eq!(tail, vals[off..].to_vec());

    // Offset + limit window
    let off = 100usize;
    let lim = 37usize;
    let win: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectU64 { out: &win };
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: false,

                limit: Some(lim),
                offset: off,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut vis,
        )
        .unwrap();
    let win = win.into_inner();
    assert_eq!(win, vals[off..off + lim].to_vec());
}

#[test]
fn pagination_sorted_u64() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let field_id = fid_user(43);
    let schema = make_schema_u64(field_id);
    let n = 2048usize;
    // Shuffle-like data: descending to ensure sort index changes order
    let rid: Vec<u64> = (0..n as u64).collect();
    let vals: Vec<u64> = (0..n as u64).rev().collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();
    // Build sort index
    store.register_index(field_id, IndexKind::Sort).unwrap();

    // Asc collector (runs are ascending; iterate forward)
    struct CollectAsc<'a> {
        out: &'a RefCell<Vec<u64>>,
    }
    impl<'a> llkv_column_map::store::scan::PrimitiveVisitor for CollectAsc<'a> {}
    impl<'a> llkv_column_map::store::scan::PrimitiveWithRowIdsVisitor for CollectAsc<'a> {}
    impl<'a> llkv_column_map::store::scan::PrimitiveSortedWithRowIdsVisitor for CollectAsc<'a> {}
    impl<'a> PrimitiveSortedVisitor for CollectAsc<'a> {
        fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            let mut v = self.out.borrow_mut();
            for i in s..e {
                v.push(a.value(i));
            }
        }
    }

    // Unbounded ascending
    let asc_all: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectAsc { out: &asc_all };
    store
        .scan(
            field_id,
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
            &mut vis,
        )
        .unwrap();
    let asc_all = asc_all.into_inner();
    assert_eq!(asc_all.len(), n);
    assert!(asc_all.windows(2).all(|w| w[0] <= w[1]));

    // Asc: offset+limit window
    let off = 123usize;
    let lim = 77usize;
    let asc_win: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectAsc { out: &asc_win };
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: true,
                reverse: false,
                with_row_ids: false,

                limit: Some(lim),
                offset: off,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut vis,
        )
        .unwrap();
    let asc_win = asc_win.into_inner();
    assert_eq!(asc_win, asc_all[off..off + lim].to_vec());

    // Desc collector (each run should be consumed descending within the run)
    struct CollectDesc<'a> {
        out: &'a RefCell<Vec<u64>>,
    }
    impl<'a> llkv_column_map::store::scan::PrimitiveVisitor for CollectDesc<'a> {}
    impl<'a> llkv_column_map::store::scan::PrimitiveWithRowIdsVisitor for CollectDesc<'a> {}
    impl<'a> llkv_column_map::store::scan::PrimitiveSortedWithRowIdsVisitor for CollectDesc<'a> {}
    impl<'a> PrimitiveSortedVisitor for CollectDesc<'a> {
        fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
            let mut i = s + l;
            let mut v = self.out.borrow_mut();
            while i > s {
                i -= 1;
                v.push(a.value(i));
            }
        }
    }

    // Unbounded descending
    let desc_all: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectDesc { out: &desc_all };
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: true,
                reverse: true,
                with_row_ids: false,

                limit: None,
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut vis,
        )
        .unwrap();
    let desc_all = desc_all.into_inner();
    assert_eq!(desc_all.len(), n);
    assert!(desc_all.windows(2).all(|w| w[0] >= w[1]));

    // Desc: limit only
    let lim = 50usize;
    let desc_take: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let mut vis = CollectDesc { out: &desc_take };
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: true,
                reverse: true,
                with_row_ids: false,

                limit: Some(lim),
                offset: 0,
                include_nulls: false,
                nulls_first: false,
                anchor_row_id_field: None,
            },
            &mut vis,
        )
        .unwrap();
    let desc_take = desc_take.into_inner();
    assert_eq!(desc_take, desc_all[..lim].to_vec());
}
