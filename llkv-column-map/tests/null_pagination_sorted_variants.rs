use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind, ROW_ID_COLUMN_NAME};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

fn seed_anchor_and_target() -> (ColumnStore<MemPager>, LogicalFieldId, LogicalFieldId) {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    // Anchor rid 0..100
    let anchor_fid = LogicalFieldId::for_default_user(11);
    let mut md_a = HashMap::new();
    md_a.insert("field_id".to_string(), u64::from(anchor_fid).to_string());
    let schema_a = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_a),
    ]));
    let r: Vec<u64> = (0..100).collect();
    let b = RecordBatch::try_new(
        schema_a.clone(),
        vec![
            Arc::new(UInt64Array::from(r.clone())),
            Arc::new(UInt64Array::from(r.clone())),
        ],
    )
    .unwrap();
    store.append(&b).unwrap();
    // Target: values only on multiples of 3
    let target_fid = LogicalFieldId::for_default_user(12);
    let mut md_t = HashMap::new();
    md_t.insert("field_id".to_string(), u64::from(target_fid).to_string());
    let schema_t = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_t),
    ]));
    let t_r: Vec<u64> = (0..100).filter(|x| x % 3 == 0).collect();
    let t_v: Vec<u64> = t_r.iter().map(|x| x * 10).collect();
    let tb = RecordBatch::try_new(
        schema_t,
        vec![
            Arc::new(UInt64Array::from(t_r)),
            Arc::new(UInt64Array::from(t_v)),
        ],
    )
    .unwrap();
    store.append(&tb).unwrap();
    store.register_index(target_fid, IndexKind::Sort).unwrap();
    (store, anchor_fid, target_fid)
}

struct Collect<'a> {
    vals: &'a std::cell::RefCell<Vec<u64>>,
    rids: &'a std::cell::RefCell<Vec<u64>>,
    rev: bool,
}
impl<'a> PrimitiveVisitor for Collect<'a> {}
impl<'a> PrimitiveWithRowIdsVisitor for Collect<'a> {}
impl<'a> PrimitiveSortedVisitor for Collect<'a> {}
impl<'a> PrimitiveSortedWithRowIdsVisitor for Collect<'a> {
    fn u64_run_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
        let mut vv = self.vals.borrow_mut();
        let mut rr = self.rids.borrow_mut();
        if self.rev {
            let mut i = s + l;
            while i > s {
                i -= 1;
                vv.push(v.value(i));
                rr.push(r.value(i));
            }
        } else {
            let e = s + l;
            for i in s..e {
                vv.push(v.value(i));
                rr.push(r.value(i));
            }
        }
    }
    fn null_run(&mut self, r: &UInt64Array, s: usize, l: usize) {
        let mut rr = self.rids.borrow_mut();
        if self.rev {
            let mut i = s + l;
            while i > s {
                i -= 1;
                rr.push(r.value(i));
            }
        } else {
            let e = s + l;
            for i in s..e {
                rr.push(r.value(i));
            }
        }
    }
}

#[test]
fn sorted_asc_nulls_first_nonpaginate() {
    let (store, anchor_fid, target_fid) = seed_anchor_and_target();
    let vals = std::cell::RefCell::new(Vec::new());
    let rids = std::cell::RefCell::new(Vec::new());
    let mut c = Collect {
        vals: &vals,
        rids: &rids,
        rev: false,
    };
    store
        .scan(
            target_fid,
            ScanOptions {
                sorted: true,
                reverse: false,
                with_row_ids: true,
                limit: None,
                offset: 0,
                include_nulls: true,
                nulls_first: true,
                anchor_row_id_field: Some(anchor_fid.with_namespace(Namespace::RowIdShadow)),
            },
            &mut c,
        )
        .unwrap();
    let inner_rids = rids.into_inner();
    // First should be nulls: verify the first 10 rids are not divisible by 3
    assert!(inner_rids.len() >= 10);
    for (i, inner_rid) in inner_rids.iter().enumerate().take(10) {
        assert_ne!(
            inner_rid % 3,
            0,
            "expected null rid at head (not multiple of 3), got {} at {}",
            inner_rid,
            i
        );
    }
}

#[test]
fn sorted_desc_nulls_last_paginated() {
    let (store, anchor_fid, target_fid) = seed_anchor_and_target();
    let vals = std::cell::RefCell::new(Vec::new());
    let rids = std::cell::RefCell::new(Vec::new());
    let mut c = Collect {
        vals: &vals,
        rids: &rids,
        rev: true,
    };
    store
        .scan(
            target_fid,
            ScanOptions {
                sorted: true,
                reverse: true,
                with_row_ids: true,
                limit: Some(10),
                offset: 0,
                include_nulls: true,
                nulls_first: false,
                anchor_row_id_field: Some(anchor_fid.with_namespace(Namespace::RowIdShadow)),
            },
            &mut c,
        )
        .unwrap();
    let rr = rids.into_inner();
    // With nulls_last and reverse, start with highest present values (multiples of 3 near 99)
    assert_eq!(rr.len(), 10);
    // Expect first is 99 if it's a multiple of 3, else 98 or 97; but nulls_last => values first => should be 99 (present)
    assert_eq!(rr[0], 99);
}
