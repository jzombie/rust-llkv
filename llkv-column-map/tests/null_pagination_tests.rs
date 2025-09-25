use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

#[test]
fn sorted_with_nulls_last_pagination() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Anchor row_id column (dense 0..100)
    let anchor_fid = fid(1);
    let mut md_anchor = HashMap::new();
    md_anchor.insert("field_id".to_string(), u64::from(anchor_fid).to_string());
    let schema_anchor = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_anchor),
    ]));
    let r0: Vec<u64> = (0..100).collect();
    let a0 = RecordBatch::try_new(
        schema_anchor.clone(),
        vec![
            Arc::new(UInt64Array::from(r0.clone())),
            Arc::new(UInt64Array::from(r0.clone())),
        ],
    )
    .unwrap();
    store.append(&a0).unwrap();

    // Target column: only even row_ids present
    let target_fid = fid(2);
    let mut md_t = HashMap::new();
    md_t.insert("field_id".to_string(), u64::from(target_fid).to_string());
    let schema_t = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_t),
    ]));
    let t_rids: Vec<u64> = (0..100).filter(|x| x % 2 == 0).collect();
    let t_vals: Vec<u64> = t_rids.iter().map(|x| x * 10).collect();
    let t0 = RecordBatch::try_new(
        schema_t.clone(),
        vec![
            Arc::new(UInt64Array::from(t_rids)),
            Arc::new(UInt64Array::from(t_vals)),
        ],
    )
    .unwrap();
    store.append(&t0).unwrap();
    store.register_index(target_fid, IndexKind::Sort).unwrap();

    struct Collect<'a> {
        vals: &'a std::cell::RefCell<Vec<u64>>,
        rids: &'a std::cell::RefCell<Vec<u64>>,
    }
    impl<'a> PrimitiveVisitor for Collect<'a> {}
    impl<'a> PrimitiveWithRowIdsVisitor for Collect<'a> {}
    impl<'a> PrimitiveSortedVisitor for Collect<'a> {}
    impl<'a> PrimitiveSortedWithRowIdsVisitor for Collect<'a> {
        fn u64_run_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            let mut vv = self.vals.borrow_mut();
            let mut rr = self.rids.borrow_mut();
            for i in s..e {
                vv.push(v.value(i));
                rr.push(r.value(i));
            }
        }
        fn null_run(&mut self, r: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            let mut rr = self.rids.borrow_mut();
            for i in s..e {
                rr.push(r.value(i));
            }
        }
    }

    let vals = std::cell::RefCell::new(Vec::new());
    let rids = std::cell::RefCell::new(Vec::new());
    let mut c = Collect {
        vals: &vals,
        rids: &rids,
    };

    // Page with offset crossing into nulls (evens are values, odds are nulls)
    store
        .scan(
            target_fid,
            ScanOptions {
                sorted: true,
                reverse: false,
                with_row_ids: true,
                limit: Some(15),
                offset: 10,
                include_nulls: true,
                nulls_first: false,
                anchor_row_id_field: Some(anchor_fid.with_namespace(Namespace::RowIdShadow)),
            },
            &mut c,
        )
        .unwrap();

    let vv = vals.into_inner();
    let rr = rids.into_inner();
    // With NULLS LAST and values sorted ascending, the combined stream starts with all even rids (values), then nulls.
    // Offset 10, limit 15 => rids 20,22,...,48
    assert_eq!(rr.len(), 15);
    let expected: Vec<u64> = (20..50).filter(|x| x % 2 == 0).collect();
    assert_eq!(rr, expected);
    assert_eq!(vv.len(), 15);
}
