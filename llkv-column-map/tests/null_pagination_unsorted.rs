use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, ROW_ID_COLUMN_NAME};
use llkv_storage::pager::MemPager;
use llkv_types::{LogicalFieldId, Namespace};

#[test]
fn unsorted_with_nulls_anchor_order() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    // Anchor rid 0..50
    let anchor_fid = LogicalFieldId::for_user_table_0(21);
    let mut md_a = HashMap::new();
    md_a.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(anchor_fid).to_string(),
    );
    let schema_a = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_a),
    ]));
    let r: Vec<u64> = (0..50).collect();
    let b = RecordBatch::try_new(
        schema_a.clone(),
        vec![
            Arc::new(UInt64Array::from(r.clone())),
            Arc::new(UInt64Array::from(r.clone())),
        ],
    )
    .unwrap();
    store.append(&b).unwrap();

    // Target: evens only
    let target_fid = LogicalFieldId::for_user_table_0(22);
    let mut md_t = HashMap::new();
    md_t.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(target_fid).to_string(),
    );
    let schema_t = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md_t),
    ]));
    let t_r: Vec<u64> = (0..50).filter(|x| x % 2 == 0).collect();
    let t_v: Vec<u64> = t_r.iter().map(|x| x * 5).collect();
    let tb = RecordBatch::try_new(
        schema_t,
        vec![
            Arc::new(UInt64Array::from(t_r)),
            Arc::new(UInt64Array::from(t_v)),
        ],
    )
    .unwrap();
    store.append(&tb).unwrap();

    struct Collect<'a> {
        rids: &'a std::cell::RefCell<Vec<u64>>,
        values: &'a std::cell::RefCell<Vec<u64>>,
    }
    impl<'a> PrimitiveVisitor for Collect<'a> {}
    impl<'a> PrimitiveSortedVisitor for Collect<'a> {}
    impl<'a> PrimitiveSortedWithRowIdsVisitor for Collect<'a> {
        fn null_run(&mut self, r: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            let mut rr = self.rids.borrow_mut();
            for i in s..e {
                rr.push(r.value(i));
            }
        }
    }
    impl<'a> PrimitiveWithRowIdsVisitor for Collect<'a> {
        fn u64_chunk_with_rids(&mut self, v: &UInt64Array, r: &UInt64Array) {
            let mut rr = self.rids.borrow_mut();
            let mut vv = self.values.borrow_mut();
            for i in 0..r.len() {
                rr.push(r.value(i));
                vv.push(v.value(i));
            }
        }
    }

    let rids = std::cell::RefCell::new(Vec::new());
    let values = std::cell::RefCell::new(Vec::new());
    let mut c = Collect {
        rids: &rids,
        values: &values,
    };
    store
        .scan(
            target_fid,
            ScanOptions {
                sorted: false,
                reverse: false,
                with_row_ids: true,
                limit: None,
                offset: 0,
                include_nulls: true,
                nulls_first: false,
                anchor_row_id_field: Some(anchor_fid.with_namespace(Namespace::RowIdShadow)),
            },
            &mut c,
        )
        .unwrap();

    let rr = rids.into_inner();
    assert_eq!(rr, (0..50).collect::<Vec<u64>>());
    let vv = values.into_inner();
    assert_eq!(vv.len(), (0..50).filter(|x| x % 2 == 0).count());
}
