use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind, ProjectionBatch};
use llkv_column_map::types::{LogicalFieldId, Namespace};
use llkv_storage::pager::MemPager;

fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

#[test]
fn scan_builder_sorted_range_u64() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(42);

    // Build schema and ingest 0..10000 (shuffled via reverse values)
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));

    let n = 10_000usize;
    let rid: Vec<u64> = (0..n as u64).collect();
    let vals: Vec<u64> = (0..n as u64).rev().collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();
    store.register_index(field_id, IndexKind::Sort).unwrap();

    // Collect values within range [2000, 8000]
    struct Collect {
        out: Vec<u64>,
    }
    impl PrimitiveVisitor for Collect {}
    impl PrimitiveWithRowIdsVisitor for Collect {}
    impl PrimitiveSortedWithRowIdsVisitor for Collect {}
    impl PrimitiveSortedVisitor for Collect {
        fn u64_run(&mut self, a: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            for i in s..e {
                self.out.push(a.value(i));
            }
        }
    }

    let mut coll = Collect { out: Vec::new() };
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            reverse: false,
            with_row_ids: false,
            ..Default::default()
        })
        .with_range::<u64, _>(2000..=8000)
        .run(&mut coll)
        .unwrap();
    assert!(coll.out.windows(2).all(|w| w[0] <= w[1]));
    assert!(!coll.out.is_empty());
    assert!(*coll.out.first().unwrap() >= 2000 && *coll.out.last().unwrap() <= 8000);
}

#[test]
fn scan_builder_sorted_with_row_ids() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(77);

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));
    let n = 100_000usize;
    let rid: Vec<u64> = (0..n as u64).collect();
    let vals: Vec<u64> = (0..n as u64).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();
    store.register_index(field_id, IndexKind::Sort).unwrap();

    struct CollectRids {
        out: Vec<u64>,
    }
    impl PrimitiveSortedWithRowIdsVisitor for CollectRids {
        fn u64_run_with_rids(&mut self, _v: &UInt64Array, r: &UInt64Array, s: usize, l: usize) {
            let e = s + l;
            for i in s..e {
                self.out.push(r.value(i));
            }
        }
    }
    impl PrimitiveVisitor for CollectRids {}
    impl PrimitiveSortedVisitor for CollectRids {}
    impl PrimitiveWithRowIdsVisitor for CollectRids {}

    let mut coll = CollectRids { out: Vec::new() };
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            reverse: false,
            with_row_ids: true,
            ..Default::default()
        })
        .with_range::<u64, _>(10_000..=20_000)
        .run(&mut coll)
        .unwrap();
    assert!(!coll.out.is_empty());
    assert_eq!(coll.out.first().copied(), Some(10_000));
    assert_eq!(coll.out.last().copied(), Some(20_000));
}

#[test]
fn project_column_streams_row_aligned_batches() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let field_id = fid(5);

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        Field::new("data", DataType::UInt64, false).with_metadata(md),
    ]));

    let rid: Vec<u64> = (0..512u64).collect();
    let vals: Vec<u64> = rid.iter().map(|r| r * 2).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid));
    let val_arr = Arc::new(UInt64Array::from(vals));
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();

    let mut collected: Vec<ProjectionBatch> = Vec::new();
    store
        .project_column(field_id, ScanOptions::default(), |proj| {
            collected.push(proj);
            Ok(())
        })
        .unwrap();

    assert!(!collected.is_empty());
    let mut expected = 0u64;
    for batch in collected {
        let rids = batch.row_ids;
        let vals = batch.values.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(rids.len(), vals.len());
        for i in 0..rids.len() {
            assert_eq!(rids.value(i), expected);
            assert_eq!(vals.value(i), expected * 2);
            expected += 1;
        }
    }
    assert_eq!(expected, 512);
}
