use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ProjectionBatch, ScanBuilder, ScanOptions,
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

fn field_with_fid(name: &str, dtype: DataType, fid: LogicalFieldId) -> Field {
    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(fid).to_string());
    Field::new(name, dtype, false).with_metadata(md)
}

fn inventory_batch(
    price_fid: LogicalFieldId,
    qty_fid: LogicalFieldId,
    row_ids: Vec<u64>,
    prices: Vec<u64>,
    quantities: Vec<u64>,
) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::UInt64, false),
        field_with_fid("price", DataType::UInt64, price_fid),
        field_with_fid("qty", DataType::UInt64, qty_fid),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
        Arc::new(UInt64Array::from(prices)) as ArrayRef,
        Arc::new(UInt64Array::from(quantities)) as ArrayRef,
    ];
    RecordBatch::try_new(schema, columns).expect("inventory batch")
}

#[test]
fn sorted_price_range_scan() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let price_fid = fid(42);
    let qty_fid = fid(43);

    let n = 10_000usize;
    let row_ids: Vec<u64> = (0..n as u64).collect();
    let prices: Vec<u64> = row_ids.iter().rev().copied().collect();
    let quantities: Vec<u64> = row_ids.iter().map(|rid| (rid % 7) + 1).collect();
    let batch = inventory_batch(
        price_fid,
        qty_fid,
        row_ids.clone(),
        prices.clone(),
        quantities,
    );
    store.append(&batch).unwrap();
    store.register_index(price_fid, IndexKind::Sort).unwrap();

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
    ScanBuilder::new(&store, price_fid)
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
fn sorted_price_scan_with_row_ids() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let price_fid = fid(77);
    let qty_fid = fid(78);

    let n = 100_000usize;
    let row_ids: Vec<u64> = (0..n as u64).collect();
    let prices: Vec<u64> = row_ids.clone();
    let quantities: Vec<u64> = row_ids.iter().map(|rid| (rid % 11) + 1).collect();
    let batch = inventory_batch(
        price_fid,
        qty_fid,
        row_ids.clone(),
        prices.clone(),
        quantities,
    );
    store.append(&batch).unwrap();
    store.register_index(price_fid, IndexKind::Sort).unwrap();

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
    ScanBuilder::new(&store, price_fid)
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
    let price_fid = fid(5);
    let qty_fid = fid(6);

    let row_ids: Vec<u64> = (0..512u64).collect();
    let prices: Vec<u64> = row_ids.iter().map(|rid| 100 + rid * 3).collect();
    let quantities: Vec<u64> = row_ids.iter().map(|rid| (rid % 8) + 1).collect();
    let batch = inventory_batch(
        price_fid,
        qty_fid,
        row_ids.clone(),
        prices,
        quantities.clone(),
    );
    store.append(&batch).unwrap();

    let mut collected: Vec<ProjectionBatch> = Vec::new();
    ScanBuilder::new(&store, qty_fid)
        .project(|proj| {
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
            assert_eq!(vals.value(i), quantities[expected as usize]);
            expected += 1;
        }
    }
    assert_eq!(expected, 512);
}
