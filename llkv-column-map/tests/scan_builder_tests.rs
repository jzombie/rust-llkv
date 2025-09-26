use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{
    MultiProjectionBatch, PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor,
    PrimitiveVisitor, PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
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

    let rows_a: Vec<u64> = (0..256u64).collect();
    let prices_a: Vec<u64> = rows_a.iter().map(|rid| 100 + rid * 3).collect();
    let qty_a: Vec<u64> = rows_a.iter().map(|rid| (rid % 8) + 1).collect();
    store
        .append(&inventory_batch(
            price_fid,
            qty_fid,
            rows_a.clone(),
            prices_a,
            qty_a.clone(),
        ))
        .unwrap();

    let rows_b: Vec<u64> = (256..512u64).collect();
    let prices_b: Vec<u64> = rows_b.iter().map(|rid| 100 + rid * 3).collect();
    let qty_b: Vec<u64> = rows_b.iter().map(|rid| (rid % 8) + 1).collect();
    store
        .append(&inventory_batch(
            price_fid,
            qty_fid,
            rows_b.clone(),
            prices_b,
            qty_b.clone(),
        ))
        .unwrap();

    let quantities: Vec<u64> = qty_a.into_iter().chain(qty_b).collect();

    let mut collected: Vec<MultiProjectionBatch> = Vec::new();
    ScanBuilder::with_columns(&store, &[qty_fid])
        .project(|proj| {
            collected.push(proj);
            Ok(())
        })
        .unwrap();

    assert!(!collected.is_empty());
    assert!(
        collected.len() >= 2,
        "expected multiple batches due to chunking"
    );
    let total_rows: usize = collected.iter().map(|b| b.record_batch.num_rows()).sum();
    assert_eq!(total_rows, 512);

    let mut expected = 0u64;
    for batch in collected {
        let row_ids = batch.row_ids;
        let rb = batch.record_batch;
        assert_eq!(rb.num_columns(), 2);
        let schema = rb.schema();
        assert_eq!(schema.field(0).name(), "__row_id");
        let expected_qty_name = format!("{:?}-{}-{}", Namespace::UserData, 0, qty_fid.field_id());
        assert_eq!(schema.field(1).name(), &expected_qty_name);
        let rb_rids = rb.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let qtys = rb.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(row_ids.len(), rb_rids.len());
        assert_eq!(rb_rids.len(), qtys.len());
        for i in 0..row_ids.len() {
            let rid = row_ids.value(i);
            assert_eq!(rb_rids.value(i), rid);
            assert_eq!(rid, expected);
            assert_eq!(qtys.value(i), quantities[expected as usize]);
            expected += 1;
        }
    }
    assert_eq!(expected, 512);
}

#[test]
fn project_multiple_columns_streams_record_batches() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let price_fid = fid(7);
    let qty_fid = fid(8);

    let row_ids: Vec<u64> = (0..256u64).collect();
    let prices: Vec<u64> = row_ids.iter().map(|rid| 1_000 + rid * 5).collect();
    let quantities: Vec<u64> = row_ids.iter().map(|rid| (rid % 5) + 1).collect();
    let batch = inventory_batch(
        price_fid,
        qty_fid,
        row_ids.clone(),
        prices.clone(),
        quantities.clone(),
    );
    store.append(&batch).unwrap();

    let mut collected: Vec<MultiProjectionBatch> = Vec::new();
    ScanBuilder::with_columns(&store, &[price_fid, qty_fid])
        .project(|proj| {
            collected.push(proj);
            Ok(())
        })
        .unwrap();

    assert!(!collected.is_empty());
    let mut expected_idx = 0usize;
    for batch in collected {
        let row_ids_arr = batch.row_ids;
        let rb = batch.record_batch;
        assert_eq!(rb.num_columns(), 3);
        let schema = rb.schema();
        assert_eq!(schema.field(0).name(), "__row_id");
        let expected_price_name =
            format!("{:?}-{}-{}", Namespace::UserData, 0, price_fid.field_id());
        let expected_qty_name = format!("{:?}-{}-{}", Namespace::UserData, 0, qty_fid.field_id());
        assert_eq!(schema.field(1).name(), &expected_price_name);
        assert_eq!(schema.field(2).name(), &expected_qty_name);

        let rb_rids = rb.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let rb_price = rb.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        let rb_qty = rb.column(2).as_any().downcast_ref::<UInt64Array>().unwrap();

        assert_eq!(row_ids_arr.len(), rb_rids.len());
        assert_eq!(rb_rids.len(), rb_price.len());
        assert_eq!(rb_price.len(), rb_qty.len());

        for i in 0..row_ids_arr.len() {
            let rid = row_ids_arr.value(i) as usize;
            assert_eq!(rid, expected_idx);
            assert_eq!(rb_rids.value(i) as usize, rid);
            assert_eq!(rb_price.value(i), prices[rid]);
            assert_eq!(rb_qty.value(i), quantities[rid]);
            expected_idx += 1;
        }
    }
    assert_eq!(expected_idx, row_ids.len());
}

#[test]
fn project_deduplicates_projection_slice() {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();
    let base_fid = fid(9);
    let extra_fid = fid(10);

    let row_ids: Vec<u64> = (0..128u64).collect();
    let base_vals: Vec<u64> = row_ids.iter().map(|rid| rid * 2).collect();
    let extra_vals: Vec<u64> = row_ids.iter().map(|rid| rid * 5).collect();
    let batch = inventory_batch(
        base_fid,
        extra_fid,
        row_ids.clone(),
        base_vals.clone(),
        extra_vals.clone(),
    );
    store.append(&batch).unwrap();

    let mut collected: Vec<MultiProjectionBatch> = Vec::new();
    ScanBuilder::with_columns(&store, &[base_fid, extra_fid, base_fid, extra_fid])
        .project(|proj| {
            collected.push(proj);
            Ok(())
        })
        .unwrap();

    assert_eq!(collected.len(), 1, "single chunk expected in test setup");
    let batch = &collected[0];
    let rb = &batch.record_batch;
    assert_eq!(rb.num_columns(), 3, "row id + unique columns");
    let schema = rb.schema();
    assert_eq!(schema.field(0).name(), "__row_id");
    let expected_base_name = format!("{:?}-{}-{}", Namespace::UserData, 0, base_fid.field_id());
    let expected_extra_name = format!("{:?}-{}-{}", Namespace::UserData, 0, extra_fid.field_id());
    assert_eq!(schema.field(1).name(), &expected_base_name);
    assert_eq!(schema.field(2).name(), &expected_extra_name);

    let row_ids_arr = rb.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
    let base_arr = rb.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
    let extra_arr = rb.column(2).as_any().downcast_ref::<UInt64Array>().unwrap();
    assert_eq!(row_ids_arr.len(), row_ids.len());
    for i in 0..row_ids_arr.len() {
        assert_eq!(row_ids_arr.value(i), row_ids[i]);
        assert_eq!(base_arr.value(i), base_vals[i]);
        assert_eq!(extra_arr.value(i), extra_vals[i]);
    }
}
