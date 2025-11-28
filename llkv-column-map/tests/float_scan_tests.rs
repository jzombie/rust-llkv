use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::store::scan::{
    PrimitiveSortedVisitor, PrimitiveSortedWithRowIdsVisitor, PrimitiveVisitor,
    PrimitiveWithRowIdsVisitor, ScanBuilder, ScanOptions,
};
use llkv_column_map::store::{ColumnStore, IndexKind, ROW_ID_COLUMN_NAME};
use llkv_storage::pager::MemPager;
use llkv_types::LogicalFieldId;

use rand::seq::SliceRandom;

fn assert_sorted_f64(values: &[f64]) {
    assert!(
        values
            .windows(2)
            .all(|w| w[0].total_cmp(&w[1]) != Ordering::Greater)
    );
}

fn assert_sorted_f32(values: &[f32]) {
    assert!(
        values
            .windows(2)
            .all(|w| w[0].total_cmp(&w[1]) != Ordering::Greater)
    );
}

struct CollectSortedF64 {
    values: Vec<f64>,
}
impl CollectSortedF64 {
    fn new() -> Self {
        Self { values: Vec::new() }
    }
}
impl PrimitiveVisitor for CollectSortedF64 {}
impl PrimitiveWithRowIdsVisitor for CollectSortedF64 {}
impl PrimitiveSortedWithRowIdsVisitor for CollectSortedF64 {}
impl PrimitiveSortedVisitor for CollectSortedF64 {
    fn f64_run(&mut self, a: &Float64Array, start: usize, len: usize) {
        let end = start + len;
        for i in start..end {
            self.values.push(a.value(i));
        }
    }
}

struct CollectSortedF32 {
    values: Vec<f32>,
}
impl CollectSortedF32 {
    fn new() -> Self {
        Self { values: Vec::new() }
    }
}
impl PrimitiveVisitor for CollectSortedF32 {}
impl PrimitiveWithRowIdsVisitor for CollectSortedF32 {}
impl PrimitiveSortedWithRowIdsVisitor for CollectSortedF32 {}
impl PrimitiveSortedVisitor for CollectSortedF32 {
    fn f32_run(&mut self, a: &Float32Array, start: usize, len: usize) {
        let end = start + len;
        for i in start..end {
            self.values.push(a.value(i));
        }
    }
}

struct CollectRowIds {
    out: Vec<u64>,
}
impl CollectRowIds {
    fn new() -> Self {
        Self { out: Vec::new() }
    }
}
impl PrimitiveVisitor for CollectRowIds {}
impl PrimitiveSortedVisitor for CollectRowIds {}
impl PrimitiveWithRowIdsVisitor for CollectRowIds {}
impl PrimitiveSortedWithRowIdsVisitor for CollectRowIds {
    fn f64_run_with_rids(&mut self, _v: &Float64Array, r: &UInt64Array, start: usize, len: usize) {
        let end = start + len;
        for i in start..end {
            self.out.push(r.value(i));
        }
    }
    fn f32_run_with_rids(&mut self, _v: &Float32Array, r: &UInt64Array, start: usize, len: usize) {
        let end = start + len;
        for i in start..end {
            self.out.push(r.value(i));
        }
    }
}

fn run_float64_case(mut values: Vec<f64>, low: f64, high: f64) {
    values.shuffle(&mut rand::rng());

    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager)).unwrap();
    let field_id = LogicalFieldId::for_user(2, 500);

    let mut md = HashMap::new();
    md.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::Float64, false).with_metadata(md),
    ]));

    let rid: Vec<u64> = (0..values.len() as u64).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid.clone()));
    let val_arr = Arc::new(Float64Array::from(values.clone())) as arrow::array::ArrayRef;
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();
    store.register_index(field_id, IndexKind::Sort).unwrap();

    // store.scan ascending
    let mut asc = CollectSortedF64::new();
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: true,
                ..Default::default()
            },
            &mut asc,
        )
        .unwrap();
    assert_sorted_f64(&asc.values);

    // Expected range
    let mut sorted_pairs: Vec<(f64, u64)> = values
        .iter()
        .enumerate()
        .map(|(idx, &v)| (v, rid[idx]))
        .collect();
    sorted_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let expected_vals: Vec<f64> = sorted_pairs
        .iter()
        .filter(|(v, _)| *v >= low && *v <= high)
        .map(|(v, _)| *v)
        .collect();
    let expected_rids: Vec<u64> = sorted_pairs
        .iter()
        .filter(|(v, _)| *v >= low && *v <= high)
        .map(|(_, r)| *r)
        .collect();
    assert!(!expected_vals.is_empty());

    let mut builder_vals = CollectSortedF64::new();
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            ..Default::default()
        })
        .with_range::<f64, _>(low..=high)
        .run(&mut builder_vals)
        .unwrap();
    assert_eq!(builder_vals.values, expected_vals);

    let mut builder_rids = CollectRowIds::new();
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            with_row_ids: true,
            ..Default::default()
        })
        .with_range::<f64, _>(low..=high)
        .run(&mut builder_rids)
        .unwrap();
    assert_eq!(builder_rids.out, expected_rids);
}

fn run_float32_case(mut values: Vec<f32>, low: f32, high: f32) {
    values.shuffle(&mut rand::rng());

    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(Arc::clone(&pager)).unwrap();
    let field_id = LogicalFieldId::for_user(2, 600);

    let mut md = HashMap::new();
    md.insert(
        llkv_column_map::store::FIELD_ID_META_KEY.to_string(),
        u64::from(field_id).to_string(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID_COLUMN_NAME, DataType::UInt64, false),
        Field::new("data", DataType::Float32, false).with_metadata(md),
    ]));

    let rid: Vec<u64> = (0..values.len() as u64).collect();
    let rid_arr = Arc::new(UInt64Array::from(rid.clone()));
    let val_arr = Arc::new(Float32Array::from(values.clone())) as arrow::array::ArrayRef;
    let batch = RecordBatch::try_new(schema, vec![rid_arr, val_arr]).unwrap();
    store.append(&batch).unwrap();
    store.register_index(field_id, IndexKind::Sort).unwrap();

    let mut asc = CollectSortedF32::new();
    store
        .scan(
            field_id,
            ScanOptions {
                sorted: true,
                ..Default::default()
            },
            &mut asc,
        )
        .unwrap();
    assert_sorted_f32(&asc.values);

    let mut sorted_pairs: Vec<(f64, u64)> = values
        .iter()
        .enumerate()
        .map(|(idx, &v)| ((v as f64), rid[idx]))
        .collect();
    sorted_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let expected_vals: Vec<f32> = sorted_pairs
        .iter()
        .filter(|(v, _)| *v >= low as f64 && *v <= high as f64)
        .map(|(v, _)| *v as f32)
        .collect();
    let expected_rids: Vec<u64> = sorted_pairs
        .iter()
        .filter(|(v, _)| *v >= low as f64 && *v <= high as f64)
        .map(|(_, r)| *r)
        .collect();
    assert!(!expected_vals.is_empty());

    let mut builder_vals = CollectSortedF32::new();
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            ..Default::default()
        })
        .with_range::<f32, _>(low..=high)
        .run(&mut builder_vals)
        .unwrap();
    assert_eq!(builder_vals.values, expected_vals);

    let mut builder_rids = CollectRowIds::new();
    ScanBuilder::new(&store, field_id)
        .options(ScanOptions {
            sorted: true,
            with_row_ids: true,
            ..Default::default()
        })
        .with_range::<f32, _>(low..=high)
        .run(&mut builder_rids)
        .unwrap();
    assert_eq!(builder_rids.out, expected_rids);
}

#[test]
fn float64_sorted_scan_and_ranges() {
    let values: Vec<f64> = vec![
        12.5, -3.2, 45.6, 0.0, 18.75, 99.9, -42.1, 5.5, 18.75, 64.0, -0.01, 23.4,
    ];
    run_float64_case(values, -10.0, 50.0);
}

#[test]
fn float32_sorted_scan_and_ranges() {
    let values: Vec<f32> = vec![
        1.25, -8.5, 12.0, 3.5, 6.75, 42.125, -16.0, 3.5, 9.0, 15.5, 27.25, -0.5,
    ];
    run_float32_case(values, -5.0, 20.0);
}
