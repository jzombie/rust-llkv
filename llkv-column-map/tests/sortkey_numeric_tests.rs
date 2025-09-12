use std::borrow::Cow;
use std::ops::Bound;
use std::sync::Arc;

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode, ValueOrderPolicy};

fn le64(v: u64) -> Vec<u8> { v.to_le_bytes().to_vec() }
fn lei64(v: i64) -> Vec<u8> { v.to_le_bytes().to_vec() }
fn lef64(v: f64) -> Vec<u8> { v.to_bits().to_le_bytes().to_vec() }

fn key_bytes(i: u64) -> Vec<u8> { format!("k{:08}", i).into_bytes() }

#[test]
fn value_sorted_u64_le_with_sort_keys() {
    let p = Arc::new(MemPager::default());
    let store = ColumnStore::open(p);
    let fid: LogicalFieldId = 42_001;

    // Two segments with overlapping ranges
    let opts = AppendOptions { mode: ValueMode::ForceFixed(8), value_order: Some(ValueOrderPolicy::UnsignedLe), ..Default::default() };
    let mut items = Vec::new();
    for i in 0..100u64 { items.push((Cow::Owned(key_bytes(i)), Cow::Owned(le64(1000 - i)))); }
    store.append_many(vec![Put { field_id: fid, items }], opts.clone());

    let mut items2 = Vec::new();
    for i in 50..150u64 { items2.push((Cow::Owned(key_bytes(i+10_000)), Cow::Owned(le64(i)))); }
    store.append_many(vec![Put { field_id: fid, items: items2 }], opts);

    // Scan full range in value order; verify monotonic increasing
    let it = store.scan_values_lww(fid, ValueScanOpts { order_by: OrderBy::Value, dir: Direction::Forward, lo: Bound::Unbounded, hi: Bound::Unbounded, ..Default::default() }).unwrap();
    let mut prev: Option<u64> = None;
    for row in it {
        let s = row.value.as_slice();
        if s.len() == 8 {
            let mut a = [0u8; 8];
            a.copy_from_slice(s);
            let v = u64::from_le_bytes(a);
            if let Some(pv) = prev { assert!(pv <= v); }
            prev = Some(v);
        }
    }
}

#[test]
fn value_sorted_i64_le_with_sort_keys() {
    let p = Arc::new(MemPager::default());
    let store = ColumnStore::open(p);
    let fid: LogicalFieldId = 42_002;
    let opts = AppendOptions { mode: ValueMode::ForceFixed(8), value_order: Some(ValueOrderPolicy::SignedLe), ..Default::default() };
    let vals = [-5i64, -1, 0, 1, 100, -1000, i64::MIN+1, i64::MAX];
    let mut items = Vec::new();
    for (i, v) in vals.iter().enumerate() { items.push((Cow::Owned(key_bytes(i as u64)), Cow::Owned(lei64(*v)))); }
    store.append_many(vec![Put { field_id: fid, items }], opts);

    let it = store.scan_values_lww(fid, ValueScanOpts { order_by: OrderBy::Value, dir: Direction::Forward, lo: Bound::Unbounded, hi: Bound::Unbounded, ..Default::default() }).unwrap();
    let mut decoded: Vec<i64> = Vec::new();
    for row in it { let s = row.value.as_slice(); if s.len()==8 { let mut a=[0u8;8]; a.copy_from_slice(s); decoded.push(i64::from_le_bytes(a)); } }
    let mut expected = vals.to_vec(); expected.sort();
    assert_eq!(decoded, expected);
}

#[test]
fn value_sorted_f64_le_with_sort_keys_monotonic() {
    let p = Arc::new(MemPager::default());
    let store = ColumnStore::open(p);
    let fid: LogicalFieldId = 42_003;
    let opts = AppendOptions { mode: ValueMode::ForceFixed(8), value_order: Some(ValueOrderPolicy::F64Le), ..Default::default() };
    let vals = [f64::NAN, -f64::INFINITY, -0.0, 0.0, 1.5, f64::INFINITY];
    let mut items = Vec::new();
    for (i, v) in vals.iter().enumerate() { items.push((Cow::Owned(key_bytes(i as u64)), Cow::Owned(lef64(*v)))); }
    store.append_many(vec![Put { field_id: fid, items }], opts);

    let it = store.scan_values_lww(fid, ValueScanOpts { order_by: OrderBy::Value, dir: Direction::Forward, lo: Bound::Unbounded, hi: Bound::Unbounded, ..Default::default() }).unwrap();
    let mut prev_ord: Option<[u8;8]> = None;
    for row in it {
        let s = row.value.as_slice();
        if s.len()==8 {
            let mut a=[0u8;8]; a.copy_from_slice(s);
            let ord = llkv_column_map::codecs::orderkey::f64_le_to_sort8(a);
            if let Some(prev) = prev_ord { assert!(prev <= ord, "not non-decreasing"); }
            prev_ord = Some(ord);
        }
    }
}
