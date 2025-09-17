//! Bounds tests for SortedMerge over pre-sorted chunks.
//!
//! Each test ingests shuffled data across batches, builds the sort
//! index, then scans with bounds and validates output.

use std::collections::HashMap;
use std::ops::Bound::{Included, Unbounded};
use std::sync::Arc;

use arrow::array::{Int32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::store::{ColumnStore, Run};
use llkv_column_map::types::{LogicalFieldId, Namespace};

use rand::rng;
use rand::seq::SliceRandom;

/// Test helper to create a standard user-data LogicalFieldId.
fn fid(id: u32) -> LogicalFieldId {
    LogicalFieldId::new()
        .with_namespace(Namespace::UserData)
        .with_table_id(0)
        .with_field_id(id)
}

/// Ingest shuffled u64 data into a new store; returns the opened store.
fn ingest_u64(
    num_rows: usize,
    num_batches: usize,
    field_id: LogicalFieldId,
) -> ColumnStore<MemPager> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());

    let data_f = Field::new("data", DataType::UInt64, false).with_metadata(md);
    let row_f = Field::new("row_id", DataType::UInt64, false);
    let schema = Arc::new(Schema::new(vec![row_f, data_f]));

    let mut data: Vec<u64> = (0..num_rows as u64).collect();
    let mut r = rng();
    data.as_mut_slice().shuffle(&mut r);

    let batch_size = num_rows / num_batches;
    for i in 0..num_batches {
        let start = i * batch_size;
        let end = start + batch_size;

        let rids: Vec<u64> = (start as u64..end as u64).collect();
        let rid_arr = Arc::new(UInt64Array::from(rids));
        let vals = &data[start..end];
        let val_arr = Arc::new(UInt64Array::from(vals.to_vec()));

        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
        store.append(&batch).unwrap();
    }

    store.create_sort_index(field_id).unwrap();
    store
}

/// Ingest shuffled i32 data into a new store; returns the opened store.
fn ingest_i32(
    num_rows: usize,
    num_batches: usize,
    field_id: LogicalFieldId,
) -> ColumnStore<MemPager> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let mut md = HashMap::new();
    md.insert("field_id".to_string(), u64::from(field_id).to_string());

    let data_f = Field::new("data", DataType::Int32, false).with_metadata(md);
    let row_f = Field::new("row_id", DataType::UInt64, false);
    let schema = Arc::new(Schema::new(vec![row_f, data_f]));

    let mut data: Vec<i32> = (0..num_rows as i32)
        .map(|i| i - (num_rows as i32 / 2))
        .collect();
    let mut r = rng();
    data.as_mut_slice().shuffle(&mut r);

    let batch_size = num_rows / num_batches;
    for i in 0..num_batches {
        let start = i * batch_size;
        let end = start + batch_size;

        let rids: Vec<u64> = (start as u64..end as u64).collect();
        let rid_arr = Arc::new(UInt64Array::from(rids));
        let vals = &data[start..end];
        let val_arr = Arc::new(Int32Array::from(vals.to_vec()));

        let batch = RecordBatch::try_new(schema.clone(), vec![rid_arr, val_arr]).unwrap();
        store.append(&batch).unwrap();
    }

    store.create_sort_index(field_id).unwrap();
    store
}

fn collect_u64_in_bounds(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    lo: std::ops::Bound<u64>,
    hi: std::ops::Bound<u64>,
) -> Vec<u64> {
    let mut m = store.scan_sorted(fid).unwrap().with_u64_range(lo, hi);
    let mut out = Vec::new();
    while let Some(run) = m.next_run() {
        match run {
            Run::U64 { arr, start, len } => {
                let end = start + len;
                for i in start..end {
                    out.push(arr.value(i));
                }
            }
            _ => panic!("unexpected run type for u64"),
        }
    }
    out
}

fn collect_i32_in_bounds(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    lo: std::ops::Bound<i32>,
    hi: std::ops::Bound<i32>,
) -> Vec<i32> {
    let mut m = store.scan_sorted(fid).unwrap().with_i32_range(lo, hi);
    let mut out = Vec::new();
    while let Some(run) = m.next_run() {
        match run {
            Run::I32 { arr, start, len } => {
                let end = start + len;
                for i in start..end {
                    out.push(arr.value(i));
                }
            }
            _ => panic!("unexpected run type for i32"),
        }
    }
    out
}

#[test]
fn u64_bounds_inclusive_middle() {
    const N: usize = 100_000;
    const B: usize = 5;
    let fid = fid(501);

    let store = ingest_u64(N, B, fid);

    let lo = 12_345u64;
    let hi = 54_321u64;
    let got = collect_u64_in_bounds(&store, fid, Included(lo), Included(hi));

    let exp: Vec<u64> = (lo..=hi).collect();
    assert_eq!(got.len(), exp.len());
    assert_eq!(got, exp);
}

#[test]
fn u64_bounds_open_ends() {
    const N: usize = 64_000;
    const B: usize = 4;
    let fid = fid(502);

    let store = ingest_u64(N, B, fid);

    // Only hi bound
    let hi = 10_000u64;
    let got_hi = collect_u64_in_bounds(&store, fid, Unbounded, Included(hi));
    let exp_hi: Vec<u64> = (0..=hi).collect();
    assert_eq!(got_hi, exp_hi);

    // Only lo bound
    let lo = 42_000u64;
    let got_lo = collect_u64_in_bounds(&store, fid, Included(lo), Unbounded);
    let exp_lo: Vec<u64> = (lo..N as u64).collect();
    assert_eq!(got_lo, exp_lo);
}

#[test]
fn u64_bounds_empty_ranges() {
    const N: usize = 32_768;
    const B: usize = 4;
    let fid = fid(503);

    let store = ingest_u64(N, B, fid);

    // Case 1: closed range entirely above the domain [N+10, N+20].
    let got1 = collect_u64_in_bounds(
        &store,
        fid,
        Included(N as u64 + 10),
        Included(N as u64 + 20),
    );
    assert!(got1.is_empty());

    // Case 2: lo above max with open hi.
    let got2 = collect_u64_in_bounds(&store, fid, Included(N as u64 + 1), Unbounded);
    assert!(got2.is_empty());

    // Case 3: lo > hi inside the domain.
    let got3 = collect_u64_in_bounds(&store, fid, Included(10_000), Included(9_999));
    assert!(got3.is_empty());
}

#[test]
fn i32_bounds_cross_zero() {
    const N: usize = 80_000;
    const B: usize = 5;
    let fid = fid(504);

    let store = ingest_i32(N, B, fid);

    // Data covers [-N/2, N/2). Choose a band across zero.
    let lo = -1234i32;
    let hi = 3456i32;
    let got = collect_i32_in_bounds(&store, fid, Included(lo), Included(hi));
    let exp: Vec<i32> = (lo..=hi).collect();

    assert_eq!(got.len(), exp.len());
    assert_eq!(got, exp);
}
