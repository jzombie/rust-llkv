//! Bounds tests for SortedMerge over pre-sorted chunks.
//!
//! Each test ingests shuffled data across batches into ONE store that
//! has two columns (u64 and i32), builds sort indices for both, then
//! scans with bounds and validates output.

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

/// Ingest shuffled u64 and i32 columns into a single store.
fn ingest_two_cols(
    num_rows: usize,
    num_batches: usize,
    fid_u64: LogicalFieldId,
    fid_i32: LogicalFieldId,
) -> ColumnStore<MemPager> {
    let pager = Arc::new(MemPager::new());
    let store = ColumnStore::open(pager).unwrap();

    let mut md_u64 = HashMap::new();
    md_u64.insert("field_id".to_string(), u64::from(fid_u64).to_string());
    let mut md_i32 = HashMap::new();
    md_i32.insert("field_id".to_string(), u64::from(fid_i32).to_string());

    let row_f = Field::new("row_id", DataType::UInt64, false);
    let data_u64_f = Field::new("data_u64", DataType::UInt64, false).with_metadata(md_u64);
    let data_i32_f = Field::new("data_i32", DataType::Int32, false).with_metadata(md_i32);
    let schema = Arc::new(Schema::new(vec![row_f, data_u64_f, data_i32_f]));

    // Prepare shuffled payloads.
    let mut data_u64: Vec<u64> = (0..num_rows as u64).collect();
    let mut data_i32: Vec<i32> = (0..num_rows as i32)
        .map(|i| i - (num_rows as i32 / 2))
        .collect();

    let mut r = rng();
    data_u64.as_mut_slice().shuffle(&mut r);
    data_i32.as_mut_slice().shuffle(&mut r);

    // Append in batches; both columns go in the same batch with row_id.
    let batch_size = num_rows / num_batches;
    for i in 0..num_batches {
        let start = i * batch_size;
        let end = start + batch_size;

        let rids: Vec<u64> = (start as u64..end as u64).collect();
        let rid_arr = Arc::new(UInt64Array::from(rids));

        let vals_u64 = Arc::new(UInt64Array::from(data_u64[start..end].to_vec()));
        let vals_i32 = Arc::new(Int32Array::from(data_i32[start..end].to_vec()));

        let batch =
            RecordBatch::try_new(schema.clone(), vec![rid_arr, vals_u64, vals_i32]).unwrap();
        store.append(&batch).unwrap();
    }

    // Build sort indices for both columns.
    store.create_sort_index(fid_u64).unwrap();
    store.create_sort_index(fid_i32).unwrap();

    store
}

fn collect_u64_in_bounds(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    lo: std::ops::Bound<u64>,
    hi: std::ops::Bound<u64>,
) -> Vec<u64> {
    let mut m = store.scan_sorted(fid).unwrap().with_u64_bounds(lo, hi);
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
    let mut m = store.scan_sorted(fid).unwrap().with_i32_bounds(lo, hi);
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
    let fid_u64 = fid(501);
    let fid_i32 = fid(901); // unused here, but present in the store

    let store = ingest_two_cols(N, B, fid_u64, fid_i32);

    let lo = 12_345u64;
    let hi = 54_321u64;
    let got = collect_u64_in_bounds(&store, fid_u64, Included(lo), Included(hi));

    let exp: Vec<u64> = (lo..=hi).collect();
    assert_eq!(got.len(), exp.len());
    assert_eq!(got, exp);
}

#[test]
fn u64_bounds_open_ends() {
    const N: usize = 64_000;
    const B: usize = 4;
    let fid_u64 = fid(502);
    let fid_i32 = fid(902); // unused here

    let store = ingest_two_cols(N, B, fid_u64, fid_i32);

    // Only hi bound
    let hi = 10_000u64;
    let got_hi = collect_u64_in_bounds(&store, fid_u64, Unbounded, Included(hi));
    let exp_hi: Vec<u64> = (0..=hi).collect();
    assert_eq!(got_hi, exp_hi);

    // Only lo bound
    let lo = 42_000u64;
    let got_lo = collect_u64_in_bounds(&store, fid_u64, Included(lo), Unbounded);
    let exp_lo: Vec<u64> = (lo..N as u64).collect();
    assert_eq!(got_lo, exp_lo);
}

#[test]
fn u64_bounds_empty_ranges() {
    const N: usize = 32_768;
    const B: usize = 4;
    let fid_u64 = fid(503);
    let fid_i32 = fid(903); // unused here

    let store = ingest_two_cols(N, B, fid_u64, fid_i32);

    // Case 1: closed range entirely above the domain [N+10, N+20].
    let got1 = collect_u64_in_bounds(
        &store,
        fid_u64,
        Included(N as u64 + 10),
        Included(N as u64 + 20),
    );
    assert!(got1.is_empty());

    // Case 2: lo above max with open hi.
    let got2 = collect_u64_in_bounds(&store, fid_u64, Included(N as u64 + 1), Unbounded);
    assert!(got2.is_empty());

    // Case 3: lo > hi inside the domain.
    let got3 = collect_u64_in_bounds(&store, fid_u64, Included(10_000), Included(9_999));
    assert!(got3.is_empty());
}

#[test]
fn i32_bounds_cross_zero() {
    const N: usize = 80_000;
    const B: usize = 5;
    let fid_u64 = fid(504); // unused here
    let fid_i32 = fid(604);

    let store = ingest_two_cols(N, B, fid_u64, fid_i32);

    // Data covers [-N/2, N/2). Choose a band across zero.
    let lo = -1234i32;
    let hi = 3456i32;
    let got = collect_i32_in_bounds(&store, fid_i32, Included(lo), Included(hi));
    let exp: Vec<i32> = (lo..=hi).collect();

    assert_eq!(got.len(), exp.len());
    assert_eq!(got, exp);
}
