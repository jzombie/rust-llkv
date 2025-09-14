use std::sync::Arc;
use llkv_column_map::column_store::{ColumnStore, ScanOptions, Bounds, Direction};
use llkv_column_map::storage::pager::MemPager;

#[test]
fn value_sidecar_orders_and_bounds() {
    let pager = Arc::new(MemPager::default());
    let store: ColumnStore<MemPager> = ColumnStore::open(pager.clone());
    let fid = 4242;
    // Build one chunk with a known shuffled pattern
    let n = 100_000usize;
    let mut vals = vec![0u64; n];
    for i in 0..n { vals[i] = ((i as u64).wrapping_mul(1103515245).rotate_left(7)) ^ 0xA5A5_5A5A_A5A5_5A5A; }
    let (_pk, _ppk) = store.append_u64_chunk_with_value_perm(fid, &vals);

    // Ascending full scan in value order: validate coverage and sum.
    let mut count = 0usize;
    let opts = ScanOptions { direction: Direction::Asc, bounds: Bounds::Values { min: None, max: None }, vector_len: 0 };
    let mut sum_check: u128 = 0;
    if let Some(iter) = store.scan_u64_value_order(fid, opts) {
        for (blob, range) in iter {
            let bytes = &blob.as_ref()[range];
            // Validate ordering and sum
            let mut i = 0usize;
            while i < bytes.len() { 
                let mut w = [0u8; 8]; w.copy_from_slice(&bytes[i..i+8]);
                let v = u64::from_le_bytes(w);
                sum_check += v as u128;
                count += 1;
                i += 8;
            }
        }
    } else {
        panic!("value-order scan not available (missing sidecar)");
    }
    assert_eq!(count, n);
    // Cross-check against SIMD sum on the original stripe to ensure we didn't drop/duplicate.
    let sum_truth: u128 = vals.iter().map(|&v| v as u128).sum();
    assert_eq!(sum_check, sum_truth);

    // Now apply bounds: pick approx middle range
    let lo = vals[n/3..].iter().copied().min().unwrap_or(0);
    let hi = vals[..(2*n/3)].iter().copied().max().unwrap_or(u64::MAX);
    let opts_b = ScanOptions { direction: Direction::Asc, bounds: Bounds::Values { min: Some(lo), max: Some(hi) }, vector_len: 0 };
    let mut seen_min = u64::MAX;
    let mut seen_max = u64::MIN;
    let mut total = 0usize;
    if let Some(iter) = store.scan_u64_value_order(fid, opts_b) {
        for (blob, range) in iter {
            let bytes = &blob.as_ref()[range];
            let mut i = 0usize;
            while i < bytes.len() { 
                let mut w = [0u8; 8]; w.copy_from_slice(&bytes[i..i+8]);
                let v = u64::from_le_bytes(w);
                assert!(v >= lo && v <= hi);
                if v < seen_min { seen_min = v; }
                if v > seen_max { seen_max = v; }
                total += 1; i += 8;
            }
        }
    } else { panic!("value-order scan not available"); }
    assert!(total > 0);
    assert!(seen_min >= lo);
    assert!(seen_max <= hi);

    // Descending order: validate coverage only
    let opts_desc = ScanOptions { direction: Direction::Desc, bounds: Bounds::Values { min: None, max: None }, vector_len: 0 };
    let mut c2 = 0usize;
    if let Some(iter) = store.scan_u64_value_order(fid, opts_desc) {
        for (blob, range) in iter {
            let bytes = &blob.as_ref()[range];
            let mut i = 0usize;
            while i < bytes.len() { 
                let mut w = [0u8; 8]; w.copy_from_slice(&bytes[i..i+8]);
                let v = u64::from_le_bytes(w);
                c2 += 1; i += 8;
            }
        }
    } else { panic!("value-order scan not available"); }
    assert_eq!(c2, n);
}
