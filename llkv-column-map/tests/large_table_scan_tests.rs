//! Integration: large table (1,048,576 rows × 10 columns) with **fewer segments**
//! and thorough scan coverage.
//!
//! What this test does:
//! - Builds 10 logical columns (field_ids base..base+9)
//! - Appends 1,048,576 rows per column in fixed-width values (u64 BE)
//! - Uses a **large `segment_max_entries`** to coalesce into big segments
//! - Re-writes a large contiguous key range in column 0 (single append) to
//!   validate strict LWW shadowing with *few* generations/segments
//! - Exercises several scan/query forms:
//!   * Full-range, value-ordered LWW scan (forward)
//!   * Full-range, value-ordered LWW scan (reverse)
//!   * Windowed value-ordered LWW scans (forward and reverse)
//!   * Point lookups across multiple columns via get_many
//!   * Storage introspection to assert a **small** segment count
//!
//! NOTE: This test is heavy (data+CPU). If needed, tag with `#[ignore]`  or
//! run with `cargo test --release`.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;

use llkv_column_map::codecs::big_endian::u64_be_array;
use llkv_column_map::column_store::read_value_scan::{Direction, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};
use llkv_column_map::{ColumnStore, storage::StorageKind};

const ROWS: u32 = 1_048_576; // 2^20
const COLS: usize = 10;
const BASE_FID: LogicalFieldId = 10_000;

// Shape knobs: coalesce into big segments
const SEGMENT_MAX_ENTRIES: usize = ROWS as usize; // allow 1 segment per column
const SEGMENT_MAX_BYTES: usize = 64 << 20; // 64 MiB generous cap

// For the rewrite window (100k rows), we also do a single append.
const REWRITE_LO: u32 = 200_000;
const REWRITE_HI: u32 = 300_000;

// TODO: Dedupe
#[inline]
fn be_u64_vec(x: u64) -> Vec<u8> {
    u64_be_array(x).to_vec()
}

// TODO: Dedupe
#[inline]
fn parse_be_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_be_bytes(a)
}

#[inline]
fn row_key(i: u32) -> Vec<u8> {
    // 8 ASCII bytes like: r0000123
    format!("r{:07}", i).into_bytes()
}

// Base value for (col, i). Each column has a disjoint value band so that
// value-ordered windows can target each column reliably without collisions.
#[inline]
fn base_value(col: usize, i: u32) -> u64 {
    // 10^7 stride per column ensures non-overlap across 1,048,576 rows.
    (col as u64) * 10_000_000 + (i as u64)
}

#[test]
#[ignore = "CPU intensive test"]
fn large_table_end_to_end_scans() {
    let pager = MemPager::default();
    let store = ColumnStore::init_empty(&pager);

    // Common append options: fixed 8B values, coalesce into large segments.
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8),
        segment_max_entries: SEGMENT_MAX_ENTRIES,
        segment_max_bytes: SEGMENT_MAX_BYTES,
        last_write_wins_in_batch: true,
    };

    // -------------------------- Build 10 columns --------------------------
    // Single append per column (one big segment per column is allowed by limits).
    for col in 0..COLS {
        let fid = BASE_FID + col as u32;
        let mut items = Vec::with_capacity(ROWS as usize);
        for i in 0..ROWS {
            items.push((
                Cow::Owned(row_key(i)),
                Cow::Owned(be_u64_vec(base_value(col, i))),
            ));
        }
        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            opts.clone(),
        );
    }

    // ---------------- Overwrite a large slice in column 0 (strict LWW) -----
    // Single append for the whole rewrite window => adds at most one new segment.
    let fid0 = BASE_FID;
    let mut items = Vec::with_capacity((REWRITE_HI - REWRITE_LO) as usize);
    for i in REWRITE_LO..REWRITE_HI {
        let v = 9_000_000_000u64 + (i as u64);
        items.push((Cow::Owned(row_key(i)), Cow::Owned(be_u64_vec(v))));
    }
    store.append_many(
        vec![Put {
            field_id: fid0,
            items,
        }],
        opts.clone(),
    );

    // Sanity: expect **few** segments for fid0 (base + one rewrite).
    let seg_count_for_fid0: usize = store
        .describe_storage()
        .into_iter()
        .map(|n| match n.kind {
            StorageKind::IndexSegment { field_id, .. } if field_id == fid0 => 1usize,
            _ => 0usize,
        })
        .sum();
    assert!(
        (1..=3).contains(&seg_count_for_fid0),
        "expect few segments for fid0 (1–3), got {seg_count_for_fid0}"
    );

    // ---------------- get_many spot checks across columns ------------------
    let sample_keys: Vec<Vec<u8>> = vec![
        row_key(0),
        row_key(1),
        row_key(123_456),
        row_key(ROWS - 1),
        row_key(REWRITE_LO),     // rewritten in fid0
        row_key(REWRITE_HI - 1), // rewritten in fid0
    ];

    let mut queries = Vec::with_capacity(COLS);
    for col in 0..COLS {
        let fid = BASE_FID + col as u32;
        queries.push((fid, sample_keys.clone()));
    }
    let got = store.get_many(queries);

    for (col, v) in got.iter().enumerate() {
        let expect0 = base_value(col, 0);
        let expect1 = base_value(col, 1);
        let expect_mid = base_value(col, 123_456);
        let expect_last = base_value(col, ROWS - 1);

        assert_eq!(parse_be_u64(v[0].as_deref().unwrap()), expect0);
        assert_eq!(parse_be_u64(v[1].as_deref().unwrap()), expect1);
        assert_eq!(parse_be_u64(v[2].as_deref().unwrap()), expect_mid);
        assert_eq!(parse_be_u64(v[3].as_deref().unwrap()), expect_last);

        let got_lo = parse_be_u64(v[4].as_deref().unwrap());
        let got_hi = parse_be_u64(v[5].as_deref().unwrap());
        if col == 0 {
            assert!(
                got_lo >= 9_000_000_000 && got_hi >= 9_000_000_000,
                "LWW rewrite must win in fid0"
            );
        } else {
            assert_eq!(got_lo, base_value(col, REWRITE_LO));
            assert_eq!(got_hi, base_value(col, REWRITE_HI - 1));
        }
    }

    // --------------- Full-range value-ordered scan (forward, LWW) ---------
    let it = store
        .scan_values_lww(
            fid0,
            ValueScanOpts {
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        )
        .expect("scan iterator");

    let mut count = 0usize;
    let mut seen_keys = BTreeSet::<Vec<u8>>::new();
    let mut last_val = None::<u64>;

    for item in it {
        let val = {
            let a = item.value.start() as usize;
            let b = item.value.end() as usize;
            parse_be_u64(&item.value.data().as_ref()[a..b])
        };

        if let Some(prev) = last_val {
            assert!(
                prev <= val,
                "forward value-ordered scan must be non-decreasing ({} > {})",
                prev,
                val
            );
        }
        last_val = Some(val);

        assert!(seen_keys.insert(item.key.clone()), "duplicate key in LWW");
        count += 1;
    }
    assert_eq!(count as u32, ROWS, "strict LWW must return 1 value per key");

    // --------------- Full-range value-ordered scan (reverse, LWW) ---------
    let it = store
        .scan_values_lww(
            fid0,
            ValueScanOpts {
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        )
        .expect("reverse scan iterator");

    let mut last_val = None::<u64>;
    let mut count_rev = 0usize;
    for item in it {
        let val = {
            let a = item.value.start() as usize;
            let b = item.value.end() as usize;
            parse_be_u64(&item.value.data().as_ref()[a..b])
        };
        if let Some(prev) = last_val {
            assert!(
                prev >= val,
                "reverse value-ordered scan must be non-increasing ({} < {})",
                prev,
                val
            );
        }
        last_val = Some(val);
        count_rev += 1;
    }
    assert_eq!(count_rev as u32, ROWS, "reverse LWW must see 1 per key");

    // ---------------- Windowed value scan (forward): base band -------------
    let w_lo_i = 50_000u32;
    let w_len = 16u32;
    let lo = be_u64_vec(base_value(0, w_lo_i));
    let hi = be_u64_vec(base_value(0, w_lo_i + w_len));

    let it = store
        .scan_values_lww(
            fid0,
            ValueScanOpts {
                dir: Direction::Forward,
                lo: Bound::Included(&lo),
                hi: Bound::Excluded(&hi),
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        )
        .expect("windowed iterator");

    let mut got_keys = BTreeSet::<u32>::new();
    let mut vals = Vec::<u64>::new();
    for item in it {
        let k = parse_row_index(&item.key);
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        vals.push(parse_be_u64(&item.value.data().as_ref()[a..b]));
        got_keys.insert(k);
    }
    let expected: BTreeSet<u32> = (w_lo_i..(w_lo_i + w_len)).collect();
    assert_eq!(got_keys, expected, "window forward keys");
    assert!(
        is_non_decreasing(&vals),
        "window forward must be sorted asc"
    );

    // ---------------- Windowed value scan (reverse): base band -------------
    let it = store
        .scan_values_lww(
            fid0,
            ValueScanOpts {
                dir: Direction::Reverse,
                lo: Bound::Included(&lo),
                hi: Bound::Excluded(&hi),
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        )
        .expect("windowed iterator (reverse)");

    let mut vals_rev = Vec::<u64>::new();
    for item in it {
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        vals_rev.push(parse_be_u64(&item.value.data().as_ref()[a..b]));
    }
    assert!(
        is_non_increasing(&vals_rev),
        "window reverse must be sorted desc"
    );

    // -------- Window that targets the rewritten band (LWW correctness) ----
    let lo2 = be_u64_vec(9_000_000_000u64 + 250_000u64);
    let hi2 = be_u64_vec(9_000_000_000u64 + 250_016u64);

    let it = store
        .scan_values_lww(
            fid0,
            ValueScanOpts {
                dir: Direction::Forward,
                lo: Bound::Included(&lo2),
                hi: Bound::Excluded(&hi2),
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        )
        .expect("windowed iterator into rewritten band");

    let mut got = BTreeMap::<u32, u64>::new();
    for item in it {
        let ki = parse_row_index(&item.key);
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        let vv = parse_be_u64(&item.value.data().as_ref()[a..b]);
        got.insert(ki, vv);
    }
    assert_eq!(got.len(), 16, "window size (rewritten band)");
    for i in 250_000u32..250_016 {
        let v = *got.get(&i).expect("missing rewritten key");
        assert!(
            v >= 9_000_000_000,
            "rewritten LWW value must be from the newest generation"
        );
    }
}

// ---------------------------- helpers ---------------------------------

#[inline]
fn parse_row_index(k: &[u8]) -> u32 {
    // "r0000123" -> 123
    let s = std::str::from_utf8(k).expect("utf8 key");
    s[1..].parse::<u32>().expect("row index")
}

#[inline]
fn is_non_decreasing(v: &[u64]) -> bool {
    v.windows(2).all(|w| w[0] <= w[1])
}

#[inline]
fn is_non_increasing(v: &[u64]) -> bool {
    v.windows(2).all(|w| w[0] >= w[1])
}
