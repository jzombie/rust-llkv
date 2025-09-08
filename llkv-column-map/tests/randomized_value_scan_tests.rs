//! Randomized value-ordered scan tests (no external deps).
//!
//! What this covers:
//! - Random u64 values across many segments (forward & reverse scans)
//! - Random two-generation rewrite (LWW correctness over randomized data)
//!
//! These are heavier than unit tests and are `#[ignore]`d by default.
//! Run with: `cargo test -- --ignored`

use std::borrow::Cow;
use std::cmp::{max, min};
use std::ops::Bound;

use llkv_column_map::ColumnStore;
use llkv_column_map::codecs::big_endian::u64_be_array;
use llkv_column_map::column_store::read_value_scan::{Direction, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

#[inline]
fn be64_vec(x: u64) -> Vec<u8> {
    u64_be_array(x).to_vec()
}

#[inline]
fn parse_be64(v: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&v[..8]);
    u64::from_be_bytes(a)
}

#[inline]
fn key_bytes(i: u64) -> Vec<u8> {
    // "r00000000" fixed width (8 digits)
    format!("r{:08}", i).into_bytes()
}

/// Tiny, fast PRNG (xorshift64*) — deterministic and dependency-free.
#[derive(Clone)]
struct Xs(u64);
impl Xs {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_in(&mut self, lo: u64, hi_excl: u64) -> u64 {
        let span = hi_excl - lo;
        lo + (self.next_u64() % span)
    }
}

fn append_random_column(
    store: &ColumnStore<'_, MemPager>,
    fid: LogicalFieldId,
    n_rows: u64,
    segment_max_entries: usize,
    seed: u64,
) -> Vec<u64> {
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8),
        segment_max_entries,
        segment_max_bytes: 32 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = Xs::new(seed);
    let mut vals = Vec::with_capacity(n_rows as usize);

    // Generate values (random), keys increasing (k00000000..)
    for i in 0..n_rows {
        vals.push(rng.next_u64());
    }

    // Write in moderate chunks; actual segmenting is controlled by segment_max_entries.
    let chunk: u64 = 50_000;
    let mut start = 0;
    while start < n_rows {
        let end = (start + chunk).min(n_rows);
        let mut items = Vec::with_capacity((end - start) as usize);
        for (i, v) in (start..end).zip(vals[start as usize..end as usize].iter().copied()) {
            items.push((Cow::Owned(key_bytes(i)), Cow::Owned(be64_vec(v))));
        }
        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            opts.clone(),
        );
        start = end;
    }

    vals
}

#[test]
#[ignore = "CPU intensive test"]
fn randomized_single_generation_forward_and_reverse() {
    let pager = MemPager::default();
    let store = ColumnStore::init_empty(&pager);
    let fid: LogicalFieldId = 10_000;

    // Tune as desired — small-ish by default for CI friendliness.
    let n_rows: u64 = 200_000;
    let segment_max_entries: usize = 10_000; // force ~20+ segments

    // Seed random values
    let values = append_random_column(&store, fid, n_rows, segment_max_entries, 0xC0FFEE);

    // Pick a randomized window [lo, hi)
    let mut rng = Xs::new(0xBADC0DE);
    let a = rng.next_in(0, n_rows);
    let b = rng.next_in(0, n_rows);
    let (ia, ib) = (min(a, b), max(a, b));
    // If indices collapse, widen a bit
    let (ia, ib) = if ia == ib {
        (0, n_rows.min(ia + 1000))
    } else {
        (ia, ib)
    };

    // Derive numeric bounds from actual values to ensure the scan window hits data.
    let bound_lo = values[ia as usize];
    let bound_hi = values[ib.saturating_sub(1) as usize].wrapping_add(1); // just above
    let lo = be64_vec(bound_lo.min(bound_hi)); // guard against wrap
    let hi = be64_vec(bound_hi.max(bound_lo));

    // Compute expected winners in the window (single generation => all keys unique).
    let mut expected: Vec<u64> = values
        .iter()
        .copied()
        .filter(|&v| v >= parse_be64(&lo) && v < parse_be64(&hi))
        .collect();
    expected.sort_unstable();

    // Forward scan
    let it = store
        .scan_values_lww(
            fid,
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
        .expect("forward iterator");

    let mut got_fwd = Vec::<u64>::new();
    for item in it {
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        got_fwd.push(parse_be64(&item.value.data().as_ref()[a..b]));
    }
    // Order and membership
    assert!(
        got_fwd.windows(2).all(|w| w[0] <= w[1]),
        "forward not non-decreasing"
    );
    assert_eq!(got_fwd, expected, "forward window mismatch vs expected");

    // Reverse scan should be the reverse order of the same set
    let it = store
        .scan_values_lww(
            fid,
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
        .expect("reverse iterator");

    let mut got_rev = Vec::<u64>::new();
    for item in it {
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        got_rev.push(parse_be64(&item.value.data().as_ref()[a..b]));
    }
    assert!(
        got_rev.windows(2).all(|w| w[0] >= w[1]),
        "reverse not non-increasing"
    );

    let mut rev_sorted = expected.clone();
    rev_sorted.reverse();
    assert_eq!(got_rev, rev_sorted, "reverse window mismatch vs expected");
}

#[test]
#[ignore = "CPU intensive test"]
fn randomized_two_generation_lww_correctness() {
    let pager = MemPager::default();
    let store = ColumnStore::init_empty(&pager);
    let fid: LogicalFieldId = 20_000;

    let n_rows: u64 = 150_000;
    let segment_max_entries: usize = 7_500; // more segments

    // Gen 0: random base
    let base = append_random_column(&store, fid, n_rows, segment_max_entries, 0xDEADBEEF);

    // Gen 1: rewrite a random ~1/3 subset with new random payloads
    let mut rng = Xs::new(0xA11CE);
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8),
        segment_max_entries,
        segment_max_bytes: 32 << 20,
        last_write_wins_in_batch: true,
    };

    let mut winners = base.clone(); // start with base, then overwrite keys we touch
    let chunk: u64 = 40_000;
    let mut start = 0;
    while start < n_rows {
        let end = (start + chunk).min(n_rows);
        let mut items = Vec::new();
        for i in start..end {
            if (rng.next_u64() & 3) == 0 {
                // ~25% overwrite density
                let v1 = rng.next_u64();
                winners[i as usize] = v1;
                items.push((Cow::Owned(key_bytes(i)), Cow::Owned(be64_vec(v1))));
            }
        }
        if !items.is_empty() {
            store.append_many(
                vec![Put {
                    field_id: fid,
                    items,
                }],
                opts.clone(),
            );
        }
        start = end;
    }

    // Full-range forward: must return all winners once, sorted by value asc.
    let it = store
        .scan_values_lww(
            fid,
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
        .expect("full forward iterator");

    let mut got_vals = Vec::<u64>::new();
    for item in it {
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        got_vals.push(parse_be64(&item.value.data().as_ref()[a..b]));
    }
    assert_eq!(
        got_vals.len() as u64,
        n_rows,
        "must yield exactly one winner per key"
    );
    assert!(
        got_vals.windows(2).all(|w| w[0] <= w[1]),
        "forward not non-decreasing"
    );

    let mut expected_sorted = winners.clone();
    expected_sorted.sort_unstable();
    assert_eq!(got_vals, expected_sorted, "LWW winners mismatch (forward)");

    // Reverse order should be exact reverse
    let it = store
        .scan_values_lww(
            fid,
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
        .expect("full reverse iterator");

    let mut got_vals_rev = Vec::<u64>::new();
    for item in it {
        let a = item.value.start() as usize;
        let b = item.value.end() as usize;
        got_vals_rev.push(parse_be64(&item.value.data().as_ref()[a..b]));
    }
    assert!(
        got_vals_rev.windows(2).all(|w| w[0] >= w[1]),
        "reverse not non-increasing"
    );

    let mut expected_rev = expected_sorted.clone();
    expected_rev.reverse();
    assert_eq!(got_vals_rev, expected_rev, "LWW winners mismatch (reverse)");
}
