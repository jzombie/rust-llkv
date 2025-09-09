//! Integration tests for key/value pagination with randomized strings.
//!
//! Uses only `rand`.
//! Heavy test is #[ignore].
//!
//! Run heavy test with:
//!   cargo test --test pagination_strings_tests -- --ignored
//!
//! You can override sizes via env vars for the ignored test:
//!   N=1000000 PAGE=4096 cargo test --test pagination_strings_tests -- --ignored

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};

use std::env;
use std::ops::Bound;

// ---------------------- random string helpers -----------------------

/// ASCII alphabet for random strings.
const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Build a random ASCII string with a unique ordinal suffix so total
/// ordering is unique and stable across runs with same seed.
fn rand_unique_ascii(rng: &mut StdRng, min_len: usize, max_len: usize, ordinal: usize) -> Vec<u8> {
    let len = if max_len >= min_len {
        rng.gen_range(min_len..=max_len)
    } else {
        min_len
    };
    let mut s = Vec::with_capacity(len + 1 + 8);
    for _ in 0..len {
        let i = rng.gen_range(0..ALPHABET.len());
        s.push(ALPHABET[i]);
    }
    s.push(b'#');
    let tag = format!("{:08x}", ordinal);
    s.extend_from_slice(tag.as_bytes());
    s
}

// --------------------------- seeders ---------------------------------

/// Append in multiple batches to ensure multiple segments/generations.
/// This avoids edge cases that can lead to empty candidate sets under bounds.
///
/// - When seeding keys: values are fixed-width (u64 BE) for simplicity.
/// - When seeding values: keys are "k{idx}", values are random ASCII.
fn seed_strings_as_keys(store: &ColumnStore<MemPager>, fid: LogicalFieldId, n: usize, seed: u64) {
    let opts = AppendOptions {
        // Fixed-width makes value slicing trivial, though we scan by KEY here.
        mode: ValueMode::ForceFixed(8),
        // Smallish limits to force multiple segments even within a single append.
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    // Build all items, then shuffle and write in chunks.
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = rand_unique_ascii(&mut rng, 0, 64, i);
            let v = (i as u64).to_be_bytes().to_vec(); // fixed 8 bytes
            (k, v)
        })
        .collect();

    // Randomize insertion order to make segment min/max more realistic.
    items.shuffle(&mut rng);

    // Append in reasonably small chunks; each call is a generation.
    const CHUNK: usize = 10_000;
    for chunk in items.chunks(CHUNK) {
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
    }
}

fn seed_strings_as_values(store: &ColumnStore<MemPager>, fid: LogicalFieldId, n: usize, seed: u64) {
    let opts = AppendOptions {
        // Variable is fine; the library already tests min/max for variable values.
        mode: ValueMode::Auto,
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = format!("k{:08x}", i).into_bytes();
            let v = rand_unique_ascii(&mut rng, 0, 64, i);
            (k, v)
        })
        .collect();

    items.shuffle(&mut rng);

    const CHUNK: usize = 10_000;
    for chunk in items.chunks(CHUNK) {
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
    }
}

// ----------------------------- tests ---------------------------------

/// Smoke: key-ordered forward pagination with random string keys.
#[test]
fn paginate_strings_as_keys_smoke() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid = 9001u32;

    let n = 50_000usize;
    let page = 1009usize;

    seed_strings_as_keys(&store, fid, n, 0xDEADBEEF);

    // Forward key-ordered pagination.
    let mut after: Option<Vec<u8>> = None;
    let mut seen: usize = 0;
    let mut last: Option<Vec<u8>> = None;

    loop {
        let lo = match after.as_deref() {
            None => Bound::Unbounded,
            Some(k) => Bound::Excluded(k),
        };

        let it = match store.scan_values_lww(
            fid,
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Forward,
                lo,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            // If bounds exclude all segments, we're done.
            Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => break,
            Err(e) => panic!("scan init error: {:?}", e),
        };

        let mut took = 0usize;
        for item in it.take(page) {
            if let Some(prev) = &last {
                assert!(item.key.as_slice() > prev.as_slice(), "keys must increase");
            }
            after = Some(item.key.clone());
            last = Some(item.key);
            seen += 1;
            took += 1;
        }
        if took == 0 {
            break;
        }
    }

    assert_eq!(seen, n, "one row per inserted key");
}

/// Smoke: value-ordered forward pagination with random string values.
#[test]
fn paginate_strings_as_values_smoke() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid = 9002u32;

    let n = 50_000usize;
    let page = 977usize;

    seed_strings_as_values(&store, fid, n, 0xBADF00D);

    // Forward value-ordered pagination.
    let mut after: Option<Vec<u8>> = None;
    let mut seen: usize = 0;
    let mut last_val: Option<Vec<u8>> = None;

    loop {
        let lo = match after.as_deref() {
            None => Bound::Unbounded,
            Some(v) => Bound::Excluded(v),
        };

        let it = match store.scan_values_lww(
            fid,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Forward,
                lo,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => break,
            Err(e) => panic!("scan init error: {:?}", e),
        };

        let mut took = 0usize;
        for item in it.take(page) {
            let a = item.value.start() as usize;
            let b = item.value.end() as usize;
            let cur = item.value.data().as_ref()[a..b].to_vec();

            if let Some(prev) = &last_val {
                assert!(cur.as_slice() > prev.as_slice(), "values must increase");
            }
            after = Some(cur.clone());
            last_val = Some(cur);
            seen += 1;
            took += 1;
        }
        if took == 0 {
            break;
        }
    }

    assert_eq!(seen, n, "one row per inserted value");
}

/// Reverse scans: key- and value-ordered pagination, decreasing.
#[test]
fn paginate_strings_reverse_smoke() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid_k = 9003u32;
    let fid_v = 9004u32;

    let n = 30_000usize;
    let page = 997usize;

    seed_strings_as_keys(&store, fid_k, n, 1234);
    seed_strings_as_values(&store, fid_v, n, 4321);

    // -------- Reverse key-ordered pagination --------
    //
    // Terminology:
    // - We keep a "before" upper bound (exclusive) that moves leftward.
    // - Hitting NoActiveSegments means our bound excluded every segment,
    //   i.e. we've paged past the smallest key. That's a *normal* end.
    // - BUT: if that happens on the very first page, it's a bug — panic.
    let mut before: Option<Vec<u8>> = None;
    let mut last_key: Option<Vec<u8>> = None;
    let mut seen_k = 0usize;

    loop {
        let hi = match before.as_deref() {
            None => Bound::Unbounded,
            Some(k) => Bound::Excluded(k),
        };

        let it = match store.scan_values_lww(
            fid_k,
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => {
                // Normal termination (we've paged past the min key),
                // but not on the very first page.
                if before.is_none() {
                    panic!("unexpected NoActiveSegments on first reverse key page");
                }
                break;
            }
            Err(e) => panic!("scan init error (reverse key): {:?}", e),
        };

        let mut took = 0usize;
        for item in it.take(page) {
            // Strict global monotonicity across page boundaries.
            if let Some(prev) = &last_key {
                assert!(
                    item.key.as_slice() < prev.as_slice(),
                    "keys must strictly decrease in reverse key order"
                );
            }
            before = Some(item.key.clone());
            last_key = Some(item.key);
            seen_k += 1;
            took += 1;
        }
        // If this page was empty, we're done.
        if took == 0 {
            break;
        }
    }
    assert_eq!(seen_k, n, "must consume all wieners (keys)");

    // -------- Reverse value-ordered pagination --------
    //
    // Same story, but ordering is by VALUE bytes and the cursor is the
    // last page's value slice (exclusive upper bound).
    let mut before_v: Option<Vec<u8>> = None;
    let mut last_val: Option<Vec<u8>> = None;
    let mut seen_v = 0usize;

    loop {
        let hi = match before_v.as_deref() {
            None => Bound::Unbounded,
            Some(v) => Bound::Excluded(v),
        };

        let it = match store.scan_values_lww(
            fid_v,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => {
                if before_v.is_none() {
                    panic!("unexpected NoActiveSegments on first reverse value page");
                }
                break;
            }
            Err(e) => panic!("scan init error (reverse value): {:?}", e),
        };

        let mut took = 0usize;
        for item in it.take(page) {
            // Pull the value bytes using accessors (fields are private).
            let a = item.value.start() as usize;
            let b = item.value.end() as usize;
            let cur = item.value.data().as_ref()[a..b].to_vec();

            if let Some(prev) = &last_val {
                assert!(
                    cur.as_slice() < prev.as_slice(),
                    "values must strictly decrease in reverse value order"
                );
            }
            before_v = Some(cur.clone());
            last_val = Some(cur);
            seen_v += 1;
            took += 1;
        }
        if took == 0 {
            break;
        }
    }
    assert_eq!(seen_v, n, "must consume all wieners (values)");
}

/// Heavy test (ignored by default): run with up to a million rows.
#[test]
#[ignore = "CPU intensive test"]
fn paginate_strings_heavy_ignored() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid_k = 9101u32;
    let fid_v = 9102u32;

    // Allow overriding sizes via env for stress runs.
    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1_000_000);
    let page = std::env::var("PAGE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4096);

    seed_strings_as_keys(&store, fid_k, n, 0xA5A5_0001);
    seed_strings_as_values(&store, fid_v, n, 0x5A5A_0002);

    // -------- Key-ordered forward pagination (stress) --------
    //
    // Using a moving "after" lower bound (exclusive). When segments are
    // fully below the bound we expect NoActiveSegments — normal end.
    let mut after: Option<Vec<u8>> = None;
    let mut last_k: Option<Vec<u8>> = None;
    let mut seen = 0usize;

    loop {
        let lo = match after.as_deref() {
            None => Bound::Unbounded,
            Some(k) => Bound::Excluded(k),
        };
        let it = match store.scan_values_lww(
            fid_k,
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Forward,
                lo,
                hi: Bound::Unbounded,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => {
                if after.is_none() {
                    panic!("unexpected NoActiveSegments on first forward key page");
                }
                break;
            }
            Err(e) => panic!("scan init error (forward key): {:?}", e),
        };

        let mut took = 0usize;
        for item in it.take(page) {
            if let Some(prev) = &last_k {
                assert!(
                    item.key.as_slice() > prev.as_slice(),
                    "keys must strictly increase in forward key order"
                );
            }
            after = Some(item.key.clone());
            last_k = Some(item.key);
            seen += 1;
            took += 1;
        }
        if took == 0 {
            break;
        }
    }
    assert_eq!(seen, n, "must consume all wieners (keys)");

    // -------- Value-ordered reverse pagination (stress) --------
    let mut before_v: Option<Vec<u8>> = None;
    let mut last_v: Option<Vec<u8>> = None;
    let mut seen_v = 0usize;

    loop {
        let hi = match before_v.as_deref() {
            None => Bound::Unbounded,
            Some(v) => Bound::Excluded(v),
        };
        let it = match store.scan_values_lww(
            fid_v,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi,
                prefix: None,
                bucket_prefix_len: 2,
                head_tag_len: 8,
                frame_predicate: None,
            },
        ) {
            Ok(it) => it,
            Err(llkv_column_map::column_store::read_scan::ScanError::NoActiveSegments) => {
                if before_v.is_none() {
                    panic!("unexpected NoActiveSegments on first reverse value page");
                }
                break;
            }
            Err(e) => panic!("scan init error (reverse value): {:?}", e),
        };

        let mut took = 0usize;
        for item in it.take(page) {
            let a = item.value.start() as usize;
            let b = item.value.end() as usize;
            let cur = item.value.data().as_ref()[a..b].to_vec();

            if let Some(prev) = &last_v {
                assert!(
                    cur.as_slice() < prev.as_slice(),
                    "values must strictly decrease in reverse value order"
                );
            }
            before_v = Some(cur.clone());
            last_v = Some(cur);
            seen_v += 1;
            took += 1;
        }
        if took == 0 {
            break;
        }
    }
    assert_eq!(seen_v, n, "must consume all wieners (values)");
}
