//! Integration tests for key/value pagination with randomized strings.
//!
//! Uses only `rand`.
//! Heavy test is #[ignore].
//!
//! Run heavy test with:
//!   cargo test --release --test pagination_strings_tests -- --ignored
//!
//! You can override sizes via env vars for the ignored test:
//!   N=1000000 PAGE=4096 cargo test --release --test pagination_strings_tests -- --ignored

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};

use std::ops::Bound;

// ---------------------- random string helpers -----------------------

/// ASCII alphabet for random strings.
const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Build a random ASCII string with a unique ordinal suffix so total
/// ordering is unique and stable across runs with same seed.
fn rand_unique_ascii(rng: &mut StdRng, min_len: usize, max_len: usize, ordinal: usize) -> Vec<u8> {
    let len = if max_len >= min_len {
        rng.random_range(min_len..=max_len)
    } else {
        min_len
    };
    let mut s = Vec::with_capacity(len + 1 + 8);
    for _ in 0..len {
        let i = rng.random_range(0..ALPHABET.len());
        s.push(ALPHABET[i]);
    }
    s.push(b'#');
    let tag = format!("{:08x}", ordinal);
    s.extend_from_slice(tag.as_bytes());
    s
}

// --------------------------- seeders ---------------------------------

/// Append in multiple batches to ensure multiple segments/generations.
fn seed_strings_as_keys(store: &ColumnStore<MemPager>, fid: LogicalFieldId, n: usize, seed: u64) {
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8),
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = rand_unique_ascii(&mut rng, 0, 64, i);
            let v = (i as u64).to_be_bytes().to_vec();
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

fn seed_strings_as_values(store: &ColumnStore<MemPager>, fid: LogicalFieldId, n: usize, seed: u64) {
    let opts = AppendOptions {
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
    let page_size = 1009usize;
    seed_strings_as_keys(&store, fid, n, 0xDEADBEEF);

    // REFACTORED: Create the iterator once for the entire scan.
    let mut it = store
        .scan_values_lww(
            fid,
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                ..Default::default()
            },
        )
        .expect("scan init should succeed");

    let mut seen: usize = 0;
    let mut last_key: Option<Vec<u8>> = None;

    // REFACTORED: Loop by consuming the single iterator instance in pages.
    loop {
        let page_items: Vec<_> = it.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break; // Iterator is exhausted.
        }

        for item in page_items {
            if let Some(prev) = &last_key {
                assert!(item.key.as_slice() > prev.as_slice(), "keys must increase");
            }
            last_key = Some(item.key);
            seen += 1;
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
    let page_size = 977usize;
    seed_strings_as_values(&store, fid, n, 0xBADF00D);

    // Create the iterator once for the entire scan.
    let mut it = store
        .scan_values_lww(
            fid,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                ..Default::default()
            },
        )
        .expect("scan init should succeed");

    let mut seen: usize = 0;
    let mut last_val: Option<Vec<u8>> = None;

    // Loop by consuming the single iterator instance in pages.
    loop {
        let page_items: Vec<_> = it.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break;
        }

        for item in page_items {
            let cur = item.value.to_vec();
            if let Some(prev) = &last_val {
                assert!(cur.as_slice() > prev.as_slice(), "values must increase");
            }
            last_val = Some(cur);
            seen += 1;
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
    let page_size = 997usize;
    seed_strings_as_keys(&store, fid_k, n, 1234);
    seed_strings_as_values(&store, fid_v, n, 4321);

    // -------- Reverse key-ordered pagination --------
    let mut it_k = store
        .scan_values_lww(
            fid_k,
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                ..Default::default()
            },
        )
        .expect("reverse key scan init should succeed");

    let mut last_key: Option<Vec<u8>> = None;
    let mut seen_k = 0usize;
    loop {
        let page_items: Vec<_> = it_k.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break;
        }
        for item in page_items {
            if let Some(prev) = &last_key {
                assert!(item.key.as_slice() < prev.as_slice(), "keys must decrease");
            }
            last_key = Some(item.key);
            seen_k += 1;
        }
    }
    assert_eq!(seen_k, n, "must consume all items (reverse keys)");

    // -------- Reverse value-ordered pagination --------
    let mut it_v = store
        .scan_values_lww(
            fid_v,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                ..Default::default()
            },
        )
        .expect("reverse value scan init should succeed");

    let mut last_val: Option<Vec<u8>> = None;
    let mut seen_v = 0usize;
    loop {
        let page_items: Vec<_> = it_v.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break;
        }
        for item in page_items {
            let cur = item.value.to_vec();
            if let Some(prev) = &last_val {
                assert!(cur.as_slice() < prev.as_slice(), "values must decrease");
            }
            last_val = Some(cur);
            seen_v += 1;
        }
    }
    assert_eq!(seen_v, n, "must consume all items (reverse values)");
}

/// Heavy test (ignored by default): run with up to a million rows.
#[test]
#[ignore = "CPU intensive test"]
fn paginate_strings_heavy_ignored() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid_k = 9101u32;
    let fid_v = 9102u32;

    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let page_size = std::env::var("PAGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    seed_strings_as_keys(&store, fid_k, n, 0xA5A5_0001);
    seed_strings_as_values(&store, fid_v, n, 0x5A5A_0002);

    // -------- Key-ordered forward pagination (stress) --------
    let mut it_k = store
        .scan_values_lww(
            fid_k,
            ValueScanOpts {
                order_by: OrderBy::Key,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                ..Default::default()
            },
        )
        .expect("forward key scan init should succeed");

    let mut last_k: Option<Vec<u8>> = None;
    let mut seen_k = 0usize;
    loop {
        let page_items: Vec<_> = it_k.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break;
        }
        for item in page_items {
            if let Some(prev) = &last_k {
                assert!(item.key.as_slice() > prev.as_slice(), "keys must increase");
            }
            last_k = Some(item.key);
            seen_k += 1;
        }
    }
    assert_eq!(seen_k, n, "must consume all items (forward keys, heavy)");

    // -------- Value-ordered reverse pagination (stress) --------
    let mut it_v = store
        .scan_values_lww(
            fid_v,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Reverse,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                ..Default::default()
            },
        )
        .expect("reverse value scan init should succeed");

    let mut last_v: Option<Vec<u8>> = None;
    let mut seen_v = 0usize;
    loop {
        let page_items: Vec<_> = it_v.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break;
        }
        for item in page_items {
            let cur = item.value.to_vec();
            if let Some(prev) = &last_v {
                assert!(cur.as_slice() < prev.as_slice(), "values must decrease");
            }
            last_v = Some(cur);
            seen_v += 1;
        }
    }
    assert_eq!(seen_v, n, "must consume all items (reverse values, heavy)");
}
