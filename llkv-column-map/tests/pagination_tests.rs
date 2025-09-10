//! Integration tests for key/value pagination with randomized var-width strings
//! and fixed-width integers.
//!
//! Uses only `rand`.
//! Heavy tests are #[ignore] and run independently so the store is
//! seeded once per process.
//!
//! Run heavy tests (examples):
//!   RUST_TEST_THREADS=1 cargo test --release \
//!     --test pagination_tests \
//!     paginate_strings_heavy_keys_1kb_ignored \
//!     -- --ignored --nocapture
//!
//!   RUST_TEST_THREADS=1 cargo test --release \
//!     --test pagination_tests \
//!     paginate_strings_heavy_vals_1kb_ignored \
//!     -- --ignored --nocapture
//!
//! You can override sizes via env vars for the ignored tests:
//!   N=1000000 PAGE=4096 MAX_TOTAL=1024 cargo test --release \
//!     --test pagination_tests -- --ignored
//!
//! To see printed timings in any test, add: -- --nocapture

use llkv_column_map::ColumnStore;
use llkv_column_map::column_store::read_scan::{Direction, OrderBy, ValueScanOpts};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};

use std::ops::Bound;
use std::time::Instant;

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

/// Build a random ASCII string with total length <= `max_total`.
/// The `#` + 8-hex tag adds 9 bytes; we cap the random part so the
/// final length never exceeds `max_total`.
fn rand_varlen_upto_total(rng: &mut StdRng, max_total: usize, ordinal: usize) -> Vec<u8> {
    let tag_overhead = 9usize;
    let usable = max_total.saturating_sub(tag_overhead);
    let len = rng.random_range(0..=usable);
    let mut s = Vec::with_capacity(len + tag_overhead);
    for _ in 0..len {
        let i = rng.random_range(0..ALPHABET.len());
        s.push(ALPHABET[i]);
    }
    s.push(b'#');
    let tag = format!("{:08x}", ordinal);
    s.extend_from_slice(tag.as_bytes());
    s
}

// --------------------------- seeders (73B) --------------------------

/// Append in multiple batches to ensure multiple segments/generations.
/// Prints per-batch creation and ingestion timings.
fn seed_strings_as_keys(store: &ColumnStore<MemPager>, fid: LogicalFieldId, n: usize, seed: u64) {
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8),
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    let t_build_all = Instant::now();
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = rand_unique_ascii(&mut rng, 0, 64, i);
            let v = (i as u64).to_be_bytes().to_vec();
            (k, v)
        })
        .collect();
    let build_all_ms = t_build_all.elapsed().as_secs_f64() * 1e3;

    let t_shuffle = Instant::now();
    items.shuffle(&mut rng);
    let shuffle_ms = t_shuffle.elapsed().as_secs_f64() * 1e3;

    println!(
        "[seed keys] built all items: {:.3} ms, shuffle: {:.3} ms, n={}",
        build_all_ms, shuffle_ms, n
    );

    const CHUNK: usize = 10_000;
    let total_batches = items.len().div_ceil(CHUNK);

    // Aggregates for "lump" reporting.
    let t_seed = Instant::now();
    let mut sum_make_ms = 0.0f64;
    let mut sum_ingest_ms = 0.0f64;
    let mut max_make_ms = 0.0f64;
    let mut max_ingest_ms = 0.0f64;
    let mut slow_ingests = 0usize; // ingest > 8 ms

    for (bi, chunk) in items.chunks(CHUNK).enumerate() {
        let t_batch_make = Instant::now();
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        let make_ms = t_batch_make.elapsed().as_secs_f64() * 1e3;

        let t_ingest = Instant::now();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
        let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1e3;

        println!(
            "[seed keys] batch {}/{}: make={:.3} ms, ingest={:.3} ms, \
items={}",
            bi + 1,
            total_batches,
            make_ms,
            ingest_ms,
            chunk.len()
        );

        sum_make_ms += make_ms;
        sum_ingest_ms += ingest_ms;
        if make_ms > max_make_ms {
            max_make_ms = make_ms;
        }
        if ingest_ms > max_ingest_ms {
            max_ingest_ms = ingest_ms;
        }
        if ingest_ms > 8.0 {
            slow_ingests += 1;
        }
    }

    let seed_ms = t_seed.elapsed().as_secs_f64() * 1e3;
    println!(
        "[seed keys] totals: batches={}, make_sum={:.3} ms, \
ingest_sum={:.3} ms, make_max={:.3} ms, ingest_max={:.3} ms, \
slow_ingests_gt8ms={}, seed_end_to_end={:.3} ms",
        total_batches,
        sum_make_ms,
        sum_ingest_ms,
        max_make_ms,
        max_ingest_ms,
        slow_ingests,
        seed_ms
    );
}

/// Append in multiple batches to ensure multiple segments/generations.
/// Prints per-batch creation and ingestion timings.
fn seed_strings_as_values(store: &ColumnStore<MemPager>, fid: LogicalFieldId, n: usize, seed: u64) {
    let opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    let t_build_all = Instant::now();
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = format!("k{:08x}", i).into_bytes();
            let v = rand_unique_ascii(&mut rng, 0, 64, i);
            (k, v)
        })
        .collect();
    let build_all_ms = t_build_all.elapsed().as_secs_f64() * 1e3;

    let t_shuffle = Instant::now();
    items.shuffle(&mut rng);
    let shuffle_ms = t_shuffle.elapsed().as_secs_f64() * 1e3;

    println!(
        "[seed vals] built all items: {:.3} ms, shuffle: {:.3} ms, n={}",
        build_all_ms, shuffle_ms, n
    );

    const CHUNK: usize = 10_000;
    let total_batches = items.len().div_ceil(CHUNK);

    // Aggregates for "lump" reporting.
    let t_seed = Instant::now();
    let mut sum_make_ms = 0.0f64;
    let mut sum_ingest_ms = 0.0f64;
    let mut max_make_ms = 0.0f64;
    let mut max_ingest_ms = 0.0f64;
    let mut slow_ingests = 0usize; // ingest > 8 ms

    for (bi, chunk) in items.chunks(CHUNK).enumerate() {
        let t_batch_make = Instant::now();
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        let make_ms = t_batch_make.elapsed().as_secs_f64() * 1e3;

        let t_ingest = Instant::now();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
        let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1e3;

        println!(
            "[seed vals] batch {}/{}: make={:.3} ms, ingest={:.3} ms, \
items={}",
            bi + 1,
            total_batches,
            make_ms,
            ingest_ms,
            chunk.len()
        );

        sum_make_ms += make_ms;
        sum_ingest_ms += ingest_ms;
        if make_ms > max_make_ms {
            max_make_ms = make_ms;
        }
        if ingest_ms > max_ingest_ms {
            max_ingest_ms = ingest_ms;
        }
        if ingest_ms > 8.0 {
            slow_ingests += 1;
        }
    }

    let seed_ms = t_seed.elapsed().as_secs_f64() * 1e3;
    println!(
        "[seed vals] totals: batches={}, make_sum={:.3} ms, \
ingest_sum={:.3} ms, make_max={:.3} ms, ingest_max={:.3} ms, \
slow_ingests_gt8ms={}, seed_end_to_end={:.3} ms",
        total_batches,
        sum_make_ms,
        sum_ingest_ms,
        max_make_ms,
        max_ingest_ms,
        slow_ingests,
        seed_ms
    );
}

// --------------- seeders for heavy 1 KiB max strings ----------------

/// Heavy seeder where the random string's total length is capped at
/// `max_total` bytes. Keys are var-len; values are fixed u64 (8B).
fn seed_keys_varlen_max_total(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    n: usize,
    seed: u64,
    max_total: usize,
) {
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8),
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    let t_build_all = Instant::now();
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = rand_varlen_upto_total(&mut rng, max_total, i);
            let v = (i as u64).to_be_bytes().to_vec();
            (k, v)
        })
        .collect();
    let build_all_ms = t_build_all.elapsed().as_secs_f64() * 1e3;

    let t_shuffle = Instant::now();
    items.shuffle(&mut rng);
    let shuffle_ms = t_shuffle.elapsed().as_secs_f64() * 1e3;

    println!(
        "[seed keys 1kb] built all: {:.3} ms, shuffle: {:.3} ms, n={}, \
max_total={}",
        build_all_ms, shuffle_ms, n, max_total
    );

    const CHUNK: usize = 10_000;
    let total_batches = items.len().div_ceil(CHUNK);

    let t_seed = Instant::now();
    let mut sum_make_ms = 0.0f64;
    let mut sum_ingest_ms = 0.0f64;
    let mut max_make_ms = 0.0f64;
    let mut max_ingest_ms = 0.0f64;
    let mut slow_ingests = 0usize;

    for (bi, chunk) in items.chunks(CHUNK).enumerate() {
        let t_batch_make = Instant::now();
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        let make_ms = t_batch_make.elapsed().as_secs_f64() * 1e3;

        let t_ingest = Instant::now();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
        let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1e3;

        println!(
            "[seed keys 1kb] batch {}/{}: make={:.3} ms, ingest={:.3} ms, \
items={}",
            bi + 1,
            total_batches,
            make_ms,
            ingest_ms,
            chunk.len()
        );

        sum_make_ms += make_ms;
        sum_ingest_ms += ingest_ms;
        if make_ms > max_make_ms {
            max_make_ms = make_ms;
        }
        if ingest_ms > max_ingest_ms {
            max_ingest_ms = ingest_ms;
        }
        if ingest_ms > 8.0 {
            slow_ingests += 1;
        }
    }

    let seed_ms = t_seed.elapsed().as_secs_f64() * 1e3;
    println!(
        "[seed keys 1kb] totals: batches={}, make_sum={:.3} ms, \
ingest_sum={:.3} ms, make_max={:.3} ms, ingest_max={:.3} ms, \
slow_ingests_gt8ms={}, seed_end_to_end={:.3} ms",
        total_batches,
        sum_make_ms,
        sum_ingest_ms,
        max_make_ms,
        max_ingest_ms,
        slow_ingests,
        seed_ms
    );
}

/// Heavy seeder where the random string's total length is capped at
/// `max_total` bytes. Values are var-len; keys are fixed 9B.
fn seed_vals_varlen_max_total(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    n: usize,
    seed: u64,
    max_total: usize,
) {
    let opts = AppendOptions {
        mode: ValueMode::Auto,
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    let t_build_all = Instant::now();
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = format!("k{:08x}", i).into_bytes();
            let v = rand_varlen_upto_total(&mut rng, max_total, i);
            (k, v)
        })
        .collect();
    let build_all_ms = t_build_all.elapsed().as_secs_f64() * 1e3;

    let t_shuffle = Instant::now();
    items.shuffle(&mut rng);
    let shuffle_ms = t_shuffle.elapsed().as_secs_f64() * 1e3;

    println!(
        "[seed vals 1kb] built all: {:.3} ms, shuffle: {:.3} ms, n={}, \
max_total={}",
        build_all_ms, shuffle_ms, n, max_total
    );

    const CHUNK: usize = 10_000;
    let total_batches = items.len().div_ceil(CHUNK);

    let t_seed = Instant::now();
    let mut sum_make_ms = 0.0f64;
    let mut sum_ingest_ms = 0.0f64;
    let mut max_make_ms = 0.0f64;
    let mut max_ingest_ms = 0.0f64;
    let mut slow_ingests = 0usize;

    for (bi, chunk) in items.chunks(CHUNK).enumerate() {
        let t_batch_make = Instant::now();
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        let make_ms = t_batch_make.elapsed().as_secs_f64() * 1e3;

        let t_ingest = Instant::now();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
        let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1e3;

        println!(
            "[seed vals 1kb] batch {}/{}: make={:.3} ms, ingest={:.3} ms, \
items={}",
            bi + 1,
            total_batches,
            make_ms,
            ingest_ms,
            chunk.len()
        );

        sum_make_ms += make_ms;
        sum_ingest_ms += ingest_ms;
        if make_ms > max_make_ms {
            max_make_ms = make_ms;
        }
        if ingest_ms > max_ingest_ms {
            max_ingest_ms = ingest_ms;
        }
        if ingest_ms > 8.0 {
            slow_ingests += 1;
        }
    }

    let seed_ms = t_seed.elapsed().as_secs_f64() * 1e3;
    println!(
        "[seed vals 1kb] totals: batches={}, make_sum={:.3} ms, \
ingest_sum={:.3} ms, make_max={:.3} ms, ingest_max={:.3} ms, \
slow_ingests_gt8ms={}, seed_end_to_end={:.3} ms",
        total_batches,
        sum_make_ms,
        sum_ingest_ms,
        max_make_ms,
        max_ingest_ms,
        slow_ingests,
        seed_ms
    );
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

    let t_scan = Instant::now();

    loop {
        let page_items: Vec<_> = it.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break;
        }

        for item in page_items {
            if let Some(prev) = &last_key {
                assert!(item.key.as_slice() > prev.as_slice(), "keys must increase");
            }
            last_key = Some(item.key);
            seen += 1;
        }
    }

    let scan_ms = t_scan.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan keys fwd] pagesize={}, items={}, total={:.3} ms",
        page_size, seen, scan_ms
    );

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

    let t_scan = Instant::now();

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

    let scan_ms = t_scan.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan vals fwd] pagesize={}, items={}, total={:.3} ms",
        page_size, seen, scan_ms
    );

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

    // Reverse key-ordered pagination.
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

    let t_scan_k = Instant::now();

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
    let scan_k_ms = t_scan_k.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan keys rev] pagesize={}, items={}, total={:.3} ms",
        page_size, seen_k, scan_k_ms
    );
    assert_eq!(seen_k, n, "must consume all items (reverse keys)");

    // Reverse value-ordered pagination.
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

    let t_scan_v = Instant::now();

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
    let scan_v_ms = t_scan_v.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan vals rev] pagesize={}, items={}, total={:.3} ms",
        page_size, seen_v, scan_v_ms
    );
    assert_eq!(seen_v, n, "must consume all items (reverse values)");
}

// --------------------- heavy, split, 1 KiB max ----------------------

/// Heavy: keys var-len (<= 1 KiB total), values fixed 8B. Seeds once.
/// Scans forward by key. #[ignore] by default.
#[test]
#[ignore = "CPU intensive test"]
fn paginate_strings_heavy_keys_1kb_ignored() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid_k = 9101u32;

    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let page_size = std::env::var("PAGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let max_total = std::env::var("MAX_TOTAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    seed_keys_varlen_max_total(&store, fid_k, n, 0xA5A5_0001, max_total);

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

    let t_scan_k = Instant::now();

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
    let scan_k_ms = t_scan_k.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan keys fwd 1kb] pagesize={}, items={}, total={:.3} ms",
        page_size, seen_k, scan_k_ms
    );
    assert_eq!(seen_k, n, "must consume all items (forward keys, heavy)");
}

/// Heavy: values var-len (<= 1 KiB total), keys fixed 9B. Seeds once.
/// Scans reverse by value. #[ignore] by default.
#[test]
#[ignore = "CPU intensive test"]
fn paginate_strings_heavy_vals_1kb_ignored() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid_v = 9102u32;

    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let page_size = std::env::var("PAGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let max_total = std::env::var("MAX_TOTAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    seed_vals_varlen_max_total(&store, fid_v, n, 0x5A5A_0002, max_total);

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

    let t_scan_v = Instant::now();

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
    let scan_v_ms = t_scan_v.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan vals rev 1kb] pagesize={}, items={}, total={:.3} ms",
        page_size, seen_v, scan_v_ms
    );
    assert_eq!(seen_v, n, "must consume all items (reverse values, heavy)");
}

// --------------- fixed-width integer seeders (u64) -------------------

/// Seeds u64 keys (big-endian) and u64 values. Values are fixed 8B.
/// Uses randomized insert order and prints per-batch timings.
fn seed_ints_as_keys_fixed(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    n: usize,
    seed: u64,
) {
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8), // values are u64 (8B)
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    let t_build_all = Instant::now();
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = (i as u64).to_be_bytes().to_vec(); // 8B key
            let v = (i as u64).to_be_bytes().to_vec(); // 8B value
            (k, v)
        })
        .collect();
    let build_all_ms = t_build_all.elapsed().as_secs_f64() * 1e3;

    let t_shuffle = Instant::now();
    items.shuffle(&mut rng);
    let shuffle_ms = t_shuffle.elapsed().as_secs_f64() * 1e3;

    println!(
        "[seed int-keys] built all: {:.3} ms, shuffle: {:.3} ms, n={}",
        build_all_ms, shuffle_ms, n
    );

    const CHUNK: usize = 10_000;
    let total_batches = items.len().div_ceil(CHUNK);

    let t_seed = Instant::now();
    let mut sum_make_ms = 0.0f64;
    let mut sum_ingest_ms = 0.0f64;
    let mut max_make_ms = 0.0f64;
    let mut max_ingest_ms = 0.0f64;
    let mut slow_ingests = 0usize;

    for (bi, chunk) in items.chunks(CHUNK).enumerate() {
        let t_batch_make = Instant::now();
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        let make_ms = t_batch_make.elapsed().as_secs_f64() * 1e3;

        let t_ingest = Instant::now();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
        let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1e3;

        println!(
            "[seed int-keys] batch {}/{}: make={:.3} ms, ingest={:.3} ms, \
items={}",
            bi + 1,
            total_batches,
            make_ms,
            ingest_ms,
            chunk.len()
        );

        sum_make_ms += make_ms;
        sum_ingest_ms += ingest_ms;
        if make_ms > max_make_ms {
            max_make_ms = make_ms;
        }
        if ingest_ms > max_ingest_ms {
            max_ingest_ms = ingest_ms;
        }
        if ingest_ms > 8.0 {
            slow_ingests += 1;
        }
    }

    let seed_ms = t_seed.elapsed().as_secs_f64() * 1e3;
    println!(
        "[seed int-keys] totals: batches={}, make_sum={:.3} ms, \
ingest_sum={:.3} ms, make_max={:.3} ms, ingest_max={:.3} ms, \
slow_ingests_gt8ms={}, seed_end_to_end={:.3} ms",
        total_batches,
        sum_make_ms,
        sum_ingest_ms,
        max_make_ms,
        max_ingest_ms,
        slow_ingests,
        seed_ms
    );
}

/// Seeds u64 values (big-endian) and u64 keys (big-endian).
/// Fixed 8B for values via ForceFixed(8). Randomized insert order.
fn seed_ints_as_values_fixed(
    store: &ColumnStore<MemPager>,
    fid: LogicalFieldId,
    n: usize,
    seed: u64,
) {
    let opts = AppendOptions {
        mode: ValueMode::ForceFixed(8), // values are u64 (8B)
        segment_max_entries: 8192,
        segment_max_bytes: 1 << 20,
        last_write_wins_in_batch: true,
    };

    let mut rng = StdRng::seed_from_u64(seed);

    let t_build_all = Instant::now();
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            let k = (i as u64).to_be_bytes().to_vec(); // 8B key
            let v = (i as u64).to_be_bytes().to_vec(); // 8B value
            (k, v)
        })
        .collect();
    let build_all_ms = t_build_all.elapsed().as_secs_f64() * 1e3;

    let t_shuffle = Instant::now();
    items.shuffle(&mut rng);
    let shuffle_ms = t_shuffle.elapsed().as_secs_f64() * 1e3;

    println!(
        "[seed int-vals] built all: {:.3} ms, shuffle: {:.3} ms, n={}",
        build_all_ms, shuffle_ms, n
    );

    const CHUNK: usize = 10_000;
    let total_batches = items.len().div_ceil(CHUNK);

    let t_seed = Instant::now();
    let mut sum_make_ms = 0.0f64;
    let mut sum_ingest_ms = 0.0f64;
    let mut max_make_ms = 0.0f64;
    let mut max_ingest_ms = 0.0f64;
    let mut slow_ingests = 0usize;

    for (bi, chunk) in items.chunks(CHUNK).enumerate() {
        let t_batch_make = Instant::now();
        let batch = chunk
            .iter()
            .cloned()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        let make_ms = t_batch_make.elapsed().as_secs_f64() * 1e3;

        let t_ingest = Instant::now();
        store.append_many(
            vec![Put {
                field_id: fid,
                items: batch,
            }],
            opts.clone(),
        );
        let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1e3;

        println!(
            "[seed int-vals] batch {}/{}: make={:.3} ms, ingest={:.3} ms, \
items={}",
            bi + 1,
            total_batches,
            make_ms,
            ingest_ms,
            chunk.len()
        );

        sum_make_ms += make_ms;
        sum_ingest_ms += ingest_ms;
        if make_ms > max_make_ms {
            max_make_ms = make_ms;
        }
        if ingest_ms > max_ingest_ms {
            max_ingest_ms = ingest_ms;
        }
        if ingest_ms > 8.0 {
            slow_ingests += 1;
        }
    }

    let seed_ms = t_seed.elapsed().as_secs_f64() * 1e3;
    println!(
        "[seed int-vals] totals: batches={}, make_sum={:.3} ms, \
ingest_sum={:.3} ms, make_max={:.3} ms, ingest_max={:.3} ms, \
slow_ingests_gt8ms={}, seed_end_to_end={:.3} ms",
        total_batches,
        sum_make_ms,
        sum_ingest_ms,
        max_make_ms,
        max_ingest_ms,
        slow_ingests,
        seed_ms
    );
}

// ---------------------- heavy, fixed-width integers ------------------

/// Heavy: u64 keys (8B), u64 values (8B). Seeds once. Scan by key fwd.
/// #[ignore] by default.
#[test]
#[ignore = "CPU intensive test"]
fn paginate_ints_heavy_keys_fixed_ignored() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid_k = 9201u32;

    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let page_size = std::env::var("PAGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    seed_ints_as_keys_fixed(&store, fid_k, n, 0x1);

    let mut it = store
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
        .expect("scan init should succeed");

    let mut seen = 0usize;
    let mut last_key: Option<Vec<u8>> = None;

    let t_scan = Instant::now();
    loop {
        let page_items: Vec<_> = it.by_ref().take(page_size).collect();
        if page_items.is_empty() {
            break;
        }
        for item in page_items {
            if let Some(prev) = &last_key {
                assert!(item.key.as_slice() > prev.as_slice(), "keys must increase");
            }
            last_key = Some(item.key);
            seen += 1;
        }
    }
    let scan_ms = t_scan.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan int-keys fwd FIXED] pagesize={}, items={}, total={:.3} ms",
        page_size, seen, scan_ms
    );
    assert_eq!(seen, n, "must consume all items (int keys, fixed)");
}

/// Heavy: u64 values (8B), u64 keys (8B). Seeds once. Scan by value fwd.
/// #[ignore] by default.
#[test]
#[ignore = "CPU intensive test"]
fn paginate_ints_heavy_vals_fixed_ignored() {
    let p = MemPager::default();
    let store = ColumnStore::init_empty(&p);
    let fid_v = 9202u32;

    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let page_size = std::env::var("PAGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    seed_ints_as_values_fixed(&store, fid_v, n, 0x1);

    let mut it = store
        .scan_values_lww(
            fid_v,
            ValueScanOpts {
                order_by: OrderBy::Value,
                dir: Direction::Forward,
                lo: Bound::Unbounded,
                hi: Bound::Unbounded,
                ..Default::default()
            },
        )
        .expect("scan init should succeed");

    let mut seen = 0usize;
    let mut last_val: Option<Vec<u8>> = None;

    let t_scan = Instant::now();
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
    let scan_ms = t_scan.elapsed().as_secs_f64() * 1e3;
    println!(
        "[scan int-vals fwd FIXED] pagesize={}, items={}, total={:.3} ms",
        page_size, seen, scan_ms
    );
    assert_eq!(seen, n, "must consume all items (int vals, fixed)");
}
