//! This example demonstrates the performance of running analytical
//! aggregations directly on the B+Tree's iterator.
//!
//! It seeds a B+Tree with a large number of random u64 keys and random
//! u64 values, then iterates through every other row to calculate common
//! aggregate functions over the VALUES: COUNT, SUM, AVG, MIN, and MAX.
//! The total time for the scan and computation is measured and printed.

use llkv_btree::bplus_tree::{BPlusTree, SharedBPlusTree};
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec};
use llkv_btree::iter::BPlusTreeIter;
use llkv_btree::pager::{MemPager64, SharedPager};
use llkv_btree::prelude::*;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

// ----------------------------- Seed Data --------------------------------

// Note: This will be especially slow if not in release mode
const ITEM_COUNT: usize = 5_000_000;

/// Inserts random u64 keys and independent random u64 values.
fn seed(
    shared: &mut SharedBPlusTree<
        SharedPager<MemPager64>,
        BigEndianKeyCodec<u64>,
        BigEndianIdCodec<u64>,
    >,
) {
    let mut rng = StdRng::seed_from_u64(1337);
    let mut items = Vec::with_capacity(ITEM_COUNT);

    for _ in 0..ITEM_COUNT {
        let key = rng.random::<u64>();
        let val = rng.random::<u64>(); // value independent of key
        items.push((key, val.to_be_bytes()));
    }

    let owned: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();

    shared.insert_many(&owned).expect("seed failed");
}

// ------------------------- Aggregation Logic ----------------------------

/// Results of the aggregation query over VALUES.
#[derive(Debug)]
struct Aggregates {
    count: u64,
    sum: u128, // u128 prevents overflow when summing many u64 values.
    min: u64,
    max: u64,
    avg: f64,
}

/// Scan every other row and aggregate over the VALUE payload.
fn calculate_aggregates(
    tree: &BPlusTree<SharedPager<MemPager64>, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>,
) -> Option<Aggregates> {
    let mut count = 0u64;
    let mut sum = 0u128;
    let mut min = u64::MAX;
    let mut max = u64::MIN;

    // Standard forward iterator over the full snapshot.
    let it = BPlusTreeIter::new(tree).unwrap();

    // Branchless "take every other row" toggle.
    let mut take = true;

    for (_k_ref, v_ref) in it {
        if !take {
            take = true;
            continue;
        }
        take = false;

        // Decode value bytes as u64 (we aggregate over values, not keys).
        let val = u64::from_be_bytes(v_ref.as_ref().try_into().unwrap());

        count += 1;
        sum += val as u128;
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    if count == 0 {
        return None;
    }

    Some(Aggregates {
        count,
        sum,
        min,
        max,
        avg: sum as f64 / count as f64,
    })
}

// --------------------------------- main ----------------------------------

fn main() {
    // 1) Setup pager and empty tree.
    let base = MemPager64::new(4096);
    let pager = SharedPager::new(base);
    let mut shared = SharedBPlusTree::create_empty(pager, None).expect("create failed");

    // 2) Seed with random keys and values.
    println!("Seeding B-Tree with {} items...", ITEM_COUNT);
    seed(&mut shared);
    println!("Seeding complete.");

    // 3) Snapshot for a stable read view.
    let snap = shared.snapshot();

    // 4) Run aggregation timed.
    println!("\nCalculating aggregates over VALUES (every other row)...");
    let now = Instant::now();
    let results = calculate_aggregates(&snap);
    let elapsed = now.elapsed();

    // 5) Print results.
    match results {
        Some(aggs) => {
            println!("\n--- Aggregation Results ---");
            println!("Processed Rows (COUNT): {}", aggs.count);
            println!("MIN:                    {}", aggs.min);
            println!("MAX:                    {}", aggs.max);
            println!("SUM:                    {}", aggs.sum);
            println!("AVG:                    {}", aggs.avg);
            println!("\nTotal time:             {:?}", elapsed);
        }
        None => {
            println!("No data found to aggregate.");
        }
    }
}
