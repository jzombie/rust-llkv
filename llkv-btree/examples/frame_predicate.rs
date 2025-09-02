//! Simple counter demo for frame_predicate.
//! Keys are u64 integers from 1 to 25.
//!
//! The frame_predicate groups keys into batches of 5. The main loop
//! processes one batch (frame) per pass, then resumes the scan from
//! where the last batch left off.

use core::ops::Bound::{Excluded, Unbounded};
use llkv_btree::bplus_tree::SharedBPlusTree;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec, KeyCodec};
use llkv_btree::errors::Error;
use llkv_btree::iter::{BPlusTreeIter, ScanOpts};
// Import the pagers used in the other example
use llkv_btree::pager::{MemPager64, SharedPager};
use llkv_btree::prelude::*;

// Added for the shuffle functionality.
use rand::seq::SliceRandom;

// ----------------------------- seed data --------------------------------

/// Inserts keys 1 through 25 into the B-Tree in a random order.
fn seed(
    // The function now accepts the tree using the standard SharedPager.
    shared: &mut SharedBPlusTree<
        SharedPager<MemPager64>,
        BigEndianKeyCodec<u64>,
        BigEndianIdCodec<u64>,
    >,
) -> Result<(), Error> {
    // Create the items in order first.
    let mut items: Vec<(u64, [u8; 8])> = (1..=25u64).map(|k| (k, k.to_be_bytes())).collect();

    // Shuffle the items into a random order before insertion.
    let mut rng = rand::rng();
    items.shuffle(&mut rng);

    // Insert the now-randomized items into the tree.
    let owned: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    shared.insert_many(&owned)
}

// ----------------------------- export logic ------------------------------

/// Iterates through the tree, processing numbers in batches of 5.
fn export_in_batches(
    // The function now accepts the tree using the standard SharedPager.
    shared: &SharedBPlusTree<
        SharedPager<MemPager64>,
        BigEndianKeyCodec<u64>,
        BigEndianIdCodec<u64>,
    >,
) -> Result<(), Error> {
    // A type alias for clarity.
    type U64KC = BigEndianKeyCodec<u64>;

    let tree = shared.snapshot();
    let mut last_key_owned: Option<u64> = None;

    println!("--- Exporting counter in batches of 5 ---");

    loop {
        // Determine the starting point for this scan.
        let lower_bound = match &last_key_owned {
            None => Unbounded,
            Some(k) => Excluded(k),
        };

        // The frame predicate groups numbers into batches of 5.
        let same_batch = |head: &[u8], current: &[u8]| -> bool {
            let head_key = u64::from_be_bytes(head.try_into().unwrap());
            let current_key = u64::from_be_bytes(current.try_into().unwrap());
            (head_key - 1) / 5 == (current_key - 1) / 5
        };

        // Create an iterator that will stop automatically at the batch boundary.
        let iter = BPlusTreeIter::with_opts(
            &tree,
            ScanOpts::forward()
                .with_bounds(lower_bound, Unbounded)
                .with_frame_predicate(same_batch),
        )?;

        // Collect all items from the current frame.
        let batch: Vec<u64> = iter
            .map(|(k, _)| U64KC::decode_from(k.as_ref()).unwrap())
            .collect();

        // If the iterator returned nothing, we are done.
        if batch.is_empty() {
            break;
        }

        // Print the batch we collected.
        println!("Batch: {:?}", batch);

        // Save the last key from this batch to resume the next scan.
        last_key_owned = batch.last().cloned();
    }

    Ok(())
}

// --------------------------------- main ----------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup is now identical to the pagination example.
    // 1. Create the base in-memory pager.
    let base = MemPager64::new(4096);
    // 2. Wrap it in the thread-safe SharedPager adapter.
    let pager = SharedPager::new(base);
    // 3. Create the tree instance.
    let mut shared = SharedBPlusTree::create_empty(pager, None)?;

    seed(&mut shared)?;
    export_in_batches(&shared)?;
    Ok(())
}
