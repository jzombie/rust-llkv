use std::ops::Bound::{Excluded, Included, Unbounded};

use llkv_btree::codecs::KeyCodec;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec};
use llkv_btree::define_mem_pager;
use llkv_btree::iter::{BPlusTreeIter, ScanOpts};
use llkv_btree::pager::SharedPager;
use llkv_btree::prelude::*;
use llkv_btree::shared_bplus_tree::SharedBPlusTree;

define_mem_pager! {
    /// In-memory pager with u64 page IDs.
    name: MemPager64,
    id: u64,
    default_page_size: 256
}

// ---------- types ----------
// Type aliases to make the code more readable.
type KC = BigEndianKeyCodec<u64>;
type IC = BigEndianIdCodec<u64>;
type SP = SharedPager<MemPager64>;
type SharedTree = SharedBPlusTree<SP, KC, IC>;

/// Fetches a single "page" of data from the tree.
///
/// # Arguments
/// * `tree`: A reference to the shared tree.
/// * `cursor`: An optional key from the previous page. If `None`, pagination starts
///   from the beginning (or end, if reversing).
/// * `reverse`: The direction of pagination.
/// * `page_size`: The maximum number of items to retrieve for the page.
///
/// # Returns
/// A tuple containing a vector of rows for the page, and an optional
/// cursor (the key of the last item) to be used for the next page request.
fn page_once(
    tree: &SharedTree,
    cursor: Option<u64>,
    reverse: bool,
    page_size: usize,
) -> (Vec<(u64, Vec<u8>)>, Option<u64>) {
    // Take a snapshot of the tree. This provides a stable, consistent view of
    // the data for the duration of this function call, even if other threads
    // are writing to the tree.
    let snap = tree.snapshot();

    // Build the basic scan options, setting the direction based on the `reverse` flag.
    let mut opts = if reverse {
        ScanOpts::<KC>::reverse()
    } else {
        ScanOpts::<KC>::forward()
    };
    opts.lower = Unbounded;
    opts.upper = Unbounded;

    // This section translates the cursor into a specific start/end boundary for the scan.
    // The `lo_guard` and `up_guard` variables are a pattern to ensure that the key
    // we borrow for the scan options lives for the entire function.
    let (lo_guard, up_guard);
    if let Some(k) = cursor {
        if reverse {
            // For a reverse scan, we want items with keys *less than* the cursor.
            up_guard = k; // `up_guard` now owns the key.
            opts.upper = Excluded(&up_guard); // The scan will end before this key.
        } else if let Some(k1) = k.checked_add(1) {
            // For a forward scan, we want items with keys *greater than* the cursor.
            // For integers, this is equivalent to starting at `cursor + 1`.
            lo_guard = k1; // `lo_guard` now owns the key.
            opts.lower = Included(&lo_guard); // The scan will start at this key.
        } else {
            // This handles the edge case where the cursor is u64::MAX and we're
            // paging forward; there are no more keys after it.
            return (Vec::new(), None);
        }
    }

    // Create a synchronous iterator using the snapshot and the configured options.
    let mut it = BPlusTreeIter::with_opts(&snap, opts).expect("iter");

    // Collect the rows for this page.
    let mut rows = Vec::with_capacity(page_size);
    // This loop will run up to `page_size` times, pulling one item from the
    // iterator at a time until the page is full or the iterator is exhausted.
    for _ in 0..page_size {
        match it.next() {
            Some((kref, vref)) => {
                let k = KC::decode_from(kref.as_ref()).expect("decode key");
                rows.push((k, vref.as_ref().to_vec()));
            }
            None => break, // Stop if the iterator has no more items.
        }
    }

    // The cursor for the *next* page is the key of the last item on *this* page.
    let next_cursor = rows.last().map(|(k, _)| *k);
    (rows, next_cursor)
}

fn main() {
    // 1. Setup: Wrap the in-memory pager with a thread-safe adapter.
    let base = MemPager64::new(4096);
    let pager = SharedPager::new(base);

    // Build a shared B+Tree that can be read from and written to.
    let tree: SharedTree = SharedTree::create_empty(pager, None).expect("create_empty");

    // 2. Seeding: Insert 20 key-value pairs into the tree.
    let items: Vec<(u64, Vec<u8>)> = (1u64..=20u64)
        .map(|k| (k, k.to_be_bytes().to_vec()))
        .collect();
    let owned: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    tree.insert_many(&owned).expect("insert_many");

    // 3. Forward Pagination: Paginate through the data in chunks of 7.
    println!("-- forward pages (size=7) --");
    // The cursor starts as `None` to begin the scan from the very first key.
    let mut cursor = None;
    loop {
        // Fetch one page of data. The `next_cursor` will be the key of the last item.
        let (page, next_cursor) = page_once(&tree, cursor, /*reverse=*/ false, 7);
        // If the returned page is empty, we've reached the end.
        if page.is_empty() {
            break;
        }
        // Print the contents of the page and verify the data.
        for (k, v) in &page {
            assert_eq!(v.len(), 8);
            let got = u64::from_be_bytes(v.as_slice().try_into().unwrap());
            assert_eq!(*k, got);
            print!("{k} ");
        }
        println!();
        // Update the cursor for the next loop iteration.
        cursor = next_cursor;
    }

    // 4. Reverse Pagination: Paginate backward from the end in chunks of 6.
    println!("-- reverse pages from end (size=6) --");
    // The cursor starts at key 20, so the first page will contain keys < 20.
    let mut cursor = Some(20u64);
    loop {
        // Fetch one page of data, this time in reverse.
        let (page, next_cursor) = page_once(&tree, cursor, /*reverse=*/ true, 6);
        if page.is_empty() {
            break;
        }
        for (k, v) in &page {
            assert_eq!(v.len(), 8);
            let got = u64::from_be_bytes(v.as_slice().try_into().unwrap());
            assert_eq!(*k, got);
            print!("{k} ");
        }
        println!();
        // Update the cursor to continue the scan backward.
        cursor = next_cursor;
    }
}
