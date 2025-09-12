use std::sync::Arc;
use std::thread;

use llkv_column_map::column_store::ColumnStore;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

/// Compile-time proof that ColumnStore<MemPager> is Send + Sync.
#[test]
fn column_store_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ColumnStore<MemPager>>();
}

/// Concurrent appends against a single shared ColumnStore must be safe.
/// Each thread appends a *disjoint* set of keys to the same field id.
/// After joining, we read all keys back and verify values.
///
/// This exercises:
/// - multi-threaded writers (serialized internally by the store),
/// - multiple segments (small segment limits),
/// - lookup correctness after interleaved appends.
#[test]
fn concurrent_appends_are_thread_safe() {
    // Shared in-memory pager + store.
    let pager = Arc::new(MemPager::default());
    let store = Arc::new(ColumnStore::open(pager));

    const THREADS: usize = 4;
    const KEYS_PER_THREAD: usize = 200;
    const FID: LogicalFieldId = 99;

    // Make batches (intentionally out of order inside each batch).
    let make_batch = |tid: usize| -> Vec<(Vec<u8>, Vec<u8>)> {
        // Keys: "t{tid}-k{idx}" in *descending* order so per-batch input is unsorted.
        // Values: "v{tid}:{idx}"
        let mut v = Vec::with_capacity(KEYS_PER_THREAD);
        for i in (0..KEYS_PER_THREAD).rev() {
            let k = format!("t{tid}-k{i:04}").into_bytes();
            let val = format!("v{tid}:{i}").into_bytes();
            v.push((k, val));
        }
        v
    };

    // Spawn writers.
    let mut handles = Vec::new();
    for tid in 0..THREADS {
        let store_cloned = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            let items = make_batch(tid)
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect::<Vec<_>>();

            // Small segment thresholds to force multiple segments and more interleaving.
            let opts = AppendOptions {
                mode: ValueMode::Auto,
                segment_max_entries: 64,
                segment_max_bytes: 8 * 1024,
                last_write_wins_in_batch: true,
                sort_key: None,
            };

            store_cloned.append_many(
                vec![Put {
                    field_id: FID,
                    items,
                }],
                opts,
            );
        }));
    }

    // Wait for all writers to finish.
    for h in handles {
        h.join().expect("writer thread panicked");
    }

    // Build the full set of keys in a deterministic order for verification.
    let mut all_keys: Vec<Vec<u8>> = Vec::with_capacity(THREADS * KEYS_PER_THREAD);
    for tid in 0..THREADS {
        for i in 0..KEYS_PER_THREAD {
            all_keys.push(format!("t{tid}-k{i:04}").into_bytes());
        }
    }

    // Read all keys back in one logical query.
    let got = store.get_many(vec![(FID, all_keys.clone())]);
    let got = &got[0];

    // Verify every (key -> value) pair we expect is present and correct.
    for (idx, key) in all_keys.iter().enumerate() {
        // Reconstruct expected "v{tid}:{i}" from the ordering we built above.
        let tid = idx / KEYS_PER_THREAD;
        let i = idx % KEYS_PER_THREAD;
        let expect = format!("v{tid}:{i}").into_bytes();

        let val = got[idx]
            .as_ref()
            .unwrap_or_else(|| panic!("missing value for key {:?}", String::from_utf8_lossy(key)));
        assert_eq!(
            val.as_slice(),
            expect.as_slice(),
            "wrong value for key {:?}",
            String::from_utf8_lossy(key)
        );
    }

    // Sanity: we should have recorded at least one batch for appends and one for the read.
    let stats = store.io_stats();
    assert!(
        stats.batches >= 2,
        "expected >=2 pager batches (writes + read), got {}",
        stats.batches
    );
}
