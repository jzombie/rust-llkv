//! Pager persistence + visibility tests for ColumnStore.
//!
//! This test focuses on **durability across handle reopen** using the same
//! pager. We write data with one handle, drop it (via lexical scope end),
//! reopen a fresh handle, and verify the data is still visible. We then
//! perform updates in a new handle, drop it, reopen again, and verify the
//! updates persisted.
//!
//! TODO: ⚠️ **Deletes are NOT thoroughly tested here.**

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::borrow::Cow;
use std::sync::Arc;

use llkv_column_map::ColumnStore;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, LogicalFieldId, Put, ValueMode};

/// Verify that segments written via `append_many` are durable and visible
/// after dropping a handle and reopening another handle on the same pager.
///
/// Why scoped handles (instead of two live handles)?
/// - We want to avoid any ambiguity around live inter-handle cache coherence
///   and focus purely on persistence/visibility after a clean reopen.
/// - Letting the handle go out of scope ensures all in-memory state owned by
///   the handle is released.
///
/// Contract checked:
/// 1) After an initial write, reopening the store yields the same values.
/// 2) After an update batch, reopening again yields the updated values for the
///    updated keys and the original values for the rest.
///
/// Assumptions:
/// - `append_many` seals and persists all written segments in one batch.
/// - The pager returns the same contents to any newly opened `ColumnStore`
///   handle sharing that pager.
#[test]
fn test_pager_persistence_on_reopen_column_map() {
    let n: u64 = std::env::var("LLKV_TEST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let pager = Arc::new(MemPager::default());
    let fid: LogicalFieldId = 42;

    // Generate & shuffle keys (outside scopes so we can reuse)
    let mut keys: Vec<u64> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    keys.shuffle(&mut rng);

    // 1) Create a store handle and write the initial batch (value = !k).
    {
        let store = ColumnStore::open(Arc::clone(&pager));

        let items: Vec<_> = keys
            .iter()
            .map(|k| {
                let key = k.to_be_bytes().to_vec();
                let val = (!k).to_be_bytes().to_vec();
                (Cow::Owned(key), Cow::Owned(val))
            })
            .collect();

        store.append_many(
            vec![Put {
                field_id: fid,
                items,
            }],
            AppendOptions {
                mode: ValueMode::Auto,
                segment_max_entries: 100_000,
                segment_max_bytes: 32 << 20,
                last_write_wins_in_batch: true,
            },
        );
        // store drops here (end of scope)
    }

    // 2) Reopen and spot-check that initial data is visible.
    {
        let store = ColumnStore::open(Arc::clone(&pager));

        for k in (0..n).step_by(((n / 10).max(1)) as usize) {
            let key = k.to_be_bytes().to_vec();
            let got = store.get_many(vec![(fid, vec![key.clone()])]);

            // got[0][0] is Option<ValueSlice>
            let found = got[0][0].as_ref().map(|vs| {
                let a = vs.start() as usize;
                let b = vs.end() as usize;
                let arc = vs.data();
                arc[a..b].to_vec()
            });

            assert_eq!(
                found.as_deref(),
                Some((!k).to_be_bytes().as_slice()),
                "after reopen, initial value mismatch for key {k}"
            );
        }
        // store drops here
    }

    // 3) Update a subset in a fresh handle (value = k ^ 0xDEADBEEF).
    {
        let store = ColumnStore::open(Arc::clone(&pager));

        let updates: Vec<_> = (0..n)
            .step_by(7)
            .map(|k| {
                let key = k.to_be_bytes().to_vec();
                let val = (k ^ 0xDEAD_BEEF).to_be_bytes().to_vec();
                (Cow::Owned(key), Cow::Owned(val))
            })
            .collect();

        store.append_many(
            vec![Put {
                field_id: fid,
                items: updates,
            }],
            AppendOptions {
                mode: ValueMode::Auto,
                segment_max_entries: 100_000,
                segment_max_bytes: 32 << 20,
                last_write_wins_in_batch: true,
            },
        );
        // store drops here
    }

    // 4) Reopen and verify updated values persisted; non-updated keys still equal !k.
    {
        let store = ColumnStore::open(Arc::clone(&pager));

        for k in (0..n).step_by(((n / 20).max(1)) as usize) {
            let key = k.to_be_bytes().to_vec();
            let got = store.get_many(vec![(fid, vec![key.clone()])]);
            let found = got[0][0].as_ref().map(|vs| {
                let a = vs.start() as usize;
                let b = vs.end() as usize;
                let arc = vs.data();
                arc[a..b].to_vec()
            });

            let expected = if k % 7 == 0 {
                (k ^ 0xDEAD_BEEF).to_be_bytes().to_vec()
            } else {
                (!k).to_be_bytes().to_vec()
            };

            assert_eq!(
                found.as_deref(),
                Some(expected.as_slice()),
                "after reopen, value mismatch for key {k}"
            );
        }
        // store drops here
    }

    eprintln!("NOTE: delete sync is NOT exercised in this test.");
}
