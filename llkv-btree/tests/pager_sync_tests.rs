// Persistence + sync test for the plain B+ tree:
//  1) Build A with randomized inserts (forces many leaf splits).
//  2) Open B from the pager's last root and verify it sees data.
//  3) Update some keys from A; verify via B.
//  4) Delete some keys from B; verify via A.
//  5) Insert a new disjoint range from B; verify via A.
//  6) Re-open C from latest root and compare full scans to ground truth.

mod common;

use common::{
    SharedPager, TreeU64, assert_dots_eq_u64 as assert_dots_eq, expect_val_u64 as expect_val,
    reopen_latest, snapshot_as_vec_u64 as snapshot_as_vec,
};
use llkv_btree::bplus_tree::BPlusTree;
use llkv_btree::traits::GraphvizExt;

#[test]
fn pager_sync_bplustree() {
    // Allow overriding N via env if you want to stress harder:
    let n: u64 = std::env::var("LLKV_TEST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);

    let pager = SharedPager::new(4096);

    // A: bulk randomized load
    let mut tree_a: TreeU64<SharedPager> =
        BPlusTree::create_empty(pager.clone(), None).expect("create_empty");

    // Generate & shuffle keys
    let mut keys: Vec<u64> = (0..n).collect();
    {
        use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        keys.shuffle(&mut rng);
    }

    // Build initial batch
    let mut initial: Vec<(u64, [u8; 8])> = Vec::with_capacity(n as usize);
    for k in keys.into_iter() {
        initial.push((k, k.to_be_bytes()));
    }
    let owned_initial: Vec<(u64, &[u8])> =
        initial.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    tree_a.insert_many(&owned_initial).expect("insert_many(A)");

    // B: open from last root
    let root = pager.last_root().unwrap_or_else(|| tree_a.root_id());
    let mut tree_b: TreeU64<SharedPager> = BPlusTree::new(pager.clone(), root, None);

    // spot-check + full count
    for &k in &[0, 1, 2, n - 2, n - 1] {
        expect_val(&tree_b, k, Some(&k.to_be_bytes()));
    }

    // Ensure structures match after initial open
    assert_dots_eq(&tree_a, &tree_b, "after_initial_open");

    let mut snap_b = snapshot_as_vec(&tree_b);
    snap_b.sort_by_key(|e| e.0);
    assert_eq!(
        snap_b.len(),
        n as usize,
        "B should see all {n} initial rows"
    );
    assert_eq!(snap_b.first().unwrap().0, 0);
    assert_eq!(snap_b.last().unwrap().0, n - 1);

    // A: updates every 7th
    let updates: Vec<(u64, [u8; 8])> = (0..n)
        .filter(|k| k % 7 == 0)
        .map(|k| (k, (!k).to_be_bytes()))
        .collect();
    let owned_updates: Vec<(u64, &[u8])> =
        updates.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    tree_a.insert_many(&owned_updates).expect("updates(A)");

    // Verify some of those updates are visible via B
    tree_b.clear_cache(); // Clear cache for remote updates
    for (k, v) in updates.iter().step_by(97).take(100) {
        expect_val(&tree_b, *k, Some(v));
    }

    // Structure should remain identical (value-size unchanged)
    assert_dots_eq(&tree_a, &tree_b, "after_updates_seen_by_B");

    // B: deletes every 5th
    let dels: Vec<u64> = (0..n).filter(|k| k % 5 == 0).collect();
    tree_b.delete_many(&dels).expect("delete_many(B)");

    // Verify on A (random-ish spread). A may still point at an older root,
    // but read paths must still see the updated pages.
    tree_a.clear_cache(); // Clear cache for remote updates
    for k in (0..n).step_by(113).take(100) {
        if k % 5 == 0 {
            expect_val(&tree_a, k, None);
        } else if k % 7 == 0 {
            expect_val(&tree_a, k, Some((!k).to_be_bytes().as_slice()));
        } else {
            expect_val(&tree_a, k, Some(k.to_be_bytes().as_slice()));
        }
    }

    // Refresh A to the latest root before comparing DOTs in case B's deletes
    // caused structural changes (e.g., merges or root shrink).
    let tree_a = reopen_latest(&pager);
    assert_dots_eq(&tree_a, &tree_b, "after_deletes_seen_by_A");

    // B: new disjoint range (scale a bit with n)
    let extra_start = n;
    let extra_end = n + 7_500; // keeps total written well past 50k
    let extra: Vec<(u64, [u8; 8])> = (extra_start..extra_end)
        .map(|k| (k, k.to_be_bytes()))
        .collect();
    let owned_extra: Vec<(u64, &[u8])> = extra.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    tree_b.insert_many(&owned_extra).expect("extra inserts(B)");

    // After B's inserts (which can cause splits/possible new root),
    // refresh A again to latest root before DOT comparison.
    let tree_a = reopen_latest(&pager);
    assert_dots_eq(&tree_a, &tree_b, "after_extra_inserts_seen_by_A");

    // Ground truth
    let mut truth = std::collections::BTreeMap::<u64, Vec<u8>>::new();
    for (k, v) in initial {
        truth.insert(k, v.to_vec());
    }
    for (k, v) in updates {
        truth.insert(k, v.to_vec());
    }
    for k in dels {
        truth.remove(&k);
    }
    for (k, v) in extra {
        truth.insert(k, v.to_vec());
    }

    // C: reopen from latest root and compare scans
    let latest_root = pager.last_root().unwrap_or(root);
    let tree_c: TreeU64<SharedPager> = BPlusTree::new(pager.clone(), latest_root, None);

    let mut got_a = snapshot_as_vec(&tree_a);
    let mut got_b = snapshot_as_vec(&tree_b);
    let mut got_c = snapshot_as_vec(&tree_c);

    got_a.sort_by_key(|e| e.0);
    got_b.sort_by_key(|e| e.0);
    got_c.sort_by_key(|e| e.0);

    let mut want: Vec<(u64, Vec<u8>)> = truth.into_iter().collect();
    want.sort_by_key(|e| e.0);

    assert_eq!(got_a, want, "A scan mismatch vs ground truth");
    assert_eq!(got_b, want, "B scan mismatch vs ground truth");
    assert_eq!(got_c, want, "C scan mismatch vs ground truth");

    // Final DOT parity across all handles
    let dot_a = tree_a.to_dot().expect("to_dot(A@final)");
    let dot_b = tree_b.to_dot().expect("to_dot(B@final)");
    let dot_c = tree_c.to_dot().expect("to_dot(C@final)");
    assert_eq!(dot_a, dot_b, "DOT mismatch A vs B @ final");
    assert_eq!(dot_a, dot_c, "DOT mismatch A vs C @ final");
}
