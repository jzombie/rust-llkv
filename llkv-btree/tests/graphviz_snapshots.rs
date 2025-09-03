mod common;

use llkv_btree::bplus_tree::SharedBPlusTree;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec};
use llkv_btree::prelude::*;
use llkv_btree::traits::GraphvizExt;

const N: u64 = 200;

/// Build a BPlusTree<TestPager> with deterministic items.
fn build_btree() -> common::TreeU64<common::TestPager> {
    let t = common::create_tree().expect("create btree");
    let items: Vec<(u64, Vec<u8>)> = (0..N)
        .map(|k| (k, format!("v{}", k).into_bytes()))
        .collect();
    let owned: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    t.insert_many(&owned).expect("insert_many btree");
    t
}

/// Build a SharedBPlusTree<SharedPager> with the same items.
fn build_shared(
    pager: common::SharedPager,
) -> SharedBPlusTree<common::SharedPager, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
    let t = SharedBPlusTree::create_empty(pager, None).expect("create");
    let items: Vec<(u64, Vec<u8>)> = (0..N)
        .map(|k| (k, format!("v{}", k).into_bytes()))
        .collect();
    let owned: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    t.insert_many(&owned).expect("insert_many shared");
    t
}

#[test]
fn graphviz_btree_vs_shared_identical_and_snapshotted() {
    // Build both trees on the same dataset.
    let t_bt = build_btree();
    let pager = common::SharedPager::new(256);
    let t_sh: SharedBPlusTree<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
        build_shared(pager);

    // Canonicalize and snapshot the BPlusTree graph.
    // To (re)generate: set UPDATE_SNAPSHOTS=1 before running tests.
    let dot_bt = t_bt.to_canonicalized_dot().expect("to_dot btree");
    common::assert_matches_golden(&dot_bt, "tests/snapshots/graphviz_200.dot");

    // Canonicalize SharedBPlusTree graph and compare to BPlusTree.
    let dot_sh = t_sh.to_canonicalized_dot().expect("to_dot shared");
    assert_eq!(dot_bt, dot_sh, "canonical DOT differs");
}

#[test]
fn graphviz_shared_reopen_is_stable() {
    // Build a shared tree, then reopen a plain BPlusTree from last_root.
    let pager = common::SharedPager::new(256);
    let t_sh: SharedBPlusTree<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
        build_shared(pager.clone());
    let t_reopened: common::TreeU64<common::SharedPager> = common::reopen_latest(&pager);

    let d_shared = t_sh.to_canonicalized_dot().expect("dot shared");
    let d_reopen = t_reopened.to_canonicalized_dot().expect("dot reopen");

    // To (re)generate: set UPDATE_SNAPSHOTS=1 before running tests.
    common::assert_matches_golden(&d_reopen, "tests/snapshots/reopen_200.dot");

    // Ensure reopen does not change structure.
    assert_eq!(d_shared, d_reopen, "DOT changed after reopen");
}
