use llkv_btree::bplus_tree::{BPlusTree, SharedBPlusTree};
use llkv_btree::codecs::KeyCodec;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec};
use llkv_btree::pager::{MemPager64, SharedPager};
use llkv_btree::traits::{BTree, GraphvizExt};

fn seed_items(n: u64) -> Vec<(u64, Vec<u8>)> {
    // Deterministic, readable payloads.
    // `insert_many` sorts keys internally, so order here is irrelevant.
    (0..n)
        .map(|k| (k, format!("v{}", k).into_bytes()))
        .collect()
}

#[test]
fn graphviz_is_identical_between_btree_and_shared() {
    // Use the same pager type and page size so node fanout behavior matches.
    let base1 = MemPager64::new(256);
    let base2 = MemPager64::new(256);
    let p1 = SharedPager::new(base1);
    let p2 = SharedPager::new(base2);

    // Build separate trees (no shared storage required for this test).
    let t1: BPlusTree<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
        BPlusTree::create_empty(p1, None).expect("create t1");
    let t2s: SharedBPlusTree<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
        SharedBPlusTree::create_empty(p2, None).expect("create t2");

    // Seed both with the same dataset to force multiple nodes and splits.
    let items = seed_items(200);
    let owned1: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let owned2 = owned1.clone();

    t1.insert_many(&owned1).expect("seed t1");
    t2s.insert_many(&owned2).expect("seed t2");

    // Render DOT for both trees.
    let c1 = t1.to_canonicalized_dot().expect("dot t1");
    let c2 = t2s.to_canonicalized_dot().expect("dot t2");

    // Canonicalize and compare.
    assert_eq!(c1, c2, "canonical DOT outputs differ");

    // Also assert the logical contents are identical.
    let rows1: Vec<(u64, Vec<u8>)> = t1
        .iter()
        .expect("iter t1")
        .map(|(kref, vref)| {
            (
                BigEndianKeyCodec::<u64>::decode_from(kref.as_ref()).unwrap(),
                vref.as_ref().to_vec(),
            )
        })
        .collect();

    // For SharedBPlusTree, iterate on a snapshot (read-only stable view).
    let snap2 = t2s.snapshot();
    let rows2: Vec<(u64, Vec<u8>)> = snap2
        .iter()
        .expect("iter t2")
        .map(|(kref, vref)| {
            (
                BigEndianKeyCodec::<u64>::decode_from(kref.as_ref()).unwrap(),
                vref.as_ref().to_vec(),
            )
        })
        .collect();

    assert_eq!(rows1, rows2, "row sequences differ");
}
