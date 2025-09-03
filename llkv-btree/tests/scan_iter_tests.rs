mod common;

use common::{TestPager, U64Tree, collect_and_validate_iter, create_tree, populate_tree};
use llkv_btree::bplus_tree::BPlusTree;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec, KeyCodec, StringKeyCodec};
use llkv_btree::iter::{BPlusTreeIter, Direction, ScanOpts};
use std::ops::Bound::{Excluded, Included, Unbounded};

// Forward full scan
#[test]
fn iter_forward_scan() -> Result<(), Box<dyn std::error::Error>> {
    let mut tree = create_tree()?;
    let expected = populate_tree(&mut tree)?;
    let keys: Vec<u64> = expected.iter().map(|(k, _)| *k).collect();
    let vals: Vec<&[u8]> = expected.iter().map(|(_, v)| v.as_slice()).collect();

    let it = tree.iter()?; // forward default
    collect_and_validate_iter::<TestPager, BigEndianKeyCodec<u64>>(it, &keys, &vals);
    Ok(())
}

// Reverse full scan
#[test]
fn iter_reverse_scan() -> Result<(), Box<dyn std::error::Error>> {
    let mut tree = create_tree()?;
    let mut expected = populate_tree(&mut tree)?;
    expected.reverse();
    let keys: Vec<u64> = expected.iter().map(|(k, _)| *k).collect();
    let vals: Vec<&[u8]> = expected.iter().map(|(_, v)| v.as_slice()).collect();

    let it = BPlusTreeIter::with_opts(
        &tree,
        ScanOpts {
            dir: Direction::Reverse,
            lower: Unbounded,
            upper: Unbounded,
            prefix: None,
            frame_predicate: None,
        },
    )?;
    collect_and_validate_iter::<TestPager, BigEndianKeyCodec<u64>>(it, &keys, &vals);
    Ok(())
}

// Range (inclusive/exclusive) forward
#[test]
fn iter_range_scan_u64() -> Result<(), Box<dyn std::error::Error>> {
    let mut tree: U64Tree =
        BPlusTree::<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>::create_empty(
            // TestPager {
            //     pages: FxHashMap::default(),
            //     next_id: 1,
            //     page_size: 256,
            // },
            TestPager::new(256),
            None,
        )?;
    let expected = common::populate_tree(&mut tree)?;

    let it = BPlusTreeIter::with_opts(
        &tree,
        ScanOpts {
            dir: Direction::Forward,
            lower: Included(&500),
            upper: Included(&1500),
            prefix: None,
            frame_predicate: None,
        },
    )?;
    let got: Vec<(u64, Vec<u8>)> = it
        .map(|(k, v)| {
            (
                BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                v.as_ref().to_vec(),
            )
        })
        .collect();

    let mut want: Vec<(u64, Vec<u8>)> = expected
        .into_iter()
        .filter(|(k, _)| *k >= 500 && *k <= 1500)
        .collect();
    want.sort_by_key(|e| e.0);
    assert_eq!(got, want);
    Ok(())
}

// Reverse + bounded range (extra safety)
#[test]
fn iter_range_reverse_scan_u64() -> Result<(), Box<dyn std::error::Error>> {
    let mut tree = create_tree()?;
    let expected = populate_tree(&mut tree)?;
    let mut want: Vec<(u64, Vec<u8>)> = expected
        .into_iter()
        .filter(|(k, _)| *k >= 500 && *k < 1500)
        .collect();
    want.sort_by_key(|e| e.0);
    want.reverse();

    let it = BPlusTreeIter::with_opts(
        &tree,
        ScanOpts {
            dir: Direction::Reverse,
            lower: Included(&500),
            upper: Excluded(&1500),
            prefix: None,
            frame_predicate: None,
        },
    )?;
    let got: Vec<(u64, Vec<u8>)> = it
        .map(|(k, v)| {
            (
                BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                v.as_ref().to_vec(),
            )
        })
        .collect();

    assert_eq!(got, want);
    Ok(())
}

// Prefix scan on String keys (forward)
#[test]
fn iter_prefix_scan_strings() -> Result<(), Box<dyn std::error::Error>> {
    let tree = BPlusTree::<TestPager, StringKeyCodec, BigEndianIdCodec<u64>>::create_empty(
        // TestPager {
        //     pages: FxHashMap::default(),
        //     next_id: 1,
        //     page_size: 256,
        // },
        TestPager::new(256),
        None,
    )?;

    let mut items = Vec::<(String, Vec<u8>)>::new();
    for (k, v) in [
        ("apple".to_string(), "red".as_bytes().to_vec()),
        ("apricot".to_string(), "orange".as_bytes().to_vec()),
        ("banana".to_string(), "yellow".as_bytes().to_vec()),
        ("avocado".to_string(), "green".as_bytes().to_vec()),
        ("apply".to_string(), "v6".as_bytes().to_vec()),
        ("app".to_string(), "v1".as_bytes().to_vec()),
    ] {
        items.push((k, v));
    }
    for i in 0..600u32 {
        items.push((format!("ap{:04}", i), format!("vap{}", i).into_bytes()));
        items.push((format!("aq{:04}", i), format!("vaq{}", i).into_bytes()));
        items.push((format!("b{:04}", i), format!("vb{}", i).into_bytes()));
    }
    let owned: Vec<_> = items
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();
    tree.insert_many(&owned)?;

    let it = BPlusTreeIter::with_opts(
        &tree,
        ScanOpts {
            dir: Direction::Forward,
            lower: Unbounded,
            upper: Unbounded,
            prefix: Some(b"ap"),
            frame_predicate: None,
        },
    )?;
    let got: Vec<(String, Vec<u8>)> = it
        .map(|(k, v)| {
            (
                StringKeyCodec::decode_from(k.as_ref()).unwrap(),
                v.as_ref().to_vec(),
            )
        })
        .collect();

    let mut want: Vec<(String, Vec<u8>)> = items
        .into_iter()
        .filter(|(k, _)| k.as_bytes().starts_with(b"ap"))
        .collect();
    want.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(got, want);
    Ok(())
}

// Reverse + prefix (extra safety)
#[test]
fn iter_prefix_reverse_scan_strings() -> Result<(), Box<dyn std::error::Error>> {
    let tree = BPlusTree::<TestPager, StringKeyCodec, BigEndianIdCodec<u64>>::create_empty(
        // TestPager {
        //     pages: FxHashMap::default(),
        //     next_id: 1,
        //     page_size: 256,
        // },
        TestPager::new(256),
        None,
    )?;

    let mut items = Vec::<(String, Vec<u8>)>::new();
    for i in 0..200u32 {
        items.push((format!("ap{:04}", i), format!("vap{}", i).into_bytes()));
        items.push((format!("aq{:04}", i), format!("vaq{}", i).into_bytes()));
    }
    let owned: Vec<_> = items
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();
    tree.insert_many(&owned)?;

    let it = BPlusTreeIter::with_opts(
        &tree,
        ScanOpts {
            lower: Unbounded,
            upper: Unbounded,
            dir: Direction::Reverse,
            prefix: Some(b"ap"),
            frame_predicate: None,
        },
    )?;
    let got: Vec<(String, Vec<u8>)> = it
        .map(|(k, v)| {
            (
                StringKeyCodec::decode_from(k.as_ref()).unwrap(),
                v.as_ref().to_vec(),
            )
        })
        .collect();

    let mut want: Vec<(String, Vec<u8>)> = items
        .into_iter()
        .filter(|(k, _)| k.as_bytes().starts_with(b"ap"))
        .collect();
    want.sort_by(|a, b| b.0.cmp(&a.0)); // descending

    assert_eq!(got, want);
    Ok(())
}
