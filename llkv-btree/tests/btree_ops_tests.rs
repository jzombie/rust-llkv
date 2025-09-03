mod common;

use common::{create_tree, diff_dump, populate_tree};
use llkv_btree::bplus_tree::BPlusTree;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec, KeyCodec, StringKeyCodec};
use llkv_btree::errors::Error;
use llkv_btree::iter::{BPlusTreeIter, Direction, ScanOpts};
use llkv_btree::pager::Pager;
use rustc_hash::FxHashMap;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::Arc;
use std::sync::Mutex;

// Upserts & deletes over u64 keys
#[test]
fn ops_upserts_and_deletes_u64() -> Result<(), Box<dyn std::error::Error>> {
    let mut t = create_tree()?;
    let expected = populate_tree(&mut t)?;
    let mut map: FxHashMap<u64, Vec<u8>> = expected.into_iter().collect();

    // Delete every 3rd
    let keys_to_delete: Vec<u64> = (0u64..2000).step_by(3).collect();
    t.delete_many(&keys_to_delete)?;
    for k in keys_to_delete {
        map.remove(&k);
    }

    // Check after deletes
    {
        let got: Vec<(u64, Vec<u8>)> = t
            .iter()?
            .map(|(k, v)| {
                (
                    BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                    v.as_ref().to_vec(),
                )
            })
            .collect();
        let mut want: Vec<(u64, Vec<u8>)> = map.iter().map(|(k, v)| (*k, v.clone())).collect();
        want.sort_by_key(|e| e.0);
        if got != want {
            diff_dump("after deletes (BPlusTree)", &got, &want);
        }
        assert_eq!(got, want);
    }

    // Upsert every even (reinserts multiples of 6)
    let mut upserts = Vec::new();
    for k in (0u64..2000).filter(|k| k % 2 == 0) {
        upserts.push((k, format!("up_{}", k).into_bytes()));
    }
    let owned: Vec<_> = upserts.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    t.insert_many(&owned)?;
    for (k, v) in upserts {
        map.insert(k, v);
    }

    // Check after upserts
    {
        let got: Vec<(u64, Vec<u8>)> = t
            .iter()?
            .map(|(k, v)| {
                (
                    BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                    v.as_ref().to_vec(),
                )
            })
            .collect();
        let mut want: Vec<(u64, Vec<u8>)> = map.iter().map(|(k, v)| (*k, v.clone())).collect();
        want.sort_by_key(|e| e.0);
        if got != want {
            diff_dump("after upserts (BPlusTree)", &got, &want);
        }
        assert_eq!(got, want);
    }

    // Insert [2000, 2200)
    let mut new_items = Vec::new();
    for k in 2000u64..2200 {
        new_items.push((k, format!("new_{}", k).into_bytes()));
    }
    let owned_new: Vec<_> = new_items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    t.insert_many(&owned_new)?;
    for (k, v) in new_items {
        map.insert(k, v);
    }

    // Final check
    {
        let got: Vec<(u64, Vec<u8>)> = t
            .iter()?
            .map(|(k, v)| {
                (
                    BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                    v.as_ref().to_vec(),
                )
            })
            .collect();
        let mut want: Vec<(u64, Vec<u8>)> = map.iter().map(|(k, v)| (*k, v.clone())).collect();
        want.sort_by_key(|e| e.0);
        if got != want {
            diff_dump("final (BPlusTree)", &got, &want);
        }
        assert_eq!(got, want);
    }
    Ok(())
}

// Range boundaries sanity (ops mutate then verify scan)
#[test]
fn ops_range_boundaries_u64() -> Result<(), Box<dyn std::error::Error>> {
    let mut t = create_tree()?;
    let expected = populate_tree(&mut t)?;
    let mut mv: FxHashMap<u64, Vec<u8>> = expected.into_iter().collect();

    // Mutate a few edges
    let muts = [
        (100u64, "edge_100"),
        (1000, "edge_1000"),
        (1500, "edge_1500"),
        (1999, "edge_1999"),
    ];
    let owned: Vec<_> = muts.iter().map(|(k, v)| (*k, v.as_bytes())).collect();
    t.insert_many(&owned)?;
    for (k, v) in muts {
        mv.insert(k, v.as_bytes().to_vec());
    }

    let check =
        |lo: Option<u64>, up: Option<u64>, incl: bool| -> Result<(), Box<dyn std::error::Error>> {
            // Translate Option bounds + `incl` into Bound<&u64>
            let lower_b = match lo.as_ref() {
                Some(l) => Included(l),
                None => Unbounded,
            };
            let upper_b = match (up.as_ref(), incl) {
                (Some(u), true) => Included(u),
                (Some(u), false) => Excluded(u),
                (None, _) => Unbounded,
            };

            let it = BPlusTreeIter::with_opts(
                &t,
                ScanOpts {
                    dir: Direction::Forward,
                    lower: lower_b,
                    upper: upper_b,
                    prefix: None,
                    frame_predicate: None,
                },
            )?;

            let mut want: Vec<(u64, Vec<u8>)> = mv
                .iter()
                .filter(|(k, _)| {
                    let ok_lo = lo.is_none_or(|l| **k >= l);
                    let ok_up = match (up, incl) {
                        (Some(u), true) => **k <= u,
                        (Some(u), false) => **k < u,
                        _ => true,
                    };
                    ok_lo && ok_up
                })
                .map(|(k, v)| (*k, v.clone()))
                .collect();
            want.sort_by_key(|e| e.0);

            let got: Vec<(u64, Vec<u8>)> = it
                .map(|(k, v)| {
                    (
                        BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                        v.as_ref().to_vec(),
                    )
                })
                .collect();
            assert_eq!(got, want, "lo={:?} up={:?} incl={}", lo, up, incl);
            Ok(())
        };

    check(Some(100), Some(1500), true)?;
    check(Some(100), Some(1500), false)?;
    check(None, Some(500), false)?;
    check(Some(1800), None, true)?;
    check(Some(1900), Some(1800), true)?; // empty
    Ok(())
}

// String-key prefix ops (mutate then verify prefix scan)
#[test]
fn ops_prefix_mutations_strings() -> Result<(), Box<dyn std::error::Error>> {
    let mut tb = BPlusTree::<_, StringKeyCodec, BigEndianIdCodec<u64>>::create_empty(
        // common::TestPager {
        //     pages: FxHashMap::default(),
        //     next_id: 1,
        //     page_size: 512,
        // },
        common::TestPager::new(512),
        None,
    )?;

    let mut kv = Vec::<(String, Vec<u8>)>::new();
    for (k, v) in [
        ("app".to_string(), b"v1".to_vec()),
        ("apple".to_string(), b"v2".to_vec()),
        ("apricot".to_string(), b"v3".to_vec()),
        ("banana".to_string(), b"v4".to_vec()),
        ("avocado".to_string(), b"v5".to_vec()),
        ("apply".to_string(), b"v6".to_vec()),
    ] {
        kv.push((k, v));
    }
    for i in 0..700u32 {
        kv.push((format!("ap{:04}", i), format!("vap{}", i).into_bytes()));
        kv.push((format!("aq{:04}", i), format!("vaq{}", i).into_bytes()));
        kv.push((format!("b{:04}", i), format!("vb{}", i).into_bytes()));
    }
    let own: Vec<_> = kv.iter().map(|(k, v)| (k.clone(), v.as_slice())).collect();
    tb.insert_many(&own)?;

    tb.delete_many(&["banana".to_string(), "apricot".to_string()])?;
    let ups = [
        ("app".to_string(), b"x1".to_vec()),
        ("apple".to_string(), b"x2".to_vec()),
        ("apex".to_string(), b"x3".to_vec()),
    ];
    let ups_owned: Vec<_> = ups.iter().map(|(k, v)| (k.clone(), v.as_slice())).collect();
    tb.insert_many(&ups_owned)?;

    let it = BPlusTreeIter::with_opts(
        &tb,
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

    let mut map = kv.into_iter().collect::<FxHashMap<_, _>>();
    map.remove("banana");
    map.remove("apricot");
    map.insert("app".into(), b"x1".to_vec());
    map.insert("apple".into(), b"x2".to_vec());
    map.insert("apex".into(), b"x3".to_vec());

    let mut want: Vec<(String, Vec<u8>)> = map
        .into_iter()
        .filter(|(k, _)| k.as_bytes().starts_with(b"ap"))
        .collect();
    want.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(got, want);
    Ok(())
}

// Randomized ops checked against ground truth map
#[test]
fn ops_randomized_against_truth() -> Result<(), Box<dyn std::error::Error>> {
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
    let mut t = create_tree()?;
    let mut rng = StdRng::seed_from_u64(12345);

    let mut all: FxHashMap<u64, Vec<u8>> = FxHashMap::default();
    let mut initial = Vec::new();
    for k in 0u64..1000 {
        let v = format!("v{}", k).into_bytes();
        initial.push((k, v.clone()));
        all.insert(k, v);
    }
    let own0: Vec<_> = initial.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    t.insert_many(&own0)?;

    let mut keys: Vec<u64> = (0..1500).collect();
    keys.shuffle(&mut rng);
    for (i, k) in keys.into_iter().enumerate() {
        match (i + k as usize) % 3 {
            0 => {
                let _ = t.delete_many(&[k]);
                all.remove(&k);
            }
            1 => {
                let v = format!("u{}_{}", i, k).into_bytes();
                t.insert_many(&[(k, v.as_slice())])?;
                all.insert(k, v);
            }
            _ => {
                let v = format!("n{}_{}", i, k).into_bytes();
                t.insert_many(&[(k, v.as_slice())])?;
                all.insert(k, v);
            }
        }

        if i % 150 == 0 {
            let it = t.iter()?;
            let mut expect: Vec<(u64, Vec<u8>)> =
                all.iter().map(|(k, v)| (*k, v.clone())).collect();
            expect.sort_by_key(|e| e.0);
            let got: Vec<(u64, Vec<u8>)> = it
                .map(|(k, v)| {
                    (
                        BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                        v.as_ref().to_vec(),
                    )
                })
                .collect();
            assert_eq!(got, expect, "mismatch at step {}", i);
        }
    }
    Ok(())
}

#[test]
fn test_phyiscal_keys_are_less_than_logical_keys_single_insert()
-> Result<(), Box<dyn std::error::Error>> {
    // TODO: Refactor shared pager into common utils
    // This is used to get access to the internals for testing purposes
    #[derive(Clone)]
    struct SharedPager {
        inner: Arc<Mutex<Inner>>,
    }
    struct Inner {
        pages: FxHashMap<u64, Arc<[u8]>>,
        next_id: u64,
        page_size: usize,
    }
    impl SharedPager {
        fn new(page_size: usize) -> Self {
            Self {
                inner: Arc::new(Mutex::new(Inner {
                    pages: FxHashMap::default(),
                    next_id: 1,
                    page_size,
                })),
            }
        }
    }
    impl Pager for SharedPager {
        type Id = u64;
        type Page = Arc<[u8]>;
        fn read_batch(&self, ids: &[Self::Id]) -> Result<FxHashMap<Self::Id, Self::Page>, Error> {
            let g = self.inner.lock().unwrap();
            Ok(ids
                .iter()
                .filter_map(|id| g.pages.get(id).map(|p| (*id, p.clone())))
                .collect())
        }
        fn write_batch(&self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error> {
            let mut g = self.inner.lock().unwrap();
            for (id, data) in pages {
                if data.len() > g.page_size {
                    return Err(Error::Corrupt("page overflow"));
                }
                g.pages.insert(*id, Arc::from(*data));
            }
            Ok(())
        }
        fn alloc_ids(&self, count: usize) -> Result<Vec<Self::Id>, Error> {
            let mut g = self.inner.lock().unwrap();
            let start = g.next_id;
            g.next_id += count as u64;
            Ok((start..g.next_id).collect())
        }
        fn dealloc_ids(&self, ids: &[Self::Id]) -> Result<(), Error> {
            let mut g = self.inner.lock().unwrap();
            for id in ids {
                g.pages.remove(id);
            }
            Ok(())
        }
        fn page_size_hint(&self) -> Option<usize> {
            Some(self.inner.lock().unwrap().page_size)
        }
        fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error> {
            Ok(Arc::from(bytes))
        }
    }

    // Use a pager we can inspect.
    let pager = SharedPager::new(256);
    let mut tree: BPlusTree<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
        BPlusTree::create_empty(pager.clone(), None)?;

    let n: u64 = 15_000;

    // Insert keys 1..=5000 exactly once.
    let items: Vec<_> = (1..=n).map(|k| (k, k.to_be_bytes())).collect();
    let owned: Vec<_> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    tree.insert_many(&owned)?; // flushes pages

    // Count physical *internal* keys by scanning pager pages:
    // header layout: [0]=tag (0=Internal, 1=Leaf), [1..5)=count (u32 LE), [5..9)=aux_len
    let pages_snapshot = { pager.inner.lock().unwrap().pages.clone() };
    let mut internal_key_count = 0usize;
    for page in pages_snapshot.values() {
        if page.len() >= 9 {
            let tag = page[0];
            if tag == 0 {
                let cnt = u32::from_le_bytes(page[1..5].try_into().unwrap()) as usize;
                internal_key_count += cnt;
            }
        }
    }

    assert_eq!(internal_key_count, 4996);
    Ok(())
}
