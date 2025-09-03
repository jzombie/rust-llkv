//! Common test utilities shared across integration tests.

#![allow(dead_code)]

use line_ending::LineEnding;
use llkv_btree::bplus_tree::BPlusTree;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec, KeyCodec};
use llkv_btree::errors::Error;
use llkv_btree::pager::Pager;
use llkv_btree::traits::GraphvizExt;
use llkv_btree::{views::key_view::KeyRef, views::value_view::ValueRef};

use rand::rngs::StdRng;
use rand::{SeedableRng, seq::SliceRandom};
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};

// ---------------- In-memory pager (simple; no root tracking) ----------------
struct TestPagerState {
    pages: FxHashMap<u64, Arc<[u8]>>,
    next_id: u64,
}

pub struct TestPager {
    page_size: usize,
    state: Mutex<TestPagerState>,
}

impl TestPager {
    pub fn new(page_size: usize) -> Self {
        Self {
            page_size,
            state: Mutex::new(TestPagerState {
                pages: FxHashMap::default(),
                next_id: 1,
            }),
        }
    }
}

impl Pager for TestPager {
    type Id = u64;
    type Page = Arc<[u8]>;
    fn read_batch(&self, ids: &[Self::Id]) -> Result<FxHashMap<Self::Id, Self::Page>, Error> {
        let state = self.state.lock().unwrap();

        Ok(ids
            .iter()
            .filter_map(|id| state.pages.get(id).map(|p| (*id, p.clone())))
            .collect())
    }
    fn write_batch(&self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error> {
        let mut state = self.state.lock().unwrap();

        for (id, data) in pages {
            state.pages.insert(*id, Arc::from(*data));
        }
        Ok(())
    }
    fn alloc_ids(&self, count: usize) -> Result<Vec<Self::Id>, Error> {
        let mut state = self.state.lock().unwrap();

        let start = state.next_id;
        state.next_id += count as u64;
        Ok((start..state.next_id).collect())
    }
    fn dealloc_ids(&self, ids: &[Self::Id]) -> Result<(), Error> {
        let mut state = self.state.lock().unwrap();

        for id in ids {
            state.pages.remove(id);
        }
        Ok(())
    }
    fn page_size_hint(&self) -> Option<usize> {
        Some(self.page_size)
    }
    fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error> {
        Ok(Arc::from(bytes))
    }
}

// ---------------- Synchronized pager (tracks last_root for reopen) ----------------
#[derive(Clone)]
pub struct SharedPager {
    inner: Arc<Mutex<Inner>>,
}
struct Inner {
    pages: FxHashMap<u64, Arc<[u8]>>,
    next_id: u64,
    page_size: usize,
    last_root: Option<u64>,
}
impl SharedPager {
    pub fn new(page_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                pages: FxHashMap::default(),
                next_id: 1,
                page_size,
                last_root: None,
            })),
        }
    }
    pub fn last_root(&self) -> Option<u64> {
        self.inner.lock().unwrap().last_root
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
    fn on_root_changed(&self, new_root: Self::Id) -> Result<(), Error> {
        self.inner.lock().unwrap().last_root = Some(new_root);
        Ok(())
    }
}

// ---------------- Generic helpers for u64-key trees ----------------
// Keep the generic alias if you still want it for helpers:
pub type TreeU64<P> = BPlusTree<P, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>;

// Make U64Tree concrete for tests so `U64Tree` (no generics) compiles:
pub type U64Tree = TreeU64<TestPager>;

/// Create a `BPlusTree<TestPager, …>` (used by many tests).
pub fn create_tree() -> Result<TreeU64<TestPager>, Error> {
    let pager = TestPager::new(256);
    BPlusTree::<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>::create_empty(pager, None)
}

/// Populate with shuffled keys and mixed values; returns sorted (k, v) ground truth.
pub fn populate_tree(tree: &mut TreeU64<TestPager>) -> Result<Vec<(u64, Vec<u8>)>, Error> {
    // Large enough to cross many leaves.
    const N: usize = 20_000;

    let mut rng = StdRng::seed_from_u64(42);
    let mut keys: Vec<u64> = (0..N as u64).collect();
    keys.shuffle(&mut rng);

    let shared_value = b"this is a duplicated value".to_vec();
    let mut items = Vec::with_capacity(N);
    let mut expected = Vec::with_capacity(N);

    for (i, &k) in keys.iter().enumerate() {
        let value = if i % 5 == 0 {
            shared_value.clone()
        } else {
            format!("unique_value_{i}").into_bytes()
        };
        items.push((k, value.clone()));
        expected.push((k, value));
    }

    expected.sort_by_key(|e| e.0);

    let owned: Vec<_> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    tree.insert_many(&owned)?;
    Ok(expected)
}

pub fn expect_val_u64<P>(tree: &TreeU64<P>, k: u64, expected: Option<&[u8]>)
where
    P: Pager<Id = u64>,
    P::Page: Clone,
{
    let got = tree.get(&k).unwrap();
    match (got, expected) {
        (None, None) => {}
        (Some(vref), Some(bytes)) => assert_eq!(&*vref, bytes),
        (g, e) => panic!(
            "for key {}, got {:?}, expected {:?}",
            k,
            g.map(|v| v.as_ref().to_vec()),
            e.map(|b| b.to_vec())
        ),
    }
}

pub fn snapshot_as_vec_u64<P>(tree: &TreeU64<P>) -> Vec<(u64, Vec<u8>)>
where
    P: Pager<Id = u64>,
    P::Page: Clone,
{
    tree.iter()
        .unwrap()
        .map(|(k, v)| {
            (
                BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                v.as_ref().to_vec(),
            )
        })
        .collect()
}

/// Compare `.to_dot()` graphs; if they differ, optionally dump with `LLKV_DUMP_DOT=<prefix>`.
pub fn assert_dots_eq_u64<P>(a: &TreeU64<P>, b: &TreeU64<P>, stage: &str)
where
    P: Pager<Id = u64>,
    P::Page: Clone,
{
    let da = a.to_dot().expect("to_dot(A)");
    let db = b.to_dot().expect("to_dot(B)");

    if da != db {
        if let Ok(prefix) = std::env::var("LLKV_DUMP_DOT") {
            let _ = std::fs::write(format!("{}_A_{}.dot", prefix, stage), &da);
            let _ = std::fs::write(format!("{}_B_{}.dot", prefix, stage), &db);
        }
        let al: Vec<_> = da.lines().collect();
        let bl: Vec<_> = db.lines().collect();
        for i in 0..al.len().max(bl.len()) {
            let l = al.get(i).copied().unwrap_or("<EOF>");
            let r = bl.get(i).copied().unwrap_or("<EOF>");
            if l != r {
                panic!(
                    "DOT mismatch @ {stage} (line {}):\nA: {l}\nB: {r}\n--- (set LLKV_DUMP_DOT to dump full graphs)",
                    i + 1
                );
            }
        }
        panic!("DOT mismatch @ {stage}, but could not locate differing line");
    }
}

/// Reopen a tree using the pager's last root (for `SharedPager`).
pub fn reopen_latest(pager: &SharedPager) -> TreeU64<SharedPager> {
    let root = pager.last_root().expect("no last_root recorded");
    BPlusTree::new(pager.clone(), root, None)
}

// ---------------- Iterator + diff helpers ----------------
pub fn collect_and_validate_iter<P: Pager, KC: KeyCodec>(
    iter: impl Iterator<Item = (KeyRef<P::Page>, ValueRef<P::Page>)>,
    expected_keys: &[KC::Key],
    expected_values: &[&[u8]],
) where
    P::Page: Clone,
    KC::Key: PartialEq + std::fmt::Debug,
{
    let results: Vec<(KC::Key, Vec<u8>)> = iter
        .map(|(k_ref, v_ref)| {
            (
                KC::decode_from(k_ref.as_ref()).unwrap(),
                v_ref.as_ref().to_vec(),
            )
        })
        .collect();

    let expected_values_vec: Vec<Vec<u8>> = expected_values.iter().map(|v| v.to_vec()).collect();
    assert_eq!(
        results.len(),
        expected_keys.len(),
        "Iterator yielded a different number of items."
    );
    for i in 0..results.len() {
        assert_eq!(
            results[i].0, expected_keys[i],
            "Keys do not match at index {}",
            i
        );
        assert_eq!(
            results[i].1, expected_values_vec[i],
            "Values do not match at index {}",
            i
        );
    }
}

pub fn diff_dump(who: &str, got: &[(u64, Vec<u8>)], want: &[(u64, Vec<u8>)]) {
    use std::collections::BTreeMap;
    let gm: BTreeMap<u64, &Vec<u8>> = got.iter().map(|(k, v)| (*k, v)).collect();
    let wm: BTreeMap<u64, &Vec<u8>> = want.iter().map(|(k, v)| (*k, v)).collect();

    let missing: Vec<u64> = wm
        .keys()
        .filter(|k| !gm.contains_key(k))
        .take(10)
        .cloned()
        .collect();
    let extra: Vec<u64> = gm
        .keys()
        .filter(|k| !wm.contains_key(k))
        .take(10)
        .cloned()
        .collect();

    let mut first_val_mismatch: Option<(u64, Vec<u8>, Vec<u8>)> = None;
    for k in wm.keys() {
        if let (Some(gv), Some(wv)) = (gm.get(k), wm.get(k))
            && *gv != *wv
        {
            first_val_mismatch = Some((*k, (*gv).clone(), (*wv).clone()));
            break;
        }
    }

    println!("DIFF [{}]:", who);
    if !missing.is_empty() {
        println!("  missing keys (sample): {:?}", missing);
    }
    if !extra.is_empty() {
        println!("  extra keys (sample):   {:?}", extra);
    }
    if let Some((k, g, w)) = first_val_mismatch {
        println!(
            "  first value mismatch at key {}: got {:?} want {:?}",
            k, g, w
        );
    }
}

/// How to (re)generate golden snapshots:
///   1) Set the env var to update snapshots:
///      UPDATE_SNAPSHOTS=1
///   2) Run just these tests (example):
///      cargo test -p llkv-btree -- graphviz_
///   3) Commit the files written under tests/snapshots/.
///
/// To debug mismatches, you can also set LLKV_DUMP_DOT to write .dot
/// files when asserts fail (see common::assert_dots_eq_u64).
pub fn assert_matches_golden(actual: &str, golden_path: &str) {
    use std::fs;
    use std::path::Path;

    let p = Path::new(golden_path);
    if std::env::var("UPDATE_SNAPSHOTS").as_deref() == Ok("1") {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, actual.as_bytes()).unwrap();
        eprintln!("updated snapshot: {golden_path}");
    } else {
        let expected = LineEnding::normalize(
            &fs::read_to_string(p).expect("missing golden; set UPDATE_SNAPSHOTS=1"),
        );
        assert_eq!(actual, expected, "snapshot mismatch: {golden_path}",);
    }
}
