//! Simple metrics dump for BPlusTree.
//!
//! Run:
//!   cargo run --example debug_metrics --release -- 200000
//! If no arg is given, N defaults to 200_000.

use llkv_btree::bplus_tree::bplus_tree::BPlusTree;
use llkv_btree::codecs::{BigEndianIdCodec, BigEndianKeyCodec};
use llkv_btree::errors::Error;
use llkv_btree::pager::Pager;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rustc_hash::FxHashMap;
use std::env;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

// ---------------- In-memory pager (example-only) ----------------

struct MemPagerState<I> {
    pages: FxHashMap<I, Arc<[u8]>>,
    next_id: I,
}

struct MemPagerU64 {
    page_size: usize,
    state: Mutex<MemPagerState<u64>>,
}

impl MemPagerU64 {
    fn new(page_size: usize) -> Self {
        Self {
            page_size,
            state: Mutex::new(MemPagerState {
                pages: FxHashMap::default(),
                next_id: 1,
            }),
        }
    }
}

impl Pager for MemPagerU64 {
    type Id = u64;
    type Page = Arc<[u8]>;

    fn read_batch(&self, ids: &[Self::Id]) -> Result<FxHashMap<Self::Id, Self::Page>, Error> {
        let s = self.state.lock().unwrap();
        Ok(ids
            .iter()
            .filter_map(|id| s.pages.get(id).map(|p| (*id, p.clone())))
            .collect())
    }

    fn write_batch(&self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error> {
        let mut s = self.state.lock().unwrap();
        for (id, data) in pages {
            if data.len() > self.page_size {
                return Err(Error::Corrupt("page overflow"));
            }
            s.pages.insert(*id, Arc::from(*data));
        }
        Ok(())
    }

    fn alloc_ids(&self, count: usize) -> Result<Vec<Self::Id>, Error> {
        let mut s = self.state.lock().unwrap();
        let start = s.next_id;
        s.next_id = s
            .next_id
            .checked_add(count as u64)
            .ok_or(Error::Corrupt("id space exhausted"))?;
        Ok((start..s.next_id).collect())
    }

    fn dealloc_ids(&self, ids: &[Self::Id]) -> Result<(), Error> {
        let mut s = self.state.lock().unwrap();
        for id in ids {
            s.pages.remove(id);
        }
        Ok(())
    }

    fn page_size_hint(&self) -> Option<usize> {
        Some(self.page_size)
    }

    fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error> {
        Ok(Arc::from(bytes))
    }

    fn on_root_changed(&self, _new_root: Self::Id) -> Result<(), Error> {
        Ok(())
    }
}

// ---------------- Example driver ----------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);

    // 8192 is your stated page size.
    let pager = MemPagerU64::new(8192);

    type KC = BigEndianKeyCodec<u64>;
    type IC = BigEndianIdCodec<u64>;

    let tree: BPlusTree<MemPagerU64, KC, IC> = BPlusTree::create_empty(pager, None)?;

    // Mixed workload: ~50% duplicates, ~50% unique.
    // (Keys are sorted inside insert_many.)
    let mut rng = StdRng::seed_from_u64(42);
    let domain = (n / 2).max(1) as u64;

    let mut items: Vec<(u64, Vec<u8>)> = Vec::with_capacity(n);
    for _ in 0..n {
        let k = rng.random::<u64>() % domain; // <- no `r#gen`
        let v = format!("val_{k}").into_bytes();
        items.push((k, v));
    }

    // Borrow-as-slices for insert_many API.
    let items_ref: Vec<(u64, &[u8])> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();

    let t0 = Instant::now();
    tree.insert_many(&items_ref)?;
    tree.flush()?; // ensure pending writes persist
    let upsert_ms = t0.elapsed().as_millis();

    // Probe a few reads just to exercise the read path.
    let probe_keys: Vec<u64> = (0..10).map(|i| i as u64 % domain).collect();
    let _ = tree.get_many(&probe_keys)?;

    let (reads_from_pager, cache_hits, write_page_calls) = tree.debug_metrics();

    println!("BPlusTree metrics:");
    println!("  inserts:             {}", n);
    println!("  upsert_time_ms:      {}", upsert_ms);
    println!("  reads_from_pager:    {}", reads_from_pager);
    println!("  cache_hits:          {}", cache_hits);
    println!("  write_page_calls:    {}", write_page_calls);

    Ok(())
}
