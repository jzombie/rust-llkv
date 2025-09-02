use criterion::{BatchSize, Bencher, Criterion, criterion_group, criterion_main};
use llkv_btree::{
    bplus_tree::BPlusTree,
    codecs::{BigEndianIdCodec, BigEndianKeyCodec},
    pager::Pager,
};
use rustc_hash::FxHashMap;
use std::hint::black_box;
use std::sync::Arc;

// --- Type Aliases for the Benchmark ---
type U64KeyCodec = BigEndianKeyCodec<u64>;
type U64IdCodec = BigEndianIdCodec<u64>;

// --- In-Memory Pager for Benchmarking ---
#[derive(Clone)]
struct BenchPager {
    pages: FxHashMap<u64, Arc<[u8]>>,
    next_id: u64,
    page_size: usize,
}

impl Pager for BenchPager {
    type Id = u64;
    type Page = Arc<[u8]>;

    fn read_batch(
        &self,
        ids: &[Self::Id],
    ) -> Result<FxHashMap<Self::Id, Self::Page>, llkv_btree::errors::Error> {
        Ok(ids
            .iter()
            .filter_map(|id| self.pages.get(id).map(|p| (*id, p.clone())))
            .collect())
    }
    fn write_batch(
        &mut self,
        pages: &[(Self::Id, &[u8])],
    ) -> Result<(), llkv_btree::errors::Error> {
        for (id, data) in pages {
            self.pages.insert(*id, Arc::from(*data));
        }
        Ok(())
    }
    fn alloc_ids(&mut self, count: usize) -> Result<Vec<Self::Id>, llkv_btree::errors::Error> {
        let start = self.next_id;
        self.next_id += count as u64;
        Ok((start..self.next_id).collect())
    }
    fn dealloc_ids(&mut self, ids: &[Self::Id]) -> Result<(), llkv_btree::errors::Error> {
        for id in ids {
            self.pages.remove(id);
        }
        Ok(())
    }
    fn page_size_hint(&self) -> Option<usize> {
        Some(self.page_size)
    }
    fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, llkv_btree::errors::Error> {
        Ok(Arc::from(bytes))
    }
}

/// Generates a dataset of `count` items where every value is unique.
fn generate_unique_data(count: usize) -> (Vec<(u64, Vec<u8>)>, Vec<u64>) {
    let keys: Vec<u64> = (0..count as u64).collect();

    // Each key gets a completely unique value.
    let data: Vec<(u64, Vec<u8>)> = keys
        .iter()
        .map(|&k| (k, format!("this-is-a-unique-long-value-{}", k).into_bytes()))
        .collect();

    (data, keys)
}

fn benchmark_upserts(c: &mut Criterion) {
    const ITEM_COUNT: usize = 1_000_000;

    let (data, _) = generate_unique_data(ITEM_COUNT);
    let owned_data: Vec<_> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();

    let mut group = c.benchmark_group("Upsert 1M items (100% unique)");
    group.sample_size(10);

    group.bench_function("BPlusTree", |b: &mut Bencher| {
        b.iter_batched(
            || {
                let pager = BenchPager {
                    pages: FxHashMap::default(),
                    next_id: 1,
                    page_size: 4096,
                };
                BPlusTree::<_, U64KeyCodec, U64IdCodec>::create_empty(pager, None).unwrap()
            },
            |mut tree| {
                tree.insert_many(black_box(&owned_data)).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn benchmark_reads(c: &mut Criterion) {
    const ITEM_COUNT: usize = 1_000_000;

    let mut group = c.benchmark_group("Read 1M items (100% unique)");
    group.sample_size(10);

    group.bench_function("BPlusTree", |b: &mut Bencher| {
        b.iter_batched(
            || {
                let (data, keys) = generate_unique_data(ITEM_COUNT);
                let owned_data: Vec<_> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
                let pager = BenchPager {
                    pages: FxHashMap::default(),
                    next_id: 1,
                    page_size: 4096,
                };
                let mut tree =
                    BPlusTree::<_, U64KeyCodec, U64IdCodec>::create_empty(pager, None).unwrap();
                tree.insert_many(&owned_data).unwrap();
                (tree, keys)
            },
            |(tree, keys)| {
                for key in &keys {
                    black_box(tree.get(key).unwrap());
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, benchmark_upserts, benchmark_reads);
criterion_main!(benches);
