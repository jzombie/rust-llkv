use criterion::{BatchSize, Bencher, Criterion, criterion_group, criterion_main};
use llkv_btree::{
    bplus_tree::BPlusTree,
    codecs::{BigEndianIdCodec, BigEndianKeyCodec},
    define_mem_pager,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;

// --- Type Aliases for the Benchmark ---
type U64KeyCodec = BigEndianKeyCodec<u64>;
type U64IdCodec = BigEndianIdCodec<u64>;

define_mem_pager! {
    /// In-memory pager with u64 page IDs.
    name: MemPager64,
    id: u64,
    default_page_size: 256
}

/// Generates a dataset of `count` items with `unique_values` distinct values.
fn generate_data(count: usize, unique_values: usize) -> (Vec<(u64, Vec<u8>)>, Vec<u64>) {
    let mut rng = StdRng::seed_from_u64(42);
    let keys: Vec<u64> = (0..count as u64).collect();

    let values: Vec<Vec<u8>> = (0..unique_values)
        .map(|i| format!("this-is-a-moderately-long-value-{}", i).into_bytes())
        .collect();

    let data: Vec<(u64, Vec<u8>)> = keys
        .iter()
        .map(|&k| (k, values[rng.random_range(0..unique_values)].clone()))
        .collect();

    (data, keys)
}

fn benchmark_upserts(c: &mut Criterion) {
    const ITEM_COUNT: usize = 1_000_000;
    const UNIQUE_VALUES: usize = ITEM_COUNT / 2;

    let (data, _) = generate_data(ITEM_COUNT, UNIQUE_VALUES);
    let owned_data: Vec<_> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();

    let mut group = c.benchmark_group("Upsert 1M items (50% duplicates)");
    group.sample_size(10);

    group.bench_function("BPlusTree", |b: &mut Bencher| {
        b.iter_batched(
            || {
                let pager = MemPager64::new(4096);
                BPlusTree::<_, U64KeyCodec, U64IdCodec>::create_empty(pager, None).unwrap()
            },
            |tree| {
                tree.insert_many(black_box(&owned_data)).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn benchmark_reads(c: &mut Criterion) {
    const ITEM_COUNT: usize = 1_000_000;
    const UNIQUE_VALUES: usize = ITEM_COUNT / 2;

    let mut group = c.benchmark_group("Read 1M items (50% duplicates)");
    group.sample_size(10);

    group.bench_function("BPlusTree", |b: &mut Bencher| {
        b.iter_batched(
            || {
                let (data, keys) = generate_data(ITEM_COUNT, UNIQUE_VALUES);
                let owned_data: Vec<_> = data.iter().map(|(k, v)| (*k, v.as_slice())).collect();
                let pager = MemPager64::new(4096);
                let tree =
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
