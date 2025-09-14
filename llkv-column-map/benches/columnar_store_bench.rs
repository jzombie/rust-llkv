use std::sync::Arc;
use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::LogicalFieldId;
use llkv_column_map::column_store::ColumnStore;
use llkv_column_map::column_store::columnar::sum_u64_le_simd;

fn bench_column_store_u64_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_store_u64_sum_1M");
    group.sample_size(30);
    group.bench_function("append_and_sum_u64_vectors", |b| {
        b.iter_batched(
            || {
                let pager = Arc::new(MemPager::default());
                let store: ColumnStore<MemPager> = ColumnStore::open(pager.clone());
                let fid: LogicalFieldId = 7777;
                let mut vals = vec![0u64; 1_000_000];
                for i in 0..vals.len() { vals[i] = (i as u64) & 0xFFFF_FFFF; }
                store.append_u64_chunk(fid, &vals);
                (store, fid)
            },
            |(store, fid)| {
                // Warm once
                let mut warm: u128 = 0;
                for (blob, range) in store.scan_u64_vectors(fid) {
                    warm += sum_u64_le_simd(&blob.as_ref()[range.clone()]);
                }
                black_box(warm);
                // Measure
                let mut acc: u128 = 0;
                for (blob, range) in store.scan_u64_vectors(fid) {
                    acc += sum_u64_le_simd(&blob.as_ref()[range.clone()]);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("append_and_sum_u64_full_chunks", |b| {
        b.iter_batched(
            || {
                let pager = Arc::new(MemPager::default());
                let store: ColumnStore<MemPager> = ColumnStore::open(pager.clone());
                let fid: LogicalFieldId = 8888;
                let mut vals = vec![0u64; 1_000_000];
                for i in 0..vals.len() { vals[i] = (i as u64) & 0xFFFF_FFFF; }
                store.append_u64_chunk(fid, &vals);
                (store, fid)
            },
            |(store, fid)| {
                // Warm once
                let mut warm: u128 = 0;
                for (blob, range) in store.scan_u64_full_chunks(fid) {
                    warm += sum_u64_le_simd(&blob.as_ref()[range.clone()]);
                }
                black_box(warm);
                // Measure
                let mut acc: u128 = 0;
                for (blob, range) in store.scan_u64_full_chunks(fid) {
                    acc += sum_u64_le_simd(&blob.as_ref()[range.clone()]);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_column_store_u64_sum);
criterion_main!(benches);
