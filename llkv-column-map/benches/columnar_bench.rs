use criterion::{criterion_group, criterion_main, Criterion, BatchSize, black_box};
use llkv_column_map::storage::pager::{MemPager, Pager};
use llkv_column_map::types::PhysicalKey;
use llkv_column_map::column_store::columnar::{write_u64_chunk, get_chunk_blob, U64ChunkView, sum_u64_le_simd};

fn bench_columnar_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("columnar_sum_1M");
    group.sample_size(50);
    group.bench_function("write_and_sum_1M_u64", |b| {
        b.iter_batched(
            || -> (MemPager, PhysicalKey, Vec<u8>) {
                // Setup: in-memory pager + allocate key + values
                let pager = MemPager::default();
                let pk: PhysicalKey = Pager::alloc_many(&pager, 1).unwrap()[0];
                let mut vals = vec![0u64; 1_000_000];
                for i in 0..vals.len() { vals[i] = (i as u64) & 0xFFFF_FFFF; }
                write_u64_chunk(&pager, pk, &vals, 16_384, 0).expect("write chunk");
                // Extract stripe bytes once for measurement-only path
                let blob = get_chunk_blob(&pager, pk).expect("blob");
                let bytes = blob.as_ref();
                let view = U64ChunkView::from_blob_le(bytes).expect("view");
                (pager, pk, view.values_le.to_vec())
            },
            |input| {
                let (_pager, _pk, stripe): (MemPager, PhysicalKey, Vec<u8>) = input;
                // Warm run
                let _ = sum_u64_le_simd(&stripe);
                // Measure
                let acc = sum_u64_le_simd(&stripe);
                black_box(acc)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_columnar_sum);
criterion_main!(benches);
