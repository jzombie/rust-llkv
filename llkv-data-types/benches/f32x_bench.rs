use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use llkv_data_types::{F32x, f32x_decode_many_into, f32x_decode_many_into_par, Codec};

const N: usize = 1024; // dimensions per vector
const CHUNK_ROWS: usize = 4096; // rows per encoded chunk (CHUNK_ROWS * N * 4 bytes)

fn make_encoded_chunk() -> Vec<u8> {
    let mut buf = Vec::with_capacity(CHUNK_ROWS * N * 4);
    // encode deterministic values to avoid NaNs
    for r in 0..CHUNK_ROWS {
        // simple pattern: x_i = (r as f32) + (i as f32) * 1e-3
        let mut row = vec![0f32; N];
        for i in 0..N { row[i] = r as f32 + (i as f32) * 1e-3; }
        F32x::<N>::encode_into(&mut buf, &row).unwrap();
    }
    buf
}

fn bench_f32x_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32x_decode");
    group.throughput(Throughput::Elements(1_000_000));

    // Prepare one encoded chunk and reuse it to simulate streaming 1,000,000 rows.
    let enc = make_encoded_chunk();

    group.bench_with_input(BenchmarkId::new("decode_1M_rows_1024d", N), &N, |b, &_n| {
        b.iter_batched(
            || vec![0f32; CHUNK_ROWS * N],
            |mut scratch| {
                let mut rows_done = 0usize;
                while rows_done < 1_000_000 {
                    f32x_decode_many_into::<N>(&mut scratch, &enc, CHUNK_ROWS).unwrap();
                    // do a tiny reduction to keep the work
                    let mut acc = 0f32;
                    for &x in &scratch[0..N] { acc += x; }
                    black_box(acc);
                    rows_done += CHUNK_ROWS;
                }
            },
            BatchSize::LargeInput,
        );
    });

    // Parallel variant
    group.bench_with_input(BenchmarkId::new("decode_1M_rows_1024d_par", N), &N, |b, &_n| {
        b.iter_batched(
            || vec![0f32; CHUNK_ROWS * N],
            |mut scratch| {
                let mut rows_done = 0usize;
                while rows_done < 1_000_000 {
                    f32x_decode_many_into_par::<N>(&mut scratch, &enc, CHUNK_ROWS).unwrap();
                    // small reduction to keep work live
                    let mut acc = 0f32;
                    for &x in &scratch[0..N] { acc += x; }
                    black_box(acc);
                    rows_done += CHUNK_ROWS;
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_f32x_decode);
criterion_main!(benches);
