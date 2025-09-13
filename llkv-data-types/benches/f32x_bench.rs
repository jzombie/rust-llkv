use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use llkv_data_types::{F32x, f32x_decode_many_into, f32x_decode_many_into_par, Codec};
use rayon::prelude::*;

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

// Native-endian encoded chunk (no byte swap per element).
fn make_encoded_chunk_ne() -> Vec<u8> {
    let mut buf = Vec::with_capacity(CHUNK_ROWS * N * 4);
    for r in 0..CHUNK_ROWS {
        for i in 0..N {
            let x = r as f32 + (i as f32) * 1e-3;
            buf.extend_from_slice(&x.to_bits().to_ne_bytes());
        }
    }
    buf
}

#[inline]
fn decode_many_into_ne<const M: usize>(dst: &mut [f32], src: &[u8], rows: usize) {
    debug_assert_eq!(dst.len(), rows * M);
    for r in 0..rows {
        let row_bytes = &src[(r * M * 4)..((r + 1) * M * 4)];
        for c in 0..M {
            let a = c * 4;
            let mut b4 = [0u8; 4];
            b4.copy_from_slice(&row_bytes[a..a + 4]);
            dst[r * M + c] = f32::from_bits(u32::from_ne_bytes(b4));
        }
    }
}

#[inline]
fn decode_many_into_ne_par<const M: usize>(dst: &mut [f32], src: &[u8], rows: usize) {
    dst.par_chunks_mut(M)
        .zip(src.par_chunks_exact(M * 4))
        .for_each(|(row_dst, row_src)| {
            for c in 0..M {
                let a = c * 4;
                let mut b4 = [0u8; 4];
                b4.copy_from_slice(&row_src[a..a + 4]);
                row_dst[c] = f32::from_bits(u32::from_ne_bytes(b4));
            }
        });
}

fn bench_f32x_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32x_decode");
    group.throughput(Throughput::Elements(1_000_000));

    // Prepare one encoded chunk and reuse it to simulate streaming 1,000,000 rows.
    let enc = make_encoded_chunk();
    let enc_ne = make_encoded_chunk_ne();

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

    // Native-endian (no byte swap) single-threaded
    group.bench_with_input(BenchmarkId::new("decode_1M_rows_1024d_ne", N), &N, |b, &_n| {
        b.iter_batched(
            || vec![0f32; CHUNK_ROWS * N],
            |mut scratch| {
                let mut rows_done = 0usize;
                while rows_done < 1_000_000 {
                    decode_many_into_ne::<N>(&mut scratch, &enc_ne, CHUNK_ROWS);
                    let mut acc = 0f32; for &x in &scratch[0..N] { acc += x; } black_box(acc);
                    rows_done += CHUNK_ROWS;
                }
            },
            BatchSize::LargeInput,
        );
    });

    // Native-endian (no byte swap) parallel
    group.bench_with_input(BenchmarkId::new("decode_1M_rows_1024d_ne_par", N), &N, |b, &_n| {
        b.iter_batched(
            || vec![0f32; CHUNK_ROWS * N],
            |mut scratch| {
                let mut rows_done = 0usize;
                while rows_done < 1_000_000 {
                    decode_many_into_ne_par::<N>(&mut scratch, &enc_ne, CHUNK_ROWS);
                    let mut acc = 0f32; for &x in &scratch[0..N] { acc += x; } black_box(acc);
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
