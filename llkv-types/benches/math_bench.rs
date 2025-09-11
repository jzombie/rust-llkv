//! A dedicated benchmark to prove the performance of batch vs. single
//! decoding for a realistic math kernel (summing u64s).

#![forbid(unsafe_code)]

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use llkv_types::{
    DataType, DecodedValue, decode_for_each, decode_for_each_reduce, decode_value,
    reduce_u64_for_each,
};

const N: usize = 1_000_000;

// Helper to generate test data.
fn make_u64s_encoded(n: usize) -> Vec<[u8; 8]> {
    let mut v = Vec::with_capacity(n);
    let mut rng = SmallRng::seed_from_u64(0xC0FF_EE00_DADA_BEEF);
    for i in 0..n as u64 {
        v.push((i ^ rng.random::<u64>()).to_be_bytes());
    }
    v
}

fn bench_math_kernels(c: &mut Criterion) {
    // Fixtures (built once).
    let enc_u64 = make_u64s_encoded(N);
    let enc_u64_slices: Vec<&[u8]> = enc_u64.iter().map(|a| a.as_slice()).collect();
    let u64_dtype = DataType::U64;

    // --- BENCHMARK 1: Your proposed method (decode one-by-one) ---
    // This is a fair, real-world test. It decodes AND collects the results
    // into a Vec before summing.
    c.bench_function("math_kernel/single_op_sum", |b| {
        b.iter_batched(
            || Vec::with_capacity(N),
            |mut out_vec| {
                // --- Part 1: Decode and Collect ---
                for encoded_bytes in &enc_u64 {
                    if let Some(DecodedValue::U64(x)) = decode_value(encoded_bytes, &u64_dtype) {
                        out_vec.push(x);
                    }
                }

                // --- Part 2: The Math Kernel ---
                let mut sum = 0u64;
                for &val in &out_vec {
                    sum = sum.wrapping_add(val);
                }
                black_box(sum);
            },
            BatchSize::PerIteration,
        );
    });

    // --- BENCHMARK 2: Using the `decode_for_each` function ---
    // This benchmark does the same work, but streams decoding and sums
    // immediately (no intermediate Vec<DecodedValue> allocation).
    c.bench_function("math_kernel/decode_many_sum", |b| {
        b.iter(|| {
            // --- Part 1: Decode (streamed) ---
            // --- Part 2: The Math Kernel (combined while decoding) ---
            let mut sum = 0u64;
            decode_for_each(enc_u64_slices.iter().copied(), &u64_dtype, |dv| {
                if let DecodedValue::U64(x) = dv {
                    sum = sum.wrapping_add(x);
                }
            })
            .unwrap();
            black_box(sum);
        });
    });

    // --- BENCHMARK 3: Non-typed reducer over DecodedValue ---
    // Same as BENCHMARK 2, but via a reducer-style helper that takes an
    // accumulator and returns it. Keeps DecodedValue in the API.
    c.bench_function("math_kernel/reducer_sum", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            let _n = decode_for_each_reduce(
                enc_u64_slices.iter().copied(),
                &u64_dtype,
                &mut sum,
                |acc, dv| {
                    if let DecodedValue::U64(x) = dv {
                        *acc = (*acc).wrapping_add(x);
                    }
                },
            )
            .unwrap();
            black_box(sum);
        });
    });

    // --- BENCHMARK 4: Typed reducer (u64) ---
    // Streams decoding as native u64 and reduces without enum or match.
    c.bench_function("math_kernel/typed_reducer_sum", |b| {
        b.iter(|| {
            let sum = reduce_u64_for_each(enc_u64_slices.iter().copied(), 0u64, |acc, x| {
                acc.wrapping_add(x)
            })
            .unwrap();
            black_box(sum);
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_math_kernels
}
criterion_main!(benches);
