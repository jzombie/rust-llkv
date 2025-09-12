use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use llkv_data_types::{
    DataType,
    DecodedValue,
    // TODO: Experimental reducers; bench only SUM variants below.
    be_u64_reduce_many_concat,
    be_u64_reduce_streaming,
    be_u64_reduce_streaming_unaligned,
    be_u64_sum_streaming_unaligned,
    // TODO: Add generic reducer comparison against baseline one-by-one.
    decode_reduce,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn make_be_bytes(n: usize, seed: u64) -> (Vec<u8>, Vec<u64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        vals.push(rng.random::<u64>());
    }
    let mut buf = Vec::with_capacity(n * 8);
    for v in &vals {
        buf.extend_from_slice(&v.to_be_bytes());
    }
    (buf, vals)
}

// TODO: Experimental reducer benches: focus on SUM at 1,000,000 elements.
// If adopted, consider moving alongside math_bench and unifying structure.
fn bench_reduce_many_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_u64_reduce_many");

    // Only benchmark the 1,000,000 element case to reduce noise.
    for &n in &[1_000_000usize] {
        let (src, _vals) = make_be_bytes(n, 777);
        group.throughput(Throughput::Elements(n as u64));

        // correctness smoke check (once)
        let mut sum_one = 0u128;
        for i in 0..n {
            let a = i * 8;
            let b = a + 8;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&src[a..b]);
            sum_one += u64::from_be_bytes(bytes) as u128;
        }
        let (sum_many, cnt) =
            be_u64_reduce_many_concat(&src, 0u128, |acc, x| acc + (x as u128)).unwrap();
        assert_eq!(cnt, n);
        assert_eq!(sum_many, sum_one, "reduce_many must match one_by_one");
        // streaming correctness
        let (sum_stream, cnt2) =
            be_u64_reduce_streaming(&src, 0u128, |acc, x| acc + (x as u128)).unwrap();
        assert_eq!(cnt2, n);
        assert_eq!(
            sum_stream, sum_one,
            "streaming reduce must match one_by_one"
        );

        group.bench_with_input(BenchmarkId::new("sum_one_by_one", n), &n, |b, &_n| {
            b.iter(|| {
                let mut acc = 0u128;
                for i in 0..n {
                    let a = i * 8;
                    let b = a + 8;
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&src[a..b]);
                    acc += u64::from_be_bytes(bytes) as u128;
                }
                black_box(acc)
            });
        });

        // group.bench_with_input(BenchmarkId::new("sum_streaming", n), &n, |b, &_n| {
        //     b.iter(|| {
        //         let (acc, _cnt) = be_u64_reduce_streaming(&src, 0u128, |acc, x| acc + (x as u128)).unwrap();
        //         black_box(acc)
        //     });
        // });

        // group.bench_with_input(BenchmarkId::new("sum_streaming_unaligned", n), &n, |b, &_n| {
        //     b.iter(|| {
        //         let (acc, _cnt) = be_u64_reduce_streaming_unaligned(&src, 0u128, |acc, x| acc + (x as u128)).unwrap();
        //         black_box(acc)
        //     });
        // });

        // // Specialized sum without closure overhead; should approach baseline.
        // group.bench_with_input(BenchmarkId::new("sum_streaming_unaligned_spec", n), &n, |b, &_n| {
        //     b.iter(|| {
        //         let (acc, _cnt) = be_u64_sum_streaming_unaligned(&src).unwrap();
        //         black_box(acc)
        //     });
        // });

        // TODO: Compare against the original generic reducer in the library.
        group.bench_with_input(BenchmarkId::new("sum_decode_reduce", n), &n, |b, &_n| {
            b.iter(|| {
                let inputs = src.chunks_exact(8);
                let (acc, _cnt) = decode_reduce(inputs, &DataType::U64, 0u128, |acc, v| match v {
                    DecodedValue::U64(x) => acc + (x as u128),
                    _ => unreachable!(),
                })
                .unwrap();
                black_box(acc)
            });
        });

        // group.bench_with_input(BenchmarkId::new("sum_many_concat", n), &n, |b, &_n| {
        //     b.iter_batched(
        //         || (),
        //         |_| {
        //             let (acc, _cnt) =
        //                 be_u64_reduce_many_concat(&src, 0u128, |acc, x| acc + (x as u128)).unwrap();
        //             black_box(acc)
        //         },
        //         BatchSize::SmallInput,
        //     );
        // });
    }

    group.finish();
}

criterion_group!(benches, bench_reduce_many_u64);
criterion_main!(benches);
