use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use llkv_data_types::{be_u64_decode_many_into, be_u64_reduce_many_concat};
use rand::{rngs::StdRng, Rng, SeedableRng};

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

fn bench_decode_many_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_u64_decode_many");

    for &n in &[8usize, 1024, 65_536, 1_000_000] {
        let (src, _vals) = make_be_bytes(n, 42);
        let mut dst = vec![0u64; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("one_by_one", n), &n, |b, &_n| {
            b.iter_batched(
                || dst.clone(),
                |mut d| {
                    for i in 0..n {
                        let a = i * 8;
                        let b = a + 8;
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(&src[a..b]);
                        d[i] = u64::from_be_bytes(bytes);
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("many_into", n), &n, |b, &_n| {
            b.iter_batched(
                || dst.clone(),
                |mut d| {
                    be_u64_decode_many_into(&mut d, &src).unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// TODO: Experimental reducer benches. If adopted, consider moving alongside math_bench
// and unifying benchmark structure.
fn bench_reduce_many_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_u64_reduce_many");

    for &n in &[1024usize, 65_536, 1_000_000] {
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
        let (sum_many, cnt) = be_u64_reduce_many_concat(&src, 0u128, |acc, x| acc + (x as u128)).unwrap();
        assert_eq!(cnt, n);
        assert_eq!(sum_many, sum_one, "reduce_many must match one_by_one");

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

        group.bench_with_input(BenchmarkId::new("sum_many_concat", n), &n, |b, &_n| {
            b.iter_batched(
                || (),
                |_| {
                    let (acc, _cnt) = be_u64_reduce_many_concat(&src, 0u128, |acc, x| acc + (x as u128)).unwrap();
                    black_box(acc)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_decode_many_u64, bench_reduce_many_u64);
criterion_main!(benches);
