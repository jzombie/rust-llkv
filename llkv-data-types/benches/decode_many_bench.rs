use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use llkv_data_types::be_u64_decode_many_into;
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

criterion_group!(benches, bench_decode_many_u64);
criterion_main!(benches);
