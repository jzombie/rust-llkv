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
// TODO: Experimental end-to-end bench using column-map scan + decode_reduce.
use llkv_column_map::ColumnStore;
use llkv_column_map::storage::pager::MemPager;
use llkv_column_map::types::{AppendOptions, Put, ValueMode, LogicalFieldId};
use llkv_column_map::column_store::read_scan::{ValueScanOpts, OrderBy};
use std::borrow::Cow;

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

// TODO: Experimental: seed a column-map store, scan 1,000,000 fixed-width
// u64 values, and reduce via decode_reduce to validate correctness and gauge
// end-to-end throughput. Keep sample size small to avoid long runs.
fn bench_column_map_scan_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("cm_scan_reduce_1M");
    group.sample_size(10);

    // Build store and seed once (outside the measured loop).
    let pager = std::sync::Arc::new(MemPager::default());
    let store = ColumnStore::open(pager);
    let fid: LogicalFieldId = 4242;

    // Seed 1,000,000 entries: key=u64_be(i), value=u64_be(i)
    {
        let mut items: Vec<(Cow<'static, [u8]>, Cow<'static, [u8]>)> = Vec::with_capacity(1_000_000);
        for i in 0u64..1_000_000u64 {
            let k = i.to_be_bytes().to_vec();
            let v = i.to_be_bytes().to_vec();
            items.push((Cow::Owned(k), Cow::Owned(v)));
        }
        store.append_many(
            vec![Put { field_id: fid, items }],
            AppendOptions { mode: ValueMode::ForceFixed(8), ..Default::default() },
        );
    }

    // Expected sum: sum 0..1_000_000-1
    let expected_sum: u128 = (0u128..1_000_000u128).sum();

    // Bench: scan full range and decode_reduce to sum.
    group.bench_function("scan_sum_decode_reduce", |b| {
        // Create a fresh iterator in the setup (not measured), then consume it inside the timing closure.
        b.iter_batched(
            || {
                store
                    .scan_values_lww(
                        fid,
                        ValueScanOpts { order_by: OrderBy::Key, ..Default::default() },
                    )
                    .expect("scan iterator")
            },
            |it| {
                // Hold ValueSlices to keep blobs alive; then feed &[u8] slices.
                let vals: Vec<_> = it.map(|x| x.value).collect();
                let inputs = vals.iter().map(|v| v.as_slice());
                let (sum, cnt) = decode_reduce(inputs, &DataType::U64, 0u128, |acc, v| match v {
                    DecodedValue::U64(x) => acc + (x as u128),
                    _ => unreachable!(),
                })
                .expect("reduce");
                assert_eq!(cnt, 1_000_000);
                assert_eq!(sum, expected_sum);
                black_box(sum)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_reduce_many_u64, bench_column_map_scan_reduce);
criterion_main!(benches);
