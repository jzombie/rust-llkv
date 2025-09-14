use criterion::{criterion_group, criterion_main, Criterion, BatchSize, black_box};

#[inline(always)]
fn sum_u64_scalar(xs: &[u64]) -> u128 {
    let mut a0: u128 = 0;
    let mut a1: u128 = 0;
    let mut a2: u128 = 0;
    let mut a3: u128 = 0;
    let mut i = 0;
    let n = xs.len();
    let unroll = 4;
    let bound = n - (n % unroll);
    while i < bound {
        a0 += xs[i] as u128;
        a1 += xs[i + 1] as u128;
        a2 += xs[i + 2] as u128;
        a3 += xs[i + 3] as u128;
        i += unroll;
    }
    let mut acc = a0 + a1 + a2 + a3;
    while i < n {
        acc += xs[i] as u128;
        i += 1;
    }
    acc
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn sum_u64_avx2(xs: &[u64]) -> u128 {
    use core::arch::x86_64::*;
    let mut i = 0;
    let n = xs.len();
    let mut v0 = _mm256_setzero_si256();
    let mut v1 = _mm256_setzero_si256();
    let mut v2 = _mm256_setzero_si256();
    let mut v3 = _mm256_setzero_si256();
    // process 16 u64 per iter
    while i + 16 <= n {
        let p = xs.as_ptr().add(i) as *const __m256i;
        let a = _mm256_loadu_si256(p);
        let b = _mm256_loadu_si256(p.add(1));
        let c = _mm256_loadu_si256(p.add(2));
        let d = _mm256_loadu_si256(p.add(3));
        v0 = _mm256_add_epi64(v0, a);
        v1 = _mm256_add_epi64(v1, b);
        v2 = _mm256_add_epi64(v2, c);
        v3 = _mm256_add_epi64(v3, d);
        i += 16;
    }
    // horizontal add vectors to 2 lanes 128b
    let s01 = _mm256_add_epi64(v0, v1);
    let s23 = _mm256_add_epi64(v2, v3);
    let s = _mm256_add_epi64(s01, s23);
    // reduce 256->128
    let hi = _mm256_extracti128_si256(s, 1);
    let lo = _mm256_castsi256_si128(s);
    let s128 = _mm_add_epi64(lo, hi);
    // reduce 128->scalar
    let mut tmp = [0u64; 2];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, s128);
    let mut acc: u128 = tmp[0] as u128 + tmp[1] as u128;
    // tail
    while i < n {
        acc += *xs.get_unchecked(i) as u128;
        i += 1;
    }
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn sum_u64_neon(xs: &[u64]) -> u128 {
    use core::arch::aarch64::*;
    let mut i = 0;
    let n = xs.len();
    let mut v0 = vdupq_n_u64(0);
    let mut v1 = vdupq_n_u64(0);
    let mut v2 = vdupq_n_u64(0);
    let mut v3 = vdupq_n_u64(0);
    while i + 8 <= n {
        let p = xs.as_ptr().add(i) as *const u64;
        let a = vld1q_u64(p);
        let b = vld1q_u64(p.add(2));
        let c = vld1q_u64(p.add(4));
        let d = vld1q_u64(p.add(6));
        v0 = vaddq_u64(v0, a);
        v1 = vaddq_u64(v1, b);
        v2 = vaddq_u64(v2, c);
        v3 = vaddq_u64(v3, d);
        i += 8;
    }
    let s01 = vaddq_u64(v0, v1);
    let s23 = vaddq_u64(v2, v3);
    let s = vaddq_u64(s01, s23);
    let lanes: [u64; 2] = core::mem::transmute(s);
    let mut acc: u128 = lanes[0] as u128 + lanes[1] as u128;
    while i < n {
        acc += *xs.get_unchecked(i) as u128;
        i += 1;
    }
    acc
}

#[inline(always)]
fn sum_u64_simd(xs: &[u64]) -> u128 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return sum_u64_avx2(xs);
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return sum_u64_neon(xs);
    }
    sum_u64_scalar(xs)
}

fn bench_simd_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sum_1M");
    group.sample_size(100);
    group.bench_function("sum_1M_u64_simd", |b| {
        b.iter_batched(
            || {
                // Allocate aligned Vec<u64> of 1,000,000 values; fill with pattern
                let mut v = vec![0u64; 1_000_000];
                for i in 0..v.len() { v[i] = (i as u64) & 0xFFFF_FFFF; }
                v
            },
            |v| {
                // Warm run
                let _w = sum_u64_simd(&v);
                // Measure
                let acc = sum_u64_simd(&v);
                black_box(acc)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_simd_sum);
criterion_main!(benches);

