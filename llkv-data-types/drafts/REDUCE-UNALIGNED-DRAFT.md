Most “iterator-y” abstractions add overhead that kills perf. The fastest thing on stable Rust is a tight, monomorphized loop over the raw pointer with **unaligned loads**, a **byte-swap**, and your kernel inlined. No `Vec`, no per-item arrays, no trait objects.

Here’s a small drop-in “reducer kernel” pattern that’s fast in practice:

* Works whether the bytes are aligned or not.
* Uses `ptr::read_unaligned::<u64>` (so no UB).
* `u64::from_be(..)` gives you the swap for free.
* Unrolled 4× to hide latency.
* Generic over the kernel so the whole thing inlines into one hot loop.
* i64 “lex” variant just XORs the sign bit.

```rust
use core::ptr;

/// Decode BE u64s from `bytes` and feed them into a reducer `step(acc, x)`.
/// Returns the final accumulator or an error if the input length is wrong.
/// - No allocations
/// - Unaligned-safe
/// - Unrolled inner loop
#[inline(always)]
pub fn reduce_be_u64_with<T, F>(bytes: &[u8], mut acc: T, mut step: F) -> Result<T, &'static str>
where
    F: FnMut(T, u64) -> T,
{
    let len = bytes.len();
    if len & 7 != 0 {
        return Err("length not multiple of 8");
    }
    // Raw pointer walk; unaligned reads are OK via read_unaligned.
    let mut p = bytes.as_ptr();
    let end = unsafe { p.add(len) };

    // 4x unroll
    while unsafe { p.add(32) } <= end {
        unsafe {
            let a = u64::from_be(ptr::read_unaligned(p as *const u64));
            let b = u64::from_be(ptr::read_unaligned(p.add(8) as *const u64));
            let c = u64::from_be(ptr::read_unaligned(p.add(16) as *const u64));
            let d = u64::from_be(ptr::read_unaligned(p.add(24) as *const u64));
            acc = step(acc, a);
            acc = step(acc, b);
            acc = step(acc, c);
            acc = step(acc, d);
            p = p.add(32);
        }
    }

    // tail
    while p < end {
        unsafe {
            let x = u64::from_be(ptr::read_unaligned(p as *const u64));
            acc = step(acc, x);
            p = p.add(8);
        }
    }
    Ok(acc)
}

/// Same idea for i64 stored in lexicographic order: value is (u ^ SIGN) as i64.
#[inline(always)]
pub fn reduce_be_i64lex_with<T, F>(bytes: &[u8], mut acc: T, mut step: F) -> Result<T, &'static str>
where
    F: FnMut(T, i64) -> T,
{
    const SIGN: u64 = 0x8000_0000_0000_0000;
    let len = bytes.len();
    if len & 7 != 0 {
        return Err("length not multiple of 8");
    }
    let mut p = bytes.as_ptr();
    let end = unsafe { p.add(len) };

    while unsafe { p.add(32) } <= end {
        unsafe {
            let a = (u64::from_be(ptr::read_unaligned(p as *const u64)) ^ SIGN) as i64;
            let b = (u64::from_be(ptr::read_unaligned(p.add(8) as *const u64)) ^ SIGN) as i64;
            let c = (u64::from_be(ptr::read_unaligned(p.add(16) as *const u64)) ^ SIGN) as i64;
            let d = (u64::from_be(ptr::read_unaligned(p.add(24) as *const u64)) ^ SIGN) as i64;
            acc = step(acc, a);
            acc = step(acc, b);
            acc = step(acc, c);
            acc = step(acc, d);
            p = p.add(32);
        }
    }

    while p < end {
        unsafe {
            let x = (u64::from_be(ptr::read_unaligned(p as *const u64)) ^ SIGN) as i64;
            acc = step(acc, x);
            p = p.add(8);
        }
    }
    Ok(acc)
}
```

### Common kernels (no allocation, fully inlined)

```rust
#[inline(always)]
pub fn sum_u64(bytes: &[u8]) -> Result<u128, &'static str> {
    // Wider accumulator to reduce overflow in real-world sums.
    reduce_be_u64_with(bytes, 0u128, |acc, x| acc + x as u128)
}

#[derive(Clone, Copy, Debug)]
pub struct MinMax<T> { pub min: T, pub max: T }

#[inline(always)]
pub fn minmax_u64(bytes: &[u8]) -> Result<Option<MinMax<u64>>, &'static str> {
    if bytes.is_empty() { return Ok(None); }
    reduce_be_u64_with(bytes, None, |acc, x| {
        match acc {
            None => Some(MinMax { min: x, max: x }),
            Some(mut mm) => {
                if x < mm.min { mm.min = x; }
                if x > mm.max { mm.max = x; }
                Some(mm)
            }
        }
    })
}

#[inline(always)]
pub fn mean_var_i64(bytes: &[u8]) -> Result<Option<(f64, f64)>, &'static str> {
    // Welford’s online algorithm: numerically stable
    #[derive(Clone, Copy)]
    struct W { n: u64, mean: f64, m2: f64 }
    let w = reduce_be_i64lex_with(bytes, W { n:0, mean:0.0, m2:0.0 }, |mut w, x| {
        w.n += 1;
        let xf = x as f64;
        let delta = xf - w.mean;
        w.mean += delta / (w.n as f64);
        w.m2 += delta * (xf - w.mean);
        w
    })?;
    if w.n == 0 { Ok(None) } else { Ok(Some((w.mean, w.m2 / w.n as f64))) }
}
```

### Why this is fast

* **No iterator object churn**: a single tight loop the optimizer can see through.
* **No `Vec` internally**: no allocs, no copies.
* **Unaligned loads**: `read_unaligned::<u64>` is efficient on x86\_64/ARM; you avoid copying to align.
* **Byte swap**: `u64::from_be` compiles to a single `bswap` on LE.
* **Unrolling**: 4× is a good default; tweak if it helps your target.

### If you want to squeeze more

* Add a cfg-gated SIMD path (e.g., AVX2/NEON) that does a 16-byte shuffle to byteswap multiple lanes at once, then feed lanes to the kernel. Keep the scalar path as the default; only enable SIMD where you’ve benched it.
* Specialize hot kernels (sum, min/max, dot) as dedicated functions (like above) instead of closures; that can give LLVM more room to reorder/unroll.
* For pairwise ops (e.g., dot product), mirror the above with two pointers and two unaligned reads per step.

### Usage

```rust
let bytes: Vec<u8> = /* N*8 bytes, big-endian u64s */;

let total = sum_u64(&bytes)?;
let mm = minmax_u64(&bytes)?.unwrap();
let (mean, var) = mean_var_i64(&bytes)?.unwrap();

// Custom reducer:
let count_ge_1e6 = reduce_be_u64_with(&bytes, 0u64, |acc, x| acc + u64::from(x >= 1_000_000))?;
```

This pattern usually matches or beats “decode inside the kernel per-iteration” because it keeps everything in one place for the optimizer, avoids allocation, and minimizes per-item work to: **unaligned load → bswap → one inlined step**.
