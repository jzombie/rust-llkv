//! Bench public enum encode_value / decode_value for 1_000_000 items.
//! Adds non-ASCII string benches to force the ICU path.

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use llkv_types::{DataType, DecodedValue};
use llkv_types::{decode_value, encode_value, encode_value_to_vec};

const N: usize = 1_000_000;

fn make_u64s(n: usize) -> Vec<u64> {
    let mut v = Vec::with_capacity(n);
    let mut rng = SmallRng::seed_from_u64(0xC0FF_EE00_DADA_BEEF);
    for i in 0..n as u64 {
        v.push(i ^ rng.random::<u64>());
    }
    v
}

fn make_bools(n: usize) -> Vec<bool> {
    let mut v = Vec::with_capacity(n);
    let mut rng = SmallRng::seed_from_u64(0xBADC_0FFE_EE00_1234);
    for _ in 0..n {
        v.push(rng.random::<bool>());
    }
    v
}

fn make_strings_ascii(n: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(format!("{:06x}", i as u32));
    }
    v
}

fn make_strings_nonascii(n: usize) -> Vec<String> {
    // Build strings that are guaranteed non-ASCII using escapes.
    // Examples produced (no literals): "re\u{00E9}sum", "Stra\u{00DF}e",
    // "\u{212B}ng", "\u{212A}elvin". Each has an ASCII suffix for
    // uniqueness.
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = String::with_capacity(16);
        match i & 3 {
            0 => {
                s.push('r');
                s.push('\u{00E9}'); // e-acute
                s.push_str("sum");
            }
            1 => {
                s.push_str("Stra");
                s.push('\u{00DF}'); // sharp s
                s.push('e');
            }
            2 => {
                s.push('\u{212B}'); // Angstrom sign
                s.push_str("ng");
            }
            _ => {
                s.push('\u{212A}'); // Kelvin sign
                s.push_str("elvin");
            }
        }
        let _ = write!(&mut s, "{:06x}", i as u32);
        v.push(s);
    }
    v
}

fn bench_datatype_enum(c: &mut Criterion) {
    // Fixtures (built once).
    let vals_u64 = make_u64s(N);
    let vals_bool = make_bools(N);
    let vals_str_ascii = make_strings_ascii(N);
    let vals_str_nonascii = make_strings_nonascii(N);

    // Pre-encode inputs for decode benches.
    // u64: fixed arrays to avoid tiny Vecs.
    let mut enc_u64: Vec<[u8; 8]> = Vec::with_capacity(N);
    for &x in &vals_u64 {
        let b = encode_value_to_vec(DecodedValue::U64(x), &DataType::U64).expect("encode u64");
        debug_assert_eq!(b.len(), 8);
        let mut a = [0u8; 8];
        a.copy_from_slice(&b);
        enc_u64.push(a);
    }

    // bool: single byte each.
    let mut enc_bool: Vec<u8> = Vec::with_capacity(N);
    for &b in &vals_bool {
        let v = encode_value_to_vec(DecodedValue::Bool(b), &DataType::Bool).expect("encode bool");
        debug_assert_eq!(v.len(), 1);
        enc_bool.push(v[0]);
    }

    // strings (ASCII): per-item Vec<u8>.
    let mut enc_str_ascii: Vec<Vec<u8>> = Vec::with_capacity(N);
    for s in &vals_str_ascii {
        let v = encode_value_to_vec(DecodedValue::Str(s.as_str()), &DataType::Utf8)
            .expect("encode str ascii");
        enc_str_ascii.push(v);
    }

    // strings (non-ASCII): per-item Vec<u8>, forces ICU path.
    let mut enc_str_nonascii: Vec<Vec<u8>> = Vec::with_capacity(N);
    for s in &vals_str_nonascii {
        let v = encode_value_to_vec(DecodedValue::Str(s.as_str()), &DataType::Utf8)
            .expect("encode str non-ascii");
        enc_str_nonascii.push(v);
    }

    // ---------------------------
    // Encode benches (ASCII)
    // ---------------------------

    c.bench_function("DataType::Utf8/encode_ascii", |b| {
        b.iter_batched(
            || Vec::with_capacity(vals_str_ascii.len() * 6),
            |mut out| {
                for s in &vals_str_ascii {
                    encode_value(DecodedValue::Str(s.as_str()), &DataType::Utf8, &mut out).unwrap();
                }
                black_box(out.len());
            },
            BatchSize::PerIteration,
        );
    });

    // ---------------------------
    // Encode benches (non-ASCII)
    // ---------------------------

    c.bench_function("DataType::Utf8/encode_non_ascii", |b| {
        b.iter_batched(
            || Vec::with_capacity(vals_str_nonascii.len() * 8),
            |mut out| {
                for s in &vals_str_nonascii {
                    encode_value(DecodedValue::Str(s.as_str()), &DataType::Utf8, &mut out).unwrap();
                }
                black_box(out.len());
            },
            BatchSize::PerIteration,
        );
    });

    // ---------------------------
    // Encode benches (u64, bool)
    // ---------------------------

    c.bench_function("DataType::U64/encode", |b| {
        b.iter_batched(
            || Vec::with_capacity(vals_u64.len() * 8),
            |mut out| {
                for &x in &vals_u64 {
                    encode_value(DecodedValue::U64(x), &DataType::U64, &mut out).unwrap();
                }
                black_box(out.len());
            },
            BatchSize::PerIteration,
        );
    });

    c.bench_function("DataType::Bool/encode", |b| {
        b.iter_batched(
            || Vec::with_capacity(vals_bool.len()),
            |mut out| {
                for &x in &vals_bool {
                    encode_value(DecodedValue::Bool(x), &DataType::Bool, &mut out).unwrap();
                }
                black_box(out.len());
            },
            BatchSize::PerIteration,
        );
    });

    // ---------------------------
    // Decode benches (ASCII)
    // ---------------------------

    c.bench_function("DataType::Utf8/decode_ascii", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for bytes in &enc_str_ascii {
                let dv = decode_value(bytes, &DataType::Utf8).expect("decode str ascii");
                if let DecodedValue::Str(s) = dv {
                    acc = acc.wrapping_add(s.len());
                } else {
                    panic!("type mismatch");
                }
            }
            black_box(acc);
        });
    });

    // ---------------------------
    // Decode benches (non-ASCII)
    // ---------------------------

    c.bench_function("DataType::Utf8/decode_non_ascii", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for bytes in &enc_str_nonascii {
                let dv = decode_value(bytes, &DataType::Utf8).expect("decode str non-ascii");
                if let DecodedValue::Str(s) = dv {
                    acc = acc.wrapping_add(s.len());
                } else {
                    panic!("type mismatch");
                }
            }
            black_box(acc);
        });
    });

    // ---------------------------
    // Decode benches (u64, bool)
    // ---------------------------

    c.bench_function("DataType::U64/decode", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for a in &enc_u64 {
                let dv = decode_value(a, &DataType::U64).expect("decode u64");
                if let DecodedValue::U64(x) = dv {
                    acc ^= x;
                } else {
                    panic!("type mismatch");
                }
            }
            black_box(acc);
        });
    });

    c.bench_function("DataType::Bool/decode", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &b1 in &enc_bool {
                let dv =
                    decode_value(core::slice::from_ref(&b1), &DataType::Bool).expect("decode bool");
                if let DecodedValue::Bool(x) = dv {
                    acc ^= x as u64;
                } else {
                    panic!("type mismatch");
                }
            }
            black_box(acc);
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_datatype_enum
}
criterion_main!(benches);
