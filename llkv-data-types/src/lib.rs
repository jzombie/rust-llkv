mod internal;
use crate::internal::{BeI64, BeU8, BeU16, BeU32, BeU64, Bool, Codec, EncodeInto, Utf8CaseFold};

pub mod errors;
pub use errors::*;

// --- Public-Facing Metadata Enums ---

/// A tag representing the physical data type for a piece of metadata.
///
/// This is a simple, C-like enum that is cheap to store and copy.
/// Its only purpose is to act as a label for the underlying storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Utf8,
    U8,
    U16,
    U32,
    U64,
    I64,
    Bool,
    Bytes,
    CategoryId,
}

/// A generic enum to hold any possible value decoded from storage.
///
/// It uses a lifetime (`'a`) to allow for zero-allocation borrowing of strings,
/// preserving the performance of the internal codecs.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DecodedValue<'a> {
    Str(&'a str),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I64(i64),
    Bool(bool),
    Bytes(&'a [u8]),
    CategoryId(u32),
}

/// Encode `value` into `out` using `dtype`.
/// Appends to `out`.
#[inline]
pub fn encode_value<'a>(
    value: DecodedValue<'a>,
    dtype: &DataType,
    out: &mut Vec<u8>,
) -> Result<(), EncodeError> {
    match (dtype, value) {
        (DataType::Utf8, DecodedValue::Str(s)) => {
            s.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::Str(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "Str",
        }),
        //
        (DataType::U8, DecodedValue::U8(x)) => {
            x.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::U8(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "U8",
        }),
        //
        (DataType::U16, DecodedValue::U16(x)) => {
            x.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::U16(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "U16",
        }),
        //
        (DataType::U32, DecodedValue::U32(x)) => {
            x.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::U32(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "U32",
        }),
        //
        (DataType::U64, DecodedValue::U64(x)) => {
            x.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::U64(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "U64",
        }),
        //
        (DataType::I64, DecodedValue::I64(x)) => {
            x.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::I64(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "I64",
        }),
        //
        (DataType::Bool, DecodedValue::Bool(b)) => {
            b.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::Bool(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "Bool",
        }),
        //
        (DataType::Bytes, DecodedValue::Bytes(b)) => {
            b.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::Bytes(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "Bytes",
        }),
        (DataType::CategoryId, DecodedValue::CategoryId(x)) => {
            BeU32::encode_into(out, &x).ok();
            Ok(())
        }
        (expected, DecodedValue::CategoryId(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "CategoryId",
        }),
    }
}

#[inline]
pub fn encode_value_to_vec<'a>(
    value: DecodedValue<'a>,
    dtype: &DataType,
) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::new();
    encode_value(value, dtype, &mut out)?;
    Ok(out)
}

/// Bridges the runtime metadata (`DataType`) to the high-performance,
/// statically-dispatched codec functions from the `internal` module.
///
/// This function is the key to the whole pattern: it looks at the tag once,
/// then calls the specific, hyper-optimized function for that type.
pub fn decode_value<'a>(bytes: &'a [u8], dtype: &DataType) -> Option<DecodedValue<'a>> {
    match dtype {
        DataType::Utf8 => Utf8CaseFold::decode_borrowed(bytes).map(DecodedValue::Str),
        DataType::U8 => BeU8::decode(bytes).ok().map(DecodedValue::U8),
        DataType::U16 => BeU16::decode(bytes).ok().map(DecodedValue::U16),
        DataType::U32 => BeU32::decode(bytes).ok().map(DecodedValue::U32),
        DataType::U64 => BeU64::decode(bytes).ok().map(DecodedValue::U64),
        DataType::I64 => BeI64::decode(bytes).ok().map(DecodedValue::I64),
        DataType::Bool => Bool::decode(bytes).ok().map(DecodedValue::Bool),
        DataType::Bytes => internal::Bytes::decode_borrowed(bytes).map(DecodedValue::Bytes),
        DataType::CategoryId => BeU32::decode(bytes).ok().map(DecodedValue::CategoryId),
    }
}

// -------------------- Public batch helpers for benches/consumers --------------------

/// Decode concatenated big-endian u64 values into `dst` in a single pass.
/// `src.len()` must equal `dst.len() * 8`.
///
/// TODO: Experimental helper. Consider moving this behind a feature flag or
/// integrating into a refined codec trait once stabilized.
#[inline]
pub fn be_u64_decode_many_into(dst: &mut [u64], src: &[u8]) -> Result<(), DecodeError> {
    internal::BeU64::decode_many_into(dst, src)
}

/// Reduce over a concatenated buffer of BE u64 values using a bulk decode.
/// Returns (accumulator, count). `src.len()` must be a multiple of 8.
///
/// TODO: Experimental specialization. If kept, generalize to other integer
/// widths and unify with `decode_reduce` via a trait or enum dispatch.
pub fn be_u64_reduce_many_concat<T, F>(src: &[u8], init: T, mut f: F) -> Result<(T, usize), DecodeError>
where
    F: FnMut(T, u64) -> T,
{
    if src.len() % 8 != 0 { return Err(DecodeError::NotEnoughData); }
    let n = src.len() / 8;
    let mut tmp = vec![0u64; n];
    internal::BeU64::decode_many_into(&mut tmp, src)?;
    let mut acc = init;
    for &x in &tmp { acc = f(acc, x); }
    Ok((acc, n))
}

/// Stream a reduce over concatenated BE u64 bytes without an intermediate buffer.
/// Returns (accumulator, count). `src.len()` must be a multiple of 8.
///
/// TODO: Experimental; if useful, consider exposing a generic streaming
/// decoder-reducer for fixed-width integers.
#[inline(always)]
pub fn be_u64_reduce_streaming<T, F>(src: &[u8], init: T, mut f: F) -> Result<(T, usize), DecodeError>
where
    F: FnMut(T, u64) -> T,
{
    if src.len() % 8 != 0 { return Err(DecodeError::NotEnoughData); }
    let n = src.len() / 8;
    let mut acc = init;
    let mut off = 0usize;
    for _ in 0..n {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&src[off..off+8]);
        let x = u64::from_be_bytes(bytes);
        acc = f(acc, x);
        off += 8;
    }
    Ok((acc, n))
}

/// Chunked reduce over concatenated BE u64 using a reusable decode buffer.
/// Decodes up to `buf.len()` elements at a time, then folds them.
/// Returns (accumulator, count). `src.len()` must be a multiple of 8.
///
/// TODO: Experimental; evaluate against streaming for various cache sizes and
/// consider a single-pass fused variant if mapping stores into an output column.
pub fn be_u64_reduce_chunked_with_buf<T, F>(
    src: &[u8],
    init: T,
    mut f: F,
    buf: &mut [u64],
) -> Result<(T, usize), DecodeError>
where
    F: FnMut(T, u64) -> T,
{
    if src.len() % 8 != 0 { return Err(DecodeError::NotEnoughData); }
    if buf.is_empty() { return Ok((init, 0)); }
    let total = src.len() / 8;
    let mut acc = init;
    let mut done = 0usize;
    let mut off_bytes = 0usize;
    while done < total {
        let take = core::cmp::min(buf.len(), total - done);
        let bytes_len = take * 8;
        let chunk = &src[off_bytes..off_bytes + bytes_len];
        be_u64_decode_many_into(&mut buf[..take], chunk)?;
        for &x in &buf[..take] {
            acc = f(acc, x);
        }
        done += take;
        off_bytes += bytes_len;
    }
    Ok((acc, total))
}

/// Streaming reduce with unsafe unaligned reads (fast path).
/// Decodes using `read_unaligned` and `u64::from_be` to avoid per-item copies.
/// Returns (accumulator, count). `src.len()` must be a multiple of 8.
///
/// TODO: Experimental fast path. Guard with a feature if kept; add similar
/// implementations for other integer widths.
#[inline(always)]
pub fn be_u64_reduce_streaming_unaligned<T, F>(
    src: &[u8],
    init: T,
    mut f: F,
) -> Result<(T, usize), DecodeError>
where
    F: FnMut(T, u64) -> T,
{
    if src.len() % 8 != 0 {
        return Err(DecodeError::NotEnoughData);
    }
    let n = src.len() / 8;
    let mut acc = init;
    let mut p = src.as_ptr();
    for _ in 0..n {
        // SAFETY: We only advance within bounds in 8-byte steps; alignment is not required.
        let word = unsafe { (p as *const u64).read_unaligned() };
        let x = u64::from_be(word);
        acc = f(acc, x);
        // SAFETY: pointer arithmetic within checked bounds above.
        p = unsafe { p.add(8) };
    }
    Ok((acc, n))
}

/// Value-returning reducer over decoded values.
/// Streams items, calling `f(acc, item)` each step.
/// Returns `(accumulator, count)`.
#[inline]
pub fn decode_reduce<'a, I, T, F>(
    inputs: I,
    dtype: &DataType,
    init: T,
    mut f: F,
) -> Result<(T, usize), DecodeError>
where
    I: IntoIterator<Item = &'a [u8]>,
    F: FnMut(T, DecodedValue<'a>) -> T,
{
    let mut acc = init;
    let mut n = 0usize;

    match dtype {
        DataType::Utf8 => {
            for b in inputs {
                let s = Utf8CaseFold::decode_borrowed(b).ok_or(DecodeError::InvalidFormat)?;
                acc = f(acc, DecodedValue::Str(s));
                n += 1;
            }
        }
        DataType::U8 => {
            for b in inputs {
                let x = BeU8::decode(b)?;
                acc = f(acc, DecodedValue::U8(x));
                n += 1;
            }
        }
        DataType::U16 => {
            for b in inputs {
                let x = BeU16::decode(b)?;
                acc = f(acc, DecodedValue::U16(x));
                n += 1;
            }
        }
        DataType::U32 => {
            for b in inputs {
                let x = BeU32::decode(b)?;
                acc = f(acc, DecodedValue::U32(x));
                n += 1;
            }
        }
        DataType::U64 => {
            for b in inputs {
                let x = BeU64::decode(b)?;
                acc = f(acc, DecodedValue::U64(x));
                n += 1;
            }
        }
        DataType::I64 => {
            for b in inputs {
                let x = BeI64::decode(b)?;
                acc = f(acc, DecodedValue::I64(x));
                n += 1;
            }
        }
        DataType::Bool => {
            for b in inputs {
                let x = Bool::decode(b)?;
                acc = f(acc, DecodedValue::Bool(x));
                n += 1;
            }
        }
        DataType::Bytes => {
            for b in inputs {
                let x = internal::Bytes::decode_borrowed(b).ok_or(DecodeError::InvalidFormat)?;
                acc = f(acc, DecodedValue::Bytes(x));
                n += 1;
            }
        }
        DataType::CategoryId => {
            for b in inputs {
                let x = BeU32::decode(b)?;
                acc = f(acc, DecodedValue::CategoryId(x));
                n += 1;
            }
        }
    }

    Ok((acc, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use internal::EncodeInto;

    /* ---------------- Utf8 tests ---------------- */

    #[test]
    fn test_utf8_encode_decode_roundtrip() {
        let dtype = DataType::Utf8;

        let mut buf = Vec::new();
        "Jeremy".encode_into(&mut buf);

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::Str("Jeremy"));

        if let DecodedValue::Str(s) = dv {
            assert!(s.starts_with('J'));
        } else {
            panic!("expected Str");
        }
    }

    /* ---------------- U8 tests ---------------- */

    #[test]
    fn test_u8_encode_decode_roundtrip() {
        let dtype = DataType::U8;

        let mut buf = Vec::new();
        200u8.encode_into(&mut buf);

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::U8(200));
    }

    #[test]
    fn test_decode_reduce_u8_collects_values() {
        let dtype = DataType::U8;
        let values = [0u8, 1, 127, 200, 255];

        let encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|v| {
                let mut buf = Vec::new();
                v.encode_into(&mut buf);
                buf
            })
            .collect();
        let slices: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();

        let (decoded, count): (Vec<DecodedValue<'_>>, usize) =
            decode_reduce(slices.iter().copied(), &dtype, Vec::new(), |mut acc, dv| {
                acc.push(dv);
                acc
            })
            .unwrap();

        assert_eq!(count, values.len());
        assert_eq!(
            decoded,
            values
                .iter()
                .copied()
                .map(DecodedValue::U8)
                .collect::<Vec<_>>()
        );
    }

    /* ---------------- U16 tests ---------------- */

    #[test]
    fn test_u16_encode_decode_roundtrip() {
        let dtype = DataType::U16;

        let mut buf = Vec::new();
        0xABCDu16.encode_into(&mut buf);

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::U16(0xABCD));
    }

    #[test]
    fn test_decode_reduce_u16_collects_values() {
        let dtype = DataType::U16;
        let values = [0u16, 1, 256, 42_000, u16::MAX];

        let encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|v| {
                let mut buf = Vec::new();
                v.encode_into(&mut buf);
                buf
            })
            .collect();
        let slices: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();

        let (decoded, count): (Vec<DecodedValue<'_>>, usize) =
            decode_reduce(slices.iter().copied(), &dtype, Vec::new(), |mut acc, dv| {
                acc.push(dv);
                acc
            })
            .unwrap();

        assert_eq!(count, values.len());
        assert_eq!(
            decoded,
            values
                .iter()
                .copied()
                .map(DecodedValue::U16)
                .collect::<Vec<_>>()
        );
    }

    /* ---------------- U32 tests ---------------- */

    #[test]
    fn test_u32_encode_decode_roundtrip() {
        let dtype = DataType::U32;

        let mut buf = Vec::new();
        0xDEADBEEFu32.encode_into(&mut buf);

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::U32(0xDEADBEEF));
    }

    #[test]
    fn test_decode_reduce_u32_collects_values() {
        let dtype = DataType::U32;
        let values = [0u32, 1, 256, 65_536, 0xDEADBEEF, u32::MAX];

        let encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|v| {
                let mut buf = Vec::new();
                v.encode_into(&mut buf);
                buf
            })
            .collect();
        let slices: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();

        let (decoded, count): (Vec<DecodedValue<'_>>, usize) =
            decode_reduce(slices.iter().copied(), &dtype, Vec::new(), |mut acc, dv| {
                acc.push(dv);
                acc
            })
            .unwrap();

        assert_eq!(count, values.len());
        assert_eq!(
            decoded,
            values
                .iter()
                .copied()
                .map(DecodedValue::U32)
                .collect::<Vec<_>>()
        );
    }

    /* ---------------- U64 tests ---------------- */

    #[test]
    fn test_u64_encode_decode_roundtrip() {
        let dtype = DataType::U64;

        let mut buf = Vec::new();
        12345u64.encode_into(&mut buf);

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::U64(12345));
    }

    #[test]
    fn test_decode_reduce_u64_collects_values() {
        let dtype = DataType::U64;
        let values = [10u64, 20, 30];

        let encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|v| {
                let mut buf = Vec::new();
                v.encode_into(&mut buf);
                buf
            })
            .collect();
        let slices: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();

        let (decoded, count): (Vec<DecodedValue<'_>>, usize) =
            decode_reduce(slices.iter().copied(), &dtype, Vec::new(), |mut acc, dv| {
                acc.push(dv);
                acc
            })
            .unwrap();

        assert_eq!(count, values.len());
        assert_eq!(
            decoded,
            vec![
                DecodedValue::U64(10),
                DecodedValue::U64(20),
                DecodedValue::U64(30)
            ]
        );
    }

    /* ---------------- I64 tests ---------------- */

    #[test]
    fn test_i64_encode_decode_roundtrip() {
        let dtype = DataType::I64;

        let mut buf = Vec::new();
        (-12345i64).encode_into(&mut buf);

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::I64(-12345));
    }

    #[test]
    fn test_decode_reduce_i64_collects_values_and_order() {
        let dtype = DataType::I64;
        // Include negatives, zero, positives, and some wide-range values
        let values = [i64::MIN + 1, -10, -1, 0, 1, 10, 123456789, i64::MAX];

        let encoded: Vec<Vec<u8>> = values
            .iter()
            .map(|v| {
                let mut buf = Vec::new();
                v.encode_into(&mut buf);
                buf
            })
            .collect();

        // Check decode_reduce path
        let slices: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();
        let (decoded, count): (Vec<DecodedValue<'_>>, usize) =
            decode_reduce(slices.iter().copied(), &dtype, Vec::new(), |mut acc, dv| {
                acc.push(dv);
                acc
            })
            .unwrap();

        assert_eq!(count, values.len());
        assert_eq!(
            decoded,
            values
                .iter()
                .copied()
                .map(DecodedValue::I64)
                .collect::<Vec<_>>()
        );

        // Also verify lex order == numeric order for the encoded bytes
        let mut bytes_sorted = encoded.clone();
        bytes_sorted.sort(); // bytewise lex
        let decoded_sorted: Vec<i64> = bytes_sorted
            .iter()
            .map(|b| match decode_value(b, &dtype).unwrap() {
                DecodedValue::I64(x) => x,
                _ => unreachable!(),
            })
            .collect();

        let mut numeric_sorted = values.to_vec();
        numeric_sorted.sort();
        assert_eq!(decoded_sorted, numeric_sorted);
    }

    /* ---------------- Bool tests ---------------- */

    #[test]
    fn test_bool_encode_decode_roundtrip() {
        let dtype = DataType::Bool;

        let mut buf = Vec::new();
        true.encode_into(&mut buf);

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::Bool(true));
    }

    /* ---------------- Bytes tests ---------------- */

    #[test]
    fn test_bytes_encode_decode_roundtrip() {
        let dtype = DataType::Bytes;

        let payload = vec![0x00u8, 0x7E, 0x00, 0xFF];

        // Encode via public API
        let mut buf = Vec::new();
        encode_value(DecodedValue::Bytes(&payload), &dtype, &mut buf).unwrap();

        // Decode back
        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::Bytes(&payload[..]));
    }

    #[test]
    fn test_bytes_decode_reduce_collects_values() {
        let dtype = DataType::Bytes;
        let payload = vec![0xAA, 0xBB, 0xCC];

        let mut buf = Vec::new();
        encode_value(DecodedValue::Bytes(&payload), &dtype, &mut buf).unwrap();
        let slices = [buf.as_slice(), buf.as_slice(), buf.as_slice()];

        let (collected, count): (Vec<DecodedValue<'_>>, usize) =
            decode_reduce(slices.iter().copied(), &dtype, Vec::new(), |mut acc, dv| {
                acc.push(dv);
                acc
            })
            .unwrap();

        assert_eq!(count, 3);
        assert_eq!(collected.len(), 3);
        for dv in collected {
            assert_eq!(dv, DecodedValue::Bytes(&payload[..]));
        }
    }

    /* ---------------- CategoryId tests ---------------- */

    #[test]
    fn test_categoryid_encode_decode_roundtrip() {
        let dtype = DataType::CategoryId;

        let mut buf = Vec::new();
        encode_value(DecodedValue::CategoryId(42), &dtype, &mut buf).unwrap();

        let dv = decode_value(&buf, &dtype).unwrap();
        assert_eq!(dv, DecodedValue::CategoryId(42));
    }
}
