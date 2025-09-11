pub mod internal;
use crate::internal::{BeU64, Bool, Codec, EncodeInto, Utf8CaseFold};

pub mod errors;
pub use errors::*;

// TODO: Add category type

// --- Public-Facing Metadata Enums ---

/// A tag representing the physical data type for a piece of metadata.
///
/// This is a simple, C-like enum that is cheap to store and copy.
/// Its only purpose is to act as a label for the underlying storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Indicates the data is a case-insensitive, order-preserving string.
    Utf8,
    /// Indicates the data is a big-endian 64-bit unsigned integer.
    U64,
    /// Indicates the data is a boolean value (0 for false, 1 for true).
    Bool,
    //
    /// Indicates the data is a collection of bytes.
    Bytes,
}

/// A generic enum to hold any possible value decoded from storage.
///
/// It uses a lifetime (`'a`) to allow for zero-allocation borrowing of strings,
/// preserving the performance of the internal codecs.
#[derive(Debug, PartialEq)]
pub enum DecodedValue<'a> {
    Str(&'a str),
    U64(u64),
    Bool(bool),
    Bytes(&'a [u8]),
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
        (DataType::U64, DecodedValue::U64(x)) => {
            x.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::U64(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "U64",
        }),
        (DataType::Bool, DecodedValue::Bool(b)) => {
            b.encode_into(out);
            Ok(())
        }

        (expected, DecodedValue::Bool(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "Bool",
        }),
        (DataType::Bytes, DecodedValue::Bytes(b)) => {
            b.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::Bytes(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "Bytes",
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
        DataType::U64 => BeU64::decode(bytes).ok().map(DecodedValue::U64),
        DataType::Bool => Bool::decode(bytes).ok().map(DecodedValue::Bool),
        DataType::Bytes => internal::Bytes::decode_borrowed(bytes).map(DecodedValue::Bytes),
    }
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
        DataType::U64 => {
            for b in inputs {
                let x = BeU64::decode(b)?;
                acc = f(acc, DecodedValue::U64(x));
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
}
