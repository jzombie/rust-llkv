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
        (DataType::U64, DecodedValue::U64(x)) => {
            x.encode_into(out);
            Ok(())
        }
        (DataType::Bool, DecodedValue::Bool(b)) => {
            b.encode_into(out);
            Ok(())
        }
        (expected, DecodedValue::Str(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "Str",
        }),
        (expected, DecodedValue::U64(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "U64",
        }),
        (expected, DecodedValue::Bool(_)) => Err(EncodeError::TypeMismatch {
            expected: *expected,
            got: "Bool",
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
    }
}

// Public streaming helper. Avoids collecting and enum Vec overhead.
#[inline]
pub fn decode_for_each<'a, I, F>(
    inputs: I,
    dtype: &DataType,
    mut f: F,
) -> Result<usize, DecodeError>
where
    I: IntoIterator<Item = &'a [u8]>,
    F: FnMut(DecodedValue<'a>),
{
    let mut n = 0usize;
    match dtype {
        DataType::Utf8 => {
            for b in inputs {
                let s =
                    internal::Utf8CaseFold::decode_borrowed(b).ok_or(DecodeError::InvalidFormat)?;
                f(DecodedValue::Str(s));
                n += 1;
            }
        }
        DataType::U64 => {
            for b in inputs {
                let x = internal::BeU64::decode(b)?;
                f(DecodedValue::U64(x));
                n += 1;
            }
        }
        DataType::Bool => {
            for b in inputs {
                let x = internal::Bool::decode(b)?;
                f(DecodedValue::Bool(x));
                n += 1;
            }
        }
    }
    Ok(n)
}

// --- Example Test ---
#[cfg(test)]
mod tests {
    use super::*;
    use internal::EncodeInto;

    #[test]
    fn test_catalog_decoding() {
        // 1. Define the data types for our system directly.
        let user_name_dtype = DataType::Utf8;
        let user_id_dtype = DataType::U64;
        let is_active_dtype = DataType::Bool;

        // 2. Create some raw encoded data.
        let mut name_bytes = Vec::new();
        "Jeremy".encode_into(&mut name_bytes);

        let mut id_bytes = Vec::new();
        12345u64.encode_into(&mut id_bytes);

        let mut is_active_bytes = Vec::new();
        true.encode_into(&mut is_active_bytes);

        // 3. Use the bridge function to decode the data based on the data type.
        let decoded_name = decode_value(&name_bytes, &user_name_dtype).unwrap();
        let decoded_id = decode_value(&id_bytes, &user_id_dtype).unwrap();
        let decoded_is_active = decode_value(&is_active_bytes, &is_active_dtype).unwrap();

        // 4. Assert that we got the correct types and values back.
        assert_eq!(decoded_name, DecodedValue::Str("Jeremy"));
        assert_eq!(decoded_id, DecodedValue::U64(12345));
        assert_eq!(decoded_is_active, DecodedValue::Bool(true));

        // 5. Use the decoded values in a type-safe way.
        if let DecodedValue::Str(s) = decoded_name {
            assert!(s.starts_with('J'));
        } else {
            panic!("Expected a string!");
        }

        if let DecodedValue::Bool(b) = decoded_is_active {
            assert!(b);
        } else {
            panic!("Expected a bool!");
        }
    }

    #[test]
    fn test_decode_for_each() {
        let u64_dtype = DataType::U64;
        let values_in = vec![10u64, 20, 30];

        // 1. Encode multiple values
        let encoded_values: Vec<Vec<u8>> = values_in
            .iter()
            .map(|v| {
                let mut buf = Vec::new();
                v.encode_into(&mut buf);
                buf
            })
            .collect();

        // 2. Decode them in a batch (streaming, then collect)
        let mut decoded_out = Vec::new();
        let encoded_slices: Vec<&[u8]> = encoded_values.iter().map(|v| v.as_slice()).collect();

        let count = decode_for_each(encoded_slices, &u64_dtype, |dv| {
            decoded_out.push(dv);
        })
        .unwrap();

        // 3. Assert results
        assert_eq!(count, 3);
        assert_eq!(
            decoded_out,
            vec![
                DecodedValue::U64(10),
                DecodedValue::U64(20),
                DecodedValue::U64(30)
            ]
        );
    }
}
