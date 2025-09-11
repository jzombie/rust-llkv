//! Defines the core, low-level types used throughout the storage engine.
//!
//! This module provides type aliases for fundamental concepts like physical keys,
//! logical field IDs, and various byte-level measurements.
//!
//! ## A Note on Endianness and Sort Order
//!
//! You will notice that this system consistently uses **big-endian** encoding for
//! all numeric types that are part of a `LogicalKeyBytes` payload. This is a
//! deliberate and critical design choice.
//!
//! The storage engine's core is built on a simple, fast, and generic
//! **lexicographical (byte-wise) sort**. To function correctly, all data types—strings,
//! numbers, and composites—must sort correctly using the same universal rule.
//!
//! - **Strings** naturally sort lexicographically.
//! - **Little-Endian** numbers (the native format for most CPUs) **do not**. For example,
//!   as bytes, `256` would sort before `1`.
//! - **Big-Endian** numbers **do**. Their byte order matches their numeric order, allowing
//!   them to be sorted correctly with the same logic as strings.
//!
//! This requires a fast byte-swap when decoding numbers for computation, but this
//! small, one-time cost is far outweighed by the massive simplification and
//! performance gain of a unified, type-agnostic sorting and searching core.

pub mod internal;
use crate::internal::Codec;

// --- Public-Facing Metadata Enums ---

/// A tag representing the physical data type for a piece of metadata.
///
/// This is a simple, C-like enum that is cheap to store and copy. Its only
/// purpose is to act as a label for the underlying storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Indicates the data is a case-insensitive, order-preserving string.
    Utf8CaseFold,
    /// Indicates the data is a big-endian 64-bit unsigned integer.
    BeU64,
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

/// Bridges the runtime metadata (`DataType`) to the high-performance,
/// statically-dispatched codec functions from the `internal` module.
///
/// This function is the key to the whole pattern: it looks at the tag once,
/// then calls the specific, hyper-optimized function for that type.
pub fn decode_value<'a>(bytes: &'a [u8], dtype: &DataType) -> Option<DecodedValue<'a>> {
    match dtype {
        DataType::Utf8CaseFold => {
            // Statically calls the allocation-free borrowed decode.
            internal::Utf8CaseFold::decode_borrowed(bytes).map(DecodedValue::Str)
        }
        DataType::BeU64 => {
            // Statically calls the optimized integer decode.
            Some(DecodedValue::U64(internal::BeU64::decode(bytes)))
        }
        DataType::Bool => {
            // Statically calls the boolean decode.
            Some(DecodedValue::Bool(internal::Bool::decode(bytes)))
        }
    }
}

// --- Example Test ---
#[cfg(test)]
mod tests {
    use super::*;
    use internal::EncodeInto;

    #[test]
    fn test_catalog_decoding() {
        // 1. Define the data types for our system directly.
        let user_name_dtype = DataType::Utf8CaseFold;
        let user_id_dtype = DataType::BeU64;
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
}
