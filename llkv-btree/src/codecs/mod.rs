//! Codecs: key and id encoding/decoding and order-compat comparisons.

use crate::errors::Error;
use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::Hash;
use std::str;

// --- Traits ---

// TODO: Rename to connotate "physical" & "logical" keys? This would also involve renaming of `KeyCodec` and `IdCodec`
pub trait KeyCodec {
    type Key: Clone + Ord + Debug;
    fn encoded_len(key: &Self::Key) -> usize;
    fn encode_into(key: &Self::Key, out: &mut Vec<u8>);
    fn decode_from(bytes: &[u8]) -> Result<Self::Key, Error>;
    fn compare_encoded(encoded: &[u8], key: &Self::Key) -> Ordering;
    fn cmp_enc_vs_enc(a: &[u8], b: &[u8]) -> Ordering;
}

pub trait IdCodec {
    type Id: Clone + Eq + Ord + Hash + Debug;
    fn encoded_len(id: &Self::Id) -> usize;
    fn encode_into(id: &Self::Id, out: &mut Vec<u8>);
    fn decode_from(bytes: &[u8]) -> Result<(Self::Id, usize), Error>;
}

// --- Generic Integer Codec ---

/// A generic codec for any integer type that can be represented as big-endian bytes.
#[derive(Debug)]
pub struct BigEndianKeyCodec<T>(std::marker::PhantomData<T>);
pub struct BigEndianIdCodec<T>(std::marker::PhantomData<T>);

// Define the necessary trait bounds for integers we can handle.
pub trait Int: Sized + Copy + Ord + Debug + Hash {
    type Bytes: AsRef<[u8]> + AsMut<[u8]> + for<'a> TryFrom<&'a [u8]>;
    fn to_be_bytes(self) -> Self::Bytes;
    fn from_be_bytes(bytes: Self::Bytes) -> Self;
}

impl Int for u32 {
    type Bytes = [u8; 4];
    fn to_be_bytes(self) -> Self::Bytes {
        self.to_be_bytes()
    }
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }
}
impl Int for u64 {
    type Bytes = [u8; 8];
    fn to_be_bytes(self) -> Self::Bytes {
        self.to_be_bytes()
    }
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }
}
impl Int for u128 {
    type Bytes = [u8; 16];
    fn to_be_bytes(self) -> Self::Bytes {
        self.to_be_bytes()
    }
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }
}

// --- Implementations ---

impl<T: Int> KeyCodec for BigEndianKeyCodec<T>
where
    T::Bytes: Debug,
    for<'a> <T::Bytes as TryFrom<&'a [u8]>>::Error: Debug,
{
    type Key = T;

    fn encoded_len(_: &Self::Key) -> usize {
        std::mem::size_of::<T>()
    }
    fn encode_into(key: &Self::Key, out: &mut Vec<u8>) {
        out.extend_from_slice(key.to_be_bytes().as_ref());
    }
    fn decode_from(bytes: &[u8]) -> Result<Self::Key, Error> {
        let array = bytes
            .try_into()
            .map_err(|_| Error::Corrupt("invalid int bytes"))?;
        Ok(T::from_be_bytes(array))
    }
    fn compare_encoded(encoded: &[u8], key: &Self::Key) -> Ordering {
        let decoded = T::from_be_bytes(encoded.try_into().unwrap());
        decoded.cmp(key)
    }
    fn cmp_enc_vs_enc(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }
}

impl<T: Int> IdCodec for BigEndianIdCodec<T>
where
    T::Bytes: Debug,
    for<'a> <T::Bytes as TryFrom<&'a [u8]>>::Error: Debug,
{
    type Id = T;

    fn encoded_len(_: &Self::Id) -> usize {
        std::mem::size_of::<T>()
    }
    fn encode_into(id: &Self::Id, out: &mut Vec<u8>) {
        out.extend_from_slice(id.to_be_bytes().as_ref());
    }
    fn decode_from(bytes: &[u8]) -> Result<(Self::Id, usize), Error> {
        let len = std::mem::size_of::<T>();
        let array = bytes[..len]
            .try_into()
            .map_err(|_| Error::Corrupt("invalid int bytes"))?;
        let val = T::from_be_bytes(array);
        Ok((val, len))
    }
}

/// A codec for `String` keys that uses UTF-8 byte representation.
pub struct StringKeyCodec;
impl KeyCodec for StringKeyCodec {
    type Key = String;
    fn encoded_len(key: &Self::Key) -> usize {
        key.len()
    }
    fn encode_into(key: &Self::Key, out: &mut Vec<u8>) {
        out.extend_from_slice(key.as_bytes());
    }
    fn decode_from(bytes: &[u8]) -> Result<Self::Key, Error> {
        str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|_| Error::Corrupt("invalid utf8"))
    }
    fn compare_encoded(encoded: &[u8], key: &Self::Key) -> Ordering {
        str::from_utf8(encoded).unwrap().cmp(key)
    }
    fn cmp_enc_vs_enc(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }
}

#[inline]
pub fn read_u32_at(b: &[u8], pos: usize) -> (u32, usize) {
    let n = u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap());
    (n, pos + 4)
}
#[inline]
pub fn push_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}
