pub use crate::{DecodeError, EncodeError};

/// A zero-overhead codec API for a single logical type.
///
/// - `Borrowed<'a>` is the borrowed view accepted by `encode_into`,
///   e.g. `&'a str` for text, `&'a u64` for u64.
/// - `Owned` is the type `decode` returns (e.g., `String`, `u64`).
pub trait Codec {
    type Borrowed<'a>: ?Sized
    where
        Self: 'a;
    type Owned;

    /// Append the encoded bytes for `v` into `dst`.
    fn encode_into(dst: &mut Vec<u8>, v: Self::Borrowed<'_>) -> Result<(), EncodeError>;

    /// Decode one value from `src` into the native owned type.
    fn decode(src: &[u8]) -> Result<Self::Owned, DecodeError>;

    /// Decode many items from an iterator of byte slices, appending to `out`.
    #[inline]
    fn decode_many_into<'a, I>(inputs: I, out: &mut Vec<Self::Owned>) -> Result<usize, DecodeError>
    where
        I: IntoIterator<Item = &'a [u8]>,
        Self: Sized,
    {
        let mut n = 0;
        for s in inputs {
            out.push(Self::decode(s)?);
            n += 1;
        }
        Ok(n)
    }

    // TODO: Scalar example of above TODO; could be replaced by more efficient SIMD versions
    // Also: For endian flips check out: https://docs.rs/zerocopy/latest/zerocopy/byteorder/big_endian/index.html
    //
    // Scalar reference path. Always available.
    // #[inline]
    // pub fn decode_be_u64_page(src: &[u8], out: &mut [u64]) -> Result<usize, DecodeError> {
    //     let n = core::cmp::min(src.len() / 8, out.len());
    //     let mut i = 0;
    //     while i < n {
    //         let off = i * 8;
    //         let a: [u8; 8] = src[off..off + 8].try_into().map_err(|_| DecodeError::NotEnoughData)?;
    //         out[i] = u64::from_be_bytes(a);
    //         i += 1;
    //     }
    //     Ok(n)
    // }
}
