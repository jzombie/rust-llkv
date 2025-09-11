pub use crate::{DecodeError, EncodeError};

/// A marker trait for codecs that operate on fixed-size types.
/// This allows for specialized, high-performance slice-based operations and
/// compile-time checks.
pub trait FixedSizeCodec: Codec {
    /// The exact number of bytes for one encoded item.
    const ENCODED_SIZE: usize;
}

/// A zero-overhead codec API for a single logical type.
pub trait Codec {
    type Borrowed<'a>: ?Sized
    where
        Self: 'a;
    type Owned;

    fn encode_into(dst: &mut Vec<u8>, v: Self::Borrowed<'_>) -> Result<(), EncodeError>;

    fn decode(src: &[u8]) -> Result<Self::Owned, DecodeError>;

    /// Decodes a source byte slice directly into a destination slice of owned types.
    ///
    /// This is the high-performance path for preparing data for math kernels.
    /// The compiler will prevent this from being called on variable-size types
    /// that do not implement `FixedSizeCodec`.
    fn decode_slice(src: &[u8], dst: &mut [Self::Owned]) -> Result<usize, DecodeError>
    where
        Self: Sized + FixedSizeCodec,
    {
        // A generic, correct default implementation.
        // Individual codecs can provide a faster, specialized version.
        let n = core::cmp::min(src.len() / Self::ENCODED_SIZE, dst.len());
        for i in 0..n {
            let offset = i * Self::ENCODED_SIZE;
            let chunk = &src[offset..offset + Self::ENCODED_SIZE];
            dst[i] = Self::decode(chunk)?;
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
