pub use crate::{DecodeError, EncodeError};

/// A zero-overhead codec API for a single logical type.
pub trait Codec {
    /// Fixed encoded width in bytes. Use `0` for variable-width codecs.
    const WIDTH: usize;

    type Borrowed<'a>: ?Sized
    where
        Self: 'a;
    type Owned;

    fn encode_into(dst: &mut Vec<u8>, v: Self::Borrowed<'_>) -> Result<(), EncodeError>;

    fn decode(src: &[u8]) -> Result<Self::Owned, DecodeError>;

    /// Bulk decode for fixed-width codecs.
    /// Implementors should provide a fast path; variable-width codecs should panic.
    fn decode_many_into(dst: &mut [Self::Owned], src: &[u8]) -> Result<(), DecodeError>;
}
