pub use crate::{DecodeError, EncodeError};

/// A zero-overhead codec API for a single logical type.
pub trait Codec {
    type Borrowed<'a>: ?Sized
    where
        Self: 'a;
    type Owned;

    fn encode_into(dst: &mut Vec<u8>, v: Self::Borrowed<'_>) -> Result<(), EncodeError>;

    fn decode(src: &[u8]) -> Result<Self::Owned, DecodeError>;
}
