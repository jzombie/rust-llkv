use crate::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// The provided value does not match the requested DataType.
    TypeMismatch {
        expected: DataType,
        got: &'static str,
    },
}

/// Error type for decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The input slice does not contain enough bytes to decode a value.
    NotEnoughData,
    /// The byte format is invalid for the target type (e.g., invalid UTF-8).
    InvalidFormat,
}
