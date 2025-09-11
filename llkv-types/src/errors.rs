use crate::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// The provided value does not match the requested DataType.
    TypeMismatch {
        expected: DataType,
        got: &'static str,
    },
}
