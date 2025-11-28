use crate::error::Error;

/// Result type alias used throughout LLKV.
///
/// This is a type alias for `std::result::Result<T, Error>`, providing a convenient
/// shorthand for functions that return LLKV errors. All LLKV operations that can fail
/// should return this type.
pub type Result<T> = std::result::Result<T, Error>;
