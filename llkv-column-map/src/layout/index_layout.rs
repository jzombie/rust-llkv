use crate::types::ByteWidth;

/// Small, cheap summary of a segment's logical layout.
///
/// This is used for sizing, accounting, and selection. It does not
/// include any large vectors and is safe to keep in RAM.
#[derive(Clone, Debug)]
pub struct IndexLayoutInfo {
    /// "fixed" when values are fixed width; "variable" otherwise.
    pub kind: &'static str,

    /// Some(width) for fixed value layout; None for variable layout.
    pub fixed_width: Option<ByteWidth>,

    /// Total length in bytes of the concatenated logical key blob.
    pub key_bytes: usize,

    /// Bytes used by the key offset directory (0 for fixed keys).
    pub key_offs_bytes: usize,

    /// Bytes used by value meta (offset dir for variable, or 4 for
    /// fixed width to store the width).
    pub value_meta_bytes: usize,
}
