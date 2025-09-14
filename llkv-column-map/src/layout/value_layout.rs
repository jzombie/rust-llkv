use crate::types::{ByteWidth, IndexEntryCount};

/// Slicing recipe for values inside the data blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueLayout {
    /// Every value is exactly `width` bytes.
    FixedWidth { width: ByteWidth },

    /// Variable width values. Prefix sum of byte offsets into data blob.
    /// Slice i is [value_offsets[i], value_offsets[i + 1]).
    Variable { value_offsets: Vec<IndexEntryCount> }, // len = n + 1
}
