use crate::types::{ByteWidth, IndexEntryCount, PhysicalKey};

/// Logical key layout for a segment. Supports streaming.
///
/// For compatibility, an in-memory variant remains, but all new code
/// should use `VariablePaged` to avoid materializing large vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyLayout {
    /// Every key is exactly `width` bytes.
    FixedWidth { width: ByteWidth },

    /// Legacy, in-memory variable-width directory (len = n + 1).
    Variable { key_offsets: Vec<IndexEntryCount> },

    /// Streamable variable-width directory stored as a blob:
    /// little-endian u32 prefix sums of length n+1.
    VariablePaged {
        offsets_pk: PhysicalKey,
        n_entries: IndexEntryCount,
    },
}

/// Compute the byte slice range for key `i` given a loader for two
/// adjacent offsets. This avoids pulling the whole directory.
#[inline(always)]
pub fn key_slice_range_from_offsets(
    i: usize,
    load_two: impl Fn(usize) -> (u32, u32),
) -> (usize, usize) {
    let (a, b) = load_two(i);
    (a as usize, b as usize)
}
