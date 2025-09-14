use crate::types::{ByteWidth, IndexEntryCount, PhysicalKey};

/// Value layout for a segment. Supports streaming.
///
/// Keep the legacy in-memory variant for compatibility, but prefer
/// `VariablePaged` for large segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueLayout {
    /// Every value is exactly `width` bytes.
    FixedWidth { width: ByteWidth },

    /// Legacy, in-memory variable-width directory (len = n + 1).
    Variable { value_offsets: Vec<IndexEntryCount> },

    /// Streamable variable-width directory stored as a blob of
    /// little-endian u32 prefix sums with length n+1.
    VariablePaged {
        offsets_pk: PhysicalKey,
        n_entries: IndexEntryCount,
    },
}
