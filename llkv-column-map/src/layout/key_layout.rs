use crate::types::{ByteWidth, IndexEntryCount};
use bitcode::{Decode, Encode};

// ── Logical key layout (how to slice logical_key_bytes) ──────────────────────
#[derive(Debug, Clone, Encode, Decode)]
pub enum KeyLayout {
    /// Every *logical key* is exactly `width` bytes → slice i is [i*w .. (i+1)*w).
    FixedWidth { width: ByteWidth },

    /// Variable width logical keys. Prefix sum of key byte offsets:
    /// key_i is [key_offsets[i], key_offsets[i+1]).
    Variable { key_offsets: Vec<IndexEntryCount> }, // len = n_entries + 1
}
