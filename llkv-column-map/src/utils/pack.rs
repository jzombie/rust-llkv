use crate::layout::KeyLayout;
use crate::types::{ByteOffset, ByteWidth, IndexEntryCount};

// Pack keys choosing the most space-efficient representation.
// - If all keys have the same non-zero length -> FixedWidth { width }
// - Otherwise -> Variable { key_offsets }
#[inline]
pub fn pack_keys_with_layout(mut keys: Vec<Vec<u8>>) -> (Vec<u8>, KeyLayout) {
    // sort exactly once here
    keys.sort_unstable();
    let n = keys.len();

    // Fixed-width path (non-zero width, all equal)
    if n > 0 {
        let w = keys[0].len();
        if w > 0 && keys.iter().all(|k| k.len() == w) {
            let mut bytes = Vec::with_capacity(n * w);
            for k in &keys {
                bytes.extend_from_slice(k);
            }
            return (
                bytes,
                KeyLayout::FixedWidth {
                    width: w as ByteWidth,
                },
            );
        }
    }

    // Variable-width path (or empty): pack bytes + offsets
    // (Inline the helper or call concat_keys_and_offsets_sorted(keys))
    let total: usize = keys.iter().map(|k| k.len()).sum();
    let mut bytes = Vec::with_capacity(total);
    let mut offs = Vec::with_capacity(n + 1);

    let mut acc: ByteOffset = 0;
    offs.push(acc);
    for k in keys {
        bytes.extend_from_slice(&k);
        acc = acc.wrapping_add(k.len() as ByteOffset);
        offs.push(acc);
    }

    (
        bytes,
        KeyLayout::Variable {
            key_offsets: offs.into_iter().map(|v| v as IndexEntryCount).collect(),
        },
    )
}
