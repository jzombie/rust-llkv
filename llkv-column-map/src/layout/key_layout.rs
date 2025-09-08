use crate::types::{ByteOffset, ByteWidth, IndexEntryCount};
use bitcode::{Decode, Encode};

// ── Logical key layout (how to slice logical_key_bytes) ──────────────────────
#[derive(Debug, Clone, Encode, Decode)]
pub enum KeyLayout {
    /// Every *logical key* is exactly `width` bytes → slice i is [i*w .. (i+1)*w).
    FixedWidth { width: ByteWidth },

    /// Variable width logical keys. Prefix sum of key byte offsets:
    /// key_i is [key_offsets\[i\], key_offsets\[i+1\]).
    Variable { key_offsets: Vec<IndexEntryCount> }, // len = n_entries + 1
}

impl KeyLayout {
    #[inline]
    pub fn binary_search_key_with_layout(
        bytes: &[u8],
        layout: &KeyLayout,
        n_entries: usize,
        target: &[u8],
    ) -> Option<usize> {
        let mut lo = 0usize;
        let mut hi = n_entries; // exclusive
        while lo < hi {
            let mid = (lo + hi) / 2;
            let k = KeyLayout::slice_key_by_layout(bytes, layout, mid);
            match k.cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    #[inline]
    pub fn slice_key_by_layout<'a>(bytes: &'a [u8], layout: &'a KeyLayout, i: usize) -> &'a [u8] {
        match layout {
            KeyLayout::FixedWidth { width } => {
                let w = *width as usize;
                let a = i * w;
                let b = a + w;
                &bytes[a..b]
            }
            KeyLayout::Variable { key_offsets } => {
                let a = key_offsets[i] as usize;
                let b = key_offsets[i + 1] as usize;
                &bytes[a..b]
            }
        }
    }

    // Pack keys choosing the most space-efficient representation.
    // - If all keys have the same non-zero length -> FixedWidth { width }
    // - Otherwise -> Variable { key_offsets }
    #[inline]
    pub fn pack_keys_with_layout<'a, T>(keys: &'a [T]) -> (Vec<u8>, KeyLayout)
    where
        T: AsRef<[u8]> + 'a,
    {
        // Create a Vec of references to sort without owning the data.
        let mut key_refs: Vec<&'a T> = keys.iter().collect();
        key_refs.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));

        let n = key_refs.len();

        // Fixed-width path (non-zero width, all equal)
        if n > 0 {
            let w = key_refs[0].as_ref().len();
            if w > 0 && key_refs.iter().all(|k| k.as_ref().len() == w) {
                let mut bytes = Vec::with_capacity(n * w);
                for k in &key_refs {
                    bytes.extend_from_slice(k.as_ref());
                }
                return (
                    bytes,
                    KeyLayout::FixedWidth {
                        width: w as ByteWidth,
                    },
                );
            }
        }

        // Variable-width path: pack bytes + offsets from the sorted references
        let total: usize = key_refs.iter().map(|k| k.as_ref().len()).sum();
        let mut bytes = Vec::with_capacity(total);
        let mut offs = Vec::with_capacity(n + 1);

        let mut acc: ByteOffset = 0;
        offs.push(acc);
        for k in key_refs {
            bytes.extend_from_slice(k.as_ref());
            acc = acc.wrapping_add(k.as_ref().len() as ByteOffset);
            offs.push(acc);
        }

        (
            bytes,
            KeyLayout::Variable {
                key_offsets: offs.into_iter().map(|v| v as IndexEntryCount).collect(),
            },
        )
    }
}
