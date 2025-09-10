use crate::types::{ByteOffset, ByteWidth, IndexEntryCount};
use bitcode::{Decode, Encode};

#[derive(Debug, Clone, Encode, Decode)]
pub enum KeyLayout {
    /// Every *logical key* is exactly `width` bytes → slice i is [i*w .. (i+1)*w).
    FixedWidth { width: ByteWidth },

    /// Variable-width logical keys. Prefix sum of key byte offsets:
    /// key_i is \[key_offsets\[i\], key_offsets[i+1]).
    Variable { key_offsets: Vec<IndexEntryCount> }, // len = n_entries + 1
}

impl KeyLayout {
    /// Returns the position of `target` if present, else None.
    /// Fast path: specialize once per layout and avoid per-iteration matches.
    #[inline]
    pub fn binary_search_key_with_layout(
        bytes: &[u8],
        layout: &KeyLayout,
        n_entries: usize,
        target: &[u8],
    ) -> Option<usize> {
        match layout {
            KeyLayout::FixedWidth { width } => {
                let w = *width as usize;
                if w != 0 && target.len() != w {
                    return None;
                }
                binary_search_fixed(bytes, w, n_entries, target)
            }
            KeyLayout::Variable { key_offsets } => {
                debug_assert_eq!(key_offsets.len(), n_entries + 1);
                binary_search_var(bytes, key_offsets, n_entries, target)
            }
        }
    }

    /// Lower bound: first index `i` where key_i >= probe.
    #[inline]
    pub fn lower_bound(bytes: &[u8], layout: &KeyLayout, n_entries: usize, probe: &[u8]) -> usize {
        match layout {
            KeyLayout::FixedWidth { width } => lb_fixed(bytes, *width as usize, n_entries, probe),
            KeyLayout::Variable { key_offsets } => {
                debug_assert_eq!(key_offsets.len(), n_entries + 1);
                lb_var(bytes, key_offsets, n_entries, probe)
            }
        }
    }

    /// Upper bound: if `include_equal == true`, returns first i with key_i > probe;
    /// otherwise returns first i with key_i >= probe.
    #[inline]
    pub fn upper_bound(
        bytes: &[u8],
        layout: &KeyLayout,
        n_entries: usize,
        probe: &[u8],
        include_equal: bool,
    ) -> usize {
        match layout {
            KeyLayout::FixedWidth { width } => {
                ub_fixed(bytes, *width as usize, n_entries, probe, include_equal)
            }
            KeyLayout::Variable { key_offsets } => {
                debug_assert_eq!(key_offsets.len(), n_entries + 1);
                ub_var(bytes, key_offsets, n_entries, probe, include_equal)
            }
        }
    }

    /// Slice the i-th logical key according to the layout.
    /// Always-inline; uses unchecked slicing in release for zero bounds overhead.
    #[inline(always)]
    pub fn slice_key_by_layout<'a>(bytes: &'a [u8], layout: &'a KeyLayout, i: usize) -> &'a [u8] {
        match layout {
            KeyLayout::FixedWidth { width } => {
                let w = *width as usize;
                let a = i * w;
                let b = a + w;
                debug_assert!(b <= bytes.len());
                unsafe { bytes.get_unchecked(a..b) }
            }
            KeyLayout::Variable { key_offsets } => {
                let a = key_offsets[i] as usize;
                let b = key_offsets[i + 1] as usize;
                debug_assert!(a <= b && b <= bytes.len());
                unsafe { bytes.get_unchecked(a..b) }
            }
        }
    }

    /// Packs and sorts keys choosing the most space-efficient representation.
    #[inline]
    pub fn pack_keys_with_layout<'a, T>(keys: &'a [T]) -> (Vec<u8>, KeyLayout)
    where
        T: AsRef<[u8]> + 'a,
    {
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

        // Variable-width path: pack bytes + offsets
        let total: usize = key_refs.iter().map(|k| k.as_ref().len()).sum();
        let mut bytes = Vec::with_capacity(total);
        let mut offs = Vec::with_capacity(n + 1);

        let mut acc: ByteOffset = 0;
        offs.push(acc);
        for k in key_refs {
            let s = k.as_ref();
            bytes.extend_from_slice(s);
            acc = acc.wrapping_add(s.len() as ByteOffset);
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

// -------- layout-specialized searches (fixed / variable) --------

#[inline(always)]
fn binary_search_fixed(bytes: &[u8], w: usize, n: usize, target: &[u8]) -> Option<usize> {
    if w == 0 || n == 0 {
        return None;
    }
    debug_assert!(n * w <= bytes.len());

    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = mid * w;
        let b = a + w;
        let k = unsafe { bytes.get_unchecked(a..b) };
        match k.cmp(target) {
            core::cmp::Ordering::Less => lo = mid + 1,
            core::cmp::Ordering::Greater => hi = mid,
            core::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}

#[inline(always)]
fn binary_search_var(
    bytes: &[u8],
    offs: &[IndexEntryCount],
    n: usize,
    target: &[u8],
) -> Option<usize> {
    debug_assert_eq!(offs.len(), n + 1);
    debug_assert!((offs[n] as usize) <= bytes.len());

    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = offs[mid] as usize;
        let b = offs[mid + 1] as usize;
        let k = unsafe { bytes.get_unchecked(a..b) };
        match k.cmp(target) {
            core::cmp::Ordering::Less => lo = mid + 1,
            core::cmp::Ordering::Greater => hi = mid,
            core::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}

#[inline(always)]
fn lb_fixed(bytes: &[u8], w: usize, n: usize, probe: &[u8]) -> usize {
    if w == 0 || n == 0 {
        return 0;
    }
    debug_assert!(n * w <= bytes.len());

    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = mid * w;
        let b = a + w;
        let k = unsafe { bytes.get_unchecked(a..b) };
        if k < probe { lo = mid + 1 } else { hi = mid }
    }
    lo
}

#[inline(always)]
fn lb_var(bytes: &[u8], offs: &[IndexEntryCount], n: usize, probe: &[u8]) -> usize {
    debug_assert_eq!(offs.len(), n + 1);
    debug_assert!((offs[n] as usize) <= bytes.len());

    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = offs[mid] as usize;
        let b = offs[mid + 1] as usize;
        let k = unsafe { bytes.get_unchecked(a..b) };
        if k < probe { lo = mid + 1 } else { hi = mid }
    }
    lo
}

#[inline(always)]
fn ub_fixed(bytes: &[u8], w: usize, n: usize, probe: &[u8], include_equal: bool) -> usize {
    if w == 0 || n == 0 {
        return 0;
    }
    debug_assert!(n * w <= bytes.len());

    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = mid * w;
        let b = a + w;
        let k = unsafe { bytes.get_unchecked(a..b) };
        let go_right = if include_equal { k <= probe } else { k < probe };
        if go_right { lo = mid + 1 } else { hi = mid }
    }
    lo
}

#[inline(always)]
fn ub_var(
    bytes: &[u8],
    offs: &[IndexEntryCount],
    n: usize,
    probe: &[u8],
    include_equal: bool,
) -> usize {
    debug_assert_eq!(offs.len(), n + 1);
    debug_assert!((offs[n] as usize) <= bytes.len());

    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = offs[mid] as usize;
        let b = offs[mid + 1] as usize;
        let k = unsafe { bytes.get_unchecked(a..b) };
        let go_right = if include_equal { k <= probe } else { k < probe };
        if go_right { lo = mid + 1 } else { hi = mid }
    }
    lo
}
