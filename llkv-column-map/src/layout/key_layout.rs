use crate::types::{ByteOffset, ByteWidth, IndexEntryCount};

/// Layout for logical keys inside a single contiguous byte buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyLayout {
    /// Keys are fixed width.
    FixedWidth { width: ByteWidth },

    /// Keys are variable width with prefix-sum offsets (len = n + 1).
    Variable { key_offsets: Vec<IndexEntryCount> },
}

#[inline(always)]
fn lb_fixed(bytes: &[u8], width: usize, n: usize, probe: &[u8], include_equal: bool) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = mid * width;
        let b = a + width;
        let k = unsafe { bytes.get_unchecked(a..b) };
        let go_right = if include_equal { k < probe } else { k <= probe };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline(always)]
fn ub_fixed(bytes: &[u8], width: usize, n: usize, probe: &[u8], include_equal: bool) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let a = mid * width;
        let b = a + width;
        let k = unsafe { bytes.get_unchecked(a..b) };
        let go_right = if include_equal { k <= probe } else { k < probe };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline(always)]
fn lb_var(
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
        let go_right = if include_equal { k < probe } else { k <= probe };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
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
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Lower bound index in [0, n] for `probe`.
#[inline(always)]
pub fn lower_bound(bytes: &[u8], layout: &KeyLayout, n: usize, probe: &[u8]) -> usize {
    match layout {
        KeyLayout::FixedWidth { width } => lb_fixed(bytes, *width as usize, n, probe, false),
        KeyLayout::Variable { key_offsets } => lb_var(bytes, key_offsets, n, probe, false),
    }
}

/// Upper bound index in [0, n] for `probe`.
#[inline(always)]
pub fn upper_bound(bytes: &[u8], layout: &KeyLayout, n: usize, probe: &[u8]) -> usize {
    match layout {
        KeyLayout::FixedWidth { width } => ub_fixed(bytes, *width as usize, n, probe, true),
        KeyLayout::Variable { key_offsets } => ub_var(bytes, key_offsets, n, probe, true),
    }
}

/// Return the key slice for entry `i` using `layout`.
#[inline(always)]
pub fn slice_key_by_layout<'a>(bytes: &'a [u8], layout: &KeyLayout, i: usize) -> &'a [u8] {
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

/// Pack keys and return (contiguous bytes, layout).
pub fn pack_keys_with_layout(keys: &[Vec<u8>]) -> (Vec<u8>, KeyLayout) {
    if keys.is_empty() {
        return (Vec::new(), KeyLayout::FixedWidth { width: 0 });
    }

    let w = keys[0].len();
    let all_fixed = keys.iter().all(|k| k.len() == w);

    if all_fixed {
        let mut out = Vec::with_capacity(w * keys.len());
        for k in keys {
            out.extend_from_slice(k);
        }
        (
            out,
            KeyLayout::FixedWidth {
                width: w as ByteWidth,
            },
        )
    } else {
        let mut offs = Vec::with_capacity(keys.len() + 1);
        offs.push(0 as IndexEntryCount);
        let mut out = Vec::new();
        for k in keys {
            out.extend_from_slice(k);
            let next = (offs.last().unwrap() as &IndexEntryCount) + (k.len() as IndexEntryCount);
            offs.push(next);
        }
        (out, KeyLayout::Variable { key_offsets: offs })
    }
}
