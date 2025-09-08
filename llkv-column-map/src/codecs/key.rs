use crate::types::LogicalKeyBytes;

/// Encode a `u64` so lexicographic order == numeric order.
#[inline]
pub fn u64_be(v: u64) -> LogicalKeyBytes {
    v.to_be_bytes().to_vec()
}

/// Encode a `u32` so lexicographic order == numeric order.
#[inline]
pub fn u32_be(v: u32) -> LogicalKeyBytes {
    v.to_be_bytes().to_vec()
}

/// Pass-through bytes. Clones into an owned `LogicalKeyBytes`.
#[inline]
pub fn bytes<B: AsRef<[u8]>>(b: B) -> LogicalKeyBytes {
    b.as_ref().to_vec()
}

/// UTF-8 string → bytes (owned).
#[inline]
pub fn utf8(s: &str) -> LogicalKeyBytes {
    s.as_bytes().to_vec()
}

// Optional zero-allocation helpers if/when you want them in hot paths:

/// Zero-alloc `u64` encoder → fixed array; useful if callers build the Vec once.
#[inline]
pub fn u64_be_arr(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

/// Zero-alloc `u32` encoder → fixed array.
#[inline]
pub fn u32_be_arr(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}
