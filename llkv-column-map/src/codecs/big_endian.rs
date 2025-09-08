//! Big-endian encoding helpers with explicit return types.
//!
//! Goals:
//! - Prefer zero-allocation array helpers in hot paths.
//! - Provide owned-Vec variants for call sites that need `Vec<u8>`.
//! - Keep old names as aliases so existing code keeps compiling.
//!
//! Naming rules:
//! - `*_be_arr` returns a fixed `[u8; N]` (no allocation).
//! - `*_be_vec` returns an owned `Vec<u8>`.
//! - `clone_bytes` clones a `&[u8]` into `Vec<u8>`.
//! - `utf8_to_vec` converts `&str` into UTF-8 bytes `Vec<u8>`.

#![forbid(unsafe_code)]

/// Encode `u64` as big-endian fixed array (no allocation).
#[inline]
pub fn u64_be_arr(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

/// Encode `u32` as big-endian fixed array (no allocation).
#[inline]
pub fn u32_be_arr(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// Encode `u16` as big-endian fixed array (no allocation).
#[inline]
pub fn u16_be_arr(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

/// Encode `u64` as big-endian owned bytes.
#[inline]
pub fn u64_be_vec(v: u64) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

/// Encode `u32` as big-endian owned bytes.
#[inline]
pub fn u32_be_vec(v: u32) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

/// Encode `u16` as big-endian owned bytes.
#[inline]
pub fn u16_be_vec(v: u16) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

/// Clone borrowed bytes to `Vec<u8>`.
#[inline]
pub fn clone_bytes(b: &[u8]) -> Vec<u8> {
    b.to_vec()
}

/// Convert `&str` to owned UTF-8 bytes.
#[inline]
pub fn utf8_to_vec(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}
