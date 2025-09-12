//! Order-key encoders: map native/LE-encoded values to lexicographically comparable bytes.
//!
//! These helpers produce fixed-width big-endian byte sequences such that
//! memcmp order matches the natural numeric order. They are intended for
//! building index-only sort keys while storing payloads in little-endian.
//!
//! Baseline usage in this repo: the writer can produce per-entry sort keys
//! using these functions and persist them inside `ValueIndex` so that the
//! read path never needs to touch payload bytes for ordering.

#![forbid(unsafe_code)]

#[inline]
pub fn u64_le_to_sort8(le_bytes: [u8; 8]) -> [u8; 8] {
    // reverse to BE representation
    let mut out = le_bytes;
    out.reverse();
    out
}

#[inline]
pub fn u32_le_to_sort4(le_bytes: [u8; 4]) -> [u8; 4] {
    let mut out = le_bytes;
    out.reverse();
    out
}

#[inline]
pub fn u16_le_to_sort2(le_bytes: [u8; 2]) -> [u8; 2] {
    let mut out = le_bytes;
    out.reverse();
    out
}

#[inline]
pub fn u8_to_sort1(b: u8) -> [u8; 1] {
    [b]
}

#[inline]
pub fn i64_le_to_sort8(le_bytes: [u8; 8]) -> [u8; 8] {
    let x = u64::from_le_bytes(le_bytes) ^ 0x8000_0000_0000_0000u64;
    x.to_be_bytes()
}

#[inline]
pub fn i32_le_to_sort4(le_bytes: [u8; 4]) -> [u8; 4] {
    let x = (u32::from_le_bytes(le_bytes) ^ 0x8000_0000u32).to_be_bytes();
    x
}

#[inline]
pub fn i16_le_to_sort2(le_bytes: [u8; 2]) -> [u8; 2] {
    let v = u16::from_le_bytes(le_bytes) ^ 0x8000u16;
    v.to_be_bytes()
}

#[inline]
pub fn i8_to_sort1(b: i8) -> [u8; 1] {
    [(b as u8) ^ 0x80]
}

#[inline]
pub fn f64_le_to_sort8(le_bytes: [u8; 8]) -> [u8; 8] {
    let mut bits = u64::from_le_bytes(le_bytes);
    // Map IEEE754 to lex order: negatives inverted, positives flip sign bit
    if bits & 0x8000_0000_0000_0000u64 != 0 {
        bits = !bits;
    } else {
        bits ^= 0x8000_0000_0000_0000u64;
    }
    bits.to_be_bytes()
}

#[inline]
pub fn f32_le_to_sort4(le_bytes: [u8; 4]) -> [u8; 4] {
    let mut bits = u32::from_le_bytes(le_bytes);
    if bits & 0x8000_0000u32 != 0 {
        bits = !bits;
    } else {
        bits ^= 0x8000_0000u32;
    }
    bits.to_be_bytes()
}

