//! Manual little-endian codecs for fixed-width structs.

#![allow(dead_code)]

// Convenience helpers for appending to Vecs and reading with an advancing offset.

#[inline(always)]
pub(crate) fn write_u32_le(dst: &mut Vec<u8>, v: u32) {
    dst.extend_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub(crate) fn write_u64_le(dst: &mut Vec<u8>, v: u64) {
    dst.extend_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub(crate) fn read_u32_le(src: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(src[*o..*o + 4].try_into().unwrap());
    *o += 4;
    v
}

#[inline(always)]
pub(crate) fn read_u64_le(src: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(src[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}
