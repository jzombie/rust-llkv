//! Manual little-endian codecs for fixed-width structs.

#![allow(dead_code)]

#[inline(always)]
pub(crate) fn get_u32(d: &[u8]) -> u32 {
    u32::from_le_bytes(d.try_into().unwrap())
}

#[inline(always)]
pub(crate) fn put_u32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

#[inline(always)]
pub(crate) fn get_u64(d: &[u8]) -> u64 {
    u64::from_le_bytes(d.try_into().unwrap())
}

#[inline(always)]
pub(crate) fn put_u64(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}
