//! Manual little-endian codecs for fixed-width structs.

#![allow(dead_code)]

// Convenience helpers for appending to Vecs and reading with an advancing offset.
// Sortable conversions are kept here so consumers share consistent encodings.

#[inline(always)]
pub fn write_u32_le(dst: &mut Vec<u8>, v: u32) {
    dst.extend_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub fn write_u64_le(dst: &mut Vec<u8>, v: u64) {
    dst.extend_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub fn read_u32_le(src: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes(src[*o..*o + 4].try_into().unwrap());
    *o += 4;
    v
}

#[inline(always)]
pub fn read_u64_le(src: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(src[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

#[inline(always)]
pub fn sortable_u64_from_f64(val: f64) -> u64 {
    let bits = val.to_bits();
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

#[inline(always)]
pub fn sortable_u64_from_f32(val: f32) -> u64 {
    sortable_u64_from_f64(val as f64)
}

#[inline(always)]
pub fn sortable_u64_from_i8(val: i8) -> u64 {
    ((val as u8) ^ 0x80) as u64
}

#[inline(always)]
pub fn sortable_u64_from_i16(val: i16) -> u64 {
    ((val as u16) ^ 0x8000) as u64
}

#[inline(always)]
pub fn sortable_u64_from_i32(val: i32) -> u64 {
    ((val as u32) ^ 0x8000_0000) as u64
}

#[inline(always)]
pub fn sortable_u64_from_i64(val: i64) -> u64 {
    (val as u64) ^ 0x8000_0000_0000_0000
}
