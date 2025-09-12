//! Defines the core, low-level types used throughout the storage engine.
//!
//! This module provides type aliases for fundamental concepts like physical keys,
//! logical field IDs, and various byte-level measurements.
//!
//! ## A Note on Endianness and Sort Order
//!
//! Payloads for all numerics/floats are stored in **little-endian (LE)**.
//! The column store builds value-order using a per-column `ValueOrderPolicy`
//! that maps LE bytes to numeric/IEEE order in registers; we never store
//! big-endian payloads or duplicate sort-key blobs in the index.

#![forbid(unsafe_code)]

//! Minimal, fast codecs + value-side `encode_into`.
//!
//!
//! - `Utf8CaseFold`: true Unicode case-insensitive, order-preserving UTF-8.
//!   Layout: \[folded_utf8\]\[original_utf8\]\[be32(len_folded)\]
//!
//! * Value-ordered stores compare the folded prefix first (CI), then original.
//!   * Decode is lossless; embedded NULs are fine.
//!
//! - `BeU64`: big-endian u64 so lexicographic order == numeric order.
//!
//!
//! - `EncodeInto` lets you write `v.encode_into(&mut buf)` using the default codec
//!
//! for that native type (String/&str → Utf8CaseFold, u64 → BeU64).

pub mod bytes;
pub use bytes::*;

pub mod codec;
pub use codec::*;

pub mod le_i64;
pub use le_i64::*;

pub mod le_u8;
pub use le_u8::*;

pub mod le_u16;
pub use le_u16::*;

pub mod le_u32;
pub use le_u32::*;

pub mod le_u64;
pub use le_u64::*;

pub mod bool;
pub use bool::*;

pub mod utf8_case_fold;
pub use utf8_case_fold::*;

/* ---------------------- Value-side encode convenience ------------------- */

/// Default, value-side encoding: `v.encode_into(&mut buf)`.
///
/// One default codec per native type to keep call sites simple.
///
/// Use specific codecs directly if you need an alternative.
pub trait EncodeInto {
    fn encode_into(&self, dst: &mut Vec<u8>);
}

// Strings → Utf8CaseFold
impl EncodeInto for str {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        // These codecs don't have a real failure path for valid inputs, so unwrap is fine.
        Utf8CaseFold::encode_into(dst, self).unwrap();
    }
}
impl EncodeInto for String {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        Utf8CaseFold::encode_into(dst, self.as_str()).unwrap();
    }
}

// i64 → LeI64
impl EncodeInto for i64 {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        LeI64::encode_into(dst, self).unwrap();
    }
}

// u8 → LeU8
impl EncodeInto for u8 {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        LeU8::encode_into(dst, self).unwrap();
    }
}

// u16 → LeU16
impl EncodeInto for u16 {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        LeU16::encode_into(dst, self).unwrap();
    }
}

// u32 → LeU32
impl EncodeInto for u32 {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        LeU32::encode_into(dst, self).unwrap();
    }
}

// u64 → LeU64
impl EncodeInto for u64 {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        LeU64::encode_into(dst, self).unwrap();
    }
}

// bool → Bool
impl EncodeInto for bool {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        Bool::encode_into(dst, self).unwrap();
    }
}

// [u8] / Vec<u8] -> Bytes
impl EncodeInto for [u8] {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        Bytes::encode_into(dst, self).unwrap();
    }
}
impl EncodeInto for Vec<u8> {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        Bytes::encode_into(dst, self.as_slice()).unwrap();
    }
}
