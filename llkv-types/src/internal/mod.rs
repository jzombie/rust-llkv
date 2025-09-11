//!
//! Defines the core, low-level types used throughout the storage engine.
//!
//!
//! This module provides type aliases for fundamental concepts like physical keys,
//! logical field IDs, and various byte-level measurements.
//!
//!
//! ## A Note on Endianness and Sort Order
//!
//! You will notice that this system consistently uses **big-endian** encoding for
//!
//! all numeric types that are part of a `LogicalKeyBytes` payload. This is a
//! deliberate and critical design choice.
//!
//!
//! The storage engine's core is built on a simple, fast, and generic
//! **lexicographical (byte-wise) sort**.
//! To function correctly, all data types—strings,
//! numbers, and composites—must sort correctly using the same universal rule.
//!
//!
//! - **Strings** naturally sort lexicographically.
//! - **Little-Endian** numbers (the native format for most CPUs) **do not**. For example,
//!
//! as bytes, `256` would sort before `1`.
//! - **Big-Endian** numbers **do**. Their byte order matches their numeric order, allowing
//!
//! them to be sorted correctly with the same logic as strings.
//!
//!
//! This requires a fast byte-swap when decoding numbers for computation, but this
//!
//! small, one-time cost is far outweighed by the massive simplification and
//!
//! performance gain of a unified, type-agnostic sorting and searching core.

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

pub mod codec;
pub use codec::*;

pub mod be_u64;
pub use be_u64::*;

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

// u64 → BeU64
impl EncodeInto for u64 {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        BeU64::encode_into(dst, self).unwrap();
    }
}

// bool → Bool
impl EncodeInto for bool {
    #[inline]
    fn encode_into(&self, dst: &mut Vec<u8>) {
        Bool::encode_into(dst, self).unwrap();
    }
}
