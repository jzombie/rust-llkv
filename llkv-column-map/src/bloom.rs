use bitcode::{Decode, Encode};
use rustc_hash::FxHasher;
use std::hash::Hasher;

// TODO: Consider (if ever using var-width or long keys): Keep the Bloom on keys but make
// its hash O(1) in key length (sample head+tail + length), and consider head_tag_len = 16
// for key-ordered scans to reduce full key memcmps.
//
/// Compact Bloom filter persisted inside `IndexSegment`.
///
/// Design:
/// - `m_bits` rounded up to the next power of two so membership can use a single
///   `& (m_bits - 1)` mask instead of a modulo — this is measurably faster in
///   tight loops.
/// - Double hashing: h(i) = h1 + i*h2 (mod m), with two seeded FxHash64 passes.
/// - False positives are possible; false negatives are not.
/// - Serialization via `bitcode` keeps this tiny on disk.
///
/// Tuning knobs:
/// - `BITS_PER_KEY` controls memory & FP-rate (~12 → ~1.5 bytes/key; ~0.35–0.5% FP).
/// - `k_hashes` is derived from (m/n) ln 2 and clamped to [1, 16].
#[derive(Debug, Clone, Encode, Decode)]
pub struct KeyBloom {
    /// Number of bits in the filter (power of two).
    pub m_bits: u32,
    /// Number of hash probes.
    pub k_hashes: u8,
    /// Seeds for double hashing; keep stable across runs.
    pub seed1: u64,
    pub seed2: u64,
    /// Packed bitset, little-endian bytes.
    pub bits: Vec<u8>,
}

impl KeyBloom {
    // Stable salts for the two base hashes.
    const SEED1: u64 = 0x9E37_79B9_7F4A_7C15;
    const SEED2: u64 = 0xD1B5_4A32_D192_ED03;

    /// Target bits/key. ~12.0 → ≈1.5 bytes/key; ~0.35–0.5% FP in practice.
    pub const BITS_PER_KEY: f64 = 12.0;

    #[inline(always)]
    fn fxhash64_with_seed(seed: u64, bytes: &[u8]) -> u64 {
        let mut h = FxHasher::default();
        h.write_u64(seed);
        h.write(bytes);
        h.finish()
    }

    #[inline(always)]
    fn index_of(bit: u32) -> (usize, u8) {
        let byte = (bit >> 3) as usize;
        let mask = 1u8 << (bit & 7);
        (byte, mask)
    }

    /// Set a bit in the bitset.
    ///
    /// SAFETY: Callers must ensure `bit < bits.len() * 8`. This is upheld because:
    /// - `from_keys` allocates `bits` to cover exactly `[0, m_bits)`,
    /// - every produced `bit` index is `(x as u32) & (m_bits - 1)`, and
    /// - `m_bits` is a power of two, so the mask never yields an out-of-range bit.
    #[inline(always)]
    fn set_bit(bits: &mut [u8], bit: u32) {
        let (byte, mask) = Self::index_of(bit);
        unsafe {
            // SAFETY: see contract above — `byte` is within bounds.
            *bits.get_unchecked_mut(byte) |= mask;
        }
    }

    /// Test a bit in the bitset (bounds-checked; still very cheap with the mask).
    #[inline(always)]
    fn test_bit(bits: &[u8], bit: u32) -> bool {
        let (byte, mask) = Self::index_of(bit);
        (bits[byte] & mask) != 0
    }

    /// Build a Bloom filter over an iterator of key byte-slices.
    ///
    /// Notes:
    /// - We gather keys to know `n`. Callers already hold keys in memory when
    ///   constructing segments, so this is cheap.
    /// - `m_bits` is rounded up to the next power of two to enable mask-based
    ///   indexing in the hot path.
    pub fn from_keys<'a, I>(keys: I) -> Self
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        // Gather to know n.
        let collected: Vec<&[u8]> = keys.into_iter().collect();
        let n = collected.len().max(1);

        // Target total bits (rounded up), then round to power of two for mask ops.
        let mut m_bits = ((n as f64) * Self::BITS_PER_KEY).ceil() as u32;
        m_bits = m_bits.max(8).next_power_of_two();
        let m_mask = m_bits - 1; // valid because m_bits is a power of two

        // k ≈ (m/n) ln 2, clamped to [1, 16].
        let kf = (m_bits as f64 / n as f64) * std::f64::consts::LN_2;
        let mut k = kf.round() as i32;
        if k <= 0 {
            k = 1;
        }
        if k > 16 {
            k = 16;
        }
        let k_hashes = k as u8;

        // Allocate bitset.
        let bytes_len = m_bits.div_ceil(8) as usize;
        let mut bits = vec![0u8; bytes_len];

        // Insert each key using double hashing.
        for &key in &collected {
            let h1 = Self::fxhash64_with_seed(Self::SEED1, key);
            let h2 = Self::fxhash64_with_seed(Self::SEED2, key);
            let mut x = h1;
            // h(i) = (h1 + i*h2) mod m  →  (x as u32) & m_mask
            for _ in 0..(k as u32) {
                let bit = (x as u32) & m_mask;
                Self::set_bit(&mut bits, bit);
                x = x.wrapping_add(h2);
            }
        }

        Self {
            m_bits,
            k_hashes,
            seed1: Self::SEED1,
            seed2: Self::SEED2,
            bits,
        }
    }

    /// Membership check (fast; may return false positives).
    ///
    /// Returns `true` if the Bloom indicates the key *may* be present, and
    /// `false` if it is definitely not present.
    #[inline(always)]
    pub fn check(&self, key: &[u8]) -> bool {
        // If disabled/minimal, treat as "maybe" to avoid false negatives.
        if self.m_bits == 0 {
            return true;
        }
        let m_mask = self.m_bits - 1;

        let h1 = Self::fxhash64_with_seed(self.seed1, key);
        let h2 = Self::fxhash64_with_seed(self.seed2, key);
        let mut x = h1;

        for _ in 0..self.k_hashes {
            let bit = (x as u32) & m_mask;
            if !Self::test_bit(&self.bits, bit) {
                return false;
            }
            x = x.wrapping_add(h2);
        }
        true
    }
}
