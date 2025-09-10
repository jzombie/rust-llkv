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
/// - `m_bits` is the number of bits in the filter (no need to be a power of two).
/// - Double hashing: h(i) = h1 + i*h2 (mod 2^64), with two seeded FxHash64 passes.
/// - Indexing uses unbiased 64→[0, m_bits) reduction via 128-bit multiply
///   (Lemire fast range reduction), avoiding low-bit degeneracy.
/// - False positives are possible; false negatives are not.
///
/// Tuning:
/// - `BITS_PER_KEY` controls memory & FP rate (~12 → ~1.5 B/key; typically ~0.3–0.6% FP).
/// - `k_hashes` ≈ (m/n) ln 2, clamped to [1, 16].
#[derive(Debug, Clone, Encode, Decode)]
pub struct KeyBloom {
    /// Number of bits in the filter.
    pub m_bits: u32,
    /// Number of hash probes.
    pub k_hashes: u8,
    /// Seeds for double hashing; kept for reproducibility.
    pub seed1: u64,
    pub seed2: u64,
    /// Packed bitset, little-endian bytes.
    pub bits: Vec<u8>,
}

impl KeyBloom {
    // Stable salts for the two base hashes.
    const SEED1: u64 = 0x9E37_79B9_7F4A_7C15;
    const SEED2: u64 = 0xD1B5_4A32_D192_ED03;

    /// Target bits/key. ~12.0 → ≈1.5 bytes/key; ~0.3–0.6% FP in practice.
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
    /// SAFETY: Callers must ensure `bit < bits.len() * 8`. This holds because:
    /// - `from_keys` allocates `bits` for exactly `m_bits`,
    /// - all indices are produced by `fast_reduce_u64_to_range(x, m_bits)`,
    ///   which guarantees `0 <= bit < m_bits`.
    #[inline(always)]
    fn set_bit(bits: &mut [u8], bit: u32) {
        let (byte, mask) = Self::index_of(bit);
        unsafe {
            // SAFETY: see contract above — `byte` is within bounds.
            *bits.get_unchecked_mut(byte) |= mask;
        }
    }

    /// Test a bit (bounds-checked; cheap).
    #[inline(always)]
    fn test_bit(bits: &[u8], bit: u32) -> bool {
        let (byte, mask) = Self::index_of(bit);
        (bits[byte] & mask) != 0
    }

    /// Unbiased reduction of a 64-bit hash into [0, m_bits) using 128-bit multiply.
    /// (Lemire: Fast Random Integer Generation in an Interval)
    #[inline(always)]
    fn fast_reduce_u64_to_range(x: u64, m_bits: u32) -> u32 {
        // ((x * m_bits) >> 64) maps uniformly from [0, 2^64) to [0, m_bits)
        ((x as u128).wrapping_mul(m_bits as u128) >> 64) as u32
    }

    /// Build a Bloom filter over an iterator of key byte-slices.
    pub fn from_keys<'a, I>(keys: I) -> Self
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        // Gather to know n (callers already have keys in memory here).
        let collected: Vec<&[u8]> = keys.into_iter().collect();
        let n = collected.len().max(1);

        // Choose total bits; no need for power-of-two anymore.
        let m_bits = ((n as f64) * Self::BITS_PER_KEY).ceil() as u32;
        let m_bits = m_bits.max(8);

        // k ≈ (m/n) ln 2, clamped to [1, 16]
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
        let bytes_len = ((m_bits + 7) / 8) as usize;
        let mut bits = vec![0u8; bytes_len];

        for &key in &collected {
            let h1 = Self::fxhash64_with_seed(Self::SEED1, key);
            let mut h2 = Self::fxhash64_with_seed(Self::SEED2, key);
            h2 |= 1; // ensure non-zero step (branchless)
            let mut x = h1;
            for _ in 0..(k as u32) {
                let bit = Self::fast_reduce_u64_to_range(x, m_bits);
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
        if self.m_bits == 0 {
            // Treat as "maybe" to avoid false negatives when disabled.
            return true;
        }

        let h1 = Self::fxhash64_with_seed(self.seed1, key);
        let mut h2 = Self::fxhash64_with_seed(self.seed2, key);
        h2 |= 1; // ensure non-zero step (branchless)
        let mut x = h1;

        for _ in 0..self.k_hashes {
            let bit = Self::fast_reduce_u64_to_range(x, self.m_bits);
            if !Self::test_bit(&self.bits, bit) {
                return false;
            }
            x = x.wrapping_add(h2);
        }
        true
    }
}

// ------------------------- tests -------------------------

#[cfg(test)]
mod bloom_fastmod_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn make_in_set_keys(n: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let len = rng.random_range(0..=32);
                let mut v = Vec::with_capacity(len + 1 + 8);
                for _ in 0..len {
                    v.push(b'a' + (rng.random::<u8>() % 26));
                }
                v.push(b'#');
                v.extend_from_slice(format!("{i:08x}").as_bytes());
                v
            })
            .collect()
    }

    fn make_out_set_keys(n: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xDEADBEEF);
        (0..n)
            .map(|i| {
                let len = rng.random_range(0..=32);
                let mut v = Vec::with_capacity(len + 1 + 8);
                for _ in 0..len {
                    v.push(b'0' + (rng.random::<u8>() % 10));
                }
                v.push(b'#');
                v.extend_from_slice(format!("{i:08x}").as_bytes());
                v
            })
            .collect()
    }

    #[test]
    fn roundtrip_membership() {
        let keys = make_in_set_keys(10_000, 42);
        let bloom = KeyBloom::from_keys(keys.iter().map(|k| k.as_slice()));

        for k in &keys {
            assert!(bloom.check(k), "inserted key must be maybe-present");
        }

        // serialization preserves behavior
        let bytes = bitcode::encode(&bloom);
        let bloom2: KeyBloom = bitcode::decode(&bytes).expect("decode");
        for k in &keys {
            assert!(bloom2.check(k), "roundtrip must keep membership");
        }
    }

    #[test]
    fn false_positive_rate_is_reasonable() {
        let n_in = 10_000usize;
        let n_out = 20_000usize;
        let keys_in = make_in_set_keys(n_in, 123);
        let keys_out = make_out_set_keys(n_out, 456);

        let bloom = KeyBloom::from_keys(keys_in.iter().map(|k| k.as_slice()));

        let mut fp = 0usize;
        for k in &keys_out {
            if bloom.check(k) {
                fp += 1;
            }
        }
        let rate = (fp as f64) / (n_out as f64);

        // With 12 bits/key, expect ≈0.3–0.6% FP. Allow up to 1.5% for safety.
        assert!(
            rate < 0.015,
            "FP too high: {:.3}% (fp={}, n={})",
            rate * 100.0,
            fp,
            n_out
        );
    }

    #[test]
    fn k_hashes_and_bits_are_sane() {
        let keys = make_in_set_keys(1234, 7);
        let bloom = KeyBloom::from_keys(keys.iter().map(|k| k.as_slice()));
        assert!(bloom.m_bits >= 8);
        assert!((1..=16).contains(&bloom.k_hashes));
        // bitset length matches m_bits
        assert_eq!(bloom.bits.len(), ((bloom.m_bits as usize + 7) / 8));
    }
}
