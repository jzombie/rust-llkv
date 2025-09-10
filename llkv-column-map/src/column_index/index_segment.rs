use crate::bounds::ValueBound;
use crate::layout::{KeyLayout, ValueLayout};
use crate::types::{ByteLen, ByteWidth, IndexEntryCount, LogicalKeyBytes, PhysicalKey};
use bitcode::{Decode, Encode};
use rustc_hash::FxHasher;
use std::hash::Hasher;

// -------------------------- Bloom (persisted) --------------------------

#[derive(Debug, Clone, Encode, Decode)]
pub struct KeyBloom {
    /// Number of bits in the filter.
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
    const SEED1: u64 = 0x9E37_79B9_7F4A_7C15;
    const SEED2: u64 = 0xD1B5_4A32_D192_ED03;

    /// Target bits/key. ~12.0 → ≈1.5 B/key; ~0.35–0.5% FP.
    const BITS_PER_KEY: f64 = 12.0;

    #[inline]
    fn fxhash64_with_seed(seed: u64, bytes: &[u8]) -> u64 {
        let mut h = FxHasher::default();
        h.write_u64(seed);
        h.write(bytes);
        h.finish()
    }

    #[inline]
    fn index_of(bit: u32) -> (usize, u8) {
        let byte = (bit >> 3) as usize;
        let mask = 1u8 << (bit & 7);
        (byte, mask)
    }

    #[inline]
    fn set_bit(bits: &mut [u8], bit: u32) {
        let (byte, mask) = Self::index_of(bit);
        unsafe {
            // safe: callers ensure capacity
            let p = bits.get_unchecked_mut(byte);
            *p |= mask;
        }
    }

    #[inline]
    fn test_bit(bits: &[u8], bit: u32) -> bool {
        let (byte, mask) = Self::index_of(bit);
        // bounds-checked (we build the filter, so it matches)
        (bits[byte] & mask) != 0
    }

    /// Build a Bloom filter over an iterator of key byte-slices.
    pub fn from_keys<'a, I>(keys: I) -> Self
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        // gather to know n
        let collected: Vec<&[u8]> = keys.into_iter().collect();
        let n = collected.len().max(1);

        let m_bits = ((n as f64) * Self::BITS_PER_KEY).ceil() as u32;
        let m_bits = m_bits.max(8); // at least 1 byte
        let m = m_bits as u64;

        // k ≈ (m/n) ln2
        let kf = (m_bits as f64 / n as f64) * std::f64::consts::LN_2;
        let mut k = kf.round() as i32;
        if k <= 0 {
            k = 1;
        }
        if k > 16 {
            k = 16; // hedge
        }
        let k_hashes = k as u8;

        let bytes_len = ((m_bits + 7) / 8) as usize;
        let mut bits = vec![0u8; bytes_len];

        for &key in &collected {
            let h1 = Self::fxhash64_with_seed(Self::SEED1, key);
            let h2 = Self::fxhash64_with_seed(Self::SEED2, key);
            let mut x = h1;
            for i in 0..(k as u32) {
                // (h1 + i*h2) % m
                let bit = ((x % m) as u32) % m_bits;
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

    /// Check membership (fast; may return false positives).
    #[inline]
    pub fn check(&self, key: &[u8]) -> bool {
        let m = self.m_bits as u64;
        if m == 0 {
            return true;
        }
        let h1 = Self::fxhash64_with_seed(self.seed1, key);
        let h2 = Self::fxhash64_with_seed(self.seed2, key);
        let mut x = h1;
        for _ in 0..self.k_hashes {
            let bit = ((x % m) as u32) % self.m_bits;
            if !Self::test_bit(&self.bits, bit) {
                return false;
            }
            x = x.wrapping_add(h2);
        }
        true
    }
}

// ----------------- IndexSegmentRef / IndexSegment / ValueIndex -----------------

/// Pointer to a sealed segment plus fast-prune metadata (all **logical**).
#[derive(Debug, Clone, Encode, Decode)]
pub struct IndexSegmentRef {
    pub index_physical_key: PhysicalKey,
    pub data_physical_key: PhysicalKey,

    pub logical_key_min: LogicalKeyBytes,
    pub logical_key_max: LogicalKeyBytes,

    pub value_min: Option<ValueBound>,
    pub value_max: Option<ValueBound>,

    pub n_entries: IndexEntryCount,
}

/// One sealed batch. Describes how to fetch values from the *data* blob.
#[derive(Debug, Clone, Encode, Decode)]
pub struct IndexSegment {
    pub data_physical_key: PhysicalKey,
    pub n_entries: IndexEntryCount,

    /// Sorted *logical* keys, stored compactly (see `key_layout`).
    pub logical_key_bytes: Vec<u8>,
    pub key_layout: KeyLayout,

    /// How to slice the *data* blob for the i-th value.
    pub value_layout: ValueLayout,

    /// Optional value index for value-ordered scans and prefix pruning.
    pub value_index: Option<ValueIndex>,

    /// Persisted Bloom filter over **logical keys** for fast conflict probes.
    pub key_bloom: KeyBloom,
}

impl IndexSegment {
    pub fn build_fixed<T: AsRef<[u8]>>(
        data_pkey: PhysicalKey,
        logical_keys: &[T],
        width: ByteWidth,
    ) -> IndexSegment {
        let n = logical_keys.len() as IndexEntryCount;

        // build compact key layout
        let (logical_key_bytes, key_layout) = KeyLayout::pack_keys_with_layout(logical_keys);
        // build Bloom (persisted)
        let bloom = KeyBloom::from_keys(logical_keys.iter().map(|k| k.as_ref()));

        IndexSegment {
            data_physical_key: data_pkey,
            n_entries: n,
            logical_key_bytes,
            key_layout,
            value_layout: ValueLayout::FixedWidth { width },
            value_index: None,
            key_bloom: bloom,
        }
    }

    pub fn build_var<T: AsRef<[u8]>>(
        data_pkey: PhysicalKey,
        logical_keys: &[T],
        value_sizes: &[ByteLen], // one per entry
    ) -> IndexSegment {
        assert_eq!(logical_keys.len(), value_sizes.len());
        let n = logical_keys.len() as IndexEntryCount;

        let (logical_key_bytes, key_layout) = KeyLayout::pack_keys_with_layout(logical_keys);

        // build offsets
        let mut value_offsets = Vec::with_capacity(value_sizes.len() + 1);
        let mut acc = 0u32;
        value_offsets.push(acc);
        for &sz in value_sizes {
            acc += sz;
            value_offsets.push(acc);
        }

        // build Bloom (persisted)
        let bloom = KeyBloom::from_keys(logical_keys.iter().map(|k| k.as_ref()));

        IndexSegment {
            data_physical_key: data_pkey,
            n_entries: n,
            logical_key_bytes,
            key_layout,
            value_layout: ValueLayout::Variable {
                value_offsets: value_offsets
                    .into_iter()
                    .map(|v| v as IndexEntryCount)
                    .collect(),
            },
            value_index: None,
            key_bloom: bloom,
        }
    }
}

/// Per-segment directory for value-ordered access.
#[derive(Debug, Clone, Encode, Decode)]
pub struct ValueDirL2 {
    pub first_byte: u8,
    pub dir257: Vec<u32>,
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct ValueIndex {
    pub value_order: Vec<IndexEntryCount>,
    pub l1_dir: Vec<IndexEntryCount>,
    pub l2_dirs: Vec<ValueDirL2>,
}
