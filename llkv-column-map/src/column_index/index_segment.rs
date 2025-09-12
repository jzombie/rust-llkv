use crate::bloom::KeyBloom;
use crate::bounds::ValueBound;
use crate::layout::{KeyLayout, ValueLayout};
use crate::types::{ByteLen, ByteWidth, IndexEntryCount, LogicalKeyBytes, PhysicalKey};
use bitcode::{Decode, Encode};

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
    /// For variable-width values only: fixed-size prefix tags per row to
    /// accelerate value-ordered scans without storing full values.
    /// If `tag_len == 0`, no tags are present.
    pub tag_len: u8,
    pub tags: Vec<u8>,
}
