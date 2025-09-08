use crate::bounds::ValueBound;
use crate::layout::{KeyLayout, ValueLayout};
use crate::types::{ByteLen, ByteWidth, IndexEntryCount, LogicalKeyBytes, PhysicalKey};
use bitcode::{Decode, Encode};

// TODO: [perf] Ensure bounds are updated as keys are dereferenced.
/// Pointer to a sealed segment plus fast-prune metadata (all **logical**).
///
/// This is intentionally compact to keep the `ColumnIndex` small and hot in
/// cache. It is enough to cheaply decide if a segment can satisfy a probe or a
/// range without opening the full `IndexSegment`.
#[derive(Debug, Clone, Encode, Decode)]
pub struct IndexSegmentRef {
    /// Physical key of the *index segment* blob.
    pub index_physical_key: PhysicalKey,
    pub data_physical_key: PhysicalKey,

    /// Quick span prune (LOGICAL key bytes).
    pub logical_key_min: LogicalKeyBytes,
    pub logical_key_max: LogicalKeyBytes,

    pub value_min: Option<ValueBound>,
    pub value_max: Option<ValueBound>,

    /// Number of entries in that segment (helps pre-alloc).
    pub n_entries: IndexEntryCount,
}

/// One sealed batch. Describes how to fetch values from the *data* blob.
/// The *data* blob contains only raw value bytes — no headers/markers.
///
/// This index stores:
///   - the *physical* key of the data blob,
///   - the *logical* keys (sorted, compact; either fixed-width or with offsets),
///   - the value layout (fixed width or var-width offsets).
#[derive(Debug, Clone, Encode, Decode)]
pub struct IndexSegment {
    /// Where the values live (physical KV key).
    pub data_physical_key: PhysicalKey,

    /// Number of entries.
    pub n_entries: IndexEntryCount,

    /// Sorted *logical* keys, stored compactly (see `key_layout`).
    pub logical_key_bytes: Vec<u8>,
    pub key_layout: KeyLayout,

    /// How to slice the *data* blob for the i-th value.
    pub value_layout: ValueLayout,
}

impl IndexSegment {
    pub fn build_fixed(
        data_pkey: PhysicalKey,
        logical_keys: Vec<Vec<u8>>,
        width: ByteWidth,
    ) -> IndexSegment {
        let n = logical_keys.len() as IndexEntryCount;
        let (logical_key_bytes, key_layout) = KeyLayout::pack_keys_with_layout(logical_keys);
        IndexSegment {
            data_physical_key: data_pkey,
            n_entries: n,
            logical_key_bytes,
            key_layout,
            value_layout: ValueLayout::FixedWidth { width },
        }
    }

    pub fn build_var(
        data_pkey: PhysicalKey,
        logical_keys: Vec<Vec<u8>>,
        value_sizes: &[ByteLen], // one per entry
    ) -> IndexSegment {
        assert_eq!(logical_keys.len(), value_sizes.len());
        let n = logical_keys.len() as IndexEntryCount;
        let (logical_key_bytes, key_layout) = KeyLayout::pack_keys_with_layout(logical_keys);

        let mut value_offsets = Vec::with_capacity(value_sizes.len() + 1);
        let mut acc = 0u32;
        value_offsets.push(acc);
        for &sz in value_sizes {
            acc += sz;
            value_offsets.push(acc);
        }

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
        }
    }
}
