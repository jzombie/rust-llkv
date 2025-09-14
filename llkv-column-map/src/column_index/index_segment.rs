use crate::layout::{KeyLayout, ValueLayout};
use crate::types::{IndexEntryCount, PhysicalKey};

/// Reference to a materialized index segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSegmentRef {
    pub index_physical_key: PhysicalKey,
    pub data_physical_key: PhysicalKey,
    pub logical_key_min: Vec<u8>,
    pub logical_key_max: Vec<u8>,
    pub n_entries: IndexEntryCount,
}

/// Secondary directory for value-ordered scans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueDirL2 {
    pub first_byte: u8,     // 0..=255
    pub dir257: [u32; 257], // prefix sums, len = 257
}

/// Optional value-order index (2-level).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueIndex {
    pub value_order: Vec<u8>,     // one byte per value
    pub l1_dir: [u32; 256],       // prefix sums, len = 256
    pub l2_dirs: Vec<ValueDirL2>, // present for buckets in use
}

/// Fully decoded index segment contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSegment {
    pub data_physical_key: PhysicalKey,
    pub n_entries: IndexEntryCount,

    pub logical_key_bytes: Vec<u8>,
    pub key_layout: KeyLayout,

    pub value_layout: ValueLayout,
    pub value_index: Option<ValueIndex>,
}

impl IndexSegment {
    /// Build a segment for fixed-width keys and fixed-width values.
    pub fn build_fixed(data_pkey: PhysicalKey, keys: &[Vec<u8>], values_width: u32) -> Self {
        let (logical_key_bytes, key_layout) = crate::layout::pack_keys_with_layout(keys);

        IndexSegment {
            data_physical_key: data_pkey,
            n_entries: keys.len() as IndexEntryCount,
            logical_key_bytes,
            key_layout,
            value_layout: ValueLayout::FixedWidth {
                width: values_width as u32,
            },
            value_index: None,
        }
    }

    /// Build a segment for variable-width keys and variable-width values.
    pub fn build_var(data_pkey: PhysicalKey, keys: &[Vec<u8>], value_offsets: Vec<u32>) -> Self {
        let (logical_key_bytes, key_layout) = crate::layout::pack_keys_with_layout(keys);

        IndexSegment {
            data_physical_key: data_pkey,
            n_entries: keys.len() as IndexEntryCount,
            logical_key_bytes,
            key_layout,
            value_layout: ValueLayout::Variable { value_offsets },
            value_index: None,
        }
    }
}
