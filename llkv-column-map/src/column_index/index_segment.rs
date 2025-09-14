use crate::layout::{KeyLayout, ValueLayout};
use crate::storage::{BatchPut, Pager};
use crate::types::{IndexEntryCount, PhysicalKey};

/// Reference to a sealed index segment plus fast-prune metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSegmentRef {
    pub index_physical_key: PhysicalKey,
    pub data_physical_key: PhysicalKey,
    pub logical_key_min: Vec<u8>,
    pub logical_key_max: Vec<u8>,
    pub n_entries: IndexEntryCount,
}

/// Secondary directory for value-ordered scans. Tiny and kept in RAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueDirL2 {
    pub first_byte: u8,     // 0..=255
    pub dir257: [u32; 257], // prefix sums (len = 257)
}

/// Optional value-order index (2-level). Large parts are paged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueIndex {
    /// Paged per-row ordering; one byte per row. Blob is raw u8s.
    pub value_order_pk: PhysicalKey,
    /// Tiny level-1 directory.
    pub l1_dir: [u32; 256],
    /// Tiny level-2 directories for used buckets.
    pub l2_dirs: Vec<ValueDirL2>,
}

/// Fully decoded index segment metadata. Large vectors are paged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSegment {
    /// Data blob pk that holds values for this segment.
    pub data_physical_key: PhysicalKey,
    /// Number of entries (key/value pairs) in the segment.
    pub n_entries: IndexEntryCount,
    /// Concatenated logical keys as a single blob (raw bytes).
    pub key_bytes_pk: PhysicalKey,
    /// Logical key layout (fixed or paged variable directory).
    pub key_layout: KeyLayout,
    /// Value layout (fixed width or paged variable directory).
    pub value_layout: ValueLayout,
    /// Optional value-order index (paged payloads).
    pub value_index: Option<ValueIndex>,
}

impl IndexSegment {
    /// Build a segment with fixed-width keys. Keys are streamed and
    /// concatenated into a single blob. Returns metadata only.
    pub fn build_fixed_streaming<P, I>(
        pager: &P,
        data_pkey: PhysicalKey,
        keys: I,
        key_width: u32,
        n_entries: IndexEntryCount,
    ) -> Result<Self, String>
    where
        P: Pager,
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        // Allocate one pk for the key blob.
        let key_bytes_pk = pager.alloc_many(1).map_err(|e| e.to_string())?[0];

        // Stream keys into a single raw byte blob; no in-RAM concat.
        let mut out = Vec::new();
        out.reserve(64 * 1024);
        for k in keys {
            let s = k.as_ref();
            if key_width != 0 && s.len() != key_width as usize {
                return Err("build_fixed_streaming: wrong key width".into());
            }
            out.extend_from_slice(s);
            if out.len() >= (1 << 20) {
                pager
                    .batch_put(&[BatchPut::Raw {
                        key: key_bytes_pk,
                        bytes: std::mem::take(&mut out),
                    }])
                    .map_err(|e| e.to_string())?;
            }
        }
        if !out.is_empty() {
            pager
                .batch_put(&[BatchPut::Raw {
                    key: key_bytes_pk,
                    bytes: out,
                }])
                .map_err(|e| e.to_string())?;
        }

        Ok(IndexSegment {
            data_physical_key: data_pkey,
            n_entries,
            key_bytes_pk,
            key_layout: KeyLayout::FixedWidth { width: key_width },
            value_layout: ValueLayout::FixedWidth { width: 0 },
            value_index: None,
        })
    }

    /// Build a segment with variable-width keys and values using paged
    /// directories. Both directories are streamed as LE u32 prefix sums.
    pub fn build_var_streaming<P, KI, VI>(
        pager: &P,
        data_pkey: PhysicalKey,
        keys: KI,
        value_sizes: VI,
        n_entries: IndexEntryCount,
    ) -> Result<Self, String>
    where
        P: Pager,
        KI: IntoIterator,
        KI::Item: AsRef<[u8]>,
        VI: IntoIterator<Item = u32>,
    {
        // Allocate pks for: key bytes, key offsets, value offsets.
        let ks = pager.alloc_many(3).map_err(|e| e.to_string())?;
        let key_bytes_pk = ks[0];
        let key_offs_pk = ks[1];
        let val_offs_pk = ks[2];

        // Stream keys and write offset prefix sums as raw LE u32 bytes.
        let mut key_bytes = Vec::with_capacity(1 << 20);
        let mut key_offs_bytes = Vec::with_capacity(1 << 16);
        let mut acc: u32 = 0;
        key_offs_bytes.extend_from_slice(&acc.to_le_bytes());
        for k in keys {
            let s = k.as_ref();
            key_bytes.extend_from_slice(s);
            acc = acc
                .checked_add(u32::try_from(s.len()).map_err(|_| "key too big")?)
                .ok_or("key bytes overflow")?;
            key_offs_bytes.extend_from_slice(&acc.to_le_bytes());

            // Flush periodically to keep memory bound.
            if key_bytes.len() >= (1 << 20) {
                pager
                    .batch_put(&[BatchPut::Raw {
                        key: key_bytes_pk,
                        bytes: std::mem::take(&mut key_bytes),
                    }])
                    .map_err(|e| e.to_string())?;
            }
            if key_offs_bytes.len() >= (1 << 20) {
                pager
                    .batch_put(&[BatchPut::Raw {
                        key: key_offs_pk,
                        bytes: std::mem::take(&mut key_offs_bytes),
                    }])
                    .map_err(|e| e.to_string())?;
            }
        }
        if !key_bytes.is_empty() {
            pager
                .batch_put(&[BatchPut::Raw {
                    key: key_bytes_pk,
                    bytes: key_bytes,
                }])
                .map_err(|e| e.to_string())?;
        }
        if !key_offs_bytes.is_empty() {
            pager
                .batch_put(&[BatchPut::Raw {
                    key: key_offs_pk,
                    bytes: key_offs_bytes,
                }])
                .map_err(|e| e.to_string())?;
        }

        // Stream value sizes into prefix sums (offsets) as LE u32 bytes.
        let mut val_offs_bytes = Vec::with_capacity(1 << 20);
        let mut vacc: u32 = 0;
        val_offs_bytes.extend_from_slice(&vacc.to_le_bytes());
        for sz in value_sizes {
            vacc = vacc.checked_add(sz).ok_or("value bytes overflow")?;
            val_offs_bytes.extend_from_slice(&vacc.to_le_bytes());
            if val_offs_bytes.len() >= (1 << 20) {
                pager
                    .batch_put(&[BatchPut::Raw {
                        key: val_offs_pk,
                        bytes: std::mem::take(&mut val_offs_bytes),
                    }])
                    .map_err(|e| e.to_string())?;
            }
        }
        if !val_offs_bytes.is_empty() {
            pager
                .batch_put(&[BatchPut::Raw {
                    key: val_offs_pk,
                    bytes: val_offs_bytes,
                }])
                .map_err(|e| e.to_string())?;
        }

        Ok(IndexSegment {
            data_physical_key: data_pkey,
            n_entries,
            key_bytes_pk,
            key_layout: KeyLayout::VariablePaged {
                offsets_pk: key_offs_pk,
                n_entries,
            },
            value_layout: ValueLayout::VariablePaged {
                offsets_pk: val_offs_pk,
                n_entries,
            },
            value_index: None,
        })
    }
}
