use bitcode::{Decode, Encode};

use crate::types::{IndexEntryCount, LogicalFieldId, LogicalKeyBytes, PhysicalKey};

// TODO: Refactor
pub fn concat_keys_and_offsets(mut keys: Vec<Vec<u8>>) -> (Vec<u8>, Vec<u32>) {
    // Ensure sorted by logical key bytes
    keys.sort_unstable();
    let mut bytes = Vec::new();
    let mut offs = Vec::with_capacity(keys.len() + 1);
    let mut acc = 0u32;
    offs.push(acc);
    for k in keys {
        bytes.extend_from_slice(&k);
        acc += k.len() as u32;
        offs.push(acc);
    }
    (bytes, offs)
}

// ── Bootstrapping ────────────────────────────────────────────────────────────
// Physical key 0 holds this tiny record so you can find the manifest.
#[derive(Debug, Clone, Encode, Decode)]
pub struct Bootstrap {
    pub manifest_physical_key: PhysicalKey,
}

// A manifest maps columns → their current ColumnIndex blob (newest version).
#[derive(Debug, Clone, Encode, Decode)]
pub struct Manifest {
    pub columns: Vec<ColumnEntry>,
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct ColumnEntry {
    pub field_id: LogicalFieldId,
    /// Physical key of the current ColumnIndex blob for this column.
    pub column_index_physical_key: PhysicalKey,
}

// ── Column index (list of immutable segments) ────────────────────────────────
// One per column. Newest-first so later segments shadow older ones.
#[derive(Debug, Clone, Encode, Decode)]
pub struct ColumnIndex {
    pub field_id: LogicalFieldId,
    pub segments: Vec<IndexSegmentRef>, // newest-first
}

/// Pointer to a sealed segment + fast-prune info (all LOGICAL).
#[derive(Debug, Clone, Encode, Decode)]
pub struct IndexSegmentRef {
    /// Physical key of the *index segment* blob.
    pub index_physical_key: PhysicalKey,

    /// Quick span prune (LOGICAL key bytes).
    pub logical_key_min: LogicalKeyBytes,
    pub logical_key_max: LogicalKeyBytes,

    /// Number of entries in that segment (helps pre-alloc).
    pub n_entries: IndexEntryCount,
}

// ── Your locked index segment & value layout ─────────────────────────────────
/// One sealed batch. Describes how to fetch values from the *data* blob.
/// The *data* blob contains only raw value bytes — no headers/markers.
///
/// This index stores:
///   - the *physical* key of the data blob,
///   - the *logical* keys (sorted, contiguous+offsets),
///   - the value layout (fixed width or var-width offsets).
#[derive(Debug, Clone, Encode, Decode)]
pub struct IndexSegment {
    /// Where the values live (physical KV key).
    pub data_physical_key: PhysicalKey,

    /// Number of entries (also equal to logical_key_offsets.len()-1).
    pub n_entries: IndexEntryCount,

    /// Sorted *logical* keys, stored compactly:
    /// `logical_key_bytes[logical_key_offsets[i]..logical_key_offsets[i+1])`
    pub logical_key_bytes: Vec<u8>,
    pub logical_key_offsets: Vec<IndexEntryCount>, // len = n_entries + 1

    /// How to slice the *data* blob for the i-th value.
    pub value_layout: ValueLayout,

    /// Redundant cached bounds for fast reject/prune.
    pub logical_key_min: LogicalKeyBytes,
    pub logical_key_max: LogicalKeyBytes,
}

impl IndexSegment {
    pub fn build_fixed(
        data_pkey: PhysicalKey,
        logical_keys: Vec<Vec<u8>>,
        width: u32,
    ) -> IndexSegment {
        let n = logical_keys.len() as IndexEntryCount;
        let (logical_key_bytes, logical_key_offsets) =
            concat_keys_and_offsets(logical_keys.clone());
        IndexSegment {
            data_physical_key: data_pkey,
            n_entries: n,
            logical_key_bytes,
            logical_key_offsets,
            value_layout: ValueLayout::FixedWidth { width },
            logical_key_min: logical_keys.first().cloned().unwrap_or_default(),
            logical_key_max: logical_keys.last().cloned().unwrap_or_default(),
        }
    }

    pub fn build_var(
        data_pkey: PhysicalKey,
        logical_keys: Vec<Vec<u8>>,
        value_sizes: &[u32], // one per entry
    ) -> IndexSegment {
        assert_eq!(logical_keys.len(), value_sizes.len());
        let n = logical_keys.len() as IndexEntryCount;
        let (logical_key_bytes, logical_key_offsets) =
            concat_keys_and_offsets(logical_keys.clone());

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
            logical_key_offsets,
            value_layout: ValueLayout::Variable { value_offsets },
            logical_key_min: logical_keys.first().cloned().unwrap_or_default(),
            logical_key_max: logical_keys.last().cloned().unwrap_or_default(),
        }
    }
}

/// Slicing recipe for values inside the *data* blob.
#[derive(Debug, Clone, Encode, Decode)]
pub enum ValueLayout {
    /// Every value is exactly `width` bytes.
    FixedWidth { width: u32 },

    /// Variable width values. Prefix sum of byte offsets into data blob.
    /// Slice i is [value_offsets[i], value_offsets[i+1]).
    Variable { value_offsets: Vec<u32> }, // len = n_entries + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::Pager;
    use bitcode::{Decode, Encode};
    use std::collections::HashMap;

    const BOOTSTRAP_PKEY: PhysicalKey = 0;

    /// Tiny in-memory KV used by the tests to simulate a pager/kv store.
    /// Values are **opaque blobs**; only indexes carry structure.
    struct MemPager {
        map: HashMap<PhysicalKey, Vec<u8>>,
        next: PhysicalKey, // simple monotonically increasing allocator
    }

    impl Default for MemPager {
        fn default() -> Self {
            Self {
                map: HashMap::new(),
                next: 1,
            } // reserve 0 for Bootstrap
        }
    }

    impl Pager for MemPager {
        // -------- allocation (batched only) --------
        fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey> {
            let start = self.next;
            self.next += n as u64;
            (0..n).map(|i| start + i as u64).collect()
        }

        // -------- raw bytes (batched only) --------
        fn batch_put_raw(&mut self, items: &[(PhysicalKey, Vec<u8>)]) {
            for (k, v) in items {
                self.map.insert(*k, v.clone());
            }
        }
        fn batch_get_raw<'a>(&'a self, keys: &[PhysicalKey]) -> Vec<&'a [u8]> {
            keys.iter()
                .map(|k| self.map.get(k).expect("missing key").as_slice())
                .collect()
        }

        // -------- typed (batched only) --------
        fn batch_put_typed<T: Encode>(&mut self, items: &[(PhysicalKey, T)]) {
            let mut enc: Vec<(PhysicalKey, Vec<u8>)> = Vec::with_capacity(items.len());
            for (k, v) in items {
                enc.push((*k, bitcode::encode(v)));
            }
            self.batch_put_raw(&enc);
        }
        fn batch_get_typed<T>(&self, keys: &[PhysicalKey]) -> Vec<T>
        where
            for<'a> T: Decode<'a>,
        {
            self.batch_get_raw(keys)
                .into_iter()
                .map(|b| bitcode::decode(b).expect("bitcode decode failed"))
                .collect()
        }
    }

    // ---------------- helpers to build index segments ----------------

    fn slice_key<'a>(bytes: &'a [u8], offs: &'a [u32], i: usize) -> &'a [u8] {
        let a = offs[i] as usize;
        let b = offs[i + 1] as usize;
        &bytes[a..b]
    }

    // Generic function (not a closure) to avoid HRTB/lifetime clash.
    fn prune<'a>(refs: &'a [IndexSegmentRef], probe: &[u8]) -> Vec<&'a IndexSegmentRef> {
        refs.iter()
            .filter(|r| {
                r.logical_key_min.as_slice() <= probe && probe <= r.logical_key_max.as_slice()
            })
            .collect()
    }

    // ---------------- core tests (ALL BATCH I/O) ----------------

    /// Boot → Manifest → ColumnIndex → IndexSegment round-trip.
    /// Shows how to:
    ///   1) discover the manifest via physical key 0,
    ///   2) jump to a column’s current index,
    ///   3) open an index segment and inspect min/max/key layout.
    #[test]
    fn bootstrap_manifest_column_index_roundtrip() {
        let mut kv = MemPager::default();

        // ----- allocate physical ids (batched) -----
        // data_100, idx_100, data_200, idx_200, col_100_idx_pkey, col_200_idx_pkey, manifest_pkey
        let ids = kv.alloc_many(7);
        let data_100 = ids[0];
        let idx_100 = ids[1];
        let data_200 = ids[2];
        let idx_200 = ids[3];
        let col_100_idx_pkey = ids[4];
        let col_200_idx_pkey = ids[5];
        let manifest_pkey = ids[6];

        // ----- build two segments for columns 100 and 200 -----
        let seg_100 = IndexSegment::build_fixed(
            data_100,
            vec![b"a".to_vec(), b"b".to_vec(), b"d".to_vec()],
            1,
        );
        let animals = vec![b"ant".to_vec(), b"wolf".to_vec(), b"zebra".to_vec()];
        let sizes: Vec<u32> = vec![5, 2, 8]; // arbitrary payload lengths
        let seg_200 = IndexSegment::build_var(data_200, animals.clone(), &sizes);

        // store index segments (batched)
        kv.batch_put_typed::<IndexSegment>(&[
            (idx_100, seg_100.clone()),
            (idx_200, seg_200.clone()),
        ]);

        // ----- ColumnIndex blobs (newest-first segments) -----
        let col_100_index = ColumnIndex {
            field_id: 100,
            segments: vec![IndexSegmentRef {
                index_physical_key: idx_100,
                logical_key_min: seg_100.logical_key_min.clone(),
                logical_key_max: seg_100.logical_key_max.clone(),
                n_entries: seg_100.n_entries,
            }],
        };
        let col_200_index = ColumnIndex {
            field_id: 200,
            segments: vec![IndexSegmentRef {
                index_physical_key: idx_200,
                logical_key_min: seg_200.logical_key_min.clone(),
                logical_key_max: seg_200.logical_key_max.clone(),
                n_entries: seg_200.n_entries,
            }],
        };

        kv.batch_put_typed::<ColumnIndex>(&[
            (col_100_idx_pkey, col_100_index),
            (col_200_idx_pkey, col_200_index),
        ]);

        // ----- Manifest maps columns → current ColumnIndex physical keys -----
        let manifest = Manifest {
            columns: vec![
                ColumnEntry {
                    field_id: 100,
                    column_index_physical_key: col_100_idx_pkey,
                },
                ColumnEntry {
                    field_id: 200,
                    column_index_physical_key: col_200_idx_pkey,
                },
            ],
        };
        kv.batch_put_typed::<Manifest>(&[(manifest_pkey, manifest)]);

        // ----- Bootstrap at physical key 0 points to the manifest -----
        kv.batch_put_typed::<Bootstrap>(&[(
            BOOTSTRAP_PKEY,
            Bootstrap {
                manifest_physical_key: manifest_pkey,
            },
        )]);

        // ======== Walk it back as a user would (batched reads only) =========

        // 0) Find the manifest
        let boot: Bootstrap = kv
            .batch_get_typed::<Bootstrap>(&[BOOTSTRAP_PKEY])
            .pop()
            .unwrap();
        let got_manifest: Manifest = kv
            .batch_get_typed::<Manifest>(&[boot.manifest_physical_key])
            .pop()
            .unwrap();

        // 1) Read ColumnIndex blobs in one batch
        let idx_keys: Vec<PhysicalKey> = got_manifest
            .columns
            .iter()
            .map(|c| c.column_index_physical_key)
            .collect();
        let mut col_indexes: Vec<ColumnIndex> = kv.batch_get_typed::<ColumnIndex>(&idx_keys);

        // map by field_id for assertions
        col_indexes.sort_by_key(|ci| ci.field_id);
        let col100_index = col_indexes.iter().find(|ci| ci.field_id == 100).unwrap();
        let col200_index = col_indexes.iter().find(|ci| ci.field_id == 200).unwrap();

        // 2) Segment refs (newest-first)
        assert_eq!(col100_index.segments.len(), 1);
        assert_eq!(col200_index.segments.len(), 1);
        let segref_100 = &col100_index.segments[0];
        let segref_200 = &col200_index.segments[0];

        // slice-coerced byte string literals (fix & [u8;N] vs &[u8])
        assert!(&b"a"[..] <= segref_100.logical_key_min.as_slice());
        assert!(&b"d"[..] >= segref_100.logical_key_max.as_slice());
        assert!(&b"ant"[..] <= segref_200.logical_key_min.as_slice());
        assert!(&b"zebra"[..] >= segref_200.logical_key_max.as_slice());

        // 3) Open both IndexSegments in a single batch read
        let segs: Vec<IndexSegment> = kv.batch_get_typed::<IndexSegment>(&[
            segref_100.index_physical_key,
            segref_200.index_physical_key,
        ]);
        let seg100 = &segs[0];
        let seg200 = &segs[1];

        assert_eq!(seg100.n_entries, 3);
        match seg100.value_layout {
            ValueLayout::FixedWidth { width } => assert_eq!(width, 1),
            _ => panic!("expected fixed width"),
        }

        // Inspect keys via offsets
        let k0 = slice_key(&seg100.logical_key_bytes, &seg100.logical_key_offsets, 0);
        let k1 = slice_key(&seg100.logical_key_bytes, &seg100.logical_key_offsets, 1);
        let k2 = slice_key(&seg100.logical_key_bytes, &seg100.logical_key_offsets, 2);
        assert_eq!(k0, &b"a"[..]);
        assert_eq!(k1, &b"b"[..]);
        assert_eq!(k2, &b"d"[..]);

        assert_eq!(seg200.n_entries, animals.len() as u32);
        match &seg200.value_layout {
            ValueLayout::Variable { value_offsets } => {
                assert_eq!(value_offsets.len(), animals.len() + 1);
                // Expected prefix sums 0,5,7,15
                let mut acc = 0u32;
                for (i, &sz) in [5u32, 2, 8].iter().enumerate() {
                    assert_eq!(value_offsets[i], acc);
                    acc += sz;
                    assert_eq!(value_offsets[i + 1], acc);
                }
            }
            _ => panic!("expected variable width"),
        }
    }

    /// Demonstrates segment pruning using min/max logical bounds,
    /// then a simple binary search inside an `IndexSegment` without any B-Tree.
    #[test]
    fn prune_and_binary_search_in_segment() {
        let mut kv = MemPager::default();

        // allocate 3 data keys + 3 index keys (batched)
        let ids = kv.alloc_many(6);
        let (data_a, data_b, data_c, p_a, p_b, p_c) =
            (ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);

        // One column with three sealed segments, newest-first.
        // Ranges: ["aa".."am"], ["b".."c"], ["wolf".."zebra"]
        let seg_a = IndexSegment::build_fixed(
            data_a,
            vec![b"aa".to_vec(), b"al".to_vec(), b"am".to_vec()],
            4,
        );
        let seg_b = IndexSegment::build_fixed(data_b, vec![b"b".to_vec(), b"c".to_vec()], 2);
        let seg_c = IndexSegment::build_var(
            data_c,
            vec![b"wolf".to_vec(), b"yak".to_vec(), b"zebra".to_vec()],
            &[7, 3, 9],
        );

        // store all three segments in one batch
        kv.batch_put_typed::<IndexSegment>(&[
            (p_a, seg_a.clone()),
            (p_b, seg_b.clone()),
            (p_c, seg_c.clone()),
        ]);

        // ColumnIndex in-memory (no IO needed)
        let col = ColumnIndex {
            field_id: 7,
            segments: vec![
                IndexSegmentRef {
                    index_physical_key: p_c,
                    logical_key_min: seg_c.logical_key_min.clone(),
                    logical_key_max: seg_c.logical_key_max.clone(),
                    n_entries: seg_c.n_entries,
                },
                IndexSegmentRef {
                    index_physical_key: p_b,
                    logical_key_min: seg_b.logical_key_min.clone(),
                    logical_key_max: seg_b.logical_key_max.clone(),
                    n_entries: seg_b.n_entries,
                },
                IndexSegmentRef {
                    index_physical_key: p_a,
                    logical_key_min: seg_a.logical_key_min.clone(),
                    logical_key_max: seg_a.logical_key_max.clone(),
                    n_entries: seg_a.n_entries,
                },
            ],
        };

        // Simple prune: keep refs whose span covers probe (inclusive).
        let cand = prune(&col.segments, &b"wolf"[..]);
        assert_eq!(cand.len(), 1);
        assert!(cand[0].logical_key_min.as_slice() <= &b"wolf"[..]);
        assert!(cand[0].logical_key_max.as_slice() >= &b"wolf"[..]);

        // Fetch the one candidate segment via a single *batch* get
        let segs: Vec<IndexSegment> =
            kv.batch_get_typed::<IndexSegment>(&[cand[0].index_physical_key]);
        let seg = &segs[0];

        // Binary search inside that segment
        let (keys, offs) = (&seg.logical_key_bytes, &seg.logical_key_offsets);
        let target = &b"wolf"[..];

        let mut lo = 0usize;
        let mut hi = seg.n_entries as usize;
        let mut hit = None;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let k = slice_key(keys, offs, mid);
            match k.cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    hit = Some(mid);
                    break;
                }
            }
        }
        let pos = hit.expect("should find wolf");
        assert_eq!(slice_key(keys, offs, pos), &b"wolf"[..]);
    }

    /// Confirms `ValueLayout` slicing rules for both FixedWidth and Variable.
    #[test]
    fn value_layout_slicing_rules() {
        let data_key = 42;

        // Fixed width = 8 → ranges are [i*8 .. (i+1)*8)
        let seg_fixed = IndexSegment::build_fixed(
            data_key,
            vec![
                b"k1".to_vec(),
                b"k2".to_vec(),
                b"k3".to_vec(),
                b"k4".to_vec(),
            ],
            8,
        );
        match seg_fixed.value_layout {
            ValueLayout::FixedWidth { width } => {
                assert_eq!(width, 8);
                for i in 0..seg_fixed.n_entries as usize {
                    let a = i as u32 * width;
                    let b = (i as u32 + 1) * width;
                    assert_eq!(a + width, b); // sanity
                }
            }
            _ => unreachable!(),
        }

        // Variable layout: explicit prefix sums
        let seg_var = IndexSegment::build_var(
            data_key,
            vec![b"a".to_vec(), b"aa".to_vec(), b"aaa".to_vec()],
            &[3, 1, 9],
        );
        match seg_var.value_layout {
            ValueLayout::Variable { ref value_offsets } => {
                assert_eq!(value_offsets, &vec![0, 3, 4, 13]);
                // i-th slice is [off[i], off[i+1])
                for i in 0..seg_var.n_entries as usize {
                    assert!(value_offsets[i] < value_offsets[i + 1]);
                }
            }
            _ => unreachable!(),
        }
    }
}
