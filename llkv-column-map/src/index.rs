use bitcode::{Decode, Encode};

use crate::layout::{KeyLayout, ValueLayout};
use crate::types::{
    ByteLen, ByteWidth, IndexEntryCount, LogicalFieldId, LogicalKeyBytes, PhysicalKey,
};

// pick a small, cache-friendly cap
pub const VALUE_BOUND_MAX: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, bitcode::Encode, bitcode::Decode)]
pub struct ValueBound {
    /// Full byte length of the value this bound represents.
    pub total_len: ByteLen,
    /// First min(VALUE_BOUND_MAX, total_len) bytes of that value.
    pub prefix: Vec<u8>,
}

impl ValueBound {
    #[inline]
    pub fn from_bytes(b: &[u8]) -> Self {
        let take = core::cmp::min(b.len(), VALUE_BOUND_MAX);
        Self {
            total_len: b.len() as u32,
            prefix: b[..take].to_vec(),
        }
    }

    #[inline]
    pub fn is_truncated(&self) -> bool {
        (self.prefix.len() as u32) < self.total_len
    }
}

// ── Bootstrapping ────────────────────────────────────────────────────────────
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

// TODO: [perf] Ensure bounds are updated as keys are dereferenced.
/// Pointer to a sealed segment + fast-prune info (all LOGICAL).
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

// ── Your locked index segment & value layout ─────────────────────────────────
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BOOTSTRAP_PKEY;
    use crate::pager::{BatchGet, BatchPut, GetResult, MemPager, Pager};
    use crate::types::{TypedKind, TypedValue};

    // ---------------- helpers to build/slice logical keys ----------------

    // TODO: Extract
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
        let kv = MemPager::default();

        // ----- allocate physical ids (batched) -----
        let ids = kv.alloc_many(7).unwrap();
        let data_100 = ids[0];
        let idx_100 = ids[1];
        let data_200 = ids[2];
        let idx_200 = ids[3];
        let col_100_idx_pkey = ids[4];
        let col_200_idx_pkey = ids[5];
        let manifest_pkey = ids[6];

        // ----- build two segments for columns 100 and 200 -----
        let keys_100 = vec![b"a".to_vec(), b"b".to_vec(), b"d".to_vec()];
        let seg_100 = IndexSegment::build_fixed(data_100, keys_100.clone(), 1);

        let animals = vec![b"ant".to_vec(), b"wolf".to_vec(), b"zebra".to_vec()];
        let sizes: Vec<u32> = vec![5, 2, 8]; // arbitrary payload lengths
        let seg_200 = IndexSegment::build_var(data_200, animals.clone(), &sizes);

        // store index segments (one batch)
        kv.batch_put(&[
            BatchPut::Typed {
                key: idx_100,
                value: TypedValue::IndexSegment(seg_100.clone()),
            },
            BatchPut::Typed {
                key: idx_200,
                value: TypedValue::IndexSegment(seg_200.clone()),
            },
        ])
        .unwrap();

        // ----- ColumnIndex blobs (newest-first segments) -----
        let col_100_index = ColumnIndex {
            field_id: 100,
            segments: vec![IndexSegmentRef {
                index_physical_key: idx_100,
                data_physical_key: data_100,
                logical_key_min: b"a".to_vec(),
                logical_key_max: b"d".to_vec(),
                value_min: None,
                value_max: None,
                n_entries: seg_100.n_entries,
            }],
        };
        let col_200_index = ColumnIndex {
            field_id: 200,
            segments: vec![IndexSegmentRef {
                index_physical_key: idx_200,
                data_physical_key: data_200,
                logical_key_min: animals.first().unwrap().clone(), // "ant"
                logical_key_max: animals.last().unwrap().clone(),  // "zebra"
                value_min: None,
                value_max: None,
                n_entries: seg_200.n_entries,
            }],
        };

        kv.batch_put(&[
            BatchPut::Typed {
                key: col_100_idx_pkey,
                value: TypedValue::ColumnIndex(col_100_index),
            },
            BatchPut::Typed {
                key: col_200_idx_pkey,
                value: TypedValue::ColumnIndex(col_200_index),
            },
        ])
        .unwrap();

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

        // ----- Bootstrap at physical key 0 points to the manifest -----
        kv.batch_put(&[
            BatchPut::Typed {
                key: manifest_pkey,
                value: TypedValue::Manifest(manifest),
            },
            BatchPut::Typed {
                key: BOOTSTRAP_PKEY,
                value: TypedValue::Bootstrap(Bootstrap {
                    manifest_physical_key: manifest_pkey,
                }),
            },
        ])
        .unwrap();

        // ======== Walk it back as a user would (batched reads only) =========

        // 0) Find the manifest
        let resp = kv
            .batch_get(&[BatchGet::Typed {
                key: BOOTSTRAP_PKEY,
                kind: TypedKind::Bootstrap,
            }])
            .unwrap();
        let boot = match &resp[0] {
            GetResult::Typed {
                value: TypedValue::Bootstrap(b),
                ..
            } => b.clone(),
            _ => panic!("bootstrap missing"),
        };

        let resp = kv
            .batch_get(&[BatchGet::Typed {
                key: boot.manifest_physical_key,
                kind: TypedKind::Manifest,
            }])
            .unwrap();
        let got_manifest = match &resp[0] {
            GetResult::Typed {
                value: TypedValue::Manifest(m),
                ..
            } => m.clone(),
            _ => panic!("manifest missing"),
        };

        // 1) Read ColumnIndex blobs in one batch
        let idx_keys: Vec<PhysicalKey> = got_manifest
            .columns
            .iter()
            .map(|c| c.column_index_physical_key)
            .collect();

        let gets = idx_keys
            .iter()
            .map(|k| BatchGet::Typed {
                key: *k,
                kind: TypedKind::ColumnIndex,
            })
            .collect::<Vec<_>>();

        let resp = kv.batch_get(&gets).unwrap();

        let mut col_indexes: Vec<ColumnIndex> = Vec::with_capacity(idx_keys.len());
        for gr in resp {
            match gr {
                GetResult::Typed {
                    value: TypedValue::ColumnIndex(ci),
                    ..
                } => col_indexes.push(ci),
                _ => panic!("ColumnIndex missing"),
            }
        }

        // map by field_id for assertions
        col_indexes.sort_by_key(|ci| ci.field_id);
        let col100_index = col_indexes.iter().find(|ci| ci.field_id == 100).unwrap();
        let col200_index = col_indexes.iter().find(|ci| ci.field_id == 200).unwrap();

        // 2) Segment refs (newest-first) – assert min/max came through
        assert_eq!(col100_index.segments.len(), 1);
        assert_eq!(col200_index.segments.len(), 1);
        let segref_100 = &col100_index.segments[0];
        let segref_200 = &col200_index.segments[0];

        assert_eq!(segref_100.logical_key_min, b"a".to_vec());
        assert_eq!(segref_100.logical_key_max, b"d".to_vec());
        assert_eq!(segref_200.logical_key_min, b"ant".to_vec());
        assert_eq!(segref_200.logical_key_max, b"zebra".to_vec());

        // 3) Open both IndexSegments in a single batch read (unchanged)
        let resp = kv
            .batch_get(&[
                BatchGet::Typed {
                    key: segref_100.index_physical_key,
                    kind: TypedKind::IndexSegment,
                },
                BatchGet::Typed {
                    key: segref_200.index_physical_key,
                    kind: TypedKind::IndexSegment,
                },
            ])
            .unwrap();

        let mut segs: Vec<IndexSegment> = Vec::new();
        for gr in resp {
            match gr {
                GetResult::Typed {
                    value: TypedValue::IndexSegment(s),
                    ..
                } => segs.push(s),
                _ => panic!("IndexSegment missing"),
            }
        }

        let seg100 = &segs[0];
        let seg200 = &segs[1];

        assert_eq!(seg100.n_entries, 3);
        match seg100.value_layout {
            ValueLayout::FixedWidth { width } => assert_eq!(width, 1),
            _ => panic!("expected fixed width"),
        }

        // Inspect keys via KeyLayout
        let k0 = KeyLayout::slice_key_by_layout(&seg100.logical_key_bytes, &seg100.key_layout, 0);
        let k1 = KeyLayout::slice_key_by_layout(&seg100.logical_key_bytes, &seg100.key_layout, 1);
        let k2 = KeyLayout::slice_key_by_layout(&seg100.logical_key_bytes, &seg100.key_layout, 2);
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
        let kv = MemPager::default();

        // allocate 3 data keys + 3 index keys (batched)
        let ids = kv.alloc_many(6).unwrap();
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
        kv.batch_put(&[
            BatchPut::Typed {
                key: p_a,
                value: TypedValue::IndexSegment(seg_a.clone()),
            },
            BatchPut::Typed {
                key: p_b,
                value: TypedValue::IndexSegment(seg_b.clone()),
            },
            BatchPut::Typed {
                key: p_c,
                value: TypedValue::IndexSegment(seg_c.clone()),
            },
        ])
        .unwrap();

        // ColumnIndex in-memory (no IO needed) — populate logical bounds here
        let col = ColumnIndex {
            field_id: 7,
            segments: vec![
                IndexSegmentRef {
                    index_physical_key: p_c,
                    data_physical_key: data_c,
                    logical_key_min: b"wolf".to_vec(),
                    logical_key_max: b"zebra".to_vec(),
                    value_min: None,
                    value_max: None,
                    n_entries: seg_c.n_entries,
                },
                IndexSegmentRef {
                    index_physical_key: p_b,
                    data_physical_key: data_b,
                    logical_key_min: b"b".to_vec(),
                    logical_key_max: b"c".to_vec(),
                    value_min: None,
                    value_max: None,
                    n_entries: seg_b.n_entries,
                },
                IndexSegmentRef {
                    index_physical_key: p_a,
                    data_physical_key: data_a,
                    logical_key_min: b"aa".to_vec(),
                    logical_key_max: b"am".to_vec(),
                    value_min: None,
                    value_max: None,
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
        let resp = kv
            .batch_get(&[BatchGet::Typed {
                key: cand[0].index_physical_key,
                kind: TypedKind::IndexSegment,
            }])
            .unwrap();
        let seg = match &resp[0] {
            GetResult::Typed {
                value: TypedValue::IndexSegment(s),
                ..
            } => s.clone(),
            _ => panic!("IndexSegment missing"),
        };

        // Binary search inside that segment using KeyLayout
        let target = &b"wolf"[..];

        let mut lo = 0usize;
        let mut hi = seg.n_entries as usize;
        let mut hit = None;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let k = KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, mid);
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
        assert_eq!(
            KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, pos),
            &b"wolf"[..]
        );
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

        // And for the key layout in both cases:
        // seg_fixed keys are all 2 bytes ("k1","k2","k3","k4")? No, the inputs are len=2 each,
        // so FixedWidth is selected; seg_var keys ("a","aa","aaa") invoke Variable (with offsets).
        match &seg_fixed.key_layout {
            KeyLayout::FixedWidth { width } => assert_eq!(*width, 2),
            _ => panic!("expected fixed-width key layout"),
        }
        match &seg_var.key_layout {
            KeyLayout::Variable { key_offsets } => {
                assert_eq!(key_offsets, &vec![0, 1, 3, 6]); // 1 + 2 + 3 = 6
            }
            _ => panic!("expected variable key layout"),
        }
    }

    #[test]
    fn min_max_fixed_width_keys() {
        // three 8-byte big-endian u64 keys: 1, 5, 9
        let keys = vec![
            1u64.to_be_bytes().to_vec(),
            9u64.to_be_bytes().to_vec(),
            5u64.to_be_bytes().to_vec(),
        ];
        let seg = IndexSegment::build_fixed(123, keys, 4 /* any width for values */);

        // bounds must be inclusive and lexicographic — derive by slicing first/last
        let first = KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, 0);
        let last = KeyLayout::slice_key_by_layout(
            &seg.logical_key_bytes,
            &seg.key_layout,
            seg.n_entries as usize - 1,
        );
        assert_eq!(first, &1u64.to_be_bytes()[..]);
        assert_eq!(last, &9u64.to_be_bytes()[..]);

        // layout must be FixedWidth for keys too (since all keys same length)
        match seg.key_layout {
            KeyLayout::FixedWidth { width } => {
                // 8 bytes per *logical key*
                assert_eq!(width, 8);
            }
            _ => panic!("expected fixed-width key layout"),
        }
    }

    #[test]
    fn min_max_variable_width_keys() {
        // mixed-length keys -> variable layout for keys
        let keys = vec![
            b"a".to_vec(),
            b"wolf".to_vec(),
            b"zebra".to_vec(),
            b"ant".to_vec(),
        ];
        let sizes = vec![1u32; keys.len()]; // dummy value sizes
        let seg = IndexSegment::build_var(777, keys, &sizes);

        // derive lexicographic min/max by slicing packed keys
        let first = KeyLayout::slice_key_by_layout(&seg.logical_key_bytes, &seg.key_layout, 0);
        let last = KeyLayout::slice_key_by_layout(
            &seg.logical_key_bytes,
            &seg.key_layout,
            seg.n_entries as usize - 1,
        );
        assert_eq!(first, b"a");
        assert_eq!(last, b"zebra");

        match &seg.key_layout {
            KeyLayout::Variable { key_offsets } => {
                // one extra offset than entries
                assert_eq!(key_offsets.len(), seg.n_entries as usize + 1);
            }
            _ => panic!("expected variable key layout"),
        }
    }
}
