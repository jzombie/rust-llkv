use super::*;
use crate::column_index::{ColumnIndex, IndexSegment};
use crate::layout::{IndexLayoutInfo, KeyLayout, ValueLayout};
use crate::storage::{BatchGet, GetResult, Pager};
use crate::types::{ByteOffset, ByteWidth, LogicalFieldId, PhysicalKey, TypedKind, TypedValue};
use rustc_hash::FxHashMap;
use std::fmt::Write;

impl<P: Pager> ColumnStore<P> {
    /// Scans bootstrap -> manifest -> column indexes -> segments -> data and
    /// returns a list of storage nodes with sizes and relationships. Uses
    /// batch-only I/O and works with paged layouts.
    pub fn describe_storage(&self) -> Vec<StorageNode> {
        let mut out: Vec<StorageNode> = Vec::new();

        // --- bootstrap + manifest (raw sizes)
        let bootstrap_key: PhysicalKey = self.bootstrap_key;
        let header_keys = [bootstrap_key, self.manifest_key];

        let gets = header_keys
            .iter()
            .map(|k| BatchGet::Raw { key: *k })
            .collect::<Vec<_>>();
        let resp = self.do_gets(gets);

        let mut raw_len_map: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
        for gr in resp {
            if let GetResult::Raw { key, bytes } = gr {
                raw_len_map.insert(key, bytes.as_ref().len());
            }
        }

        // bootstrap
        out.push(StorageNode {
            pk: bootstrap_key,
            stored_len: *raw_len_map.get(&bootstrap_key).unwrap_or(&0),
            kind: StorageKind::Bootstrap,
        });

        // manifest
        let column_count = self.manifest.read().unwrap().columns.len();
        out.push(StorageNode {
            pk: self.manifest_key,
            stored_len: *raw_len_map.get(&self.manifest_key).unwrap_or(&0),
            kind: StorageKind::Manifest { column_count },
        });

        // --- column indexes (typed + raw)
        let col_index_pks: Vec<PhysicalKey> = {
            let man = self.manifest.read().unwrap();
            man.columns
                .iter()
                .map(|c| c.column_index_physical_key)
                .collect()
        };

        let mut colindex_nodes: Vec<StorageNode> = Vec::new();
        let mut field_for_colindex: FxHashMap<PhysicalKey, LogicalFieldId> = FxHashMap::default();
        let mut all_seg_index_pks: Vec<PhysicalKey> = Vec::new();
        let mut seg_owner_colindex: FxHashMap<PhysicalKey, (LogicalFieldId, PhysicalKey)> =
            FxHashMap::default();

        if !col_index_pks.is_empty() {
            // raw + typed for each ColumnIndex key
            let mut gets: Vec<BatchGet> = Vec::with_capacity(col_index_pks.len() * 2);
            for pk in &col_index_pks {
                gets.push(BatchGet::Raw { key: *pk });
                gets.push(BatchGet::Typed {
                    key: *pk,
                    kind: TypedKind::ColumnIndex,
                });
            }
            let resp = self.do_gets(gets);

            let mut colindex_raw_len: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
            let mut colindices_by_pk: FxHashMap<PhysicalKey, ColumnIndex> = FxHashMap::default();

            for gr in resp {
                match gr {
                    GetResult::Raw { key, bytes } => {
                        colindex_raw_len.insert(key, bytes.as_ref().len());
                    }
                    GetResult::Typed {
                        key,
                        value: TypedValue::ColumnIndex(ci),
                    } => {
                        colindices_by_pk.insert(key, ci);
                    }
                    _ => {}
                }
            }

            for pk in &col_index_pks {
                if let Some(ci) = colindices_by_pk.get(pk) {
                    field_for_colindex.insert(*pk, ci.field_id);
                    colindex_nodes.push(StorageNode {
                        pk: *pk,
                        stored_len: *colindex_raw_len.get(pk).unwrap_or(&0),
                        kind: StorageKind::ColumnIndex {
                            field_id: ci.field_id,
                            n_segments: ci.segments.len(),
                        },
                    });

                    for sref in &ci.segments {
                        all_seg_index_pks.push(sref.index_physical_key);
                        seg_owner_colindex.insert(sref.index_physical_key, (ci.field_id, *pk));
                    }
                }
            }
        }

        out.extend(colindex_nodes);

        // --- index segments (typed + raw) and collect paged pk blobs
        let mut data_pkeys: Vec<PhysicalKey> = Vec::new();
        let mut key_bytes_pks: Vec<PhysicalKey> = Vec::new();
        let mut key_offs_pks: Vec<PhysicalKey> = Vec::new();
        let mut val_offs_pks: Vec<PhysicalKey> = Vec::new();

        let mut seg_nodes: Vec<StorageNode> = Vec::new();

        if !all_seg_index_pks.is_empty() {
            let mut gets: Vec<BatchGet> = Vec::with_capacity(all_seg_index_pks.len() * 2);
            for pk in &all_seg_index_pks {
                gets.push(BatchGet::Raw { key: *pk });
                gets.push(BatchGet::Typed {
                    key: *pk,
                    kind: TypedKind::IndexSegment,
                });
            }
            let resp = self.do_gets(gets);

            let mut seg_raw_len: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
            let mut segs_by_pk: FxHashMap<PhysicalKey, IndexSegment> = FxHashMap::default();

            for gr in resp {
                match gr {
                    GetResult::Raw { key, bytes } => {
                        seg_raw_len.insert(key, bytes.as_ref().len());
                    }
                    GetResult::Typed {
                        key,
                        value: TypedValue::IndexSegment(seg),
                    } => {
                        segs_by_pk.insert(key, seg);
                    }
                    _ => {}
                }
            }

            // Stage second round of gets to size blobs referenced by
            // the typed IndexSegments (paged fields).
            let mut second_gets: Vec<BatchGet> = Vec::new();

            for pk in &all_seg_index_pks {
                if let Some(seg) = segs_by_pk.get(pk) {
                    // book-keep owners and pks to size
                    data_pkeys.push(seg.data_physical_key);
                    key_bytes_pks.push(seg.key_bytes_pk);
                    second_gets.push(BatchGet::Raw {
                        key: seg.key_bytes_pk,
                    });

                    match &seg.key_layout {
                        KeyLayout::FixedWidth { .. } => {}
                        KeyLayout::Variable { key_offsets } => {
                            let _ = key_offsets;
                        }
                        KeyLayout::VariablePaged { offsets_pk, .. } => {
                            key_offs_pks.push(*offsets_pk);
                            second_gets.push(BatchGet::Raw { key: *offsets_pk });
                        }
                    }
                    match &seg.value_layout {
                        ValueLayout::FixedWidth { .. } => {}
                        ValueLayout::Variable { value_offsets } => {
                            let _ = value_offsets;
                        }
                        ValueLayout::VariablePaged { offsets_pk, .. } => {
                            val_offs_pks.push(*offsets_pk);
                            second_gets.push(BatchGet::Raw { key: *offsets_pk });
                        }
                    }
                }
            }

            let resp2 = if second_gets.is_empty() {
                Vec::new()
            } else {
                self.do_gets(second_gets)
            };

            let mut blob_len: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
            for gr in resp2 {
                if let GetResult::Raw { key, bytes } = gr {
                    blob_len.insert(key, bytes.as_ref().len());
                }
            }

            for pk in &all_seg_index_pks {
                if let Some(seg) = segs_by_pk.get(pk) {
                    let (field_id, owner_colindex_pk) =
                        seg_owner_colindex.get(pk).cloned().unwrap();

                    // Compute layout sizes from paged blobs.
                    let key_bytes = *blob_len.get(&seg.key_bytes_pk).unwrap_or(&0);

                    let key_offs_bytes = match &seg.key_layout {
                        KeyLayout::FixedWidth { .. } => 0,
                        KeyLayout::Variable { key_offsets } => {
                            key_offsets.len() * std::mem::size_of::<ByteOffset>()
                        }
                        KeyLayout::VariablePaged { offsets_pk, .. } => {
                            *blob_len.get(&offsets_pk).unwrap_or(&0)
                        }
                    };

                    let (kind, fixed_width, value_meta_bytes) = match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => (
                            "fixed",
                            Some(*width as ByteWidth),
                            std::mem::size_of::<ByteWidth>(),
                        ),
                        ValueLayout::Variable { value_offsets } => (
                            "variable",
                            None,
                            value_offsets.len() * std::mem::size_of::<ByteOffset>(),
                        ),
                        ValueLayout::VariablePaged { offsets_pk, .. } => {
                            ("variable", None, *blob_len.get(&offsets_pk).unwrap_or(&0))
                        }
                    };

                    let layout = IndexLayoutInfo {
                        kind,
                        fixed_width,
                        key_bytes,
                        key_offs_bytes,
                        value_meta_bytes,
                    };

                    seg_nodes.push(StorageNode {
                        pk: *pk,
                        stored_len: *seg_raw_len.get(pk).unwrap_or(&0),
                        kind: StorageKind::IndexSegment {
                            field_id,
                            n_entries: seg.n_entries,
                            layout,
                            data_pkey: seg.data_physical_key,
                            owner_colindex_pk,
                        },
                    });
                }
            }
        }

        out.extend(seg_nodes);

        // --- data blobs (raw only)
        if !data_pkeys.is_empty() {
            let gets = data_pkeys
                .iter()
                .map(|k| BatchGet::Raw { key: *k })
                .collect::<Vec<_>>();
            let resp = self.do_gets(gets);

            let mut data_len: FxHashMap<PhysicalKey, usize> = FxHashMap::default();
            for gr in resp {
                if let GetResult::Raw { key, bytes } = gr {
                    data_len.insert(key, bytes.as_ref().len());
                }
            }

            for dpk in data_pkeys {
                let owner = self.find_segment_owner_slow(&dpk).unwrap_or(0);
                out.push(StorageNode {
                    pk: dpk,
                    stored_len: *data_len.get(&dpk).unwrap_or(&0),
                    kind: StorageKind::DataBlob {
                        owner_index_pk: owner,
                    },
                });
            }
        }

        out.sort_by_key(|n| n.pk);
        out
    }

    /// Reverse edge: data pk -> owning segment pk (best-effort).
    ///
    /// No dependency on a non-existent `load_column_index`; instead, fetch
    /// typed ColumnIndex objects on demand and scan segments.
    fn find_segment_owner_slow(&self, data_pk: &PhysicalKey) -> Option<PhysicalKey> {
        let man = self.manifest.read().ok()?;
        if man.columns.is_empty() {
            return None;
        }

        // Batch fetch all column indexes as typed values.
        let gets = man
            .columns
            .iter()
            .map(|c| BatchGet::Typed {
                key: c.column_index_physical_key,
                kind: TypedKind::ColumnIndex,
            })
            .collect::<Vec<_>>();

        let resp = self.do_gets(gets);

        for gr in resp {
            if let GetResult::Typed {
                value: TypedValue::ColumnIndex(ci),
                ..
            } = gr
            {
                for s in ci.segments {
                    if s.data_physical_key == *data_pk {
                        return Some(s.index_physical_key);
                    }
                }
            }
        }
        None
    }
}
