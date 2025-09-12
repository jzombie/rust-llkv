use super::*;
use crate::types::PhysicalKey;
use rustc_hash::FxHashMap;

// TODO: Put behind feature flag?
// TODO: Verify that per-node byte measurements are accurate despite schema changes
impl<P: Pager> ColumnStore<P> {
    /// Scans the manifest -> ColumnIndex -> IndexSegments -> Data blobs,
    /// returns one entry per physical key with sizes and relationships.
    /// Batch-only I/O.
    pub fn describe_storage(&self) -> Vec<StorageNode> {
        let mut out: Vec<StorageNode> = Vec::new();

        // --- bootstrap + manifest (raw sizes)
        let bootstrap_key: PhysicalKey = self.bootstrap_key;
        let header_keys = [bootstrap_key, self.manifest_key];

        // Fetch raw bytes for bootstrap + manifest in one batch.
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
        let column_count = { self.manifest.read().unwrap().columns.len() };
        out.push(StorageNode {
            pk: self.manifest_key,
            stored_len: *raw_len_map.get(&self.manifest_key).unwrap_or(&0),
            kind: StorageKind::Manifest { column_count },
        });

        // --- column indexes (typed + raw in a single pass)
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
            // Combined batch: raw + typed for each ColumnIndex key
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

                    // collect segment index pkeys and owners
                    for sref in &ci.segments {
                        all_seg_index_pks.push(sref.index_physical_key);
                        seg_owner_colindex.insert(sref.index_physical_key, (ci.field_id, *pk));
                    }
                }
            }
        }

        out.extend(colindex_nodes);

        // --- index segments (typed + raw) and discover data pkeys
        let mut data_pkeys: Vec<PhysicalKey> = Vec::new();
        let mut owner_for_data: FxHashMap<PhysicalKey, PhysicalKey> = FxHashMap::default(); // data_pkey -> index_segment_pk
        let mut seg_nodes: Vec<StorageNode> = Vec::new();

        if !all_seg_index_pks.is_empty() {
            // Combined batch: raw + typed for each IndexSegment key
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

            for pk in &all_seg_index_pks {
                if let Some(seg) = segs_by_pk.get(pk) {
                    let (field_id, owner_colindex_pk) =
                        seg_owner_colindex.get(pk).cloned().unwrap();

                    // book-keep data blob for later
                    data_pkeys.push(seg.data_physical_key);
                    owner_for_data.insert(seg.data_physical_key, *pk);

                    // compute layout info
                    let key_bytes = seg.logical_key_bytes.len();
                    let key_offs_bytes = match &seg.key_layout {
                        KeyLayout::FixedWidth { .. } => 0,
                        KeyLayout::Variable { key_offsets } => {
                            key_offsets.len() * std::mem::size_of::<ByteOffset>()
                        }
                    };
                    let (kind, fixed_width, value_meta_bytes) = match &seg.value_layout {
                        ValueLayout::FixedWidth { width } => {
                            ("fixed", Some(*width), std::mem::size_of::<ByteWidth>())
                        }
                        ValueLayout::Variable { value_offsets } => (
                            "variable",
                            None,
                            value_offsets.len() * std::mem::size_of::<ByteOffset>(),
                        ),
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
                let owner = *owner_for_data.get(&dpk).unwrap();
                out.push(StorageNode {
                    pk: dpk,
                    stored_len: *data_len.get(&dpk).unwrap_or(&0),
                    kind: StorageKind::DataBlob {
                        owner_index_pk: owner,
                    },
                });
            }
        }

        // deterministic order
        out.sort_by_key(|n| n.pk);
        out
    }

    /// Renders a compact ASCII table of the current storage layout.
    /// (bytes column moved before details to keep the table aligned)
    pub fn render_storage_ascii(&self) -> String {
        let nodes = self.describe_storage();
        let mut s = String::new();

        // Header: phys_key | kind | field | bytes | details
        let header = format!(
            "{:<10} {:<12} {:<9} {:>10}  {}",
            "phys_key", "kind", "field", "bytes", "details"
        );
        let _ = writeln!(&mut s, "{header}");

        // Divider length covers the fixed-width columns (details is free-width at the end)
        let _ = writeln!(&mut s, "{}", "-".repeat(header.len()));

        for n in nodes {
            match n.kind {
                StorageKind::Bootstrap => {
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  -",
                        n.pk, "bootstrap", "-", n.stored_len,
                    );
                }
                StorageKind::Manifest { column_count } => {
                    let det = format!("columns={}", column_count);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "manifest", "-", n.stored_len, det
                    );
                }
                StorageKind::ColumnIndex {
                    field_id,
                    n_segments,
                } => {
                    let det = format!("segments={}", n_segments);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "col_index", field_id, n.stored_len, det
                    );
                }
                StorageKind::IndexSegment {
                    field_id,
                    n_entries,
                    layout,
                    data_pkey,
                    owner_colindex_pk,
                } => {
                    let det = match layout.fixed_width {
                        Some(w) => format!(
                            "entries={} layout=fixed({}) key_bytes={} key_offs={} val_meta={} data_pk={} colidx_pk={}",
                            n_entries,
                            w,
                            layout.key_bytes,
                            layout.key_offs_bytes,
                            layout.value_meta_bytes,
                            data_pkey,
                            owner_colindex_pk
                        ),
                        None => format!(
                            "entries={} layout=variable key_bytes={} key_offs={} val_meta={} data_pk={} colidx_pk={}",
                            n_entries,
                            layout.key_bytes,
                            layout.key_offs_bytes,
                            layout.value_meta_bytes,
                            data_pkey,
                            owner_colindex_pk
                        ),
                    };
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "idx_segment", field_id, n.stored_len, det
                    );
                }
                StorageKind::DataBlob { owner_index_pk } => {
                    let det = format!("owner_idx_pk={}", owner_index_pk);
                    let _ = writeln!(
                        &mut s,
                        "{:<10} {:<12} {:<9} {:>10}  {}",
                        n.pk, "data_blob", "-", n.stored_len, det
                    );
                }
            }
        }
        s
    }

    /// Renders a Graphviz DOT graph showing edges:
    /// bootstrap -> manifest -> ColumnIndex -> IndexSegment -> DataBlob
    pub fn render_storage_dot(&self) -> String {
        let nodes = self.describe_storage();

        // index nodes by pk for quick lookup
        let mut map: FxHashMap<PhysicalKey, &StorageNode> = FxHashMap::default();
        for n in &nodes {
            map.insert(n.pk, n);
        }

        let mut s = String::new();
        let _ = writeln!(&mut s, "digraph storage {{");
        let _ = writeln!(&mut s, "  node [shape=box, fontname=\"monospace\"];");

        // emit nodes
        for n in &nodes {
            match &n.kind {
                StorageKind::Bootstrap => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"Bootstrap pk={} bytes={}\"];",
                        n.pk, n.pk, n.stored_len
                    );
                    // edge to manifest
                    let _ = writeln!(&mut s, "  n{} -> n{};", n.pk, self.manifest_key);
                }
                StorageKind::Manifest { column_count } => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"Manifest pk={} columns={} bytes={}\"];",
                        n.pk, n.pk, column_count, n.stored_len
                    );
                    // edges to column indexes
                    for e in &self.manifest.read().unwrap().columns {
                        let _ =
                            writeln!(&mut s, "  n{} -> n{};", n.pk, e.column_index_physical_key);
                    }
                }
                StorageKind::ColumnIndex {
                    field_id,
                    n_segments,
                } => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"ColumnIndex pk={} field={} segs={} bytes={}\"];",
                        n.pk, n.pk, field_id, n_segments, n.stored_len
                    );
                }
                StorageKind::IndexSegment {
                    field_id,
                    n_entries,
                    layout,
                    data_pkey,
                    owner_colindex_pk,
                } => {
                    let lay = match layout.fixed_width {
                        Some(w) => format!("fixed({})", w),
                        None => "variable".to_string(),
                    };
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"IndexSegment pk={} field={} entries={} layout={} idx_bytes={} (key_bytes={}, key_offs={}, val_meta={})\"];",
                        n.pk,
                        n.pk,
                        field_id,
                        n_entries,
                        lay,
                        n.stored_len,
                        layout.key_bytes,
                        layout.key_offs_bytes,
                        layout.value_meta_bytes
                    );
                    // owner colindex and data edge
                    let _ = writeln!(&mut s, "  n{} -> n{};", owner_colindex_pk, n.pk);
                    let _ = writeln!(&mut s, "  n{} -> n{};", n.pk, data_pkey);
                }
                StorageKind::DataBlob { owner_index_pk } => {
                    let _ = writeln!(
                        &mut s,
                        "  n{} [label=\"DataBlob pk={} bytes={}\"];",
                        n.pk, n.pk, n.stored_len
                    );
                    let _ = writeln!(&mut s, "  n{} -> n{};", owner_index_pk, n.pk);
                }
            }
        }

        let _ = writeln!(&mut s, "}}");
        s
    }
}
