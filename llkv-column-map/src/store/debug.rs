use super::ColumnStore;
use crate::{
    storage::pager::{BatchGet, GetResult, Pager},
    store::{
        catalog::ColumnCatalog,
        descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorPageHeader},
    },
    types::{CATALOG_ROOT_PKEY, PhysicalKey},
};
use arrow::{
    array::{StringBuilder, UInt64Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::pretty,
};
use std::collections::HashMap;
use std::sync::Arc;

/// An extension trait for providing debug utilities on a ColumnStore.
pub trait ColumnStoreDebug {
    /// Renders the entire physical layout of the store into a formatted ASCII table string.
    fn render_storage_as_formatted_string(&self) -> String;

    /// Renders the physical layout of the store into a Graphviz `.dot` file string.
    /// The caller provides a map to color nodes based on an external category, like batch number.
    fn render_storage_as_dot(&self, batch_colors: &HashMap<PhysicalKey, usize>) -> String;
}

/// A standalone utility to discover all physical keys reachable from the catalog.
pub fn discover_all_pks<P: Pager>(pager: &P) -> Vec<PhysicalKey> {
    let mut out = Vec::new();
    out.push(CATALOG_ROOT_PKEY);

    if let Some(GetResult::Raw {
        bytes: cat_blob, ..
    }) = pager
        .batch_get(&[BatchGet::Raw {
            key: CATALOG_ROOT_PKEY,
        }])
        .unwrap()
        .pop()
    {
        let cat = ColumnCatalog::from_bytes(cat_blob.as_ref()).unwrap();
        for (_fid, desc_pk) in cat.map.iter() {
            out.push(*desc_pk);

            // Walk descriptor pages
            let desc_blob = pager
                .batch_get(&[BatchGet::Raw { key: *desc_pk }])
                .unwrap()
                .pop()
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .unwrap();
            let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
            let mut page_pk = desc.head_page_pk;
            while page_pk != 0 {
                out.push(page_pk);
                let page_blob = pager
                    .batch_get(&[BatchGet::Raw { key: page_pk }])
                    .unwrap()
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .unwrap();
                let bytes = page_blob.as_ref();
                let hdr_sz = DescriptorPageHeader::DISK_SIZE;
                let hd = DescriptorPageHeader::from_le_bytes(&bytes[..hdr_sz]);

                // Collect chunk-related pkeys
                for i in 0..(hd.entry_count as usize) {
                    let off = hdr_sz + i * ChunkMetadata::DISK_SIZE;
                    let end = off + ChunkMetadata::DISK_SIZE;
                    let meta = ChunkMetadata::from_le_bytes(&bytes[off..end]);
                    out.push(meta.chunk_pk);
                    if meta.value_order_perm_pk != 0 {
                        out.push(meta.value_order_perm_pk);
                    }
                }

                page_pk = hd.next_page_pk;
            }
        }
    }

    out.sort_unstable();
    out.dedup();
    out
}

fn color_for_batch(b: usize) -> &'static str {
    match b {
        0 => "white", // bootstrap/manifest
        1 => "lightskyblue",
        2 => "palegreen",
        3 => "khaki",
        4 => "lightpink",
        _ => "lightgray",
    }
}

impl<P: Pager> ColumnStoreDebug for ColumnStore<P> {
    fn render_storage_as_formatted_string(&self) -> String {
        let mut type_builder = StringBuilder::new();
        let mut logical_id_builder = StringBuilder::new();
        let mut pk_builder = UInt64Builder::new();
        let mut details_builder = StringBuilder::new();

        let schema = Arc::new(Schema::new(vec![
            Field::new("ObjectType", DataType::Utf8, false),
            Field::new("LogicalID", DataType::Utf8, true),
            Field::new("PhysicalKey", DataType::UInt64, false),
            Field::new("Details", DataType::Utf8, false),
        ]));

        let catalog = self.catalog.read().unwrap();

        type_builder.append_value("Catalog");
        logical_id_builder.append_null();
        pk_builder.append_value(CATALOG_ROOT_PKEY);
        details_builder.append_value(format!("{} entries", catalog.map.len()));

        for (fid, desc_pk) in catalog.map.iter() {
            if let Some(GetResult::Raw {
                bytes: desc_blob, ..
            }) = self
                .pager
                .batch_get(&[BatchGet::Raw { key: *desc_pk }])
                .ok()
                .and_then(|mut r| r.pop())
            {
                let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

                type_builder.append_value("  L Field");
                logical_id_builder.append_value(format!("{:?}", fid));
                pk_builder.append_value(*desc_pk);
                details_builder.append_value(format!(
                    "Rows: {}, Chunks: {}",
                    desc.total_row_count, desc.total_chunk_count
                ));

                let mut page_pk = desc.head_page_pk;
                while page_pk != 0 {
                    if let Some(GetResult::Raw {
                        bytes: page_blob, ..
                    }) = self
                        .pager
                        .batch_get(&[BatchGet::Raw { key: page_pk }])
                        .ok()
                        .and_then(|mut r| r.pop())
                    {
                        let bytes = page_blob.as_ref();
                        let hdr_sz = DescriptorPageHeader::DISK_SIZE;
                        let hd = DescriptorPageHeader::from_le_bytes(&bytes[..hdr_sz]);

                        type_builder.append_value("    L Page");
                        logical_id_builder.append_null();
                        pk_builder.append_value(page_pk);
                        details_builder.append_value(format!("Entries: {}", hd.entry_count));

                        for i in 0..(hd.entry_count as usize) {
                            let off = hdr_sz + i * ChunkMetadata::DISK_SIZE;
                            let end = off + ChunkMetadata::DISK_SIZE;
                            let meta = ChunkMetadata::from_le_bytes(&bytes[off..end]);

                            type_builder.append_value("      L Chunk");
                            logical_id_builder.append_null();
                            pk_builder.append_value(meta.chunk_pk);
                            details_builder.append_value(format!(
                                "Rows: {}, Est. Bytes: {}",
                                meta.row_count, meta.serialized_bytes
                            ));
                        }
                        page_pk = hd.next_page_pk;
                    } else {
                        page_pk = 0; // Break loop if page is missing
                    }
                }
            }
        }

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(type_builder.finish()),
                Arc::new(logical_id_builder.finish()),
                Arc::new(pk_builder.finish()),
                Arc::new(details_builder.finish()),
            ],
        )
        .unwrap();

        pretty::pretty_format_batches(&[batch]).unwrap().to_string()
    }

    fn render_storage_as_dot(&self, batch_colors: &HashMap<PhysicalKey, usize>) -> String {
        use std::fmt::Write;
        let pager = &self.pager;
        let mut s = String::new();

        writeln!(&mut s, "digraph storage {{").unwrap();
        writeln!(&mut s, "  rankdir=LR;").unwrap();
        writeln!(&mut s, "  node [shape=box, fontname=\"monospace\"];").unwrap();

        let cat_pk = CATALOG_ROOT_PKEY;
        let cat_color = color_for_batch(*batch_colors.get(&cat_pk).unwrap_or(&0));

        let catalog = self.catalog.read().unwrap();
        writeln!(
            &mut s,
            "  n{} [label=\"Catalog pk={} entries={}\" style=filled fillcolor={}];",
            cat_pk,
            cat_pk,
            catalog.map.len(),
            cat_color
        )
        .unwrap();

        for (fid, desc_pk) in catalog.map.iter() {
            if let Some(GetResult::Raw {
                bytes: desc_blob, ..
            }) = pager
                .batch_get(&[BatchGet::Raw { key: *desc_pk }])
                .unwrap()
                .pop()
            {
                let desc = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());
                let dcol = color_for_batch(*batch_colors.get(desc_pk).unwrap_or(&0));
                writeln!(
                    &mut s,
                    "  n{} [label=\"ColumnDescriptor pk={} field={:?} rows={} chunks={}\" style=filled fillcolor={}];",
                    desc_pk, desc_pk, fid, desc.total_row_count, desc.total_chunk_count, dcol
                )
                .unwrap();
                writeln!(&mut s, "  n{} -> n{};", cat_pk, desc_pk).unwrap();

                let mut page_pk = desc.head_page_pk;
                let mut prev_page: Option<PhysicalKey> = None;
                while page_pk != 0 {
                    let pcol = color_for_batch(*batch_colors.get(&page_pk).unwrap_or(&0));
                    if let Some(GetResult::Raw {
                        bytes: page_blob, ..
                    }) = pager
                        .batch_get(&[BatchGet::Raw { key: page_pk }])
                        .unwrap()
                        .pop()
                    {
                        let bytes = page_blob.as_ref();
                        let hdr_sz = DescriptorPageHeader::DISK_SIZE;
                        let hd = DescriptorPageHeader::from_le_bytes(&bytes[..hdr_sz]);

                        writeln!(
                            &mut s,
                            "  n{} [label=\"DescPage pk={} entries={}\" style=filled fillcolor={}];",
                            page_pk, page_pk, hd.entry_count, pcol
                        )
                        .unwrap();

                        if let Some(ppk) = prev_page {
                            writeln!(&mut s, "  n{} -> n{};", ppk, page_pk).unwrap();
                        } else {
                            writeln!(&mut s, "  n{} -> n{};", desc_pk, page_pk).unwrap();
                        }

                        for i in 0..(hd.entry_count as usize) {
                            let off = hdr_sz + i * ChunkMetadata::DISK_SIZE;
                            let end = off + ChunkMetadata::DISK_SIZE;
                            let meta = ChunkMetadata::from_le_bytes(&bytes[off..end]);

                            if let Some(GetResult::Raw { bytes: b, .. }) = pager
                                .batch_get(&[BatchGet::Raw { key: meta.chunk_pk }])
                                .unwrap()
                                .pop()
                            {
                                let len = b.as_ref().len();
                                let col = color_for_batch(
                                    *batch_colors.get(&meta.chunk_pk).unwrap_or(&0),
                                );
                                writeln!(
                                    &mut s,
                                    "  n{} [label=\"Data pk={} bytes={}\" style=filled fillcolor={}];",
                                    meta.chunk_pk, meta.chunk_pk, len, col
                                )
                                .unwrap();
                                writeln!(&mut s, "  n{} -> n{};", page_pk, meta.chunk_pk).unwrap();
                            }

                            if meta.value_order_perm_pk != 0
                                && let Some(GetResult::Raw { bytes: b, .. }) = pager
                                    .batch_get(&[BatchGet::Raw {
                                        key: meta.value_order_perm_pk,
                                    }])
                                    .unwrap()
                                    .pop()
                            {
                                let len = b.as_ref().len();
                                let col = color_for_batch(
                                    *batch_colors.get(&meta.value_order_perm_pk).unwrap_or(&0),
                                );
                                writeln!(
                                        &mut s,
                                        "  n{} [label=\"Perm pk={} bytes={}\" style=filled fillcolor={}];",
                                        meta.value_order_perm_pk, meta.value_order_perm_pk, len, col
                                    )
                                    .unwrap();
                                writeln!(
                                    &mut s,
                                    "  n{} -> n{};",
                                    page_pk, meta.value_order_perm_pk
                                )
                                .unwrap();
                            }
                        }
                        prev_page = Some(page_pk);
                        page_pk = hd.next_page_pk;
                    } else {
                        page_pk = 0;
                    }
                }
            }
        }

        writeln!(&mut s, "  subgraph cluster_legend {{").unwrap();
        writeln!(&mut s, "    label=\"Batch legend\";").unwrap();

        let max_batch = batch_colors.values().max().cloned().unwrap_or(0);
        for b in 0..=max_batch {
            writeln!(
                &mut s,
                "    l{} [label=\"batch {}\" shape=box style=filled fillcolor={}];",
                b,
                b,
                color_for_batch(b)
            )
            .unwrap();
        }

        if max_batch > 0 {
            let legend_nodes: Vec<String> = (0..=max_batch).map(|b| format!("l{}", b)).collect();
            writeln!(&mut s, "    {} [style=invis];", legend_nodes.join(" -> ")).unwrap();
        }

        writeln!(&mut s, "  }}").unwrap();
        writeln!(&mut s, "}}").unwrap();
        s
    }
}
