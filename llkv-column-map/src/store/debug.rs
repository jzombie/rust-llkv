use super::ColumnStore;
use crate::{
    storage::pager::{BatchGet, GetResult, Pager},
    store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorPageHeader},
    types::CATALOG_ROOT_PKEY,
};
use arrow::{
    array::{StringBuilder, UInt64Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::pretty, // Import the pretty printer
};
use std::sync::Arc;

/// An extension trait for providing debug utilities on a ColumnStore.
pub trait ColumnStoreDebug {
    /// Renders the entire physical layout of the store into a formatted ASCII table string.
    fn render_storage_as_table(&self) -> String;
}

impl<P: Pager> ColumnStoreDebug for ColumnStore<P> {
    fn render_storage_as_table(&self) -> String {
        // Builders for our output RecordBatch
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
            let desc_blob_res = self
                .pager
                .batch_get(&[BatchGet::Raw { key: *desc_pk }])
                .ok()
                .and_then(|mut r| r.pop())
                .and_then(|r| match r {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                });

            if let Some(desc_blob) = desc_blob_res {
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
                    let page_blob_res = self
                        .pager
                        .batch_get(&[BatchGet::Raw { key: page_pk }])
                        .ok()
                        .and_then(|mut r| r.pop())
                        .and_then(|r| match r {
                            GetResult::Raw { bytes, .. } => Some(bytes),
                            _ => None,
                        });

                    if let Some(page_blob) = page_blob_res {
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

        // Format the batch into a string and return it.
        pretty::pretty_format_batches(&[batch]).unwrap().to_string()
    }
}
