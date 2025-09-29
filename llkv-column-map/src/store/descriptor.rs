//! Streamable, paged metadata for a single column.
//!
//! This module avoids deserialization by defining fixed-layout structs that
//! can be interpreted directly from byte buffers provided by the pager.

use crate::store::indexing::IndexKind;
use crate::types::LogicalFieldId;
use llkv_result::{Error, Result};
use rkyv::{Archive, Deserialize, Serialize};
use rkyv::api::high::{to_bytes, from_bytes_unchecked};
use llkv_storage::{
    codecs::{read_u32_le, read_u64_le, write_u32_le, write_u64_le},
    pager::{BatchGet, BatchPut, GetResult, Pager},
    types::PhysicalKey,
};
use std::mem;
use std::sync::Arc;

// All values are stored as little-endian.

/// Fixed-size metadata for a single Arrow array chunk.
/// This struct's memory layout IS the on-disk format.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(crate) struct ChunkMetadata {
    pub(crate) chunk_pk: PhysicalKey,
    pub(crate) value_order_perm_pk: PhysicalKey, // Using 0 as None
    pub(crate) row_count: u64,
    pub(crate) serialized_bytes: u64,
    pub(crate) min_val_u64: u64, // For pruning, assumes u64-comparable values
    pub(crate) max_val_u64: u64,
}

impl ChunkMetadata {
    pub(crate) const DISK_SIZE: usize = mem::size_of::<Self>();

    /// Single allocation; append fields as LE without growth churn.
    pub(crate) fn to_le_bytes(self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::DISK_SIZE);
        write_u64_le(&mut buf, self.chunk_pk);
        write_u64_le(&mut buf, self.value_order_perm_pk);
        write_u64_le(&mut buf, self.row_count);
        write_u64_le(&mut buf, self.serialized_bytes);
        write_u64_le(&mut buf, self.min_val_u64);
        write_u64_le(&mut buf, self.max_val_u64);
        buf
    }

    pub(crate) fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut o = 0usize;
        let chunk_pk = read_u64_le(bytes, &mut o);
        let value_order_perm_pk = read_u64_le(bytes, &mut o);
        let row_count = read_u64_le(bytes, &mut o);
        let serialized_bytes = read_u64_le(bytes, &mut o);
        let min_val_u64 = read_u64_le(bytes, &mut o);
        let max_val_u64 = read_u64_le(bytes, &mut o);

        Self {
            chunk_pk,
            value_order_perm_pk,
            row_count,
            serialized_bytes,
            min_val_u64,
            max_val_u64,
        }
    }
}

/// The root metadata object for a single column.
/// Points to the head/tail of a linked list of DescriptorPages.
#[derive(Clone, Debug, Default)]
pub(crate) struct ColumnDescriptor {
    pub(crate) field_id: LogicalFieldId,
    pub(crate) head_page_pk: PhysicalKey,
    pub(crate) tail_page_pk: PhysicalKey,
    pub(crate) total_row_count: u64,
    pub(crate) total_chunk_count: u64,
    /// Optional Arrow data type code (0 => unknown). Persisted to avoid peeking chunks.
    pub(crate) data_type_code: u32,
    pub(crate) _padding: u32,
    /// Serialized index metadata.
    pub(crate) index_metadata: Vec<u8>,
}

// Simple header: 4-byte magic + 1-byte version
const RKYV_MAGIC: &[u8; 4] = b"RKYD";
const RKYV_VERSION: u8 = 1;

impl ColumnDescriptor {
    /// Encode descriptor with rkyv and a small header (magic + version).
    pub(crate) fn encode_rkyv(&self) -> Result<Vec<u8>> {
        let helper = ColumnDescriptorRkyv {
            field_id_u64: self.field_id.into(),
            head_page_pk: self.head_page_pk,
            tail_page_pk: self.tail_page_pk,
            total_row_count: self.total_row_count,
            total_chunk_count: self.total_chunk_count,
            data_type_code: self.data_type_code,
            _padding: self._padding,
            index_metadata: self.index_metadata.clone(),
        };

        // Use the rkyv high-level API to produce a byte buffer. rkyv expects
        // a `rancor::Source` error type; map its errors into our Error type.
        let bytes = to_bytes::<rkyv::rancor::Error>(&helper)
            .map_err(|e| Error::Internal(format!("rkyv serialize error: {}", e)))?;

        let payload: &[u8] = bytes.as_ref();

        // rkyv requires the archived root to be aligned. We store a small
        // header before the payload, so insert padding bytes so the payload
        // begins at an offset aligned for the archived type.
        let required_align = std::mem::align_of::<<ColumnDescriptorRkyv as Archive>::Archived>();
        let header_len = 4 + 1; // magic (4) + version (1)
        let pad = (required_align - (header_len % required_align)) % required_align;

        let mut out = Vec::with_capacity(header_len + pad + payload.len());
        out.extend_from_slice(RKYV_MAGIC);
        out.push(RKYV_VERSION);
        if pad > 0 {
            out.extend(std::iter::repeat(0u8).take(pad));
        }
        out.extend_from_slice(payload);
        Ok(out)
    }

    /// Decode the rkyv-encoded descriptor (validates before deserializing).
    pub(crate) fn decode_rkyv(bytes: &[u8]) -> Result<ColumnDescriptor> {
        if bytes.len() < 5 || &bytes[..4] != RKYV_MAGIC {
            return Err(Error::Internal("Invalid rkyv header".into()));
        }
        let version = bytes[4];
        if version != RKYV_VERSION {
            return Err(Error::Internal("Unsupported rkyv version".into()));
        }
        // Determine padding used at write time so we can locate the aligned
        // payload start.
        let header_len = 4 + 1;
        let required_align = std::mem::align_of::<<ColumnDescriptorRkyv as Archive>::Archived>();
        let pad = (required_align - (header_len % required_align)) % required_align;
        let payload_start = header_len + pad;
        if bytes.len() < payload_start {
            return Err(Error::Internal("Invalid rkyv header/padding".into()));
        }
        let payload = &bytes[payload_start..];

        // NOTE: validation is not performed here; we trust the payload for now.
        // Deserialize using the rkyv high-level API. The unsafe variant avoids
        // an extra validation pass.
        let helper_owned: ColumnDescriptorRkyv = unsafe {
            from_bytes_unchecked::<ColumnDescriptorRkyv, rkyv::rancor::Error>(payload)
        }
        .map_err(|_e| Error::Internal("rkyv deserialize failed".into()))?;

        Ok(ColumnDescriptor {
            field_id: LogicalFieldId::from(helper_owned.field_id_u64),
            head_page_pk: helper_owned.head_page_pk,
            tail_page_pk: helper_owned.tail_page_pk,
            total_row_count: helper_owned.total_row_count,
            total_chunk_count: helper_owned.total_chunk_count,
            data_type_code: helper_owned.data_type_code,
            _padding: helper_owned._padding,
            index_metadata: helper_owned.index_metadata,
        })
    }
}

// rkyv-friendly helper struct that uses primitive types so we don't need
// Archive impls for workspace-specific types like LogicalFieldId.
#[derive(Archive, Serialize, Deserialize)]
#[rkyv(attr(derive(Debug)))]
struct ColumnDescriptorRkyv {
    field_id_u64: u64,
    head_page_pk: u64,
    tail_page_pk: u64,
    total_row_count: u64,
    total_chunk_count: u64,
    data_type_code: u32,
    _padding: u32,
    index_metadata: Vec<u8>,
}

impl ColumnDescriptor {
    pub(crate) const FIXED_DISK_SIZE_WITHOUT_INDEX_META: usize = 48;
    pub(crate) const FIXED_DISK_SIZE: usize = Self::FIXED_DISK_SIZE_WITHOUT_INDEX_META + 4; // + index_meta_len

    // TODO: Separate between `data` and `rid` states?
    /// (Internal) Loads the full state for a descriptor, creating it if it
    /// doesn't exist.
    pub(crate) fn load_or_create<P: Pager>(
        pager: Arc<P>,
        descriptor_pk: PhysicalKey,
        field_id: LogicalFieldId,
    ) -> Result<(ColumnDescriptor, Vec<u8>)> {
        match pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
        {
            Some(GetResult::Raw { bytes, .. }) => {
                let descriptor = ColumnDescriptor::from_le_bytes(bytes.as_ref());
                let tail_page_bytes = pager
                    .batch_get(&[BatchGet::Raw {
                        key: descriptor.tail_page_pk,
                    }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?
                    .as_ref()
                    .to_vec();
                Ok((descriptor, tail_page_bytes))
            }
            _ => {
                let first_page_pk = pager.alloc_many(1)?[0];
                let descriptor = ColumnDescriptor {
                    field_id,
                    head_page_pk: first_page_pk,
                    tail_page_pk: first_page_pk,
                    ..Default::default()
                };
                let header = DescriptorPageHeader {
                    next_page_pk: 0,
                    entry_count: 0,
                    _padding: [0; 4],
                };
                let tail_page_bytes = header.to_le_bytes().to_vec();
                Ok((descriptor, tail_page_bytes))
            }
        }
    }

    /// Rewrite all descriptor pages for a column and update totals.
    pub(crate) fn rewrite_pages<P: Pager>(
        &mut self,
        pager: Arc<P>,
        descriptor_pk: PhysicalKey,
        metas: &mut [ChunkMetadata],
        puts: &mut Vec<BatchPut>,
    ) -> Result<()> {
        let mut current_page_pk = self.head_page_pk;
        let mut page_start_idx = 0usize;

        while current_page_pk != 0 {
            // This is a sequential walk, so a single-item batch is appropriate.
            let page_blob = pager
                .batch_get(&[BatchGet::Raw {
                    key: current_page_pk,
                }])?
                .pop()
                .and_then(|res| match res {
                    GetResult::Raw { bytes, .. } => Some(bytes),
                    _ => None,
                })
                .ok_or(Error::NotFound)?
                .as_ref()
                .to_vec();

            let header =
                DescriptorPageHeader::from_le_bytes(&page_blob[..DescriptorPageHeader::DISK_SIZE]);
            let n_on_page = header.entry_count as usize;
            let end_idx = page_start_idx + n_on_page;

            let mut new_page_data = Vec::new();
            for m in &metas[page_start_idx..end_idx] {
                new_page_data.extend_from_slice(&m.to_le_bytes());
            }

            let mut final_page_bytes =
                Vec::with_capacity(DescriptorPageHeader::DISK_SIZE + new_page_data.len());
            final_page_bytes.extend_from_slice(&header.to_le_bytes());
            final_page_bytes.extend_from_slice(&new_page_data);

            puts.push(BatchPut::Raw {
                key: current_page_pk,
                bytes: final_page_bytes,
            });
            current_page_pk = header.next_page_pk;
            page_start_idx = end_idx;
        }

        let mut total_rows = 0u64;
        for m in metas.iter() {
            total_rows += m.row_count;
        }
        self.total_row_count = total_rows;
        puts.push(BatchPut::Raw {
            key: descriptor_pk,
            bytes: self.to_le_bytes(),
        });
        Ok(())
    }

    /// Single allocation; append fields as LE without growth churn.
    pub(crate) fn to_le_bytes(&self) -> Vec<u8> {
        // Prefer the rkyv-encoded on-disk format for new data. Keep the old
        // fixed-layout legacy format code around for compatibility, but the
        // public `to_le_bytes` API now produces the rkyv payload so callers
        // don't need widespread changes. Errors here are treated as
        // unrecoverable internal errors.
        self.encode_rkyv().unwrap_or_else(|e| {
            panic!("rkyv encode error in to_le_bytes: {:?}", e)
        })
    }

    pub(crate) fn from_le_bytes(bytes: &[u8]) -> Self {
        // If the blob begins with the rkyv header, use the rkyv decoder. This
        // allows the on-disk format to evolve without requiring edits across
        // the codebase: callers using `from_le_bytes` will transparently read
        // the rkyv payload.
        if bytes.len() >= 5 && &bytes[..4] == RKYV_MAGIC {
            // We expect decode_rkyv to return a Result; treat failures as
            // unrecoverable for now (shouldn't happen for valid data).
            return ColumnDescriptor::decode_rkyv(bytes)
                .expect("failed to decode rkyv ColumnDescriptor");
        }

        // Legacy fixed-layout format (little-endian)
        let mut o = 0usize;
        let field_id = LogicalFieldId::from(read_u64_le(bytes, &mut o));
        let head_page_pk = read_u64_le(bytes, &mut o);
        let tail_page_pk = read_u64_le(bytes, &mut o);
        let total_row_count = read_u64_le(bytes, &mut o);
        let total_chunk_count = read_u64_le(bytes, &mut o);

        // For forwards compatibility, check if the newer fields exist.
        let (data_type_code, padding, index_meta_len) = if bytes.len() >= o + 12 {
            let dtc = read_u32_le(bytes, &mut o);
            let pad = read_u32_le(bytes, &mut o);
            let iml = read_u32_le(bytes, &mut o) as usize;
            (dtc, pad, iml)
        } else {
            (0, 0, 0)
        };

        let index_metadata = if index_meta_len > 0 && bytes.len() >= o + index_meta_len {
            bytes[o..o + index_meta_len].to_vec()
        } else {
            Vec::new()
        };

        Self {
            field_id,
            head_page_pk,
            tail_page_pk,
            total_row_count,
            total_chunk_count,
            data_type_code,
            _padding: padding,
            index_metadata,
        }
    }

    /// Deserializes index kinds from metadata using a simple byte format.
    /// Format: [num_indexes: u32_le] [kind1: u8] [kind2: u8]...
    pub(crate) fn get_indexes(&self) -> Result<Vec<IndexKind>> {
        if self.index_metadata.is_empty() {
            return Ok(Vec::new());
        }
        let mut o = 0usize;
        let data = &self.index_metadata;
        if data.len() < 4 {
            return Err(Error::Internal("Invalid index metadata: too short".into()));
        }
        let num_indexes = read_u32_le(data, &mut o) as usize;
        let expected_len = 4 + num_indexes;
        if data.len() < expected_len {
            return Err(Error::Internal(
                "Invalid index metadata: unexpected eof".into(),
            ));
        }

        let mut indexes = Vec::with_capacity(num_indexes);
        for _ in 0..num_indexes {
            let kind_u8 = data[o];
            indexes.push(IndexKind::try_from(kind_u8)?);
            o += 1;
        }
        Ok(indexes)
    }

    /// Serializes and sets index kinds to metadata.
    pub(crate) fn set_indexes(&mut self, indexes: &[IndexKind]) -> Result<()> {
        // Allocate enough space for the count (u32) and one byte per kind.
        let mut buf = Vec::with_capacity(4 + indexes.len());
        write_u32_le(&mut buf, indexes.len() as u32);
        for index in indexes {
            buf.push(u8::from(*index));
        }
        self.index_metadata = buf;
        Ok(())
    }
}

/// The header for a page in the descriptor chain.
/// The on-disk layout is this header, followed immediately by a packed
/// array of `ChunkMetadata` entries.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DescriptorPageHeader {
    pub(crate) next_page_pk: PhysicalKey, // Using 0 as None
    pub(crate) entry_count: u32,
    pub(crate) _padding: [u8; 4],
}

impl DescriptorPageHeader {
    pub(crate) const DISK_SIZE: usize = mem::size_of::<Self>();

    pub(crate) fn to_le_bytes(self) -> [u8; Self::DISK_SIZE] {
        let mut v = Vec::with_capacity(Self::DISK_SIZE);
        write_u64_le(&mut v, self.next_page_pk);
        write_u32_le(&mut v, self.entry_count);
        if v.len() < Self::DISK_SIZE {
            v.extend(std::iter::repeat_n(0u8, Self::DISK_SIZE - v.len()));
        }
        let mut buf = [0u8; Self::DISK_SIZE];
        buf.copy_from_slice(&v);
        buf
    }

    pub(crate) fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut o = 0usize;
        let next_page_pk = read_u64_le(bytes, &mut o);
        let entry_count = read_u32_le(bytes, &mut o);
        Self {
            next_page_pk,
            entry_count,
            _padding: [0; 4],
        }
    }
}

// TODO: Rename to `ColumnDescriptorIterator`?
/// An iterator that streams `ChunkMetadata` by walking the descriptor page
/// chain. It only holds one page blob in memory at a time.
pub(crate) struct DescriptorIterator<'a, P: Pager> {
    pager: &'a P,
    current_page_pk: PhysicalKey,
    current_blob: Option<P::Blob>,
    cursor_in_page: usize,
}

impl<'a, P: Pager> DescriptorIterator<'a, P> {
    pub(crate) fn new(pager: &'a P, head_page_pk: PhysicalKey) -> Self {
        Self {
            pager,
            current_page_pk: head_page_pk,
            current_blob: None,
            cursor_in_page: 0,
        }
    }
}

impl<'a, P: Pager> Iterator for DescriptorIterator<'a, P> {
    type Item = Result<ChunkMetadata>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we do not have a page loaded, try to load one.
            if self.current_blob.is_none() {
                if self.current_page_pk == 0 {
                    return None; // End of the chain
                }
                match self.pager.batch_get(&[BatchGet::Raw {
                    key: self.current_page_pk,
                }]) {
                    Ok(mut results) => match results.pop() {
                        Some(GetResult::Raw { bytes, .. }) => {
                            self.current_blob = Some(bytes);
                            self.cursor_in_page = 0;
                        }
                        Some(GetResult::Missing { .. }) => return Some(Err(Error::NotFound)),
                        None => return Some(Err(Error::Internal("batch_get empty result".into()))),
                    },
                    Err(e) => return Some(Err(e)),
                }
            }

            let blob_bytes = self.current_blob.as_ref().unwrap().as_ref();
            let hdr_sz = DescriptorPageHeader::DISK_SIZE;
            let header = DescriptorPageHeader::from_le_bytes(&blob_bytes[..hdr_sz]);

            if self.cursor_in_page < header.entry_count as usize {
                let off = hdr_sz + self.cursor_in_page * ChunkMetadata::DISK_SIZE;
                let end = off + ChunkMetadata::DISK_SIZE;
                let entry_bytes = &blob_bytes[off..end];
                let metadata = ChunkMetadata::from_le_bytes(entry_bytes);
                self.cursor_in_page += 1;
                return Some(Ok(metadata));
            } else {
                // Page exhausted, move to the next one.
                self.current_page_pk = header.next_page_pk;
                self.current_blob = None;
                // Loop again to load the next page or terminate.
            }
        }
    }
}

// TODO: Keep?
// Build a descriptor page payload (header + packed entries) with a single
// allocation. Useful when rewriting a page after metadata updates.
// pub(crate) fn build_descriptor_page_bytes(
//     header: &DescriptorPageHeader,
//     entries: &[ChunkMetadata],
// ) -> Vec<u8> {
//     let cap = DescriptorPageHeader::DISK_SIZE + entries.len() * ChunkMetadata::DISK_SIZE;
//     let mut page = Vec::with_capacity(cap);
//     page.extend_from_slice(&header.to_le_bytes());
//     for m in entries {
//         page.extend_from_slice(&m.to_le_bytes());
//     }
//     page
// }
