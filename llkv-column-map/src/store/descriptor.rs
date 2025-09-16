//! Streamable, paged metadata for a single column.
//!
//! This module avoids deserialization by defining fixed-layout structs that
//! can be interpreted directly from byte buffers provided by the pager.

use crate::codecs::{get_u32, get_u64, put_u32, put_u64};
use crate::error::{Error, Result};
use crate::storage::pager::{BatchGet, GetResult, Pager};
use crate::types::{LogicalFieldId, PhysicalKey};
use std::mem;

// All values are stored as little-endian.

/// Fixed-size metadata for a single Arrow array chunk.
/// This struct's memory layout IS the on-disk format.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ChunkMetadata {
    pub chunk_pk: PhysicalKey,
    pub value_order_perm_pk: PhysicalKey, // Using 0 as None
    pub row_count: u64,
    pub serialized_bytes: u64,
    pub min_val_u64: u64, // For pruning, assumes u64-comparable values
    pub max_val_u64: u64,
}

impl ChunkMetadata {
    pub const DISK_SIZE: usize = mem::size_of::<Self>();

    /// Single allocation; append fields as LE without growth churn.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::DISK_SIZE);
        buf.extend_from_slice(&self.chunk_pk.to_le_bytes());
        buf.extend_from_slice(&self.value_order_perm_pk.to_le_bytes());
        buf.extend_from_slice(&self.row_count.to_le_bytes());
        buf.extend_from_slice(&self.serialized_bytes.to_le_bytes());
        buf.extend_from_slice(&self.min_val_u64.to_le_bytes());
        buf.extend_from_slice(&self.max_val_u64.to_le_bytes());
        buf
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut o = 0usize;
        let rd = |off: &mut usize| {
            let v = get_u64(&bytes[*off..*off + 8]);
            *off += 8;
            v
        };
        let chunk_pk = rd(&mut o);
        let value_order_perm_pk = rd(&mut o);
        let row_count = rd(&mut o);
        let serialized_bytes = rd(&mut o);
        let min_val_u64 = rd(&mut o);
        let max_val_u64 = rd(&mut o);

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
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ColumnDescriptor {
    pub field_id: LogicalFieldId,
    pub head_page_pk: PhysicalKey,
    pub tail_page_pk: PhysicalKey,
    pub total_row_count: u64,
    pub total_chunk_count: u64,
}

impl ColumnDescriptor {
    pub const DISK_SIZE: usize = mem::size_of::<Self>();

    /// Single allocation; append fields as LE without growth churn.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::DISK_SIZE);
        // Convert LogicalFieldId struct into a u64 for serialization.
        let field_id_u64: u64 = self.field_id.into();
        buf.extend_from_slice(&field_id_u64.to_le_bytes());
        buf.extend_from_slice(&self.head_page_pk.to_le_bytes());
        buf.extend_from_slice(&self.tail_page_pk.to_le_bytes());
        buf.extend_from_slice(&self.total_row_count.to_le_bytes());
        buf.extend_from_slice(&self.total_chunk_count.to_le_bytes());
        buf
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut o = 0usize;
        let rd = |off: &mut usize| {
            let v = get_u64(&bytes[*off..*off + 8]);
            *off += 8;
            v
        };
        // Read the u64 from bytes and convert it into the LogicalFieldId struct.
        let field_id = LogicalFieldId::from(rd(&mut o));
        let head_page_pk = rd(&mut o);
        let tail_page_pk = rd(&mut o);
        let total_row_count = rd(&mut o);
        let total_chunk_count = rd(&mut o);
        Self {
            field_id,
            head_page_pk,
            tail_page_pk,
            total_row_count,
            total_chunk_count,
        }
    }
}

/// The header for a page in the descriptor chain.
/// The on-disk layout is this header, followed immediately by a packed
/// array of `ChunkMetadata` entries.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DescriptorPageHeader {
    pub next_page_pk: PhysicalKey, // Using 0 as None
    pub entry_count: u32,
    pub _padding: [u8; 4],
}

impl DescriptorPageHeader {
    pub const DISK_SIZE: usize = mem::size_of::<Self>();

    pub fn to_le_bytes(&self) -> [u8; Self::DISK_SIZE] {
        let mut buf = [0u8; Self::DISK_SIZE];
        buf[0..8].copy_from_slice(&put_u64(self.next_page_pk));
        buf[8..12].copy_from_slice(&put_u32(self.entry_count));
        buf
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let next_page_pk = get_u64(&bytes[0..8]);
        let entry_count = get_u32(&bytes[8..12]);
        Self {
            next_page_pk,
            entry_count,
            _padding: [0; 4],
        }
    }
}

/// An iterator that streams `ChunkMetadata` by walking the descriptor page
/// chain. It only holds one page blob in memory at a time.
pub struct DescriptorIterator<'a, P: Pager> {
    pager: &'a P,
    current_page_pk: PhysicalKey,
    current_blob: Option<P::Blob>,
    cursor_in_page: usize,
}

impl<'a, P: Pager> DescriptorIterator<'a, P> {
    pub fn new(pager: &'a P, head_page_pk: PhysicalKey) -> Self {
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
                    Err(e) => return Some(Err(e.into())),
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

/// Build a descriptor page payload (header + packed entries) with a single
/// allocation. Useful when rewriting a page after metadata updates.
pub fn build_descriptor_page_bytes(
    header: &DescriptorPageHeader,
    entries: &[ChunkMetadata],
) -> Vec<u8> {
    let cap = DescriptorPageHeader::DISK_SIZE + entries.len() * ChunkMetadata::DISK_SIZE;
    let mut page = Vec::with_capacity(cap);
    page.extend_from_slice(&header.to_le_bytes());
    for m in entries {
        page.extend_from_slice(&m.to_le_bytes());
    }
    page
}
