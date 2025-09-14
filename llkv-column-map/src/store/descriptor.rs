// File: src/store/descriptor.rs
//! Streamable, paged metadata for a single column.
//!
//! This module avoids deserialization by defining fixed-layout structs that can be
//! interpreted directly from byte buffers provided by the pager.

use crate::codecs::{get_u32, get_u64, put_u32, put_u64};
use crate::error::{Error, Result};
use crate::storage::pager::Pager;
use crate::types::{LogicalFieldId, PhysicalKey};
use std::mem;

// All values are stored as little-endian.

/// Fixed-size metadata for a single Arrow array chunk.
/// This struct's memory layout IS the on-disk format.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ChunkMetadata {
    pub chunk_pk: PhysicalKey,
    pub tombstone_pk: PhysicalKey,        // Using 0 as None
    pub value_order_perm_pk: PhysicalKey, // Using 0 as None
    pub row_count: u64,
    pub serialized_bytes: u64,
    pub min_val_u64: u64, // For pruning, assumes u64-comparable values
    pub max_val_u64: u64,
}

impl ChunkMetadata {
    pub const DISK_SIZE: usize = mem::size_of::<Self>();

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::DISK_SIZE);
        buf.extend_from_slice(&self.chunk_pk.to_le_bytes());
        buf.extend_from_slice(&self.tombstone_pk.to_le_bytes());
        buf.extend_from_slice(&self.value_order_perm_pk.to_le_bytes());
        buf.extend_from_slice(&self.row_count.to_le_bytes());
        buf.extend_from_slice(&self.serialized_bytes.to_le_bytes());
        buf.extend_from_slice(&self.min_val_u64.to_le_bytes());
        buf.extend_from_slice(&self.max_val_u64.to_le_bytes());
        buf
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut offset = 0;
        let chunk_pk = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let tombstone_pk = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let value_order_perm_pk = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let row_count = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let serialized_bytes = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let min_val_u64 = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let max_val_u64 = get_u64(&bytes[offset..offset + 8]);

        Self {
            chunk_pk,
            tombstone_pk,
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

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::DISK_SIZE);
        buf.extend_from_slice(&self.field_id.to_le_bytes());
        buf.extend_from_slice(&self.head_page_pk.to_le_bytes());
        buf.extend_from_slice(&self.tail_page_pk.to_le_bytes());
        buf.extend_from_slice(&self.total_row_count.to_le_bytes());
        buf.extend_from_slice(&self.total_chunk_count.to_le_bytes());
        buf
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut offset = 0;
        let field_id = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let head_page_pk = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let tail_page_pk = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let total_row_count = get_u64(&bytes[offset..offset + 8]);
        offset += 8;
        let total_chunk_count = get_u64(&bytes[offset..offset + 8]);
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
    pub _padding: [u8; 4], // FIX: Make field public
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

/// An iterator that streams `ChunkMetadata` by walking the descriptor page chain.
/// It only holds one page blob in memory at a time.
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
            // If we don't have a page loaded, try to load one.
            if self.current_blob.is_none() {
                if self.current_page_pk == 0 {
                    return None; // End of the chain
                }
                match self.pager.get_raw(self.current_page_pk) {
                    Ok(Some(blob)) => {
                        self.current_blob = Some(blob);
                        self.cursor_in_page = 0;
                    }
                    Ok(None) => return Some(Err(Error::NotFound)),
                    Err(e) => return Some(Err(e.into())),
                }
            }

            let blob_bytes = self.current_blob.as_ref().unwrap().as_ref();
            let header =
                DescriptorPageHeader::from_le_bytes(&blob_bytes[..DescriptorPageHeader::DISK_SIZE]);

            if self.cursor_in_page < header.entry_count as usize {
                let offset = DescriptorPageHeader::DISK_SIZE
                    + self.cursor_in_page * ChunkMetadata::DISK_SIZE;
                let entry_bytes = &blob_bytes[offset..offset + ChunkMetadata::DISK_SIZE];
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
