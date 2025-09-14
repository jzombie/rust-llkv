// File: src/store/catalog.rs
//! The top-level directory for all columns in the store.

use crate::codecs::{get_u64, put_u64};
use crate::error::{Error, Result};
use crate::types::{LogicalFieldId, PhysicalKey};
use rustc_hash::FxHashMap;

/// An in-memory mapping from logical field IDs to the physical key of the
/// column's root descriptor.
#[derive(Debug, Clone, Default)]
pub struct ColumnCatalog {
    pub map: FxHashMap<LogicalFieldId, PhysicalKey>,
}

impl ColumnCatalog {
    /// Deserializes the catalog from a byte buffer.
    /// Format: [entry_count: u64] [field_id: u64, desc_pk: u64]...
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::Internal(
                "Invalid catalog blob: too short".to_string(),
            ));
        }
        let entry_count = get_u64(&bytes[0..8]) as usize;
        let expected_len = 8 + entry_count * 16;
        if bytes.len() < expected_len {
            return Err(Error::Internal(
                "Invalid catalog blob: unexpected eof".to_string(),
            ));
        }

        let mut map = FxHashMap::with_capacity_and_hasher(entry_count, Default::default());
        let mut offset = 8;
        for _ in 0..entry_count {
            let field_id = get_u64(&bytes[offset..offset + 8]);
            offset += 8;
            let desc_pk = get_u64(&bytes[offset..offset + 8]);
            offset += 8;
            map.insert(field_id, desc_pk);
        }
        Ok(Self { map })
    }

    /// Serializes the catalog into a byte vector for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let entry_count = self.map.len();
        let mut buf = Vec::with_capacity(8 + entry_count * 16);
        buf.extend_from_slice(&put_u64(entry_count as u64));
        for (&field_id, &desc_pk) in &self.map {
            buf.extend_from_slice(&put_u64(field_id));
            buf.extend_from_slice(&put_u64(desc_pk));
        }
        buf
    }
}
