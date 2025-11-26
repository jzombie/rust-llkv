//! The top-level directory for all columns in the store.

use crate::codecs::{read_u64_le, write_u64_le};
use llkv_result::{Error, Result};
use llkv_storage::types::PhysicalKey;
use llkv_types::ids::LogicalFieldId;
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
        let mut o = 0usize;
        if bytes.len() < 8 {
            return Err(Error::Internal(
                "Invalid catalog blob: too short".to_string(),
            ));
        }
        let entry_count = read_u64_le(bytes, &mut o) as usize;
        let expected_len = 8 + entry_count * 16;
        if bytes.len() < expected_len {
            return Err(Error::Internal(
                "Invalid catalog blob: unexpected eof".to_string(),
            ));
        }

        let mut map = FxHashMap::with_capacity_and_hasher(entry_count, Default::default());
        for _ in 0..entry_count {
            // Read the u64 from bytes and convert it into the LogicalFieldId struct.
            let field_id = LogicalFieldId::from(read_u64_le(bytes, &mut o));
            let desc_pk = read_u64_le(bytes, &mut o);
            map.insert(field_id, desc_pk);
        }
        Ok(Self { map })
    }

    /// Serializes the catalog into a byte vector for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let entry_count = self.map.len();
        let mut buf = Vec::with_capacity(8 + entry_count * 16);
        write_u64_le(&mut buf, entry_count as u64);
        for (&field_id, &desc_pk) in &self.map {
            // Convert LogicalFieldId struct into a u64 for serialization.
            let field_id_u64: u64 = field_id.into();
            write_u64_le(&mut buf, field_id_u64);
            write_u64_le(&mut buf, desc_pk);
        }
        buf
    }
}
