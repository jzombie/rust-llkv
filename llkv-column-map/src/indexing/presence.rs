//! The presence index, implemented as a shadow column of row IDs.

use super::traits::{BackfillContext, Index, IndexPlan};
use crate::codecs::write_u64_le;
use crate::error::{Error, Result};
use crate::storage::pager::Pager;
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::ChunkMetadata;
use crate::types::{LogicalFieldId, Namespace};
use arrow::array::{ArrayRef, UInt64Array};
use simd_r_drive_entry_handle::EntryHandle;

/// An index that tracks the presence of rows in a column via a shadow row-id column.
#[derive(Default)]
pub struct PresenceIndex;

impl PresenceIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for PresenceIndex {
    fn name(&self) -> &'static str {
        "presence"
    }

    /// The "plan" for a presence index is simple: serialize the sorted row IDs.
    fn plan_append(
        &self,
        _data_for_chunk: &ArrayRef,
        rids_for_chunk: &ArrayRef,
    ) -> Result<Option<IndexPlan>> {
        let rids = rids_for_chunk
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal("Presence index requires UInt64 row IDs".into()))?;

        if rids.is_empty() {
            return Ok(None);
        }

        let mut min = u64::MAX;
        let mut max = 0;
        // The raw payload is just the tightly packed u64 row IDs.
        let mut bytes = Vec::with_capacity(rids.len() * 8);
        for i in 0..rids.len() {
            let v = rids.value(i);
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            write_u64_le(&mut bytes, v);
        }

        let plan = IndexPlan {
            metadata: ChunkMetadata {
                chunk_pk: 0,            // Will be filled in by the ColumnStore
                value_order_perm_pk: 0, // Presence indexes are always sorted by value (row ID)
                row_count: rids.len() as u64,
                serialized_bytes: bytes.len() as u64,
                min_val_u64: min,
                max_val_u64: max,
            },
            bytes,
        };

        Ok(Some(plan))
    }

    /// Find all fields that have a corresponding shadow RowId column.
    fn discover(
        &self,
        _pager: &dyn Pager<Blob = EntryHandle>,
        catalog: &ColumnCatalog,
    ) -> Result<Vec<LogicalFieldId>> {
        let mut found = Vec::new();
        for &fid in catalog.map.keys() {
            if fid.namespace() == Namespace::RowIdShadow {
                // This is a presence index. The field it's for is the same ID
                // but in the UserData namespace.
                found.push(fid.with_namespace(Namespace::UserData));
            }
        }
        Ok(found)
    }

    /// Backfilling the presence index is a no-op because it's always created
    /// and updated on the append path.
    fn backfill(
        &self,
        _context: &mut BackfillContext<'_>,
        _field_id: LogicalFieldId,
    ) -> Result<()> {
        Ok(())
    }
}
