//! Presence index: type and ops live here.

use super::IndexKind;
use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator};
use crate::store::indexing::{Index, IndexOps, IndexUpdateHint};
use crate::store::rowid::rowid_fid; // Helper to get the shadow fid
use crate::types::LogicalFieldId;
use arrow::array::UInt64Array;
use arrow::compute::{SortColumn, lexsort_to_indices};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

/// Public marker type for the presence index.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct PresenceIndex;

impl PresenceIndex {
    pub fn new() -> Self {
        Self
    }
}

impl Index for PresenceIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Presence
    }
}

/// Concrete implementation of presence index operations.
pub struct PresenceIndexOps;

impl<P> IndexOps<P> for PresenceIndexOps
where
    P: Pager<Blob = EntryHandle>,
{
    /// Builds the sorting permutation for all chunks in a column's shadow row_id column.
    fn build_all(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()> {
        todo!("Implement");
    }

    fn update(
        &self,
        _pager: &Arc<P>,
        _catalog: &Arc<RwLock<ColumnCatalog>>,
        _field: LogicalFieldId,
        _hint: &IndexUpdateHint,
    ) -> Result<()> {
        todo!("Implement");
    }

    /// Clears all row_id sorting permutations for the given field.
    fn drop_index(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()> {
        todo!("Implement");
    }
}
