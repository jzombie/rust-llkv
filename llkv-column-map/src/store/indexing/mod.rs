//! Indexing: shared traits, manager, and dispatcher only.
//! Concrete indexes live in their own files.

use crate::error::{Error, Result};
use crate::storage::pager::{BatchPut, Pager};
use crate::store::descriptor::{ChunkMetadata, ColumnDescriptor};
use crate::types::{LogicalFieldId, PhysicalKey};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;

pub mod presence;
pub mod sort;

/// Marker trait carried by concrete index types.
pub trait Index {
    /// Stable kind of the index (e.g., `IndexKind::Sort`, `IndexKind::Presence`).
    fn kind(&self) -> IndexKind;
}

/// Kinds of indexes supported by the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexKind {
    Sort,
    Presence,
}

impl From<IndexKind> for u8 {
    fn from(kind: IndexKind) -> Self {
        match kind {
            IndexKind::Presence => 0,
            IndexKind::Sort => 1,
        }
    }
}

impl TryFrom<u8> for IndexKind {
    type Error = crate::error::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(IndexKind::Presence),
            1 => Ok(IndexKind::Sort),
            _ => Err(Error::Internal("Invalid IndexKind integer!".to_string())),
        }
    }
}

/// Uniform ops each concrete index must implement.
pub(crate) trait IndexOps<P: Pager>: Send + Sync {
    /// Stages the necessary updates for a brand new chunk being added during an append.
    fn stage_update_for_new_chunk(
        &self,
        pager: &Arc<P>,
        value_slice: &arrow::array::ArrayRef,
        rid_slice: &arrow::array::ArrayRef,
        value_meta: &mut ChunkMetadata,
        rid_meta: &mut ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()>;

    /// Stages an index build for a single, existing chunk of data.
    fn stage_build_for_chunk(
        &self,
        pager: &Arc<P>, // Only used for pager.alloc_many()
        meta: &mut ChunkMetadata,
        chunk_blob: &P::Blob,
    ) -> Result<Option<BatchPut>>;

    /// Stages the removal of a physical index from a chunk's metadata.
    fn stage_drop_index(
        &self,
        pager: &Arc<P>,
        metas: &mut [ChunkMetadata],
    ) -> Result<Vec<PhysicalKey>>;
}

/// A container for the store's index implementations.
pub struct IndexManager<P: Pager> {
    pub(crate) pager: Arc<P>,
    pub(crate) presence_ops: presence::PresenceIndexOps,
    pub(crate) sort_ops: sort::SortIndexOps,
}

impl<P: Pager> IndexManager<P> {
    pub fn new(pager: Arc<P>) -> Self {
        Self {
            pager,
            presence_ops: presence::PresenceIndexOps,
            sort_ops: sort::SortIndexOps,
        }
    }

    /// Dispatches updates for a new data chunk to all registered indexes for that column.
    #[allow(clippy::too_many_arguments)] // TODO: Refactor
    pub(crate) fn stage_updates_for_new_chunk(
        &self,
        _field_id: LogicalFieldId,
        descriptor: &ColumnDescriptor,
        value_slice: &arrow::array::ArrayRef,
        rid_slice: &arrow::array::ArrayRef,
        value_meta: &mut ChunkMetadata,
        rid_meta: &mut ChunkMetadata,
        puts: &mut Vec<BatchPut>,
    ) -> Result<()>
    where
        P: Pager<Blob = EntryHandle>,
    {
        let active_indexes = descriptor.get_indexes()?;

        for index_kind in active_indexes {
            match index_kind {
                IndexKind::Presence => {
                    self.presence_ops.stage_update_for_new_chunk(
                        &self.pager,
                        value_slice,
                        rid_slice,
                        value_meta,
                        rid_meta,
                        puts,
                    )?;
                }
                IndexKind::Sort => {
                    self.sort_ops.stage_update_for_new_chunk(
                        &self.pager,
                        value_slice,
                        rid_slice,
                        value_meta,
                        rid_meta,
                        puts,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Prepares the logical registration of an index on an in-memory descriptor.
    pub(crate) fn stage_index_registration(
        &self,
        descriptor: &mut ColumnDescriptor,
        kind: IndexKind,
    ) -> Result<()> {
        let mut indexes = descriptor.get_indexes()?;
        if !indexes.contains(&kind) {
            indexes.push(kind);
            descriptor.set_indexes(&indexes)?;
        }
        Ok(())
    }

    /// Prepares the logical un-registration of an index on an in-memory descriptor.
    pub(crate) fn stage_index_unregistration(
        &self,
        descriptor: &mut ColumnDescriptor,
        kind: IndexKind,
    ) -> Result<()> {
        let mut indexes = descriptor.get_indexes()?;
        let original_len = indexes.len();
        indexes.retain(|&k| k != kind);
        if indexes.len() == original_len {
            return Err(Error::InvalidArgumentError(format!(
                "Index '{:?}' not found for this column.",
                kind
            )));
        }
        descriptor.set_indexes(&indexes)?;
        Ok(())
    }
}
