//! Indexing: shared traits, manager, and dispatcher only.
//!
//! Concrete indexes live in their own files.

use crate::store::{
    ColumnStore,
    catalog::ColumnCatalog,
    descriptor::{ChunkMetadata, ColumnDescriptor},
};
use crate::types::LogicalFieldId;
use llkv_result::{Error, Result};
use llkv_storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use llkv_storage::types::PhysicalKey;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;
use std::sync::RwLockReadGuard;

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
    type Error = llkv_result::Error;
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
    /// Stages the full build of an index for a column with existing data.
    #[allow(clippy::too_many_arguments)] // TODO: Use struct for args
    fn stage_registration(
        &self,
        index_manager: &IndexManager<P>,
        column_store: &ColumnStore<P>,
        catalog: RwLockReadGuard<'_, ColumnCatalog>,
        field_id: LogicalFieldId,
        descriptor: &mut ColumnDescriptor,
        descriptor_pk: PhysicalKey,
        puts: &mut Vec<BatchPut>,
        frees: &mut Vec<PhysicalKey>,
    ) -> Result<()>;

    /// Stages the full removal of an index from a column.
    #[allow(clippy::too_many_arguments)] // TODO: Use struct for args
    fn stage_unregistration(
        &self,
        index_manager: &IndexManager<P>,
        column_store: &ColumnStore<P>,
        catalog: RwLockReadGuard<'_, ColumnCatalog>,
        field_id: LogicalFieldId,
        descriptor: &mut ColumnDescriptor,
        descriptor_pk: PhysicalKey,
        puts: &mut Vec<BatchPut>,
        frees: &mut Vec<PhysicalKey>,
    ) -> Result<()>;

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

impl<P: Pager<Blob = EntryHandle>> IndexManager<P> {
    pub(crate) fn new(pager: Arc<P>) -> Self {
        Self {
            pager,
            presence_ops: presence::PresenceIndexOps,
            sort_ops: sort::SortIndexOps,
        }
    }

    /// Registers an index for a given column, building it for existing data atomically and with low memory usage.
    pub(crate) fn register_index(
        &self,
        column_store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        kind: IndexKind,
    ) -> Result<()> {
        // This vector will collect all write operations to be committed in one batch.
        let mut puts: Vec<BatchPut> = Vec::new();
        // This vector collects keys of any descriptor pages that are deallocated.
        let mut frees: Vec<PhysicalKey> = Vec::new();

        // --- PHASE 1: READ DESCRIPTORS ---
        // Load the root descriptor for the main data column, which is always needed for logical registration.
        let catalog = column_store.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // --- PHASE 2: STAGE PHYSICAL INDEX DATA (CHUNK-BY-CHUNK) ---
        match kind {
            IndexKind::Sort => self.sort_ops.stage_registration(
                self,
                column_store,
                catalog,
                field_id,
                &mut descriptor,
                descriptor_pk,
                &mut puts,
                &mut frees,
            ),
            IndexKind::Presence => self.presence_ops.stage_registration(
                self,
                column_store,
                catalog,
                field_id,
                &mut descriptor,
                descriptor_pk,
                &mut puts,
                &mut frees,
            ),
        }?;

        // --- PHASE 3: COMMIT ---
        // Atomically write all staged operations to the pager.
        if !puts.is_empty() {
            self.pager.batch_put(&puts)?;
        }
        if !frees.is_empty() {
            self.pager.free_many(&frees)?;
        }
        Ok(())
    }

    /// Unregisters a persisted index from a given column atomically.
    pub fn unregister_index(
        &self,
        column_store: &ColumnStore<P>,
        field_id: LogicalFieldId,
        kind: IndexKind,
    ) -> Result<()> {
        let mut puts = Vec::new();
        let mut frees = Vec::new();
        let catalog = column_store.catalog.read().unwrap();
        let descriptor_pk = *catalog.map.get(&field_id).ok_or(Error::NotFound)?;
        let desc_blob = self
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        match kind {
            IndexKind::Sort => self.sort_ops.stage_unregistration(
                self,
                column_store,
                catalog,
                field_id,
                &mut descriptor,
                descriptor_pk,
                &mut puts,
                &mut frees,
            ),
            IndexKind::Presence => self.presence_ops.stage_unregistration(
                self,
                column_store,
                catalog,
                field_id,
                &mut descriptor,
                descriptor_pk,
                &mut puts,
                &mut frees,
            ),
        }?;

        // Commit all staged metadata writes and free the old index data.
        if !puts.is_empty() {
            self.pager.batch_put(&puts)?;
        }
        if !frees.is_empty() {
            self.pager.free_many(&frees)?;
        }
        Ok(())
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
