//! Indexing: shared traits, manager, and dispatcher only.
//!
//! Concrete indexes live in their own files.

use crate::error::{Error, Result};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::{
    ColumnStore,
    descriptor::{ChunkMetadata, ColumnDescriptor, DescriptorIterator},
    rowid_fid,
};
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
        let desc_blob = column_store
            .pager
            .batch_get(&[BatchGet::Raw { key: descriptor_pk }])?
            .pop()
            .and_then(|r| match r {
                GetResult::Raw { bytes, .. } => Some(bytes),
                _ => None,
            })
            .ok_or(Error::NotFound)?;
        let mut descriptor = ColumnDescriptor::from_le_bytes(desc_blob.as_ref());

        // TODO: Put logic directly in index plugin
        // --- PHASE 2: STAGE PHYSICAL INDEX DATA (CHUNK-BY-CHUNK) ---
        match kind {
            IndexKind::Sort => {
                // The Sort index operates on the main data column's chunks.
                // This vector will hold the modified metadata to rewrite the descriptor pages at the end.
                let mut modified_metas = Vec::new();

                // Iterate through the column's metadata one chunk at a time to keep memory usage low.
                for meta_result in
                    DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk)
                {
                    let mut meta = meta_result?;
                    // Fetch only the data for the current chunk.
                    let chunk_blob = column_store
                        .pager
                        .batch_get(&[BatchGet::Raw { key: meta.chunk_pk }])?
                        .pop()
                        .and_then(|r| match r {
                            GetResult::Raw { bytes, .. } => Some(bytes),
                            _ => None,
                        })
                        .ok_or(Error::NotFound)?;
                    // Call the specialist to build the index for this single chunk.
                    // This modifies `meta` in-place and returns a `BatchPut` for any new permutation blob.
                    if let Some(put) = <sort::SortIndexOps as IndexOps<P>>::stage_build_for_chunk(
                        &self.sort_ops,
                        &self.pager,
                        &mut meta,
                        &chunk_blob,
                    )? {
                        puts.push(put);
                    }
                    modified_metas.push(meta);
                }

                // Logically register the index on the in-memory descriptor FIRST.
                self.stage_index_registration(&mut descriptor, kind)?;
                // NOW rewrite the descriptor chain, which will serialize the updated descriptor and its pages.
                column_store.write_descriptor_chain(
                    descriptor_pk,
                    &mut descriptor,
                    &modified_metas,
                    &mut puts,
                    &mut frees,
                )?;
            }
            IndexKind::Presence => {
                // The Presence index operates on the shadow row_id column.
                let rid_fid = rowid_fid(field_id);
                let rid_pk = *catalog.map.get(&rid_fid).ok_or(Error::NotFound)?;
                let rid_desc_blob = self
                    .pager
                    .batch_get(&[BatchGet::Raw { key: rid_pk }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?;
                let mut rid_desc = ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());

                let mut modified_rid_metas = Vec::new();
                for meta_result in
                    DescriptorIterator::new(self.pager.as_ref(), rid_desc.head_page_pk)
                {
                    let mut meta = meta_result?;
                    let chunk_blob = self
                        .pager
                        .batch_get(&[BatchGet::Raw { key: meta.chunk_pk }])?
                        .pop()
                        .and_then(|r| match r {
                            GetResult::Raw { bytes, .. } => Some(bytes),
                            _ => None,
                        })
                        .ok_or(Error::NotFound)?;
                    if let Some(put) =
                        <presence::PresenceIndexOps as IndexOps<P>>::stage_build_for_chunk(
                            &self.presence_ops,
                            &self.pager,
                            &mut meta,
                            &chunk_blob,
                        )?
                    {
                        puts.push(put);
                    }
                    modified_rid_metas.push(meta);
                }

                // Logically register the index on the main data descriptor.
                self.stage_index_registration(&mut descriptor, kind)?;
                // Stage a write for the main descriptor, as it's now changed.
                puts.push(BatchPut::Raw {
                    key: descriptor_pk,
                    bytes: descriptor.to_le_bytes(),
                });
                // Separately, rewrite the descriptor chain for the shadow row_id column.
                column_store.write_descriptor_chain(
                    rid_pk,
                    &mut rid_desc,
                    &modified_rid_metas,
                    &mut puts,
                    &mut frees,
                )?;
            }
        };

        drop(catalog);
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

        // TODO: Put logic directly in index plugin
        match kind {
            IndexKind::Sort => {
                // The Sort index lives on the main data column.
                let mut metas: Vec<ChunkMetadata> =
                    DescriptorIterator::new(self.pager.as_ref(), descriptor.head_page_pk)
                        .collect::<Result<_>>()?;

                // `stage_drop_index` modifies `metas` in-place (clearing perm_pk) and returns a list of keys to free.
                let sort_frees = <sort::SortIndexOps as IndexOps<P>>::stage_drop_index(
                    &self.sort_ops,
                    &self.pager,
                    &mut metas,
                )?;
                frees.extend(sort_frees);

                // Logically unregister the index first.
                self.stage_index_unregistration(&mut descriptor, kind)?;
                // Then rewrite the descriptor chain, which will serialize the updated descriptor and its pages.
                column_store.write_descriptor_chain(
                    descriptor_pk,
                    &mut descriptor,
                    &metas,
                    &mut puts,
                    &mut frees,
                )?;
            }
            IndexKind::Presence => {
                // The Presence index lives on the shadow row_id column.
                let rid_fid = rowid_fid(field_id);
                let rid_pk = *catalog.map.get(&rid_fid).ok_or(Error::NotFound)?;
                let rid_desc_blob = self
                    .pager
                    .batch_get(&[BatchGet::Raw { key: rid_pk }])?
                    .pop()
                    .and_then(|r| match r {
                        GetResult::Raw { bytes, .. } => Some(bytes),
                        _ => None,
                    })
                    .ok_or(Error::NotFound)?;
                let mut rid_desc = ColumnDescriptor::from_le_bytes(rid_desc_blob.as_ref());

                let mut rid_metas: Vec<ChunkMetadata> =
                    DescriptorIterator::new(self.pager.as_ref(), rid_desc.head_page_pk)
                        .collect::<Result<_>>()?;

                let presence_frees = <presence::PresenceIndexOps as IndexOps<P>>::stage_drop_index(
                    &self.presence_ops,
                    &self.pager,
                    &mut rid_metas,
                )?;
                frees.extend(presence_frees);

                // Logically unregister on the main descriptor first.
                self.stage_index_unregistration(&mut descriptor, kind)?;
                // Stage the write for the main descriptor.
                puts.push(BatchPut::Raw {
                    key: descriptor_pk,
                    bytes: descriptor.to_le_bytes(),
                });
                // Then rewrite the now-modified shadow descriptor chain.
                column_store.write_descriptor_chain(
                    rid_pk,
                    &mut rid_desc,
                    &rid_metas,
                    &mut puts,
                    &mut frees,
                )?;
            }
        }

        drop(catalog);

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
