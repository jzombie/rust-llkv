//! Indexing: shared traits, manager, and dispatcher only.
//! Concrete indexes live in their own files.

use crate::error::{Error, Result};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::ColumnDescriptor;
use crate::types::LogicalFieldId;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

pub mod presence;
pub mod sort;

/* ======================= Public discovery surface ====================== */

/// Marker trait carried by concrete index types.
pub trait Index {
    /// Stable kind of the index (e.g., `IndexKind::Sort`, `IndexKind::Presence`).
    fn kind(&self) -> IndexKind;
}

/* ========== Manager: register / unregister / list (generic Pager) ====== */

/// Manages registration and discovery of indexes on columns.
pub struct IndexManager<P: Pager> {
    pub(crate) pager: Arc<P>,
    pub(crate) catalog: Arc<RwLock<ColumnCatalog>>,
}

impl<P: Pager> IndexManager<P> {
    /// Creates a new `IndexManager`.
    pub fn new(pager: Arc<P>, catalog: Arc<RwLock<ColumnCatalog>>) -> Self {
        Self { pager, catalog }
    }

    /// Registers an index name for a given column.
    pub fn register_index(&self, field_id: LogicalFieldId, kind: IndexKind) -> Result<()> {
        self.update_indexes(field_id, |indexes| {
            if indexes.contains(&kind) {
                return Err(Error::InvalidArgumentError(format!(
                    "Index '{:?}' already exists for this column.",
                    kind
                )));
            }
            indexes.push(kind);
            Ok(())
        })
    }

    /// Unregisters an index name from a given column.
    pub fn unregister_index(&self, field_id: LogicalFieldId, kind: IndexKind) -> Result<()> {
        self.update_indexes(field_id, |indexes| {
            let original_len = indexes.len();
            indexes.retain(|loc_kind| *loc_kind != kind);
            if indexes.len() == original_len {
                return Err(Error::InvalidArgumentError(format!(
                    "Index '{:?}' not found for this column.",
                    kind
                )));
            }
            Ok(())
        })
    }

    /// Retrieves the names of registered indexes for the given columns.
    pub fn get_column_indexes(
        &self,
        field_ids: &[LogicalFieldId],
    ) -> Result<FxHashMap<LogicalFieldId, Vec<IndexKind>>> {
        let catalog = self.catalog.read().unwrap();
        let mut gets = Vec::new();
        let mut pk_to_fid = FxHashMap::default();

        for &field_id in field_ids {
            if let Some(&pk) = catalog.map.get(&field_id) {
                gets.push(BatchGet::Raw { key: pk });
                pk_to_fid.insert(pk, field_id);
            }
        }

        if gets.is_empty() {
            return Ok(FxHashMap::default());
        }

        let results = self.pager.batch_get(&gets)?;
        let mut all_indexes = FxHashMap::default();

        for result in results {
            if let GetResult::Raw { key, bytes } = result
                && let Some(&field_id) = pk_to_fid.get(&key)
            {
                let descriptor = ColumnDescriptor::from_le_bytes(bytes.as_ref());
                all_indexes.insert(field_id, descriptor.get_indexes()?);
            }
        }

        Ok(all_indexes)
    }

    /// Helper to load, modify, and save descriptor indexes.
    fn update_indexes<F>(&self, field_id: LogicalFieldId, mut modifier: F) -> Result<()>
    where
        F: FnMut(&mut Vec<IndexKind>) -> Result<()>,
    {
        let catalog = self.catalog.read().unwrap();
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
        let mut indexes = descriptor.get_indexes()?;
        modifier(&mut indexes)?;
        descriptor.set_indexes(&indexes)?;

        self.pager.batch_put(&[BatchPut::Raw {
            key: descriptor_pk,
            bytes: descriptor.to_le_bytes(),
        }])
    }
}

/* ======================= Uniform execution interface =================== */

/// Kinds of indexes supported by the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexKind {
    Sort,
    Presence,
}

// 1. Enum -> integer (infallible)
impl From<IndexKind> for u8 {
    fn from(kind: IndexKind) -> Self {
        match kind {
            IndexKind::Presence => 0,
            IndexKind::Sort => 1,
        }
    }
}

// 2. Integer -> Enum (fallible, so we use TryFrom)
impl TryFrom<u8> for IndexKind {
    type Error = crate::error::Error; // Error type for invalid integer

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(IndexKind::Presence),
            1 => Ok(IndexKind::Sort),
            _ => Err(Error::Internal("Invalid IndexKind integer!".to_string())),
        }
    }
}

/// Hints for incremental updates to an index.
#[derive(Clone, Debug, Default)]
pub struct IndexUpdateHint {
    /// Recompute only these chunk pages. Empty => unknown/all.
    pub changed_chunk_pks: Vec<crate::types::PhysicalKey>,
    /// If true, builder may ignore hints and rebuild all.
    pub full_rebuild_ok: bool,
}

/// Uniform ops each concrete index must implement.
///
/// These do the physical work (compute/alloc/write). They DO NOT change
/// descriptor index-name lists; call register/unregister separately.
pub trait IndexOps<P: Pager> {
    fn build_all(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()>;

    fn update(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
        hint: &IndexUpdateHint,
    ) -> Result<()>;

    fn drop_index(
        &self,
        pager: &Arc<P>,
        catalog: &Arc<RwLock<ColumnCatalog>>,
        field: LogicalFieldId,
    ) -> Result<()>;
}

/* ===== Dispatcher: requires Pager blobs to be EntryHandle for sort ===== */

impl<P> IndexManager<P>
where
    P: Pager<Blob = EntryHandle>,
{
    /// Build from scratch for a populated column.
    pub fn build_index(&self, kind: IndexKind, field_id: LogicalFieldId) -> Result<()> {
        match kind {
            IndexKind::Sort => sort::SortIndexOps.build_all(&self.pager, &self.catalog, field_id),
            IndexKind::Presence => {
                presence::PresenceIndexOps.build_all(&self.pager, &self.catalog, field_id)
            }
        }
    }

    /// Incrementally update an existing index.
    pub fn update_index(
        &self,
        kind: IndexKind,
        field_id: LogicalFieldId,
        hint: &IndexUpdateHint,
    ) -> Result<()> {
        match kind {
            IndexKind::Sort => {
                sort::SortIndexOps.update(&self.pager, &self.catalog, field_id, hint)
            }
            IndexKind::Presence => {
                presence::PresenceIndexOps.update(&self.pager, &self.catalog, field_id, hint)
            }
        }
    }

    /// Remove the index and its metadata from the descriptor.
    pub fn drop_index(&self, kind: IndexKind, field_id: LogicalFieldId) -> Result<()> {
        match kind {
            IndexKind::Sort => sort::SortIndexOps.drop_index(&self.pager, &self.catalog, field_id),
            IndexKind::Presence => {
                presence::PresenceIndexOps.drop_index(&self.pager, &self.catalog, field_id)
            }
        }
    }
}
