//! Presence index: type and ops live here.

use super::IndexKind;
use crate::error::Result;
use crate::storage::pager::Pager;
use crate::store::catalog::ColumnCatalog;
use crate::store::indexing::{Index, IndexOps, IndexUpdateHint};
use crate::types::LogicalFieldId;
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
///
/// NOTE: Bodies are placeholders to avoid guessing. Wire in your real
/// presence build/update/drop code where TODOs are.
pub struct PresenceIndexOps;

impl Default for PresenceIndexOps {
    fn default() -> Self {
        PresenceIndexOps
    }
}

impl<P: Pager> IndexOps<P> for PresenceIndexOps {
    fn build_all(
        &self,
        _pager: &Arc<P>,
        _catalog: &Arc<RwLock<ColumnCatalog>>,
        _field: LogicalFieldId,
    ) -> Result<()> {
        // TODO: compute presence metadata and rewrite pages.
        Ok(())
    }

    fn update(
        &self,
        _pager: &Arc<P>,
        _catalog: &Arc<RwLock<ColumnCatalog>>,
        _field: LogicalFieldId,
        _hint: &IndexUpdateHint,
    ) -> Result<()> {
        // TODO: recompute presence for changed chunks.
        Ok(())
    }

    fn drop_index(
        &self,
        _pager: &Arc<P>,
        _catalog: &Arc<RwLock<ColumnCatalog>>,
        _field: LogicalFieldId,
    ) -> Result<()> {
        // TODO: clear presence refs in descriptor pages.
        Ok(())
    }
}
