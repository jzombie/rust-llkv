//! Generic traits for column indexes.

use crate::error::Result;
use crate::storage::pager::{BatchPut, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::ChunkMetadata;
use crate::types::{LogicalFieldId, PhysicalKey};
use arrow::array::ArrayRef;
use simd_r_drive_entry_handle::EntryHandle;

/// Context passed to index methods during store operations.
pub struct IndexOpContext<'a> {
    pub pager: &'a dyn Pager<Blob = EntryHandle>,
    pub catalog: &'a mut ColumnCatalog,
    pub puts: &'a mut Vec<BatchPut>,
    pub frees: &'a mut Vec<PhysicalKey>,
}

/// A generic interface for a column index.
///
/// Indexes are managed by an `IndexManager` and respond to events in the store
/// like appends, deletes, and rewrites.
pub trait Index: Send + Sync {
    /// Returns a unique name for the index type (e.g., "presence", "sort").
    fn name(&self) -> &'static str;

    /// Called when new data is appended to a column.
    ///
    /// `chunks` contains the newly created data and row-id chunks that need to be indexed.
    fn on_append(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        new_chunks: &[(ChunkMetadata, ArrayRef)], // (data_meta, corresponding_rids)
    ) -> Result<()>;

    /// Called when rows are deleted from a column.
    ///
    /// `rewritten_chunks` provides the new metadata and in-memory arrays for chunks
    /// that were modified by the delete operation.
    fn on_rewrite(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        rewritten_chunks: &[(ChunkMetadata, ArrayRef)], // (new_data_meta, new_data_array)
    ) -> Result<()>;

    /// Called when a field is being compacted.
    ///
    /// This hook allows the index to update its own metadata in response to the
    /// data column's layout changing.
    fn on_compact(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        new_metas: &[ChunkMetadata], // The complete new set of metas for the data column
    ) -> Result<()>;

    /// Backfills the index for a column that already has data.
    fn backfill(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
    ) -> Result<()>;
}

/// Manages all registered indexes for the `ColumnStore`.
pub trait IndexManager: Send + Sync {
    /// Registers a new index type.
    fn register_index(&mut self, index: Box<dyn Index>);

    /// Dispatches an `on_append` event to all relevant indexes for a field.
    fn dispatch_append(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        new_chunks: &[(ChunkMetadata, ArrayRef)],
    ) -> Result<()>;

    /// Dispatches an `on_rewrite` event.
    fn dispatch_rewrite(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        rewritten_chunks: &[(ChunkMetadata, ArrayRef)],
    ) -> Result<()>;

    /// Dispatches an `on_compact` event.
    fn dispatch_compact(
        &mut self,
        context: &mut IndexOpContext<'_>,
        field_id: LogicalFieldId,
        new_metas: &[ChunkMetadata],
    ) -> Result<()>;
}
