//! Generic traits for column indexes.

use crate::error::Result;
use crate::storage::pager::{BatchPut, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::ChunkMetadata;
use crate::types::{LogicalFieldId, PhysicalKey};
use arrow::array::ArrayRef;
use simd_r_drive_entry_handle::EntryHandle;

/// Describes a planned write for a new index segment.
/// This is returned by an `Index` implementation and is purely data.
pub struct IndexPlan {
    /// The metadata for the new index chunk.
    pub metadata: ChunkMetadata,
    /// The raw, serialized bytes of the new index chunk.
    pub bytes: Vec<u8>,
}

/// Context for backfill operations.
pub struct BackfillContext<'a> {
    pub pager: &'a dyn Pager<Blob = EntryHandle>,
    pub catalog: &'a mut ColumnCatalog,
    pub puts: &'a mut Vec<BatchPut>,
    pub frees: &'a mut Vec<PhysicalKey>,
}

/// A generic interface for a column index.
pub trait Index: Send + Sync {
    /// Returns a unique name for the index type (e.g., "presence", "sort").
    fn name(&self) -> &'static str;

    /// Given new data and row IDs for a chunk, plan the corresponding index write.
    ///
    /// This method is purely computational and must not perform any I/O. It returns
    /// an `Option` because some indexes (like a sort index) may not apply to all
    /// data types.
    fn plan_append(
        &self,
        data_for_chunk: &ArrayRef,
        rids_for_chunk: &ArrayRef,
    ) -> Result<Option<IndexPlan>>;

    /// Discovers all fields for which this index is present based on on-disk metadata.
    fn discover(
        &self,
        pager: &dyn Pager<Blob = EntryHandle>,
        catalog: &ColumnCatalog,
    ) -> Result<Vec<LogicalFieldId>>;

    /// Backfills the index for a column that already has data.
    fn backfill(&self, context: &mut BackfillContext<'_>, field_id: LogicalFieldId) -> Result<()>;
}
