use std::sync::Arc;

use arrow::array::RecordBatch;

use crate::types::RowId;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_column_map::{ColumnStore, types::LogicalFieldId};
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

/// Streaming view over a set of row IDs for selected logical fields.
///
/// `ColumnStream` keeps a reusable gather context so repeated calls avoid
/// reparsing column descriptors or re-fetching chunk metadata. Each call to
/// [`ColumnStream::next_batch`] returns at most `STREAM_BATCH_ROWS` values,
/// backed by Arrow arrays without copying the column data.
pub struct ColumnStream<'table, P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: &'table ColumnStore<P>,
    ctx: MultiGatherContext,
    row_ids: Vec<RowId>,
    position: usize,
    chunk_size: usize,
    policy: GatherNullPolicy,
    logical_fields: Arc<[LogicalFieldId]>,
}

/// Single batch produced by [`ColumnStream`].
pub struct ColumnStreamBatch<'stream> {
    start: usize,
    row_ids: &'stream [RowId],
    batch: RecordBatch,
}

impl<'table, P> ColumnStream<'table, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(
        store: &'table ColumnStore<P>,
        ctx: MultiGatherContext,
        row_ids: Vec<RowId>,
        chunk_size: usize,
        policy: GatherNullPolicy,
        logical_fields: Arc<[LogicalFieldId]>,
    ) -> Self {
        Self {
            store,
            ctx,
            row_ids,
            position: 0,
            chunk_size,
            policy,
            logical_fields,
        }
    }

    /// Total number of row IDs covered by this stream.
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.row_ids.len()
    }

    /// Remaining number of row IDs that have not yet been yielded.
    #[inline]
    pub fn remaining_rows(&self) -> usize {
        self.row_ids.len().saturating_sub(self.position)
    }

    /// Logical fields produced by this stream.
    #[inline]
    pub fn logical_fields(&self) -> &[LogicalFieldId] {
        &self.logical_fields
    }

    /// Fetch the next chunk of rows, if any remain.
    pub fn next_batch(&mut self) -> LlkvResult<Option<ColumnStreamBatch<'_>>> {
        while self.position < self.row_ids.len() {
            let start = self.position;
            let end = (start + self.chunk_size).min(self.row_ids.len());
            let window = &self.row_ids[start..end];

            let batch =
                self.store
                    .gather_rows_with_reusable_context(&mut self.ctx, window, self.policy)?;

            self.position = end;

            if batch.num_rows() == 0 && matches!(self.policy, GatherNullPolicy::DropNulls) {
                // All rows dropped; continue to the next chunk to avoid yielding empties.
                continue;
            }

            return Ok(Some(ColumnStreamBatch {
                start,
                row_ids: window,
                batch,
            }));
        }

        Ok(None)
    }
}

impl<'stream> ColumnStreamBatch<'stream> {
    #[inline]
    pub fn row_ids(&self) -> &'stream [RowId] {
        self.row_ids
    }

    #[inline]
    pub fn row_offset(&self) -> usize {
        self.start
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    #[inline]
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    #[inline]
    pub fn into_batch(self) -> RecordBatch {
        self.batch
    }
}
