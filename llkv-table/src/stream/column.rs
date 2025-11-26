use std::sync::Arc;

use arrow::array::RecordBatch;

use crate::types::RowId;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_column_map::{ColumnStore, types::LogicalFieldId};
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

use roaring::RoaringTreemap;

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
    row_ids: roaring::treemap::IntoIter,
    position: usize,
    total_rows: usize,
    chunk_size: usize,
    policy: GatherNullPolicy,
    logical_fields: Arc<[LogicalFieldId]>,
}

/// Single batch produced by [`ColumnStream`].
pub struct ColumnStreamBatch {
    start: usize,
    row_ids: Vec<RowId>,
    batch: RecordBatch,
}

impl<'table, P> ColumnStream<'table, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(
        store: &'table ColumnStore<P>,
        ctx: MultiGatherContext,
        row_ids: RoaringTreemap,
        chunk_size: usize,
        policy: GatherNullPolicy,
        logical_fields: Arc<[LogicalFieldId]>,
    ) -> Self {
        let total_rows = row_ids.len() as usize;
        Self {
            store,
            ctx,
            row_ids: row_ids.into_iter(),
            position: 0,
            total_rows,
            chunk_size,
            policy,
            logical_fields,
        }
    }

    /// Total number of row IDs covered by this stream.
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Remaining number of row IDs that have not yet been yielded.
    #[inline]
    pub fn remaining_rows(&self) -> usize {
        self.total_rows.saturating_sub(self.position)
    }

    /// Logical fields produced by this stream.
    #[inline]
    pub fn logical_fields(&self) -> &[LogicalFieldId] {
        &self.logical_fields
    }

    /// Fetch the next chunk of rows, if any remain.
    pub fn next_batch(&mut self) -> LlkvResult<Option<ColumnStreamBatch>> {
        loop {
            let mut window = Vec::with_capacity(self.chunk_size);
            for _ in 0..self.chunk_size {
                if let Some(rid) = self.row_ids.next() {
                    window.push(rid);
                } else {
                    break;
                }
            }

            if window.is_empty() {
                return Ok(None);
            }

            let start = self.position;
            self.position += window.len();

            let batch =
                self.store
                    .gather_rows_with_reusable_context(&mut self.ctx, &window, self.policy)?;

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
    }
}

impl ColumnStreamBatch {
    #[inline]
    pub fn row_ids(&self) -> &[RowId] {
        &self.row_ids
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

