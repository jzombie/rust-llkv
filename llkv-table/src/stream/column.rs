use std::sync::Arc;

use arrow::array::RecordBatch;

use crate::types::RowId;
use llkv_column_map::store::{GatherNullPolicy, MultiGatherContext};
use llkv_column_map::{ColumnStore, types::LogicalFieldId};
use llkv_result::Result as LlkvResult;
use llkv_storage::pager::{MemPager, Pager};
use simd_r_drive_entry_handle::EntryHandle;

use roaring::RoaringTreemap;

/// Iterator over row IDs that can own or borrow the underlying bitmap.
pub enum RowIdIter<'a> {
    Owned(roaring::treemap::IntoIter),
    Borrowed(roaring::treemap::Iter<'a>),
}

impl<'a> Iterator for RowIdIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(iter) => iter.next(),
            Self::Borrowed(iter) => iter.next(),
        }
    }
}

impl From<RoaringTreemap> for RowIdIter<'static> {
    fn from(map: RoaringTreemap) -> Self {
        Self::Owned(map.into_iter())
    }
}

impl<'a> From<&'a RoaringTreemap> for RowIdIter<'a> {
    fn from(map: &'a RoaringTreemap) -> Self {
        Self::Borrowed(map.iter())
    }
}

pub trait RowIdStreamSource<'a> {
    fn count(&self) -> u64;
    fn into_iter_source(self) -> RowIdIter<'a>;
}

impl RowIdStreamSource<'static> for RoaringTreemap {
    fn count(&self) -> u64 {
        self.len()
    }
    fn into_iter_source(self) -> RowIdIter<'static> {
        RowIdIter::Owned(self.into_iter())
    }
}

impl<'a> RowIdStreamSource<'a> for &'a RoaringTreemap {
    fn count(&self) -> u64 {
        self.len()
    }
    fn into_iter_source(self) -> RowIdIter<'a> {
        RowIdIter::Borrowed(self.iter())
    }
}

/// Streaming view over a set of row IDs for selected logical fields.
///
/// `ColumnStream` keeps a reusable gather context so repeated calls avoid
/// reparsing column descriptors or re-fetching chunk metadata. Each call to
/// [`ColumnStream::next_batch`] returns at most `STREAM_BATCH_ROWS` values,
/// backed by Arrow arrays without copying the column data.
pub struct ColumnStream<'table, 'a, P = MemPager>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    store: &'table ColumnStore<P>,
    ctx: MultiGatherContext,
    row_ids: RowIdIter<'a>,
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

impl<'table, 'a, P> ColumnStream<'table, 'a, P>
where
    P: Pager<Blob = EntryHandle> + Send + Sync,
{
    pub(crate) fn new(
        store: &'table ColumnStore<P>,
        ctx: MultiGatherContext,
        row_ids: RowIdIter<'a>,
        total_rows: usize,
        chunk_size: usize,
        policy: GatherNullPolicy,
        logical_fields: Arc<[LogicalFieldId]>,
    ) -> Self {
        Self {
            store,
            ctx,
            row_ids,
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

