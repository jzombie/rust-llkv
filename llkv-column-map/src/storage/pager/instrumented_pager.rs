use crate::error::Result; // Use the project's Result type
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::types::PhysicalKey;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A thread-safe container for I/O statistics.
#[derive(Debug, Default)]
pub struct IoStats {
    // --- Total individual items ---
    pub physical_gets: AtomicU64,
    pub physical_puts: AtomicU64,
    pub physical_frees: AtomicU64,
    pub physical_allocs: AtomicU64,

    // --- Total batch operations (i.e., calls to the pager) ---
    pub get_batches: AtomicU64,
    pub put_batches: AtomicU64,
    pub free_batches: AtomicU64,
    pub alloc_batches: AtomicU64,
}

/// A wrapper around any Pager implementation that instruments I/O operations.
#[derive(Debug)]
pub struct InstrumentedPager<P: Pager> {
    inner: P,
    stats: Arc<IoStats>,
}

impl<P> InstrumentedPager<P>
where
    P: Pager,
{
    /// Wraps a Pager and returns the instrumented version along with a handle
    /// to its statistics.
    pub fn new(inner: P) -> (Self, Arc<IoStats>) {
        let stats = Arc::new(IoStats::default());
        (
            Self {
                inner,
                stats: Arc::clone(&stats),
            },
            stats,
        )
    }
}

impl<P> Pager for InstrumentedPager<P>
where
    P: Pager<Blob = EntryHandle>,
{
    type Blob = P::Blob;

    fn batch_get(&self, gets: &[BatchGet]) -> Result<Vec<GetResult<Self::Blob>>> {
        self.stats
            .physical_gets
            .fetch_add(gets.len() as u64, Ordering::Relaxed);
        self.stats.get_batches.fetch_add(1, Ordering::Relaxed);
        self.inner.batch_get(gets)
    }

    fn batch_put(&self, puts: &[BatchPut]) -> Result<()> {
        self.stats
            .physical_puts
            .fetch_add(puts.len() as u64, Ordering::Relaxed);
        self.stats.put_batches.fetch_add(1, Ordering::Relaxed);
        self.inner.batch_put(puts)
    }

    fn alloc_many(&self, count: usize) -> Result<Vec<PhysicalKey>> {
        self.stats
            .physical_allocs
            .fetch_add(count as u64, Ordering::Relaxed);
        self.stats.alloc_batches.fetch_add(1, Ordering::Relaxed);
        self.inner.alloc_many(count)
    }

    fn free_many(&self, keys: &[PhysicalKey]) -> Result<()> {
        self.stats
            .physical_frees
            .fetch_add(keys.len() as u64, Ordering::Relaxed);
        self.stats.free_batches.fetch_add(1, Ordering::Relaxed);
        self.inner.free_many(keys)
    }
}
