use crate::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::types::PhysicalKey;
use llkv_result::Result;
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::Arc;
use std::sync::Mutex;
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

    // --- Write classifications ---
    pub fresh_puts: AtomicU64,
    pub fresh_put_bytes: AtomicU64,
    pub overwritten_puts: AtomicU64,
    pub overwritten_put_bytes: AtomicU64,
    pub unknown_puts: AtomicU64,
    pub unknown_put_bytes: AtomicU64,
}

impl IoStats {
    fn record_put(&self, classification: KeyWriteClassification, bytes: usize) {
        let bytes = bytes as u64;
        match classification {
            KeyWriteClassification::Fresh => {
                self.fresh_puts.fetch_add(1, Ordering::Relaxed);
                self.fresh_put_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
            KeyWriteClassification::Overwrite => {
                self.overwritten_puts.fetch_add(1, Ordering::Relaxed);
                self.overwritten_put_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            KeyWriteClassification::Unknown => {
                self.unknown_puts.fetch_add(1, Ordering::Relaxed);
                self.unknown_put_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
        }
    }

    /// Capture a point-in-time snapshot of the accumulated metrics.
    pub fn snapshot(&self) -> IoStatsSnapshot {
        IoStatsSnapshot {
            physical_gets: self.physical_gets.load(Ordering::Relaxed),
            physical_puts: self.physical_puts.load(Ordering::Relaxed),
            physical_frees: self.physical_frees.load(Ordering::Relaxed),
            physical_allocs: self.physical_allocs.load(Ordering::Relaxed),
            get_batches: self.get_batches.load(Ordering::Relaxed),
            put_batches: self.put_batches.load(Ordering::Relaxed),
            free_batches: self.free_batches.load(Ordering::Relaxed),
            alloc_batches: self.alloc_batches.load(Ordering::Relaxed),
            fresh_puts: self.fresh_puts.load(Ordering::Relaxed),
            fresh_put_bytes: self.fresh_put_bytes.load(Ordering::Relaxed),
            overwritten_puts: self.overwritten_puts.load(Ordering::Relaxed),
            overwritten_put_bytes: self.overwritten_put_bytes.load(Ordering::Relaxed),
            unknown_puts: self.unknown_puts.load(Ordering::Relaxed),
            unknown_put_bytes: self.unknown_put_bytes.load(Ordering::Relaxed),
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&self) {
        self.physical_gets.store(0, Ordering::Relaxed);
        self.physical_puts.store(0, Ordering::Relaxed);
        self.physical_frees.store(0, Ordering::Relaxed);
        self.physical_allocs.store(0, Ordering::Relaxed);
        self.get_batches.store(0, Ordering::Relaxed);
        self.put_batches.store(0, Ordering::Relaxed);
        self.free_batches.store(0, Ordering::Relaxed);
        self.alloc_batches.store(0, Ordering::Relaxed);
        self.fresh_puts.store(0, Ordering::Relaxed);
        self.fresh_put_bytes.store(0, Ordering::Relaxed);
        self.overwritten_puts.store(0, Ordering::Relaxed);
        self.overwritten_put_bytes.store(0, Ordering::Relaxed);
        self.unknown_puts.store(0, Ordering::Relaxed);
        self.unknown_put_bytes.store(0, Ordering::Relaxed);
    }
}

/// Immutable copy of [`IoStats`] counters captured at a specific moment.
#[derive(Debug, Clone, Copy, Default)]
pub struct IoStatsSnapshot {
    pub physical_gets: u64,
    pub physical_puts: u64,
    pub physical_frees: u64,
    pub physical_allocs: u64,
    pub get_batches: u64,
    pub put_batches: u64,
    pub free_batches: u64,
    pub alloc_batches: u64,
    pub fresh_puts: u64,
    pub fresh_put_bytes: u64,
    pub overwritten_puts: u64,
    pub overwritten_put_bytes: u64,
    pub unknown_puts: u64,
    pub unknown_put_bytes: u64,
}

impl IoStatsSnapshot {
    /// Compute the delta between two snapshots (`newer - older`). Saturates at zero.
    pub fn delta_since(&self, older: &Self) -> Self {
        macro_rules! delta {
            ($field:ident) => {
                self.$field.saturating_sub(older.$field)
            };
        }

        Self {
            physical_gets: delta!(physical_gets),
            physical_puts: delta!(physical_puts),
            physical_frees: delta!(physical_frees),
            physical_allocs: delta!(physical_allocs),
            get_batches: delta!(get_batches),
            put_batches: delta!(put_batches),
            free_batches: delta!(free_batches),
            alloc_batches: delta!(alloc_batches),
            fresh_puts: delta!(fresh_puts),
            fresh_put_bytes: delta!(fresh_put_bytes),
            overwritten_puts: delta!(overwritten_puts),
            overwritten_put_bytes: delta!(overwritten_put_bytes),
            unknown_puts: delta!(unknown_puts),
            unknown_put_bytes: delta!(unknown_put_bytes),
        }
    }

    fn bytes_to_mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    fn ops_per_batch(ops: u64, batches: u64) -> f64 {
        if batches == 0 {
            0.0
        } else {
            ops as f64 / batches as f64
        }
    }

    /// Fresh-write bytes converted to mebibytes.
    pub fn fresh_mib(&self) -> f64 {
        Self::bytes_to_mib(self.fresh_put_bytes)
    }

    /// Overwrite bytes converted to mebibytes.
    pub fn overwrite_mib(&self) -> f64 {
        Self::bytes_to_mib(self.overwritten_put_bytes)
    }

    /// Unknown-write bytes converted to mebibytes.
    pub fn unknown_mib(&self) -> f64 {
        Self::bytes_to_mib(self.unknown_put_bytes)
    }

    /// Overwrite percentage relative to fresh + overwrite bytes.
    pub fn overwrite_pct(&self) -> f64 {
        let total = self.fresh_put_bytes + self.overwritten_put_bytes;
        if total == 0 {
            0.0
        } else {
            (self.overwritten_put_bytes as f64 / total as f64) * 100.0
        }
    }

    /// Average physical put operations per batch.
    pub fn puts_per_batch(&self) -> f64 {
        Self::ops_per_batch(self.physical_puts, self.put_batches)
    }

    /// Average physical get operations per batch.
    pub fn gets_per_batch(&self) -> f64 {
        Self::ops_per_batch(self.physical_gets, self.get_batches)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyWriteClassification {
    Fresh,
    Overwrite,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyState {
    Allocated,
    Written,
}

#[derive(Debug, Default)]
struct KeyTracker {
    state: Mutex<FxHashMap<PhysicalKey, KeyState>>,
}

impl KeyTracker {
    fn mark_allocated(&self, key: PhysicalKey) {
        let mut guard = self.state.lock().unwrap();
        guard.insert(key, KeyState::Allocated);
    }

    fn mark_freed(&self, key: &PhysicalKey) {
        let mut guard = self.state.lock().unwrap();
        guard.remove(key);
    }

    fn classify_put(&self, key: PhysicalKey) -> KeyWriteClassification {
        let mut guard = self.state.lock().unwrap();
        match guard.get_mut(&key) {
            Some(state @ KeyState::Allocated) => {
                *state = KeyState::Written;
                KeyWriteClassification::Fresh
            }
            Some(KeyState::Written) => KeyWriteClassification::Overwrite,
            None => {
                // Track the key so subsequent writes are treated as overwrites.
                guard.insert(key, KeyState::Written);
                KeyWriteClassification::Unknown
            }
        }
    }
}

/// A wrapper around any Pager implementation that instruments I/O operations.
#[derive(Debug)]
pub struct InstrumentedPager<P: Pager> {
    inner: P,
    stats: Arc<IoStats>,
    tracker: KeyTracker,
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
                tracker: KeyTracker::default(),
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
        for put in puts {
            match put {
                BatchPut::Raw { key, bytes } => {
                    let classification = self.tracker.classify_put(*key);
                    self.stats.record_put(classification, bytes.len());
                }
            }
        }
        self.inner.batch_put(puts)
    }

    fn alloc_many(&self, count: usize) -> Result<Vec<PhysicalKey>> {
        self.stats
            .physical_allocs
            .fetch_add(count as u64, Ordering::Relaxed);
        self.stats.alloc_batches.fetch_add(1, Ordering::Relaxed);
        let keys = self.inner.alloc_many(count)?;
        for key in &keys {
            self.tracker.mark_allocated(*key);
        }
        Ok(keys)
    }

    fn free_many(&self, keys: &[PhysicalKey]) -> Result<()> {
        self.stats
            .physical_frees
            .fetch_add(keys.len() as u64, Ordering::Relaxed);
        self.stats.free_batches.fetch_add(1, Ordering::Relaxed);
        for key in keys {
            self.tracker.mark_freed(key);
        }
        self.inner.free_many(keys)
    }
}
