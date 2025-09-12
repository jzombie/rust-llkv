use std::io;

use crate::types::{BlobLike, PhysicalKey, TypedKind, TypedValue};

pub mod mem_pager;
pub use mem_pager::*;

/// Put operations inside a single batch.
#[derive(Clone, Debug)]
pub enum BatchPut {
    Raw { key: PhysicalKey, bytes: Vec<u8> },
    Typed { key: PhysicalKey, value: TypedValue },
}

/// Get operations inside a single batch.
#[derive(Clone, Debug)]
pub enum BatchGet {
    Raw { key: PhysicalKey },
    Typed { key: PhysicalKey, kind: TypedKind },
}

// TODO: Clippy recommends boxing `TypedValue`, but I'm not yet sure of the
// performance impact, so I'm skipping it for now.
//
/// Result for a single get. For raw reads, the pager returns a **blob handle**
/// chosen by the pager implementation (see `Pager::Blob`). This blob must be
/// cheap to clone and must expose its bytes via `AsRef<[u8]>`.
///
/// Examples:
/// - In-memory pager: `Blob = Arc<[u8]>`
/// - File-backed pager: `Blob = EntryHandle` (which wraps `Arc<Mmap> + Range`)
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum GetResult<B> {
    Raw { key: PhysicalKey, bytes: B },
    Typed { key: PhysicalKey, value: TypedValue },
    Missing { key: PhysicalKey },
}

/// Unified pager interface with separate put/get passes.
/// - `batch_put` applies all writes atomically w.r.t. this pager.
/// - `batch_get` serves a mixed list of typed/raw reads in one round-trip.
///
/// Returning **owned blob handles** allows `&self` for reads (good for
/// RwLock read guards) while remaining zero-copy when the blob is mmap-backed.
///
/// ## Zero-copy blobs
/// Implementations choose `type Blob`:
/// - In-memory: `type Blob = Arc<[u8]>`
/// - mmap-backed: `type Blob = EntryHandle` (your type that wraps `Arc<Mmap>`)
pub trait Pager: Send + Sync + 'static {
    /// The blob handle type returned for raw reads. Must be:
    /// - `AsRef<[u8]>` so callers can view bytes without copying,
    /// - `Clone` so the handle can be duplicated cheaply (refcount bump),
    /// - `Send + Sync + 'static` for cross-thread safety.
    type Blob: AsRef<[u8]> + Clone + Send + Sync + 'static;

    /// Key allocation can remain separate; typically needed ahead of
    /// a write batch.
    fn alloc_many(&self, n: usize) -> io::Result<Vec<PhysicalKey>>;

    /// Apply all puts in one batch.
    fn batch_put(&self, puts: &[BatchPut]) -> io::Result<()>;

    /// Serve all gets (raw + typed) in one batch.
    fn batch_get(&self, gets: &[BatchGet]) -> io::Result<Vec<GetResult<Self::Blob>>>;

    /// Used for deletions (typically after GC, since each physical node may
    /// represent multiple entries)
    fn free_many(&self, keys: &[PhysicalKey]) -> io::Result<()>;
}
