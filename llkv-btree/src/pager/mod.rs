use crate::errors::Error;
use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;
pub mod shared_adapter;
pub use shared_adapter::SharedPager;
pub mod mem;

pub const DEFAULT_PAGE_SIZE_HINT: usize = 4096;

/// Pager contract used by the B+Tree.
///
/// - I/O is **batch-only** (no single read/write).
/// - `Page` is a *zero-copy readable handle* (e.g., `Arc<[u8]>`, mmap-backed, or your
///   `EntryHandle`) that derefs to `&[u8]`.
pub trait Pager {
    type Id: Clone + Eq + Ord + Hash + Debug;
    type Page: Clone + core::ops::Deref<Target = [u8]>;

    /// Batch read. Missing pages are omitted from the map.
    fn read_batch(&self, ids: &[Self::Id]) -> Result<FxHashMap<Self::Id, Self::Page>, Error>;

    /// Batch write. Caller keeps page-size invariant.
    fn write_batch(&self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error>;

    /// Allocate `count` new page IDs.
    fn alloc_ids(&self, count: usize) -> Result<Vec<Self::Id>, Error>;

    /// Batch deallocate page IDs.
    fn dealloc_ids(&self, ids: &[Self::Id]) -> Result<(), Error>;

    /// Optional page-size hint (for splitting / underflow checks).
    fn page_size_hint(&self) -> Option<usize>;

    /// Create a readable `Page` handle from owned bytes (used to surface staged writes
    /// through the same type as real pager reads; stays zero-copy for the caller).
    fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error>;

    /// Optional notification; e.g. persist new root in a meta page.
    fn on_root_changed(&self, _new_root: Self::Id) -> Result<(), Error> {
        Ok(())
    }
}
