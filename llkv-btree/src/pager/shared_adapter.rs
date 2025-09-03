use crate::errors::Error;
use crate::pager::Pager;
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

/// Thread-safe wrapper around any `Pager`.
/// - Many readers (`read_batch`, `materialize_owned`) can proceed in parallel.
/// - Writers (`write_batch`, `alloc_ids`, `dealloc_ids`, `on_root_changed`) are serialized.
/// - Publishes the latest root for callers that keep a concrete `SharedPager<P>`.
#[derive(Debug)]
pub struct SharedPager<P: Pager> {
    inner: Arc<RwLock<P>>,
}

impl<P: Pager> SharedPager<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }
}

// IMPORTANT: no `P: Clone` bound here.
// We clone the Arc, not the inner pager.
impl<P: Pager> Clone for SharedPager<P> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<P: Pager> Pager for SharedPager<P> {
    type Id = P::Id;
    type Page = P::Page;

    fn read_batch(&self, ids: &[Self::Id]) -> Result<FxHashMap<Self::Id, Self::Page>, Error> {
        self.inner.read().unwrap().read_batch(ids)
    }
    fn write_batch(&self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error> {
        self.inner.write().unwrap().write_batch(pages)
    }
    fn alloc_ids(&self, n: usize) -> Result<Vec<Self::Id>, Error> {
        self.inner.write().unwrap().alloc_ids(n)
    }
    fn dealloc_ids(&self, ids: &[Self::Id]) -> Result<(), Error> {
        self.inner.write().unwrap().dealloc_ids(ids)
    }
    fn page_size_hint(&self) -> Option<usize> {
        self.inner.read().unwrap().page_size_hint()
    }
    fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error> {
        self.inner.read().unwrap().materialize_owned(bytes)
    }
}
