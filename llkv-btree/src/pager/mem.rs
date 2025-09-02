// TODO: Define using a macro... more flexibility

//! In-memory pager with u64 page IDs.
//! Intended for tests, examples, and ephemeral stores.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::errors::Error;
use crate::pager::Pager;

/// Simple in-memory pager using u64 page IDs.
/// Not thread-safe by itself; wrap in `SharedPager` to share across threads.
pub struct MemPager64 {
    pages: FxHashMap<u64, Arc<[u8]>>,
    next_id: u64,
    page_size: usize,
}

impl Default for MemPager64 {
    fn default() -> Self {
        Self {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 256,
        }
    }
}

impl MemPager64 {
    /// Create with a maximum page size (soft-checked on writes).
    pub fn new(page_size: usize) -> Self {
        Self {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size,
        }
    }

    /// Number of pages currently stored (for tests/tools).
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// True if no pages (for tests/tools).
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

impl Pager for MemPager64 {
    type Id = u64;
    type Page = Arc<[u8]>;

    fn read_batch(&self, ids: &[Self::Id]) -> Result<FxHashMap<Self::Id, Self::Page>, Error> {
        Ok(ids
            .iter()
            .filter_map(|id| self.pages.get(id).map(|p| (*id, p.clone())))
            .collect())
    }

    fn write_batch(&mut self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error> {
        for (id, data) in pages {
            if data.len() > self.page_size {
                return Err(Error::Corrupt("page overflow"));
            }
            self.pages.insert(*id, Arc::from(*data));
        }
        Ok(())
    }

    fn alloc_ids(&mut self, count: usize) -> Result<Vec<Self::Id>, Error> {
        let start = self.next_id;
        // checked add to avoid wrap
        self.next_id = self
            .next_id
            .checked_add(count as u64)
            .ok_or(Error::Corrupt("id overflow"))?;
        Ok((start..self.next_id).collect())
    }

    fn dealloc_ids(&mut self, ids: &[Self::Id]) -> Result<(), Error> {
        for id in ids {
            self.pages.remove(id);
        }
        Ok(())
    }

    fn page_size_hint(&self) -> Option<usize> {
        Some(self.page_size)
    }

    fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error> {
        Ok(Arc::from(bytes))
    }
}
