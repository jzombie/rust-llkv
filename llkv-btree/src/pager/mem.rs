//! In-memory pager with user-defined page IDs.
//! Intended for tests, examples, and ephemeral stores.
#[macro_export]
macro_rules! define_mem_pager {
    (
        $(#[$meta:meta])*
        name: $name:ident,
        id: $id:ty,
        default_page_size: $default_ps:expr
    ) => {
        $(#[$meta])*
        /// Simple in-memory pager backed by a mutexed FxHashMap.
        /// Not thread-safe by itself; wrap in `SharedPager` to share across threads.
        pub struct $name {
            page_size: usize,
            // (pages, next_id)
            state: ::std::sync::Mutex<(::rustc_hash::FxHashMap<$id, ::std::sync::Arc<[u8]>>, $id)>,
        }

        impl ::core::default::Default for $name {
            fn default() -> Self {
                Self::new($default_ps)
            }
        }

        impl $name {
            /// Create with a maximum page size (soft-checked on writes).
            pub fn new(page_size: usize) -> Self {
                Self {
                    page_size,
                    state: ::std::sync::Mutex::new((::rustc_hash::FxHashMap::default(), 1 as $id)),
                }
            }

            /// Number of pages currently stored (for tests/tools).
            pub fn len(&self) -> usize {
                let state = self.state.lock().unwrap();
                state.0.len()
            }

            /// True if no pages (for tests/tools).
            pub fn is_empty(&self) -> bool {
                let state = self.state.lock().unwrap();
                state.0.is_empty()
            }
        }

        impl $crate::pager::Pager for $name {
            type Id = $id;
            type Page = ::std::sync::Arc<[u8]>;

            fn read_batch(&self, ids: &[Self::Id])
                -> Result<::rustc_hash::FxHashMap<Self::Id, Self::Page>, $crate::errors::Error>
            {
                let state = self.state.lock().unwrap();
                Ok(ids.iter()
                    .filter_map(|id| state.0.get(id).map(|p| (*id, p.clone())))
                    .collect())
            }

            fn write_batch(&self, pages: &[(Self::Id, &[u8])]) -> Result<(), $crate::errors::Error> {
                let mut state = self.state.lock().unwrap();

                for (id, data) in pages {
                    if data.len() > self.page_size {
                        return Err($crate::errors::Error::Corrupt("page overflow"));
                    }
                    state.0.insert(*id, ::std::sync::Arc::from(*data));
                }
                Ok(())
            }

            fn alloc_ids(&self, count: usize) -> Result<Vec<Self::Id>, $crate::errors::Error> {
                let mut state = self.state.lock().unwrap();

                let start = state.1;
                state.1 = state.1
                    .checked_add(count as $id)
                    .ok_or($crate::errors::Error::Corrupt("id overflow"))?;
                Ok((start..state.1).collect())
            }

            fn dealloc_ids(&self, ids: &[Self::Id]) -> Result<(), $crate::errors::Error> {
                let mut state = self.state.lock().unwrap();
                for id in ids {
                    state.0.remove(id);
                }
                Ok(())
            }

            fn page_size_hint(&self) -> Option<usize> {
                Some(self.page_size)
            }

            fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, $crate::errors::Error> {
                Ok(::std::sync::Arc::from(bytes))
            }
        }
    }
}
