use crate::codec::hash::hash64;
use crate::traits::KeyValueStore;
use memmap2::{Mmap, MmapMut};
use rustc_hash::FxHashMap;
use simd_r_drive::storage_engine::{EntryHandle, EntryMetadata};
use std::hash::Hash;
use std::sync::Arc;

/// In-memory arena store that returns zero-copy EntryHandles.
#[derive(Default)]
pub struct InMemoryStore<StorageKeyType>
where
    StorageKeyType: Eq + Hash + Copy,
{
    // Finalized handles by key. Each handle holds an Arc<Mmap>.
    inner: FxHashMap<StorageKeyType, EntryHandle>,
}

impl<StorageKeyType> InMemoryStore<StorageKeyType>
where
    StorageKeyType: Eq + Hash + Copy,
{
    pub fn new() -> Self {
        Self {
            inner: FxHashMap::default(),
        }
    }
}

impl<StorageKeyType> KeyValueStore<StorageKeyType> for InMemoryStore<StorageKeyType>
where
    StorageKeyType: Send + Sync + 'static + Eq + Hash + Clone + Copy,
{
    fn insert_batch(&mut self, items: FxHashMap<StorageKeyType, Vec<u8>>) {
        if items.is_empty() {
            return;
        }

        // 1) Compute total size for a single contiguous mmap
        let total: usize = items.values().map(|v| v.len()).sum();

        // 2) Allocate anon mmap and copy payloads contiguously
        let mut mm = MmapMut::map_anon(total).expect("anon mmap allocation failed");
        let mut offset = 0usize;

        // We need to remember (key, range, checksum). We compute checksum while copying.
        struct Pending<StorageKeyType> {
            storage_key: StorageKeyType,
            range: std::ops::Range<usize>,
            checksum: [u8; 4],
        }
        let mut staged: Vec<Pending<StorageKeyType>> = Vec::with_capacity(items.len());

        // NOTE: Items iteration order is arbitrary; if you care about stable order,
        // collect & sort keys first. For arena packing it usually doesn't matter.
        for (storage_key, value) in items {
            let len = value.len();
            if len == 0 {
                // Empty value still gets a handle (empty range); checksum of empty is defined
                let checksum = (hash64(&[], 0) as u32).to_le_bytes();
                staged.push(Pending {
                    storage_key,
                    range: offset..offset,
                    checksum,
                });
                continue;
            }

            // Copy bytes into the mmap
            mm[offset..offset + len].copy_from_slice(&value);

            // Compute checksum of this entry using xxhash
            let checksum = (hash64(&value, 0) as u32).to_le_bytes();

            staged.push(Pending {
                storage_key,
                range: offset..offset + len,
                checksum,
            });
            offset += len;
        }

        // 3) Freeze to read-only and wrap in Arc<Mmap>
        let ro: Mmap = mm.make_read_only().expect("freeze read-only failed");
        let arc = Arc::new(ro);

        // 4) Build EntryHandles for each staged item and insert
        for Pending {
            storage_key,
            range,
            checksum,
        } in staged
        {
            // Fill metadata. We don’t have a meaningful key_hash here without a policy,
            // so we can set it to 0 or derive from StorageKeyType if it's u64.
            let metadata = EntryMetadata::new(
                0, // If StorageKeyType == u64, you could set this to storage_key as needed.
                0, // unused in this in-memory variant
                checksum,
            );
            let handle = EntryHandle::from_arc_mmap(arc.clone(), range, metadata);
            self.inner.insert(storage_key, handle);
        }
    }

    fn get_batch(&self, keys: &[StorageKeyType]) -> FxHashMap<StorageKeyType, EntryHandle> {
        let mut out = FxHashMap::with_capacity_and_hasher(keys.len(), Default::default());
        for &storage_key in keys {
            if let Some(handle) = self.inner.get(&storage_key) {
                // Cheap: just bumps Arc<Mmap> refcount
                out.insert(storage_key, handle.clone());
            }
        }
        out
    }

    fn remove_batch(&mut self, keys: &[StorageKeyType]) {
        for storage_key in keys {
            self.inner.remove(storage_key);
        }
    }

    fn contains_key(&self, key: &StorageKeyType) -> bool {
        self.inner.contains_key(key)
    }

    #[cfg(feature = "test-utils")]
    fn keys(&self) -> Vec<StorageKeyType> {
        self.inner.keys().copied().collect()
    }

    /// Test-only: enumerate all (key, len) pairs.
    #[cfg(feature = "test-utils")]
    fn inspect_all(&self) -> Vec<(StorageKeyType, usize)> {
        let mut out = Vec::with_capacity(self.inner.len());
        for (storage_key, handle) in self.inner.iter() {
            out.push((*storage_key, handle.size()));
        }
        out
    }

    /// Test-only: enumerate (key, len) for a subset.
    #[cfg(feature = "test-utils")]
    fn inspect_subset(&self, keys: &[StorageKeyType]) -> Vec<(StorageKeyType, usize)> {
        let mut out = Vec::with_capacity(keys.len());
        for storage_key in keys {
            if let Some(handle) = self.inner.get(storage_key) {
                out.push((*storage_key, handle.size()));
            }
        }
        out
    }
}
