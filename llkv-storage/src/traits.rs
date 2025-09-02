use rustc_hash::FxHashMap;
use simd_r_drive::EntryHandle;
use std::io;

pub trait KeyValueStore<StorageKeyType>: Send + Sync
where
    StorageKeyType: Send + Sync + 'static + Eq + std::hash::Hash + Clone + Copy,
{
    fn insert_batch(&mut self, items: FxHashMap<StorageKeyType, Vec<u8>>) -> io::Result<()>;

    fn get_batch(
        &self,
        keys: &[StorageKeyType],
    ) -> io::Result<FxHashMap<StorageKeyType, EntryHandle>>;

    fn remove_batch(&mut self, keys: &[StorageKeyType]) -> io::Result<()>;

    fn contains_key(&self, key: &StorageKeyType) -> io::Result<bool>;

    /// Test-only. Obtaining a vector of keys in a production implementation may is
    /// not memory efficient.
    #[cfg(feature = "debug")]
    fn keys(&self) -> io::Result<Vec<StorageKeyType>>;

    /// Test-only: return (key, len) for all entries in the store.
    #[cfg(feature = "debug")]
    fn inspect_all(&self) -> io::Result<Vec<(StorageKeyType, usize)>>;

    /// Test-only: return (key, len) for the provided subset.
    #[cfg(feature = "debug")]
    fn inspect_subset(&self, keys: &[StorageKeyType]) -> io::Result<Vec<(StorageKeyType, usize)>>;
}
