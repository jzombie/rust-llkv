use rustc_hash::FxHashMap;
use simd_r_drive::EntryHandle;

// TODO: Return slices for zero-copy

pub trait KeyValueStore<StorageKeyType>: Send + Sync
where
    StorageKeyType: Send + Sync + 'static + Eq + std::hash::Hash + Clone + Copy,
{
    // TODO: Use Result type
    fn insert_batch(&mut self, items: FxHashMap<StorageKeyType, Vec<u8>>);

    // TODO: Use Result type
    fn get_batch(&self, keys: &[StorageKeyType]) -> FxHashMap<StorageKeyType, EntryHandle>;

    // TODO: Use Result type
    fn remove_batch(&mut self, keys: &[StorageKeyType]);

    // TODO: Use Result type
    fn contains_key(&self, key: &StorageKeyType) -> bool;

    // TODO: Use Result type
    /// Test-only. Obtaining a vector of keys in a production implementation may is
    /// not memory efficient.
    #[cfg(feature = "test-utils")]
    fn keys(&self) -> Vec<StorageKeyType>;

    // TODO: Use Result type
    /// Test-only: return (key, len) for all entries in the store.
    #[cfg(feature = "test-utils")]
    fn inspect_all(&self) -> Vec<(StorageKeyType, usize)>;

    // TODO: Use Result type
    /// Test-only: return (key, len) for the provided subset.
    #[cfg(feature = "test-utils")]
    fn inspect_subset(&self, keys: &[StorageKeyType]) -> Vec<(StorageKeyType, usize)>;
}
