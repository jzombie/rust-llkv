use crate::traits::KeyValueStore;
use crate::types::StorageKey;
use rustc_hash::FxHashMap;
use simd_r_drive::{
    DataStore,
    storage_engine::EntryHandle,
    traits::{DataStoreReader, DataStoreWriter},
};
use std::path::PathBuf;

/// Adapter over `simd_r_drive::DataStore` that implements `KeyValueStore<StorageKey>`.
pub struct SimdRDriveStore {
    inner: DataStore,
}

impl SimdRDriveStore {
    /// Open (or create) a datastore at `path`.
    pub fn new(path: &PathBuf) -> Self {
        Self {
            inner: DataStore::open(&path).expect("Failed to open SimdRDrive DataStore"),
        }
    }

    /// Optional helper if you want zero-copy iteration of all entries.
    /// (Not part of the trait.)
    pub fn iter_handles(&self) -> impl Iterator<Item = (StorageKey, EntryHandle)> + '_ {
        self.inner
            .iter_entries()
            .map(|h: EntryHandle| (h.key_hash(), h))
    }
}

impl KeyValueStore<StorageKey> for SimdRDriveStore {
    fn insert_batch(&mut self, items: FxHashMap<StorageKey, Vec<u8>>) {
        let entries: Vec<(StorageKey, &[u8])> = items
            .iter()
            .map(|(k, v)| (k.clone(), v.as_slice()))
            .collect();

        self.inner
            .batch_write_with_key_hashes(entries, false)
            .expect("SimdRDriveStore: batch_write failed");
    }

    // *** ZERO-COPY: return EntryHandle, not Vec<u8> ***
    fn get_batch(&self, keys: &[StorageKey]) -> FxHashMap<StorageKey, EntryHandle> {
        let mut out: FxHashMap<StorageKey, EntryHandle> =
            FxHashMap::with_capacity_and_hasher(keys.len(), Default::default());

        let handles = self
            .inner
            .batch_read_hashed_keys(&keys, None)
            .expect("SimdRDrive: get_batch failed");

        for (i, maybe_h) in handles.into_iter().enumerate() {
            if let Some(h) = maybe_h {
                // Move the EntryHandle directly into the map (no copy).
                out.insert(keys[i], h);
            }
        }

        out
    }

    fn remove_batch(&mut self, keys: &[StorageKey]) {
        self.inner
            .batch_delete_key_hashes(keys)
            .expect("SimdRDriveStore: delete failed");
    }

    fn contains_key(&self, key: &StorageKey) -> bool {
        self.inner
            .exists_with_key_hash(key.to_owned())
            .expect("SimdRDriveStore: exists failed")
    }

    // ---------- TEST-UTILS (feature-gated) ----------
    #[cfg(feature = "test-utils")]
    fn keys(&self) -> Vec<StorageKey> {
        self.inner.iter_entries().map(|e| e.key_hash()).collect()
    }

    #[cfg(feature = "test-utils")]
    fn inspect_all(&self) -> Vec<(StorageKey, usize)> {
        self.inner
            .iter_entries()
            .map(|e| (e.key_hash(), e.as_slice().len()))
            .collect()
    }

    #[cfg(feature = "test-utils")]
    fn inspect_subset(&self, keys: &[StorageKey]) -> Vec<(StorageKey, usize)> {
        let handles = self
            .inner
            .batch_read_hashed_keys(&keys, None)
            .expect("SimdRDrive: get_batch failed");

        handles
            .into_iter()
            .enumerate()
            .filter_map(|(i, h)| h.map(|eh| (keys[i], eh.as_slice().len())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    //! What these tests prove (read first):
    //!
    //! - Insert: after insert_batch, each key:
    //!     - exists via contains_key
    //!     - is returned by get_batch with correct bytes
    //! - Read: get_batch returns only existing keys.
    //! - Update: reinsert replaces prior value.
    //! - Delete: remove_batch makes keys disappear from both
    //!   contains_key and get_batch.
    //! - Persist and reopen: data and deletions survive reopen.
    //! - iter_handles(): exactly the stored keys, correct payload sizes.
    //! - Inspect helpers (feature "test-utils"):
    //!     - keys() returns live keys
    //!     - inspect_all() returns all keys with sizes
    //!     - inspect_subset() returns requested live keys with sizes

    use super::{KeyValueStore, SimdRDriveStore};
    use rustc_hash::FxHashMap;
    use std::collections::HashSet;
    use tempfile::TempDir;

    type K = super::StorageKey;

    fn map_from(pairs: &[(K, &[u8])]) -> FxHashMap<K, Vec<u8>> {
        let mut m = FxHashMap::default();
        for (k, v) in pairs {
            m.insert(*k, v.to_vec());
        }
        m
    }

    fn bytes_map_from_handles(
        m: &FxHashMap<K, simd_r_drive::storage_engine::EntryHandle>,
    ) -> FxHashMap<K, Vec<u8>> {
        let mut out = FxHashMap::default();
        for (k, h) in m {
            out.insert(*k, h.as_slice().to_vec());
        }
        out
    }

    #[test]
    fn crud_roundtrip_and_contains() {
        // Use a FILE path inside the temp directory.
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("store.srd");
        let mut store = SimdRDriveStore::new(&path);

        // Create three keys with differing sizes.
        let to_write = map_from(&[
            (1u64, b"alpha".as_slice()),
            (2u64, b"bravo-123".as_slice()),
            (3u64, &[0u8; 1024][..]),
        ]);
        store.insert_batch(to_write.clone());

        // After insert: all keys exist.
        for k in [1u64, 2, 3] {
            assert!(store.contains_key(&k), "missing key after insert");
        }

        // Read back via zero-copy EntryHandle.
        let got = store.get_batch(&[1u64, 2, 3]);
        assert_eq!(got.len(), 3, "get_batch should return 3 keys");
        let got_bytes = bytes_map_from_handles(&got);
        assert_eq!(got_bytes.get(&1).unwrap().as_slice(), b"alpha");
        assert_eq!(got_bytes.get(&2).unwrap().as_slice(), b"bravo-123");
        assert_eq!(got_bytes.get(&3).unwrap().len(), 1024);

        // Overwrite key 2 and verify replacement.
        store.insert_batch(map_from(&[(2u64, b"new".as_slice())]));
        let got2 = store.get_batch(&[2u64]);
        let got2_bytes = bytes_map_from_handles(&got2);
        assert_eq!(got2_bytes.get(&2).unwrap().as_slice(), b"new");

        // Delete keys 1 and 3, keep 2.
        store.remove_batch(&[1u64, 3u64]);
        assert!(!store.contains_key(&1));
        assert!(store.contains_key(&2));
        assert!(!store.contains_key(&3));

        // get_batch should now return only key 2.
        let after_del = store.get_batch(&[1u64, 2u64, 3u64]);
        assert_eq!(after_del.len(), 1, "only key 2 should remain");
        assert!(after_del.contains_key(&2));
        assert!(!after_del.contains_key(&1));
        assert!(!after_del.contains_key(&3));
    }

    #[test]
    fn get_batch_skips_missing_and_iter_handles_matches_set() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("store.srd");
        let mut store = SimdRDriveStore::new(&path);

        store.insert_batch(map_from(&[
            (10u64, b"x".as_slice()),
            (11u64, b"yy".as_slice()),
            (12u64, b"zzz".as_slice()),
        ]));

        // Ask for present and missing keys. Only present should appear.
        let got = store.get_batch(&[9u64, 10, 11, 12, 13]);
        let present: HashSet<K> = got.keys().cloned().collect();
        assert_eq!(
            present,
            HashSet::from([10u64, 11u64, 12u64]),
            "get_batch must only include existing keys"
        );

        // iter_handles must expose exactly the same key set.
        let iter_keys: HashSet<K> = store.iter_handles().map(|(k, _)| k).collect();
        assert_eq!(
            iter_keys, present,
            "iter_handles should cover exactly the live keys"
        );

        // And sizes must match the bytes we wrote.
        let size_map: FxHashMap<K, usize> = store
            .iter_handles()
            .map(|(k, h)| (k, h.as_slice().len()))
            .collect();
        assert_eq!(size_map.get(&10), Some(&1usize));
        assert_eq!(size_map.get(&11), Some(&2usize));
        assert_eq!(size_map.get(&12), Some(&3usize));
    }

    #[test]
    fn persistence_across_reopen_and_delete_persists() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("store.srd");

        {
            let mut store = SimdRDriveStore::new(&path);
            store.insert_batch(map_from(&[
                (1001u64, b"a".as_slice()),
                (1002u64, b"bb".as_slice()),
            ]));
            assert!(store.contains_key(&1001));
            assert!(store.contains_key(&1002));
        } // drop to flush/close

        {
            let store = SimdRDriveStore::new(&path);
            assert!(store.contains_key(&1001));
            assert!(store.contains_key(&1002));
            let got = store.get_batch(&[1001u64, 1002u64]);
            let got_bytes = bytes_map_from_handles(&got);
            assert_eq!(got_bytes.get(&1001).unwrap().as_slice(), b"a");
            assert_eq!(got_bytes.get(&1002).unwrap().as_slice(), b"bb");
        }

        {
            let mut store = SimdRDriveStore::new(&path);
            store.remove_batch(&[1001u64]);
            assert!(!store.contains_key(&1001));
            assert!(store.contains_key(&1002));
        }

        {
            let store = SimdRDriveStore::new(&path);
            assert!(!store.contains_key(&1001));
            assert!(store.contains_key(&1002));
        }
    }

    // --------- Feature-gated tests for inspect helpers -----------------

    #[cfg(feature = "test-utils")]
    #[test]
    fn inspect_all_and_keys_cover_and_size_correctly() {
        use super::super::KeyValueStore; // ensure trait is in scope

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("store.srd");
        let mut store = SimdRDriveStore::new(&path);

        // NOTE: DataStore rejects empty payloads, so use non-empty values.
        store.insert_batch(map_from(&[
            (2001u64, b"mm".as_slice()),   // size 2
            (2002u64, b"nnn".as_slice()),  // size 3
            (2003u64, b"oooo".as_slice()), // size 4
        ]));

        // keys() should list exactly these three keys (order-free).
        let mut ks = store.keys();
        ks.sort_unstable();
        assert_eq!(ks, vec![2001u64, 2002u64, 2003u64]);

        // inspect_all() should report sizes that match what we wrote.
        let mut all = store.inspect_all();
        all.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(all[0], (2001u64, 2usize));
        assert_eq!(all[1], (2002u64, 3usize));
        assert_eq!(all[2], (2003u64, 4usize));
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn inspect_subset_returns_only_requested_live_keys_with_sizes() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("store.srd");
        let mut store = SimdRDriveStore::new(&path);

        store.insert_batch(map_from(&[
            (3001u64, b"a".as_slice()),
            (3002u64, b"bbbb".as_slice()),
        ]));

        // Request a mix of present and absent keys.
        let mut subset = store.inspect_subset(&[3000u64, 3001u64, 3002u64]);
        subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(subset, vec![(3001u64, 1usize), (3002u64, 4usize)]);

        // After deleting 3001, it should no longer appear.
        store.remove_batch(&[3001u64]);
        let mut subset2 = store.inspect_subset(&[3001u64, 3002u64]);
        subset2.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(subset2, vec![(3002u64, 4usize)]);
    }
}
