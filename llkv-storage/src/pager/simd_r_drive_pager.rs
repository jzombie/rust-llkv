//! Pager backed by `simd_r_drive::DataStore`.
//!
//! - Zero-copy reads: returns `EntryHandle` blobs.
//! - Persistent backing file via SIMD R Drive.
//! - Key allocator is initialized by scanning existing entries once on open.

use super::{BatchGet, BatchPut, GetResult, Pager};
use crate::constants::CATALOG_ROOT_PKEY;
use crate::types::PhysicalKey;

use llkv_result::{Error, Result};
use simd_r_drive::{
    DataStore,
    traits::{DataStoreReader, DataStoreWriter},
};
use simd_r_drive_entry_handle::EntryHandle;

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct SimdRDrivePager {
    ds: DataStore,
    next_key: AtomicU64,
}

impl fmt::Debug for SimdRDrivePager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Avoid requiring Debug on DataStore; just show a lightweight view.
        let next = self.next_key.load(Ordering::Relaxed);
        f.debug_struct("SimdRDrivePager")
            .field("next_key", &next)
            .finish()
    }
}

impl SimdRDrivePager {
    /// Open (or create) a SIMD R Drive datastore at `path` and seed
    /// the allocator by scanning existing entries.
    pub fn open(path: &Path) -> Result<Self> {
        let ds = DataStore::open(path)
            .map_err(|e| Error::Io(std::io::Error::other(format!("open DataStore failed: {e}"))))?;

        // Seed the allocator: find the max existing key (key_hash) and start from max+1.
        // If file is empty, start after the reserved catalog root key.
        let mut max_key = CATALOG_ROOT_PKEY;
        for eh in ds.iter_entries() {
            let k = eh.key_hash();
            if k > max_key {
                max_key = k;
            }
        }

        Ok(Self {
            ds,
            next_key: AtomicU64::new(max_key.saturating_add(1)),
        })
    }

    /// Expose the underlying DataStore if callers need advanced ops.
    pub fn datastore(&self) -> &DataStore {
        &self.ds
    }
}

impl Pager for SimdRDrivePager {
    type Blob = EntryHandle;

    fn alloc_many(&self, n: usize) -> Result<Vec<PhysicalKey>> {
        let n = u64::try_from(n)
            .map_err(|_| Error::Internal("alloc_many: n does not fit in u64".into()))?;

        let start = self
            .next_key
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                cur.checked_add(n)
            })
            .map_err(|_| Error::Internal("physical key space overflow".into()))?;

        Ok((start..start + n).collect())
    }

    fn batch_put(&self, puts: &[BatchPut]) -> Result<()> {
        // Use SIMD R Drive batch write-with-prehashed-keys to avoid re-hashing.
        let mut entries: Vec<(u64, &[u8])> = Vec::with_capacity(puts.len());
        for p in puts {
            match p {
                BatchPut::Raw { key, bytes } => {
                    if bytes.is_empty() {
                        return Err(Error::Internal("empty payload not allowed".into()));
                    }
                    entries.push((*key, bytes.as_slice()));
                }
            }
        }

        self.ds
            .batch_write_with_key_hashes(entries, false)
            .map_err(|e| {
                Error::Io(std::io::Error::other(format!(
                    "DataStore write failed: {e}"
                )))
            })?;

        Ok(())
    }

    fn batch_get(&self, gets: &[BatchGet]) -> Result<Vec<GetResult<Self::Blob>>> {
        let mut out = Vec::with_capacity(gets.len());
        for g in gets {
            match *g {
                BatchGet::Raw { key } => {
                    match self.ds.read_with_key_hash(key).map_err(|e| {
                        Error::Io(std::io::Error::other(format!("DataStore read failed: {e}")))
                    })? {
                        Some(h) => out.push(GetResult::Raw { key, bytes: h }),
                        None => out.push(GetResult::Missing { key }),
                    }
                }
            }
        }
        Ok(out)
    }

    fn free_many(&self, keys: &[PhysicalKey]) -> Result<()> {
        self.ds.batch_delete_key_hashes(keys).map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "DataStore delete failed: {e}"
            )))
        })?;
        Ok(())
    }

    fn enumerate_keys(&self) -> Result<Vec<PhysicalKey>> {
        let mut keys: Vec<PhysicalKey> = self
            .ds
            .iter_entries()
            .map(|entry| entry.key_hash())
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }
}
