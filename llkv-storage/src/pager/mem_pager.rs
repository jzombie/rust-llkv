use super::*;
use crate::constants::CATALOG_ROOT_PKEY;
use crate::types::PhysicalKey;
use llkv_result::{Error, Result};
use rustc_hash::FxHashMap;
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{
    RwLock,
    atomic::{AtomicU64, Ordering},
};

/// In-memory pager used for tests/benchmarks.
#[allow(clippy::module_name_repetitions)]
pub struct MemPager {
    next_key: AtomicU64,
    blobs: RwLock<FxHashMap<PhysicalKey, EntryHandle>>,
}

impl Default for MemPager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemPager {
    pub fn new() -> Self {
        Self {
            next_key: AtomicU64::new(CATALOG_ROOT_PKEY + 1),
            blobs: RwLock::new(FxHashMap::default()),
        }
    }
}

impl Pager for MemPager {
    type Blob = EntryHandle;

    fn alloc_many(&self, n: usize) -> Result<Vec<PhysicalKey>> {
        let n_u64 = n as u64;
        let start = self
            .next_key
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                cur.checked_add(n_u64)
            })
            .map_err(|_| Error::Internal("physical key space overflow".to_string()))?;
        Ok((start..start + n_u64).collect())
    }

    fn batch_put(&self, puts: &[BatchPut]) -> Result<()> {
        let mut map = self
            .blobs
            .write()
            .expect("MemPager blobs write lock poisoned");
        for p in puts {
            match p {
                BatchPut::Raw { key, bytes } => {
                    // One O(len) copy on write into an anonymous mmap; reads are
                    // zero-copy via EntryHandle::as_slice().
                    let eh = EntryHandle::from_owned_bytes_anon(bytes, *key)
                        .map_err(|e| Error::Internal(format!("anon mmap failed: {e:?}")))?;
                    map.insert(*key, eh);
                }
            }
        }
        Ok(())
    }

    fn batch_get(&self, gets: &[BatchGet]) -> Result<Vec<GetResult<Self::Blob>>> {
        let map = self
            .blobs
            .read()
            .expect("MemPager blobs read lock poisoned");
        let mut out = Vec::with_capacity(gets.len());
        for g in gets {
            match *g {
                BatchGet::Raw { key } => {
                    if let Some(b) = map.get(&key) {
                        out.push(GetResult::Raw {
                            key,
                            bytes: b.clone(),
                        });
                    } else {
                        out.push(GetResult::Missing { key });
                    }
                }
            }
        }
        Ok(out)
    }

    fn free_many(&self, keys: &[PhysicalKey]) -> Result<()> {
        let mut map = self
            .blobs
            .write()
            .expect("MemPager blobs write lock poisoned");
        for &k in keys {
            map.remove(&k);
        }
        Ok(())
    }
}
