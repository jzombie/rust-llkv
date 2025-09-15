use super::*;
use crate::error::{Error, Result};
use crate::types::{CATALOG_ROOT_PKEY, PhysicalKey};
use bytes::Bytes;
use rustc_hash::FxHashMap;
use std::sync::{
    RwLock,
    atomic::{AtomicU64, Ordering},
};

/// In-memory pager used for tests/benchmarks.
#[allow(clippy::module_name_repetitions)]
pub struct MemPager {
    next_key: AtomicU64,
    blobs: RwLock<FxHashMap<PhysicalKey, Bytes>>,
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
    type Blob = Bytes;

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

    fn get_raw(&self, key: PhysicalKey) -> Result<Option<Self::Blob>> {
        let map = self
            .blobs
            .read()
            .expect("MemPager blobs read lock poisoned");
        Ok(map.get(&key).cloned())
    }

    fn batch_put(&self, puts: &[BatchPut]) -> Result<()> {
        let mut map = self
            .blobs
            .write()
            .expect("MemPager blobs write lock poisoned");
        for p in puts {
            match p {
                BatchPut::Raw { key, bytes } => {
                    map.insert(*key, Bytes::from(bytes.clone()));
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
