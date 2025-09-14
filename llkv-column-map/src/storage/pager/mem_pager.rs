// File: src/storage/pager/mem_pager.rs
use super::*;
use crate::types::{CATALOG_ROOT_PKEY, PhysicalKey};
use rustc_hash::FxHashMap;
use std::io::{self, Error};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

/// In-memory pager used for tests/benchmarks.
#[allow(clippy::module_name_repetitions)]
pub struct MemPager {
    /// Next physical key to hand out.
    next_key: AtomicU64,
    // PhysicalKey -> Arc<[u8]>
    blobs: RwLock<FxHashMap<PhysicalKey, Arc<[u8]>>>,
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
    type Blob = Arc<[u8]>;

    fn alloc_many(&self, n: usize) -> io::Result<Vec<PhysicalKey>> {
        let n_u64 = n as u64;
        let start = self
            .next_key
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                cur.checked_add(n_u64)
            })
            .map_err(|_| Error::other("physical key space overflow in MemPager::alloc_many"))?;
        let end = start + n_u64;
        Ok((start..end).collect())
    }

    fn batch_put(&self, puts: &[BatchPut]) -> io::Result<()> {
        let mut map = self
            .blobs
            .write()
            .expect("MemPager blobs write lock poisoned");
        for p in puts {
            match p {
                BatchPut::Raw { key, bytes } => {
                    map.insert(*key, Arc::from(bytes.clone()));
                }
            }
        }
        Ok(())
    }

    fn batch_get(&self, gets: &[BatchGet]) -> io::Result<Vec<GetResult<Self::Blob>>> {
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
                            bytes: Arc::clone(b),
                        });
                    } else {
                        out.push(GetResult::Missing { key });
                    }
                }
            }
        }
        Ok(out)
    }

    fn free_many(&self, keys: &[PhysicalKey]) -> io::Result<()> {
        let mut map = self
            .blobs
            .write()
            .expect("MemPager blobs write lock poisoned");
        for key in keys {
            map.remove(key);
        }
        Ok(())
    }
}
