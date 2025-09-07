use super::*;
use crate::codecs::{decode_typed, encode_typed};
use crate::constants::BOOTSTRAP_PKEY;
use rustc_hash::FxHashMap;
use std::io::{self, Error, ErrorKind};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

/// In-memory pager used for tests/benchmarks.
/// Thread-safe:
/// - key allocation uses an AtomicU64
/// - blob storage is protected by an RwLock
#[allow(clippy::module_name_repetitions)]
pub struct MemPager {
    /// Next physical key to hand out. We reserve 0 for bootstrap.
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
            next_key: AtomicU64::new(BOOTSTRAP_PKEY + 1), // reserve initial for bootstrap
            blobs: RwLock::new(FxHashMap::default()),
        }
    }
}

impl Pager for MemPager {
    /// For the in-memory backend, a blob is just an Arc to an immutable byte slice.
    type Blob = Arc<[u8]>;

    fn alloc_many(&self, n: usize) -> io::Result<Vec<PhysicalKey>> {
        // Simple monotonic allocator (thread-safe).
        // We use fetch_update to check overflow and advance atomically.
        let n_u64 = n as u64;
        let start = self
            .next_key
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                cur.checked_add(n_u64)
            })
            .map_err(|_| {
                Error::new(
                    ErrorKind::Other,
                    "physical key space overflow in MemPager::alloc_many",
                )
            })?;
        let end = start + n_u64;
        Ok((start..end).collect())
    }

    fn batch_put(&self, puts: &[BatchPut]) -> io::Result<()> {
        // Single write lock for the whole batch so it’s atomic w.r.t. readers.
        let mut map = self
            .blobs
            .write()
            .expect("MemPager blobs write lock poisoned");
        for p in puts {
            match p {
                BatchPut::Raw { key, bytes } => {
                    // Store as Arc<[u8]> so readers can share without copying.
                    // We must clone here because `puts` only gives us &Vec<u8>.
                    map.insert(*key, Arc::from(bytes.clone()));
                }
                BatchPut::Typed { key, value } => {
                    let enc = encode_typed(value); // Vec<u8>
                    map.insert(*key, Arc::from(enc.into_boxed_slice()));
                }
            }
        }
        Ok(())
    }

    fn batch_get(&self, gets: &[BatchGet]) -> io::Result<Vec<GetResult<Self::Blob>>> {
        // Single read lock for the entire batch; multiple readers can proceed concurrently.
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
                BatchGet::Typed { key, kind } => {
                    if let Some(b) = map.get(&key) {
                        // Arc<[u8]> derefs to [u8], so &b[..] is a &[u8]
                        let tv = decode_typed(kind, &b[..])?;
                        out.push(GetResult::Typed { key, value: tv });
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
            // The correct method to remove a key-value pair from a HashMap is `remove`.
            map.remove(key);
        }

        Ok(())
    }
}
