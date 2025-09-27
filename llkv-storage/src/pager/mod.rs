//! Minimal pager trait + in-memory implementation returning `Bytes` blobs.
//!
//! Returning `bytes::Bytes` lets readers build Arrow `Buffer`s that borrow
//! the pager memory with zero copying.

use crate::types::PhysicalKey;
use llkv_result::Result;

pub mod instrumented_pager;
pub use instrumented_pager::*;

pub mod mem_pager;
pub use mem_pager::*;

#[cfg(feature = "simd-r-drive-support")]
pub mod simd_r_drive_pager;
#[cfg(feature = "simd-r-drive-support")]
pub use simd_r_drive_pager::*;

#[derive(Clone, Debug)]
pub enum BatchPut {
    Raw { key: PhysicalKey, bytes: Vec<u8> },
}

#[derive(Clone, Debug)]
pub enum BatchGet {
    Raw { key: PhysicalKey },
}

#[derive(Clone, Debug)]
pub enum GetResult<B> {
    Raw { key: PhysicalKey, bytes: B },
    Missing { key: PhysicalKey },
}

pub trait Pager: Send + Sync + 'static {
    type Blob: AsRef<[u8]> + Clone + Send + Sync + 'static;

    /// Allocate `n` new physical keys.
    fn alloc_many(&self, n: usize) -> Result<Vec<PhysicalKey>>;

    /// Batch get blobs; returns one `GetResult` per request in order.
    fn batch_get(&self, gets: &[BatchGet]) -> Result<Vec<GetResult<Self::Blob>>>;

    /// Batch put blobs at fixed keys.
    fn batch_put(&self, puts: &[BatchPut]) -> Result<()>;

    /// Batch free physical keys (best-effort). Implementations may ignore
    /// unknown keys. This enables real deletion from the store.
    fn free_many(&self, keys: &[PhysicalKey]) -> Result<()>;
}
