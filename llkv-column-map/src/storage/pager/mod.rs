// File: src/storage/pager/mod.rs
use crate::types::PhysicalKey;
use std::io;
use std::sync::Arc;

pub mod mem_pager;
pub use mem_pager::*;

/// The blob handle type returned for raw reads.
pub type Blob = Arc<[u8]>;

/// Put operations inside a single batch.
#[derive(Clone, Debug)]
pub enum BatchPut {
    Raw { key: PhysicalKey, bytes: Vec<u8> },
}

/// Get operations inside a single batch.
#[derive(Clone, Debug)]
pub enum BatchGet {
    Raw { key: PhysicalKey },
}

/// Result for a single get operation.
#[derive(Debug)]
pub enum GetResult<B> {
    Raw { key: PhysicalKey, bytes: B },
    Missing { key: PhysicalKey },
}

/// Unified pager interface for abstracting over storage.
pub trait Pager: Send + Sync + 'static {
    /// The blob handle type returned for raw reads.
    type Blob: AsRef<[u8]> + Clone + Send + Sync + 'static;

    /// Allocates `n` new, unique physical keys.
    fn alloc_many(&self, n: usize) -> io::Result<Vec<PhysicalKey>>;

    /// Applies all puts in one batch.
    fn batch_put(&self, puts: &[BatchPut]) -> io::Result<()>;

    /// Serves all gets in one batch.
    fn batch_get(&self, gets: &[BatchGet]) -> io::Result<Vec<GetResult<Self::Blob>>>;

    /// Frees a list of physical keys, making them available for reuse.
    fn free_many(&self, keys: &[PhysicalKey]) -> io::Result<()>;

    /// Helper for getting a single raw value.
    fn get_raw(&self, key: PhysicalKey) -> io::Result<Option<Self::Blob>> {
        match self.batch_get(&[BatchGet::Raw { key }])?.pop() {
            Some(GetResult::Raw { bytes, .. }) => Ok(Some(bytes)),
            _ => Ok(None),
        }
    }
}
