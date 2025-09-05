use std::io;
use std::sync::Arc;

use crate::types::PhysicalKey;

pub mod mem_pager;
pub use mem_pager::*;

/// Typed kinds the pager knows how to decode/encode. Keep this set
/// scoped to storage types you persist via `bitcode`.
#[derive(Clone, Copy, Debug)]
pub enum TypedKind {
    Bootstrap,
    Manifest,
    ColumnIndex,
    IndexSegment,
}

/// Type-erased typed value. Concrete structs live in your crate's
/// `index` module. This avoids `Any`/downcasts at callers.
#[derive(Clone, Debug)]
pub enum TypedValue {
    Bootstrap(crate::index::Bootstrap),
    Manifest(crate::index::Manifest),
    ColumnIndex(crate::index::ColumnIndex),
    IndexSegment(crate::index::IndexSegment),
}

impl TypedValue {
    /// Helper to get the kind tag for a value.
    pub fn kind(&self) -> TypedKind {
        match self {
            TypedValue::Bootstrap(_) => TypedKind::Bootstrap,
            TypedValue::Manifest(_) => TypedKind::Manifest,
            TypedValue::ColumnIndex(_) => TypedKind::ColumnIndex,
            TypedValue::IndexSegment(_) => TypedKind::IndexSegment,
        }
    }
}

/// Put operations inside a single batch.
#[derive(Clone, Debug)]
pub enum BatchPut {
    Raw { key: PhysicalKey, bytes: Vec<u8> },
    Typed { key: PhysicalKey, value: TypedValue },
}

/// Get operations inside a single batch.
#[derive(Clone, Debug)]
pub enum BatchGet {
    Raw { key: PhysicalKey },
    Typed { key: PhysicalKey, kind: TypedKind },
}

/// Result for a single get. Raw returns an owned buffer (Arc slice) so callers
/// can use `&self` and share across threads without tying lifetimes to the pager.
#[derive(Debug)]
pub enum GetResult {
    Raw { key: PhysicalKey, bytes: Arc<[u8]> },
    Typed { key: PhysicalKey, value: TypedValue },
    Missing { key: PhysicalKey },
}

/// Unified pager interface with separate put/get passes.
/// - `batch_put` applies all writes atomically w.r.t. this pager.
/// - `batch_get` serves a mixed list of typed/raw reads in one round-trip.
/// Returning owned buffers allows `&self` for reads (good for RwLock read guards).
pub trait Pager {
    /// Key allocation can remain separate; typically needed ahead of
    /// a write batch.
    fn alloc_many(&self, n: usize) -> io::Result<Vec<PhysicalKey>>;

    /// Apply all puts in one batch.
    fn batch_put(&self, puts: &[BatchPut]) -> io::Result<()>;

    /// Serve all gets (raw + typed) in one batch.
    fn batch_get(&self, gets: &[BatchGet]) -> io::Result<Vec<GetResult>>;
}

// =================== Encoding helpers (typed) ======================

pub fn encode_typed(v: &TypedValue) -> Vec<u8> {
    match v {
        TypedValue::Bootstrap(x) => bitcode::encode(x),
        TypedValue::Manifest(x) => bitcode::encode(x),
        TypedValue::ColumnIndex(x) => bitcode::encode(x),
        TypedValue::IndexSegment(x) => bitcode::encode(x),
    }
}

pub fn decode_typed(kind: TypedKind, bytes: &[u8]) -> io::Result<TypedValue> {
    use std::io::{Error, ErrorKind};

    match kind {
        TypedKind::Bootstrap => {
            let v: crate::index::Bootstrap =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::Bootstrap(v))
        }
        TypedKind::Manifest => {
            let v: crate::index::Manifest =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::Manifest(v))
        }
        TypedKind::ColumnIndex => {
            let v: crate::index::ColumnIndex =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::ColumnIndex(v))
        }
        TypedKind::IndexSegment => {
            let v: crate::index::IndexSegment =
                bitcode::decode(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            Ok(TypedValue::IndexSegment(v))
        }
    }
}
