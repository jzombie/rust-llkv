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

/// Request for one unified batch. Semantics: all `puts` are applied
/// before any `gets` in the same batch.
#[derive(Clone, Debug, Default)]
pub struct BatchRequest {
    pub puts: Vec<BatchPut>,
    pub gets: Vec<BatchGet>,
}

/// Result for a single get.
#[derive(Debug)]
pub enum GetResult<'a> {
    Raw { key: PhysicalKey, bytes: &'a [u8] },
    Typed { key: PhysicalKey, value: TypedValue },
    Missing { key: PhysicalKey },
}

/// Response for a batch. `get_results` aligns with `req.gets` order.
#[derive(Debug)]
pub struct BatchResponse<'a> {
    pub get_results: Vec<GetResult<'a>>,
}

/// Unified pager. `batch` performs raw+typed puts/gets in one roundtrip.
/// Lifetimes of `Raw` get slices are tied to `&'a self`.
pub trait Pager {
    /// Key allocation can remain separate; typically needed ahead of
    /// a write batch.
    fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey>;

    /// Apply all puts, then serve all gets, in one batch.
    fn batch<'a>(&'a mut self, req: &BatchRequest) -> BatchResponse<'a>;
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

pub fn decode_typed(kind: TypedKind, bytes: &[u8]) -> Result<TypedValue, ()> {
    match kind {
        TypedKind::Bootstrap => {
            let v: crate::index::Bootstrap = bitcode::decode(bytes).map_err(|_| ())?;
            Ok(TypedValue::Bootstrap(v))
        }
        TypedKind::Manifest => {
            let v: crate::index::Manifest = bitcode::decode(bytes).map_err(|_| ())?;
            Ok(TypedValue::Manifest(v))
        }
        TypedKind::ColumnIndex => {
            let v: crate::index::ColumnIndex = bitcode::decode(bytes).map_err(|_| ())?;
            Ok(TypedValue::ColumnIndex(v))
        }
        TypedKind::IndexSegment => {
            let v: crate::index::IndexSegment = bitcode::decode(bytes).map_err(|_| ())?;
            Ok(TypedValue::IndexSegment(v))
        }
    }
}
