use crate::types::PhysicalKey;
use rustc_hash::FxHashMap;

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

// =================== Example in-memory Pager =======================

/// Minimal in-memory pager showing how to implement the unified API.
pub struct MemPager {
    map: FxHashMap<PhysicalKey, Vec<u8>>,
    next: PhysicalKey,
}

impl Default for MemPager {
    fn default() -> Self {
        Self {
            map: FxHashMap::with_hasher(Default::default()),
            next: 1, // reserve 0 for bootstrap
        }
    }
}

impl Pager for MemPager {
    fn alloc_many(&mut self, n: usize) -> Vec<PhysicalKey> {
        let start = self.next;
        self.next += n as u64;
        (0..n).map(|i| start + i as u64).collect()
    }

    fn batch<'a>(&'a mut self, req: &BatchRequest) -> BatchResponse<'a> {
        // 1) Apply puts
        for p in &req.puts {
            match p {
                BatchPut::Raw { key, bytes } => {
                    self.map.insert(*key, bytes.clone());
                }
                BatchPut::Typed { key, value } => {
                    let enc = encode_typed(value);
                    self.map.insert(*key, enc);
                }
            }
        }

        // 2) Serve gets
        let mut out: Vec<GetResult<'a>> = Vec::with_capacity(req.gets.len());
        for g in &req.gets {
            match g {
                BatchGet::Raw { key } => {
                    if let Some(v) = self.map.get(key) {
                        out.push(GetResult::Raw {
                            key: *key,
                            bytes: v.as_slice(),
                        });
                    } else {
                        out.push(GetResult::Missing { key: *key });
                    }
                }
                BatchGet::Typed { key, kind } => {
                    if let Some(v) = self.map.get(key) {
                        match decode_typed(*kind, v.as_slice()) {
                            Ok(tv) => out.push(GetResult::Typed {
                                key: *key,
                                value: tv,
                            }),
                            Err(_) => out.push(GetResult::Missing { key: *key }),
                        }
                    } else {
                        out.push(GetResult::Missing { key: *key });
                    }
                }
            }
        }

        BatchResponse { get_results: out }
    }
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
