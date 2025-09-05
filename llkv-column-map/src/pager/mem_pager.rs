use super::{
    BatchGet, BatchPut, BatchRequest, BatchResponse, GetResult, Pager, decode_typed, encode_typed,
};
use crate::types::PhysicalKey;
use rustc_hash::FxHashMap;

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
