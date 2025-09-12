use crate::column_index::IndexSegment;
use crate::types::PhysicalKey;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Very small LRU for decoded `IndexSegment`s keyed by `PhysicalKey`.
///
/// Simplicity over ultimate performance: operations are O(1) for map lookups
/// and O(n) for moving keys in the MRU list. With tiny capacities (e.g., 256),
/// this keeps overhead negligible while avoiding repeated decodes across calls.
pub struct SegLru {
    cap: usize,
    map: FxHashMap<PhysicalKey, Arc<IndexSegment>>, // key -> value
    // MRU at front (index 0), LRU at back.
    order: Vec<PhysicalKey>,
}

impl SegLru {
    pub fn new(capacity: usize) -> Self {
        Self {
            cap: capacity.max(1),
            map: FxHashMap::default(),
            order: Vec::new(),
        }
    }

    #[inline]
    pub fn get(&mut self, key: &PhysicalKey) -> Option<Arc<IndexSegment>> {
        if let Some(v) = self.map.get(key).cloned() {
            // Move key to front (MRU).
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                let k = self.order.remove(pos);
                self.order.insert(0, k);
            }
            Some(v)
        } else {
            None
        }
    }

    #[inline]
    pub fn put(&mut self, key: PhysicalKey, val: Arc<IndexSegment>) {
        if self.map.contains_key(&key) {
            // Update and move to front.
            self.map.insert(key, val);
            if let Some(pos) = self.order.iter().position(|k| *k == key) {
                let k = self.order.remove(pos);
                self.order.insert(0, k);
            } else {
                self.order.insert(0, key);
            }
            return;
        }

        // Insert new at front.
        self.map.insert(key, val);
        self.order.insert(0, key);

        // Evict if needed.
        if self.order.len() > self.cap {
            if let Some(old) = self.order.pop() {
                self.map.remove(&old);
            }
        }
    }
}

