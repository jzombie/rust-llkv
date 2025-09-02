//! llkv-btree: generic, paged B+Tree with SIMD-friendly leaf pages.
//!
//! PERFORMANCE NOTES
//! -----------------
//!
//! 1) Columnar leaf format (from prior patch):
//!    [Header][Aux][Keys block][Values block][Index]
//!    * Keys are contiguous; values are contiguous.
//!    * Index (per entry): k_off, k_len, v_off, v_len (u32 each).
//!
//! 2) SIMD equality fast path for u64 keys on leaves:
//!    * If a leaf's keys are fixed-width 8 and the queried key encodes
//!      to 8 bytes, we run an AVX2/AVX-512 equality scan over the
//!      keys_block. Falls back automatically if unsupported.
//!    * This is O(n/width) but avoids per-step decode and branches.
//!
//! 3) Scalar fallback remains the same: binary search using
//!    KC::compare_encoded(k_enc, &key) to preserve semantics for any
//!    encoding (including non-lexicographic encodings like LE u64).
//!
//! 4) Minor: page helpers for fixed-width detection and contiguous
//!    block access; light prefetch of value bytes on hit.
//!
//! 5) New: LRU page cache for read hot paths (no unsafe, O(1) ops).
//!    * Eviction only on insert, not on hit.
//!    * Writes/dealloc invalidate affected pages immediately.
//!    * Pending writes bypass the cache (serve from in-memory bytes).

use crate::{
    codecs::push_u32,
    codecs::{IdCodec, KeyCodec},
    errors::Error,
    iter::{BPlusTreeIter, ScanOpts},
    node_cache::NodeCache,
    pager::{DEFAULT_PAGE_SIZE_HINT, Pager},
    traits::BTree,
    views::node_view::{NodeTag, NodeView},
    views::value_view::ValueRef,
};
use core::cmp::Ordering;
use core::marker::PhantomData;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};

// ============================ Node representation ============================

#[derive(Clone)]
pub enum Node<K, I> {
    Internal {
        entries: Vec<(K, I)>,
    },
    Leaf {
        entries: Vec<(K, Vec<u8>)>,
        next: Option<I>,
    },
}

impl<K: Ord + Clone, I: Clone + Eq + std::hash::Hash> Node<K, I> {
    #[inline]
    pub(crate) fn entry_count(&self) -> usize {
        match self {
            Node::Leaf { entries, .. } => entries.len(),
            Node::Internal { entries } => entries.len(),
        }
    }
}

// ============================== Node builders ================================

struct LeafBuilder<'a, KC, IC>
where
    KC: KeyCodec + 'a,
    IC: IdCodec + 'a,
{
    entries: &'a [(KC::Key, Vec<u8>)],
    next: &'a Option<IC::Id>,
    _p: PhantomData<(&'a KC, &'a IC)>,
}

impl<'a, KC: KeyCodec, IC: IdCodec> LeafBuilder<'a, KC, IC> {
    fn encode(&self) -> Vec<u8> {
        let count = self.entries.len();
        let next_len = self.next.as_ref().map_or(0, |id| IC::encoded_len(id));
        let aux_len = 4 + next_len;

        let mut keys_block = Vec::new();
        let mut values_block = Vec::new();
        keys_block.reserve(self.entries.iter().map(|(k, _)| KC::encoded_len(k)).sum());
        values_block.reserve(self.entries.iter().map(|(_, v)| v.len()).sum());

        let mut k_offs = Vec::with_capacity(count);
        let mut v_offs = Vec::with_capacity(count);
        let mut k_lens = Vec::with_capacity(count);
        let mut v_lens = Vec::with_capacity(count);
        for (k, v) in self.entries {
            let ko = keys_block.len() as u32;
            KC::encode_into(k, &mut keys_block);
            let kl = (keys_block.len() as u32) - ko;

            let vo = values_block.len() as u32;
            values_block.extend_from_slice(v);
            let vl = (values_block.len() as u32) - vo;

            k_offs.push(ko);
            k_lens.push(kl);
            v_offs.push(vo);
            v_lens.push(vl);
        }

        let mut out =
            Vec::with_capacity(9 + aux_len + keys_block.len() + values_block.len() + 16 * count);
        out.push(NodeTag::Leaf as u8);
        push_u32(&mut out, count as u32);
        push_u32(&mut out, aux_len as u32);

        push_u32(&mut out, next_len as u32);
        if let Some(id) = self.next {
            IC::encode_into(id, &mut out);
        }

        out.extend_from_slice(&keys_block);
        out.extend_from_slice(&values_block);
        for i in 0..count {
            push_u32(&mut out, k_offs[i]);
            push_u32(&mut out, k_lens[i]);
            push_u32(&mut out, v_offs[i]);
            push_u32(&mut out, v_lens[i]);
        }
        out
    }
}

struct InternalBuilder<'a, KC, IC>
where
    KC: KeyCodec + 'a,
    IC: IdCodec + 'a,
{
    entries: &'a [(KC::Key, IC::Id)],
    _p: PhantomData<(&'a KC, &'a IC)>,
}

impl<'a, KC: KeyCodec, IC: IdCodec> InternalBuilder<'a, KC, IC> {
    fn encode(&self) -> Vec<u8> {
        let count = self.entries.len();
        let var_len: usize = self
            .entries
            .iter()
            .map(|(k, id)| 4 + KC::encoded_len(k) + 4 + IC::encoded_len(id))
            .sum();
        let mut out = Vec::with_capacity(9 + var_len + 4 * count);
        out.push(NodeTag::Internal as u8);
        push_u32(&mut out, count as u32);
        push_u32(&mut out, 0);
        let entries_start = out.len();
        let mut offsets = Vec::with_capacity(count);
        for (k, id) in self.entries {
            offsets.push((out.len() - entries_start) as u32);
            push_u32(&mut out, KC::encoded_len(k) as u32);
            KC::encode_into(k, &mut out);
            push_u32(&mut out, IC::encoded_len(id) as u32);
            IC::encode_into(id, &mut out);
        }
        for off in offsets {
            push_u32(&mut out, off);
        }
        out
    }
}

// =============================== B+Tree core =================================

enum InsertResult<K, I> {
    Ok,
    Split { separator: K, right_id: I },
}
enum DeleteResult {
    Ok,
    Underflow,
}

pub struct BPlusTree<P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    pub(crate) pager: P,
    pub(crate) root: P::Id,
    page_size: usize,

    pending_writes: FxHashMap<P::Id, Vec<u8>>,
    pending_deletes: FxHashSet<P::Id>,

    // Page cache for reads (safe, O(1)).
    node_cache: Arc<RwLock<NodeCache<P>>>,

    _p: PhantomData<(KC, IC)>,
}

impl<P, KC, IC> BPlusTree<P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    pub fn new(pager: P, root: P::Id, opt_node_cache: Option<Arc<RwLock<NodeCache<P>>>>) -> Self {
        let page_size = pager.page_size_hint().unwrap_or(DEFAULT_PAGE_SIZE_HINT);
        Self {
            pager,
            root,
            page_size,
            pending_writes: FxHashMap::default(),
            pending_deletes: FxHashSet::default(),
            node_cache: Self::init_node_cache(opt_node_cache),
            _p: PhantomData,
        }
    }

    pub fn create_empty(
        mut pager: P,
        opt_node_cache: Option<Arc<RwLock<NodeCache<P>>>>,
    ) -> Result<Self, Error> {
        let page_size = pager.page_size_hint().unwrap_or(DEFAULT_PAGE_SIZE_HINT);
        let root_id = pager.alloc_ids(1)?.remove(0);
        let node: Node<KC::Key, P::Id> = Node::Leaf {
            entries: vec![],
            next: None,
        };
        let node_bytes = Self::encode_node(&node);
        pager.write_batch(&[(root_id.clone(), &node_bytes)])?;
        pager.on_root_changed(root_id.clone())?;

        Ok(Self {
            pager,
            root: root_id,
            page_size,
            pending_writes: FxHashMap::default(),
            pending_deletes: FxHashSet::default(),
            node_cache: Self::init_node_cache(opt_node_cache),
            _p: PhantomData,
        })
    }

    /// Initializes with the given state for read-safe operations.
    /// Pager Write operations are not safe using this approach.
    /// This is used primarily with the `SharedBPlusTree` for streaming
    /// operations.
    pub fn with_readonly_pager_state(
        pager: P,
        root: P::Id,
        node_cache: Arc<RwLock<NodeCache<P>>>,
        page_size: usize,
    ) -> Self {
        Self {
            pager,
            root,
            page_size,
            pending_writes: FxHashMap::default(),
            pending_deletes: FxHashSet::default(),
            node_cache,
            _p: PhantomData,
        }
    }

    pub(crate) fn init_node_cache(
        opt_node_cache: Option<Arc<RwLock<NodeCache<P>>>>,
    ) -> Arc<RwLock<NodeCache<P>>> {
        match opt_node_cache {
            Some(node_cache) => node_cache,
            None => Arc::new(RwLock::new(NodeCache::<P>::default())),
        }
    }

    #[inline]
    pub fn root_id(&self) -> P::Id {
        self.root.clone()
    }

    // ----------------------------- public ops --------------------------------

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[inline]
    pub fn get(&self, key: &KC::Key) -> Result<Option<ValueRef<P::Page>>, Error>
    where
        KC::Key: Clone,
    {
        let mut v = self.get_many(&[key.clone()])?;
        Ok(v.pop().flatten())
    }

    #[inline]
    fn descend_to_leaf_for_key(
        &self,
        key: &KC::Key,
    ) -> Result<(P::Page, NodeView<P>, Option<P::Id>), Error> {
        let mut id = self.root.clone();
        loop {
            let page = self.read_one(&id)?;
            let view = NodeView::<P>::new(page.clone())?;
            match view.tag()? {
                NodeTag::Internal => {
                    let idx = self.find_child_idx(&view, key)?;
                    let (_, child_raw) = view.internal_entry_slices(idx);
                    let (child_id, _) = IC::decode_from(child_raw)?;
                    id = child_id;
                }
                NodeTag::Leaf => {
                    // leaf.next (if any)
                    let next = if view.leaf_next_aux().is_empty() {
                        None
                    } else {
                        Some(IC::decode_from(view.leaf_next_aux())?.0)
                    };
                    return Ok((page, view, next));
                }
            }
        }
    }

    #[inline]
    fn leaf_lower_bound(&self, view: &NodeView<P>, key: &KC::Key, mut lo: usize) -> (usize, bool)
    where
        KC: KeyCodec,
    {
        let mut hi = view.count();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (k_enc, _) = view.leaf_entry_slices(mid);
            match KC::compare_encoded(k_enc, key) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => return (mid, true),
            }
        }
        (lo, false) // insertion point, not found
    }

    #[inline]
    pub fn get_many(&self, keys: &[KC::Key]) -> Result<Vec<Option<ValueRef<P::Page>>>, Error>
    where
        KC::Key: Ord,
    {
        let n = keys.len();
        let mut out: Vec<Option<ValueRef<P::Page>>> = vec![None; n];
        if n == 0 {
            return Ok(out);
        }

        // Sort indices by key so we can walk leaves once, in order.
        let mut idxs: Vec<usize> = (0..n).collect();
        idxs.sort_by(|&i, &j| keys[i].cmp(&keys[j]));

        // Current leaf state (we advance forward via `next` as keys grow).
        let mut cur_page: Option<P::Page> = None;
        let mut cur_view: Option<NodeView<P>> = None;
        let mut next_id: Option<P::Id> = None;
        let mut hint_lo: usize = 0; // lower-bound hint within the current leaf

        for i in idxs {
            let k = &keys[i];

            // Ensure we have a leaf that could contain `k`.
            if cur_view.is_none() {
                let (p, v, nid) = self.descend_to_leaf_for_key(k)?;
                next_id = nid;
                hint_lo = 0;
                cur_page = Some(p);
                cur_view = Some(v);
            } else {
                // If k is greater than the last key in the current leaf, hop along `next`.
                let mut must_advance = {
                    let v = cur_view.as_ref().unwrap();
                    let cnt = v.count();
                    if cnt == 0 {
                        true
                    } else {
                        let (last_enc, _) = v.leaf_entry_slices(cnt - 1);
                        KC::compare_encoded(last_enc, k) == core::cmp::Ordering::Less
                    }
                };

                while must_advance {
                    let nid = match next_id.take() {
                        Some(id) => id,
                        None => break, // No further leaves; nothing can match
                    };
                    let p = self.read_one(&nid)?;
                    let v = NodeView::<P>::new(p.clone())?;
                    next_id = if v.leaf_next_aux().is_empty() {
                        None
                    } else {
                        Some(IC::decode_from(v.leaf_next_aux())?.0)
                    };
                    cur_page = Some(p);
                    cur_view = Some(v);
                    hint_lo = 0; // new leaf, reset search hint

                    let vv = cur_view.as_ref().unwrap();
                    let cnt = vv.count();
                    if cnt == 0 {
                        must_advance = true;
                    } else {
                        let (last_enc, _) = vv.leaf_entry_slices(cnt - 1);
                        must_advance =
                            KC::compare_encoded(last_enc, k) == core::cmp::Ordering::Less;
                    }
                }
            }

            // Binary-search within current leaf for `k` using the shared helper.
            if let (Some(page), Some(view)) = (cur_page.as_ref(), cur_view.as_ref()) {
                let (pos, found) = self.leaf_lower_bound(view, k, hint_lo);
                hint_lo = pos; // monotonic within a leaf for non-decreasing queries
                if found {
                    let range = view.leaf_entry_value_range(pos);
                    prefetch_value(&view.as_slice()[range.clone()]); // micro-prefetch
                    out[i] = Some(ValueRef::new(page.clone(), range));
                }
            }
        }

        Ok(out)
    }

    // pub fn contains_key(&self, key: &KC::Key) -> Result<bool, Error> {
    //     Ok(self.get(key)?.is_some())
    // }

    pub fn contains_key(&self, key: &KC::Key) -> Result<bool, Error>
    where
        KC::Key: Clone,
    {
        // Use the batched lookup with a single-element slice.
        let mut res = self.get_many(&[key.clone()])?;
        Ok(res.pop().flatten().is_some())
    }

    pub fn insert_many(&mut self, items: &[(KC::Key, &[u8])]) -> Result<(), Error> {
        // Sort by key to improve locality during splits and writes.
        let mut owned: Vec<_> = items.iter().map(|(k, v)| (k.clone(), *v)).collect();
        owned.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (k, v) in owned {
            self.upsert_internal(&k, v)?;
        }
        self.flush()
    }

    pub fn delete_many(&mut self, keys: &[KC::Key]) -> Result<(), Error> {
        for k in keys {
            match self.delete_internal(k) {
                Ok(_) | Err(Error::KeyNotFound) => {}
                Err(e) => return Err(e),
            }
        }
        self.flush()
    }

    pub fn iter<'a>(&'a self) -> Result<BPlusTreeIter<'a, P, KC, IC>, Error> {
        BPlusTreeIter::with_opts(self, ScanOpts::forward())
    }

    pub fn flush(&mut self) -> Result<(), Error> {
        if self.pending_writes.is_empty() && self.pending_deletes.is_empty() {
            return Ok(());
        }
        let to_write: Vec<_> = self
            .pending_writes
            .iter()
            .map(|(id, bytes)| (id.clone(), bytes.as_slice()))
            .collect();
        self.pager.write_batch(&to_write)?;
        if !self.pending_deletes.is_empty() {
            let ids: Vec<_> = self.pending_deletes.iter().cloned().collect();
            self.pager.dealloc_ids(&ids)?;
        }
        self.pending_writes.clear();
        self.pending_deletes.clear();
        Ok(())
    }

    // --------------------------- internal ops --------------------------------

    fn upsert_internal(&mut self, key: &KC::Key, value: &[u8]) -> Result<(), Error> {
        let root_id = self.root.clone();
        match self._insert(&root_id, key, value)? {
            InsertResult::Ok => Ok(()),
            InsertResult::Split {
                separator,
                right_id,
            } => {
                let old_root = self.root.clone();
                let new_root = self.alloc_ids(1)?.remove(0);
                let max_left = self.find_max_key(&old_root)?;
                let entries = vec![(max_left, old_root), (separator, right_id)];
                let node: Node<KC::Key, P::Id> = Node::Internal { entries };
                self.write_page(new_root.clone(), Self::encode_node(&node));
                self.root = new_root.clone();
                self.pager.on_root_changed(new_root)?;
                Ok(())
            }
        }
    }

    fn _insert(
        &mut self,
        node_id: &P::Id,
        key: &KC::Key,
        value: &[u8],
    ) -> Result<InsertResult<KC::Key, P::Id>, Error> {
        let page = self.read_one(node_id)?;
        let view = NodeView::<P>::new(page)?;

        match view.tag()? {
            NodeTag::Internal => {
                let child_idx = self.find_child_idx(&view, key)?;
                let (_, child_raw) = view.internal_entry_slices(child_idx);
                let (child_id, _) = IC::decode_from(child_raw)?;
                let result = self._insert(&child_id, key, value)?;

                if let InsertResult::Split {
                    separator,
                    right_id,
                } = result
                {
                    let mut node = self.read_node(node_id)?;
                    if let Node::Internal { entries } = &mut node {
                        entries[child_idx].0 = self.find_max_key(&entries[child_idx].1)?;
                        entries.insert(child_idx + 1, (separator, right_id));
                        return self.split_if_needed(node_id.clone(), node);
                    }
                    unreachable!();
                }
                Ok(InsertResult::Ok)
            }
            NodeTag::Leaf => {
                let mut lo = 0usize;
                let mut hi = view.count();
                let mut found_pos = None;

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let (k_enc, _) = view.leaf_entry_slices(mid);
                    match KC::compare_encoded(k_enc, key) {
                        Ordering::Less => lo = mid + 1,
                        Ordering::Greater => hi = mid,
                        Ordering::Equal => {
                            found_pos = Some(mid);
                            break;
                        }
                    }
                }

                let new_page_bytes = if let Some(pos) = found_pos {
                    Self::encode_leaf_with_update(&view, pos, value)?
                } else {
                    Self::encode_leaf_with_insert(&view, lo, key, value)?
                };

                self.split_if_needed_bytes(node_id.clone(), new_page_bytes)
            }
        }
    }

    fn delete_internal(&mut self, key: &KC::Key) -> Result<(), Error> {
        let root_id = self.root.clone();
        self._delete(&root_id, key)?;
        let root_node = self.read_node(&self.root)?;
        if let Node::Internal { entries } = root_node {
            if entries.len() == 1 {
                let old_root = self.root.clone();
                self.root = entries[0].1.clone();
                self.dealloc_page(old_root.clone());
                self.pager.on_root_changed(self.root.clone())?;
            }
        }
        Ok(())
    }

    fn _delete(&mut self, node_id: &P::Id, key: &KC::Key) -> Result<DeleteResult, Error> {
        let mut node = self.read_node(node_id)?;
        let mut child_pos_opt = None;

        let result = match &mut node {
            Node::Leaf { entries, .. } => match entries.binary_search_by_key(&key, |(k, _)| k) {
                Ok(pos) => {
                    entries.remove(pos);
                    let enc = Self::encode_node(&node);
                    self.write_page(node_id.clone(), enc.clone());
                    if enc.len() < self.page_size / 2 {
                        Ok(DeleteResult::Underflow)
                    } else {
                        Ok(DeleteResult::Ok)
                    }
                }
                Err(_) => Err(Error::KeyNotFound),
            },
            Node::Internal { entries } => {
                let pos = entries
                    .binary_search_by(|(k, _)| k.cmp(key))
                    .unwrap_or_else(|e| e);
                let child_pos = if pos >= entries.len() {
                    entries.len() - 1
                } else {
                    pos
                };
                child_pos_opt = Some(child_pos);
                let child_id = entries[child_pos].1.clone();

                let delete_result = self._delete(&child_id, key);
                if let Ok(_) = &delete_result {
                    if let Ok(new_max_key) = self.find_max_key(&child_id) {
                        entries[child_pos].0 = new_max_key;
                    }
                }
                delete_result
            }
        };
        if let Ok(DeleteResult::Underflow) = result {
            if let (Some(child_pos), Node::Internal { entries }) = (child_pos_opt, &mut node) {
                return self.rebalance_or_merge(node_id.clone(), entries, child_pos);
            }
        }

        if child_pos_opt.is_some() {
            self.write_page(node_id.clone(), Self::encode_node(&node));
        }

        result
    }

    fn rebalance_or_merge(
        &mut self,
        parent_id: P::Id,
        parent_entries: &mut Vec<(KC::Key, P::Id)>,
        child_pos: usize,
    ) -> Result<DeleteResult, Error> {
        if child_pos + 1 < parent_entries.len() {
            let right_id = parent_entries[child_pos + 1].1.clone();
            if self.node_size(&right_id)? > self.page_size / 2 {
                self.rebalance(parent_entries, child_pos, child_pos + 1)?;
                let node = Node::Internal {
                    entries: parent_entries.clone(),
                };
                self.write_page(parent_id, Self::encode_node(&node));
                return Ok(DeleteResult::Ok);
            }
        }
        if child_pos > 0 {
            let left_id = parent_entries[child_pos - 1].1.clone();
            if self.node_size(&left_id)? > self.page_size / 2 {
                self.rebalance(parent_entries, child_pos - 1, child_pos)?;
                let node = Node::Internal {
                    entries: parent_entries.clone(),
                };
                self.write_page(parent_id, Self::encode_node(&node));
                return Ok(DeleteResult::Ok);
            }
        }
        if child_pos > 0 {
            self.merge(parent_entries, child_pos - 1)?;
        } else {
            self.merge(parent_entries, child_pos)?;
        }
        let node = Node::Internal {
            entries: parent_entries.clone(),
        };
        let enc = Self::encode_node(&node);
        self.write_page(parent_id, enc.clone());
        if enc.len() < self.page_size / 2 {
            Ok(DeleteResult::Underflow)
        } else {
            Ok(DeleteResult::Ok)
        }
    }

    fn rebalance(
        &mut self,
        parent_entries: &mut [(KC::Key, P::Id)],
        left_pos: usize,
        right_pos: usize,
    ) -> Result<(), Error> {
        let (left_id, right_id) = (
            parent_entries[left_pos].1.clone(),
            parent_entries[right_pos].1.clone(),
        );
        let mut left_node = self.read_node(&left_id)?;
        let mut right_node = self.read_node(&right_id)?;
        match (&mut left_node, &mut right_node) {
            (Node::Internal { entries: l }, Node::Internal { entries: r }) => {
                if l.len() < r.len() {
                    let kv = r.remove(0);
                    l.push(kv);
                } else {
                    let kv = l.pop().unwrap();
                    r.insert(0, kv);
                }
            }
            (Node::Leaf { entries: l, .. }, Node::Leaf { entries: r, .. }) => {
                if l.len() < r.len() {
                    let kv = r.remove(0);
                    l.push(kv);
                } else {
                    let kv = l.pop().unwrap();
                    r.insert(0, kv);
                }
            }
            _ => return Err(Error::Corrupt("mismatched node types")),
        }
        parent_entries[left_pos].0 = self.find_max_key_in_node(&left_node)?;
        parent_entries[right_pos].0 = self.find_max_key_in_node(&right_node)?;
        self.write_page(left_id, Self::encode_node(&left_node));
        self.write_page(right_id, Self::encode_node(&right_node));
        Ok(())
    }

    fn merge(
        &mut self,
        parent_entries: &mut Vec<(KC::Key, P::Id)>,
        left_pos: usize,
    ) -> Result<(), Error> {
        let (_sep_key, right_id) = parent_entries.remove(left_pos + 1);
        let (_left_key, left_id) = parent_entries[left_pos].clone();
        let mut left_node = self.read_node(&left_id)?;
        let right_node = self.read_node(&right_id)?;
        self.dealloc_page(right_id);
        match (&mut left_node, right_node) {
            (
                Node::Leaf {
                    entries: l,
                    next: l_next,
                },
                Node::Leaf {
                    entries: r,
                    next: r_next,
                },
            ) => {
                *l_next = r_next;
                l.extend(r);
            }
            (Node::Internal { entries: l }, Node::Internal { entries: r }) => {
                l.extend(r);
            }
            _ => return Err(Error::Corrupt("mismatched node types")),
        }

        parent_entries[left_pos].0 = self.find_max_key_in_node(&left_node)?;
        self.write_page(left_id, Self::encode_node(&left_node));
        Ok(())
    }

    fn find_child_idx(&self, view: &NodeView<P>, key: &KC::Key) -> Result<usize, Error> {
        let n = view.count();
        if n == 0 {
            return Ok(0);
        }
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (k_enc, _) = view.internal_entry_slices(mid);
            if KC::compare_encoded(k_enc, key) == Ordering::Less {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        Ok(lo.min(n - 1))
    }

    fn split_if_needed(
        &mut self,
        id: P::Id,
        mut node: Node<KC::Key, P::Id>,
    ) -> Result<InsertResult<KC::Key, P::Id>, Error> {
        let enc_len = Self::encode_node(&node).len();
        if enc_len <= self.page_size {
            self.write_page(id, Self::encode_node(&node));
            return Ok(InsertResult::Ok);
        }

        let right_id = self.alloc_ids(1)?.remove(0);
        let (separator, right_node) = match &mut node {
            Node::Internal { entries } => {
                let mid = entries.len() / 2;
                let right_entries = entries.split_off(mid);
                let sep = right_entries.last().unwrap().0.clone();
                (
                    sep,
                    Node::Internal {
                        entries: right_entries,
                    },
                )
            }
            Node::Leaf { entries, next } => {
                let mid = entries.len() / 2;
                let right_entries = entries.split_off(mid);
                let sep = right_entries.last().unwrap().0.clone();
                let right_next = next.clone();
                *next = Some(right_id.clone());
                (
                    sep,
                    Node::Leaf {
                        entries: right_entries,
                        next: right_next,
                    },
                )
            }
        };
        self.write_page(id, Self::encode_node(&node));
        self.write_page(right_id.clone(), Self::encode_node(&right_node));
        Ok(InsertResult::Split {
            separator,
            right_id,
        })
    }

    pub(crate) fn read_node(&self, id: &P::Id) -> Result<Node<KC::Key, P::Id>, Error> {
        let page = self.read_one(id)?;
        let view = NodeView::<P>::new(page)?;
        Ok(match view.tag()? {
            NodeTag::Internal => {
                let entries = (0..view.count())
                    .map(|i| -> Result<_, Error> {
                        let (k_enc, id_raw) = view.internal_entry_slices(i);
                        let key = KC::decode_from(k_enc)?;
                        let (cid, _) = IC::decode_from(id_raw)?;
                        Ok((key, cid))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Node::Internal { entries }
            }
            NodeTag::Leaf => {
                let entries = (0..view.count())
                    .map(|i| -> Result<_, Error> {
                        let (k_enc, v_raw) = view.leaf_entry_slices(i);
                        let key = KC::decode_from(k_enc)?;
                        Ok((key, v_raw.to_vec()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let next = if view.leaf_next_aux().is_empty() {
                    None
                } else {
                    Some(IC::decode_from(view.leaf_next_aux())?.0)
                };
                Node::Leaf { entries, next }
            }
        })
    }

    fn encode_node(node: &Node<KC::Key, P::Id>) -> Vec<u8> {
        match node {
            Node::Leaf { entries, next } => LeafBuilder::<KC, IC> {
                entries,
                next,
                _p: PhantomData,
            }
            .encode(),
            Node::Internal { entries } => InternalBuilder::<KC, IC> {
                entries,
                _p: PhantomData,
            }
            .encode(),
        }
    }

    // ------------------- Pager I/O with cache integration --------------------

    // TODO: Rename to `read_page`?
    pub(crate) fn read_one(&self, id: &P::Id) -> Result<P::Page, Error> {
        // Pending mutations always win.
        if self.pending_deletes.contains(id) {
            return Err(Error::MissingPage);
        }
        if let Some(bytes) = self.pending_writes.get(id) {
            return self.pager.materialize_owned(bytes);
        }

        // Fast path: shared lock + peek (no LRU mutation).
        if let Some(p) = self.node_cache.read().unwrap().peek(id) {
            // Opportunistic promotion to MRU. Avoids taking a blocking write lock on hot paths.
            if let Ok(mut c) = self.node_cache.try_write() {
                c.touch(id);
            }
            return Ok(p);
        }

        // Miss: fetch from pager.
        let page = self
            .pager
            .read_batch(&[id.clone()])?
            .remove(id)
            .ok_or(Error::MissingPage)?;

        // Opportunistic insert into cache. If contended, skip; correctness is unchanged.
        if let Ok(mut c) = self.node_cache.try_write() {
            c.insert(id.clone(), page.clone());
        } else {
            // If you prefer guaranteed caching (slightly higher tail latency), use:
            // self.node_cache.write().unwrap().insert(id.clone(), page.clone());
        }

        Ok(page)
    }

    fn find_max_key(&self, node_id: &P::Id) -> Result<KC::Key, Error> {
        let node = self.read_node(node_id)?;
        self.find_max_key_in_node(&node)
    }
    fn find_max_key_in_node(&self, node: &Node<KC::Key, P::Id>) -> Result<KC::Key, Error> {
        match node {
            Node::Leaf { entries, .. } => Ok(entries
                .last()
                .ok_or(Error::Corrupt("empty leaf"))?
                .0
                .clone()),
            Node::Internal { entries, .. } => {
                self.find_max_key(&entries.last().ok_or(Error::Corrupt("empty internal"))?.1)
            }
        }
    }

    fn node_size(&self, id: &P::Id) -> Result<usize, Error> {
        Ok(self.read_one(id)?.len())
    }

    #[inline]
    fn write_page(&mut self, id: P::Id, data: Vec<u8>) {
        // Invalidate any cached copy to avoid stale reads.
        self.node_cache.write().unwrap().invalidate(&id);
        self.pending_writes.insert(id, data);
    }

    #[inline]
    fn dealloc_page(&mut self, id: P::Id) {
        self.node_cache.write().unwrap().invalidate(&id);
        self.pending_deletes.insert(id);
    }

    #[inline]
    fn alloc_ids(&mut self, count: usize) -> Result<Vec<P::Id>, Error> {
        self.pager.alloc_ids(count)
    }
}

// ---------------------- Byte-level leaf operations ---------------------------

impl<P, KC, IC> BPlusTree<P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    /// Manually clear this handle's page cache.
    #[inline]
    pub fn clear_cache(&self) {
        self.node_cache.write().unwrap().clear();
    }

    fn encode_leaf_with_insert(
        view: &NodeView<P>,
        pos: usize,
        key: &KC::Key,
        value: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let count = view.count();
        let new_count = count + 1;
        let mut k_enc = Vec::with_capacity(KC::encoded_len(key));
        KC::encode_into(key, &mut k_enc);

        let mut new_keys = Vec::new();
        let mut new_vals = Vec::new();
        let mut new_idx = Vec::with_capacity(new_count * 16);

        let mut k_off: u32 = 0;
        let mut v_off: u32 = 0;

        for i in 0..new_count {
            let (k_bytes, v_bytes) = if i < pos {
                view.leaf_entry_slices(i)
            } else if i == pos {
                (k_enc.as_slice(), value)
            } else {
                view.leaf_entry_slices(i - 1)
            };

            let k_len = k_bytes.len() as u32;
            let v_len = v_bytes.len() as u32;

            push_u32(&mut new_idx, k_off);
            push_u32(&mut new_idx, k_len);
            push_u32(&mut new_idx, v_off);
            push_u32(&mut new_idx, v_len);

            new_keys.extend_from_slice(k_bytes);
            new_vals.extend_from_slice(v_bytes);

            k_off += k_len;
            v_off += v_len;
        }

        let mut out = Vec::new();
        out.push(NodeTag::Leaf as u8);
        push_u32(&mut out, new_count as u32);
        push_u32(&mut out, view.aux_len() as u32);
        out.extend_from_slice(&view.as_slice()[view.aux_range()]);
        out.extend_from_slice(&new_keys);
        out.extend_from_slice(&new_vals);
        out.extend_from_slice(&new_idx);

        Ok(out)
    }

    fn encode_leaf_with_update(
        view: &NodeView<P>,
        pos: usize,
        value: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let count = view.count();
        let mut new_vals = Vec::new();
        let mut new_idx = Vec::with_capacity(count * 16);
        let mut v_off: u32 = 0;

        for i in 0..count {
            let (_k_bytes, v_bytes) = if i == pos {
                let (k, _) = view.leaf_entry_slices(i);
                (k, value)
            } else {
                view.leaf_entry_slices(i)
            };

            let (k_off, k_len, _, _) = view.leaf_index_fields(i);
            let v_len = v_bytes.len() as u32;

            push_u32(&mut new_idx, k_off as u32);
            push_u32(&mut new_idx, k_len as u32);
            push_u32(&mut new_idx, v_off);
            push_u32(&mut new_idx, v_len);

            new_vals.extend_from_slice(v_bytes);
            v_off += v_len;
        }

        let mut out = Vec::new();
        out.push(NodeTag::Leaf as u8);
        push_u32(&mut out, count as u32);
        push_u32(&mut out, view.aux_len() as u32);
        out.extend_from_slice(&view.as_slice()[view.aux_range()]);
        out.extend_from_slice(view.leaf_keys_block());
        out.extend_from_slice(&new_vals);
        out.extend_from_slice(&new_idx);

        Ok(out)
    }

    fn split_leaf_bytes(
        &mut self,
        page_bytes: &[u8],
        right_id: P::Id,
    ) -> Result<(Vec<u8>, KC::Key, Vec<u8>), Error> {
        let view = NodeView::<P>::new(self.pager.materialize_owned(page_bytes)?)?;
        let count = view.count();
        let mid = count / 2;

        let separator_key_enc = view.leaf_entry_slices(count - 1).0;
        let separator = KC::decode_from(separator_key_enc)?;

        let mut left_keys = Vec::new();
        let mut left_vals = Vec::new();
        let mut left_idx = Vec::new();
        let mut k_off: u32 = 0;
        let mut v_off: u32 = 0;
        for i in 0..mid {
            let (k, v) = view.leaf_entry_slices(i);
            left_keys.extend_from_slice(k);
            left_vals.extend_from_slice(v);
            push_u32(&mut left_idx, k_off);
            push_u32(&mut left_idx, k.len() as u32);
            push_u32(&mut left_idx, v_off);
            push_u32(&mut left_idx, v.len() as u32);
            k_off += k.len() as u32;
            v_off += v.len() as u32;
        }

        let mut left_aux = Vec::new();
        let next_id_enc_len = IC::encoded_len(&right_id);
        push_u32(&mut left_aux, next_id_enc_len as u32);
        IC::encode_into(&right_id, &mut left_aux);

        let mut left_bytes = Vec::new();
        left_bytes.push(NodeTag::Leaf as u8);
        push_u32(&mut left_bytes, mid as u32);
        push_u32(&mut left_bytes, left_aux.len() as u32);
        left_bytes.extend_from_slice(&left_aux);
        left_bytes.extend_from_slice(&left_keys);
        left_bytes.extend_from_slice(&left_vals);
        left_bytes.extend_from_slice(&left_idx);

        let mut right_keys = Vec::new();
        let mut right_vals = Vec::new();
        let mut right_idx = Vec::new();
        k_off = 0;
        v_off = 0;
        for i in mid..count {
            let (k, v) = view.leaf_entry_slices(i);
            right_keys.extend_from_slice(k);
            right_vals.extend_from_slice(v);
            push_u32(&mut right_idx, k_off);
            push_u32(&mut right_idx, k.len() as u32);
            push_u32(&mut right_idx, v_off);
            push_u32(&mut right_idx, v.len() as u32);
            k_off += k.len() as u32;
            v_off += v.len() as u32;
        }

        let old_aux = &view.as_slice()[view.aux_range()];
        let mut right_bytes = Vec::new();
        right_bytes.push(NodeTag::Leaf as u8);
        push_u32(&mut right_bytes, (count - mid) as u32);
        push_u32(&mut right_bytes, old_aux.len() as u32);
        right_bytes.extend_from_slice(old_aux);
        right_bytes.extend_from_slice(&right_keys);
        right_bytes.extend_from_slice(&right_vals);
        right_bytes.extend_from_slice(&right_idx);

        Ok((left_bytes, separator, right_bytes))
    }

    fn split_if_needed_bytes(
        &mut self,
        id: P::Id,
        page_bytes: Vec<u8>,
    ) -> Result<InsertResult<KC::Key, P::Id>, Error> {
        if page_bytes.len() <= self.page_size {
            self.write_page(id, page_bytes);
            return Ok(InsertResult::Ok);
        }

        let right_id = self.alloc_ids(1)?.remove(0);
        let tag = NodeTag::from_u8(page_bytes[0])?;

        let (left_bytes, separator, right_bytes) = if tag == NodeTag::Leaf {
            self.split_leaf_bytes(&page_bytes, right_id.clone())?
        } else {
            let node = self.read_node(&id)?;
            let mut node_clone = node.clone();
            let (sep, right_node) = match &mut node_clone {
                Node::Internal { entries } => {
                    let mid = entries.len() / 2;
                    let right_entries = entries.split_off(mid);
                    let sep = right_entries.last().unwrap().0.clone();
                    (
                        sep,
                        Node::Internal {
                            entries: right_entries,
                        },
                    )
                }
                _ => unreachable!(),
            };
            (
                Self::encode_node(&node_clone),
                sep,
                Self::encode_node(&right_node),
            )
        };

        self.write_page(id, left_bytes);
        self.write_page(right_id.clone(), right_bytes);

        Ok(InsertResult::Split {
            separator,
            right_id,
        })
    }
}

// ------------------------------- SIMD helpers --------------------------------

#[inline]
fn prefetch_value(slice: &[u8]) {
    if slice.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(slice.as_ptr() as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) slice.as_ptr(),
            options(nostack, preserves_flags)
        );
    }
}

fn simd_find_u64_equal_on_leaf<P: Pager>(view: &NodeView<P>, target_be: u64) -> Option<usize> {
    let n = view.count();
    if n == 0 {
        return None;
    }
    if view.leaf_fixed_key_width() != Some(8) {
        return None;
    }
    let keys = view.leaf_keys_block();
    debug_assert_eq!(keys.len(), n * 8);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe {
                use core::arch::x86_64::{
                    _mm512_cmpeq_epi64_mask, _mm512_loadu_si512, _mm512_set1_epi64,
                };
                let mut i = 0usize;
                let lanes = 8usize;
                let needle = _mm512_set1_epi64(target_be as i64);
                while i + lanes <= n {
                    let ptr = keys.as_ptr().add(i * 8) as *const _;
                    let v = _mm512_loadu_si512(ptr);
                    let m = _mm512_cmpeq_epi64_mask(v, needle);
                    if m != 0 {
                        let offset = m.trailing_zeros() as usize;
                        return Some(i + offset);
                    }
                    i += lanes;
                }
                while i < n {
                    let off = i * 8;
                    let k = u64::from_be_bytes(keys[off..off + 8].try_into().unwrap());
                    if k == target_be {
                        return Some(i);
                    }
                    i += 1;
                }
                return None;
            }
        }
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_loadu_si256,
                    _mm256_movemask_pd, _mm256_set1_epi64x,
                };
                let mut i = 0usize;
                let lanes = 4usize;
                let needle = _mm256_set1_epi64x(target_be as i64);
                while i + lanes <= n {
                    let ptr = keys.as_ptr().add(i * 8) as *const __m256i;
                    let v = _mm256_loadu_si256(ptr);
                    let eq = _mm256_cmpeq_epi64(v, needle);
                    let mask = _mm256_movemask_pd(_mm256_castsi256_pd(eq));
                    if mask != 0 {
                        let offset = mask.trailing_zeros() as usize;
                        return Some(i + offset);
                    }
                    i += lanes;
                }
                while i < n {
                    let off = i * 8;
                    let k = u64::from_be_bytes(keys[off..off + 8].try_into().unwrap());
                    if k == target_be {
                        return Some(i);
                    }
                    i += 1;
                }
                return None;
            }
        }
    }

    let mut i = 0usize;
    while i < n {
        let off = i * 8;
        let k = u64::from_be_bytes(keys[off..off + 8].try_into().unwrap());
        if k == target_be {
            return Some(i);
        }
        i += 1;
    }
    None
}

// ------------------------------- BTree trait ---------------------------------

impl<'a, P, KC, IC> BTree<'a, KC::Key> for BPlusTree<P, KC, IC>
where
    P: Pager + 'a,
    KC: KeyCodec + 'a,
    IC: IdCodec<Id = P::Id> + 'a,
{
    type Value = ValueRef<P::Page>;

    fn get_many(&'a self, keys: &[KC::Key]) -> Result<Vec<Option<Self::Value>>, Error> {
        BPlusTree::get_many(self, keys)
    }

    fn contains_key(&'a self, key: &KC::Key) -> Result<bool, Error> {
        BPlusTree::contains_key(self, key)
    }
    fn insert_many(&mut self, items: &[(KC::Key, &[u8])]) -> Result<(), Error>
    where
        KC::Key: Clone,
    {
        BPlusTree::insert_many(self, items)
    }
    fn delete_many(&mut self, keys: &[KC::Key]) -> Result<(), Error>
    where
        KC::Key: Clone,
    {
        BPlusTree::delete_many(self, keys)
    }
}

// --------------------------------- Tests -------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::{
        BigEndianIdCodec, BigEndianKeyCodec, IdCodec, KeyCodec, StringKeyCodec, read_u32_at,
    };
    use rand::rngs::StdRng;
    use rand::{SeedableRng, seq::SliceRandom};
    use rustc_hash::FxHashMap;
    use std::sync::Arc;

    // --- Pager ---
    #[derive(Clone)]
    struct TestPager {
        pages: FxHashMap<u64, Arc<[u8]>>,
        next_id: u64,
        page_size: usize,
    }

    impl Pager for TestPager {
        type Id = u64;
        type Page = Arc<[u8]>;
        fn read_batch(&self, ids: &[Self::Id]) -> Result<FxHashMap<Self::Id, Self::Page>, Error> {
            Ok(ids
                .iter()
                .filter_map(|id| self.pages.get(id).map(|p| (*id, p.clone())))
                .collect())
        }
        fn write_batch(&mut self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error> {
            for (id, data) in pages {
                if data.len() > self.page_size {
                    panic!(
                        "Page overflow: size {} > max {}",
                        data.len(),
                        self.page_size
                    );
                }
                self.pages.insert(*id, Arc::from(*data));
            }
            Ok(())
        }
        fn alloc_ids(&mut self, count: usize) -> Result<Vec<Self::Id>, Error> {
            let start = self.next_id;
            self.next_id += count as u64;
            Ok((start..self.next_id).collect())
        }
        fn dealloc_ids(&mut self, ids: &[Self::Id]) -> Result<(), Error> {
            for id in ids {
                self.pages.remove(id);
            }
            Ok(())
        }
        fn page_size_hint(&self) -> Option<usize> {
            Some(self.page_size)
        }
        fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error> {
            Ok(Arc::from(bytes))
        }
    }

    // --- Manual Construction Tests using builders ---

    #[test]
    fn point_lookup_works() {
        let mut pager = TestPager {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 4096,
        };
        let ids = pager.alloc_ids(3).unwrap();
        let root = ids[0];
        let left = ids[1];
        let right = ids[2];
        let l_entries = vec![
            (1u64, b"a".to_vec()),
            (2, b"bb".to_vec()),
            (3, b"ccc".to_vec()),
        ];
        let r_entries = vec![
            (5u64, b"eee".to_vec()),
            (6, b"ffff".to_vec()),
            (7, b"gg".to_vec()),
        ];
        let i_entries = vec![(3u64, left), (7, right)];

        let lbytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &l_entries,
            next: &Some(right),
            _p: PhantomData,
        }
        .encode();
        let rbytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &r_entries,
            next: &None,
            _p: PhantomData,
        }
        .encode();
        let ibytes = super::InternalBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &i_entries,
            _p: PhantomData,
        }
        .encode();
        pager
            .write_batch(&[(left, &lbytes), (right, &rbytes), (root, &ibytes)])
            .unwrap();
        let tree: BPlusTree<TestPager, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
            BPlusTree::new(pager, root, None);

        assert_eq!(
            tree.get(&1).unwrap().as_deref().map(|v| &*v),
            Some(&b"a"[..])
        );
        assert_eq!(
            tree.get(&2).unwrap().as_deref().map(|v| &*v),
            Some(&b"bb"[..])
        );
        assert_eq!(
            tree.get(&3).unwrap().as_deref().map(|v| &*v),
            Some(&b"ccc"[..])
        );
        assert!(tree.get(&4).unwrap().is_none());
        assert_eq!(
            tree.get(&5).unwrap().as_deref().map(|v| &*v),
            Some(&b"eee"[..])
        );
        assert_eq!(
            tree.get(&6).unwrap().as_deref().map(|v| &*v),
            Some(&b"ffff"[..])
        );
        assert_eq!(
            tree.get(&7).unwrap().as_deref().map(|v| &*v),
            Some(&b"gg"[..])
        );
    }

    #[test]
    fn iterator_walks_leaves() {
        let mut pager = TestPager {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 4096,
        };
        let ids = pager.alloc_ids(3).unwrap();
        let root = ids[0];
        let l1 = ids[1];
        let l2 = ids[2];
        let entries1 = vec![(10u64, b"x".to_vec()), (11, b"y".to_vec())];
        let entries2 = vec![(12u64, b"z".to_vec())];
        let l1_bytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &entries1,
            next: &Some(l2),
            _p: PhantomData,
        }
        .encode();
        let l2_bytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &entries2,
            next: &None,
            _p: PhantomData,
        }
        .encode();
        let i_bytes = super::InternalBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &vec![(11, l1), (12, l2)],
            _p: PhantomData,
        }
        .encode();
        pager
            .write_batch(&[(root, &i_bytes), (l1, &l1_bytes), (l2, &l2_bytes)])
            .unwrap();
        let tree: BPlusTree<TestPager, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
            BPlusTree::new(pager, root, None);

        let got: Vec<(u64, Vec<u8>)> = tree
            .iter()
            .unwrap()
            .map(|(k, v)| {
                (
                    BigEndianKeyCodec::<u64>::decode_from(k.as_ref()).unwrap(),
                    v.as_ref().to_vec(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (10, b"x".to_vec()),
                (11, b"y".to_vec()),
                (12, b"z".to_vec())
            ]
        );
    }

    #[test]
    fn get_many_batched() {
        let mut pager = TestPager {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 4096,
        };
        let ids = pager.alloc_ids(3).unwrap();
        let root = ids[0];
        let left = ids[1];
        let right = ids[2];
        let l_entries = vec![(1u64, b"a".to_vec()), (3, b"ccc".to_vec())];
        let r_entries = vec![(5u64, b"eee".to_vec()), (7, b"gg".to_vec())];
        let i_entries = vec![(3u64, left), (7, right)];

        let lbytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &l_entries,
            next: &Some(right),
            _p: PhantomData,
        }
        .encode();
        let rbytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &r_entries,
            next: &None,
            _p: PhantomData,
        }
        .encode();
        let ibytes = super::InternalBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &i_entries,
            _p: PhantomData,
        }
        .encode();
        pager
            .write_batch(&[(left, &lbytes), (right, &rbytes), (root, &ibytes)])
            .unwrap();
        let tree: BPlusTree<TestPager, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> =
            BPlusTree::new(pager, root, None);

        let q = vec![7u64, 4, 1, 6, 3];
        let ans = tree.get_many(&q).unwrap();
        let as_opt_bytes: Vec<Option<Vec<u8>>> = ans
            .into_iter()
            .map(|o| o.map(|r| r.as_ref().to_vec()))
            .collect();
        assert_eq!(as_opt_bytes[0].as_deref(), Some(&b"gg"[..]));
        assert_eq!(as_opt_bytes[1], None);
        assert_eq!(as_opt_bytes[2].as_deref(), Some(&b"a"[..]));
        assert_eq!(as_opt_bytes[3], None);
        assert_eq!(as_opt_bytes[4].as_deref(), Some(&b"ccc"[..]));
    }

    // --- Randomized Insertion/Deletion Test ---
    #[test]
    fn test_random_inserts_and_deletes() {
        let pager = TestPager {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 256,
        };
        let mut tree = BPlusTree::<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>>::create_empty(
            pager, None,
        )
        .unwrap();

        let mut shadow = std::collections::BTreeMap::new();
        let mut rng = StdRng::seed_from_u64(42);
        let mut keys: Vec<u64> = (0..1000).collect();
        keys.shuffle(&mut rng);

        // Insert
        let items: Vec<_> = keys.iter().map(|k| (*k, k.to_be_bytes())).collect();
        let owned: Vec<_> = items.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        tree.insert_many(&owned).unwrap();
        for (k, v) in items {
            shadow.insert(k, v.to_vec());
        }

        validate_structure(&tree).unwrap();
        for (k, v) in &shadow {
            assert_eq!(tree.get(k).unwrap().unwrap().as_ref(), v.as_slice());
        }

        // Update
        keys.shuffle(&mut rng);
        let updates: Vec<_> = keys
            .iter()
            .take(200)
            .map(|k| (*k, format!("updated_{k}").into_bytes()))
            .collect();
        let owned_u: Vec<_> = updates.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        tree.insert_many(&owned_u).unwrap();
        for (k, v) in updates {
            shadow.insert(k, v);
        }

        validate_structure(&tree).unwrap();
        for (k, v) in &shadow {
            assert_eq!(tree.get(k).unwrap().unwrap().as_ref(), v.as_slice());
        }

        // Delete
        keys.shuffle(&mut rng);
        let dels: Vec<_> = keys.iter().take(500).cloned().collect();
        tree.delete_many(&dels).unwrap();
        for k in dels {
            shadow.remove(&k);
        }

        validate_structure(&tree).unwrap();
        assert_eq!(tree.iter().unwrap().count(), shadow.len());
        for (k, v) in &shadow {
            assert_eq!(tree.get(k).unwrap().unwrap().as_ref(), v.as_slice());
        }
    }

    #[test]
    fn test_32bit_physical_64bit_logical_keys() {
        // 32-bit page IDs, 64-bit logical keys
        #[derive(Clone)]
        struct TestPager32 {
            pages: FxHashMap<u32, Arc<[u8]>>,
            next_id: u32,
            page_size: usize,
        }

        impl Pager for TestPager32 {
            type Id = u32;
            type Page = Arc<[u8]>;
            fn read_batch(
                &self,
                ids: &[Self::Id],
            ) -> Result<FxHashMap<Self::Id, Self::Page>, Error> {
                Ok(ids
                    .iter()
                    .filter_map(|id| self.pages.get(id).map(|p| (*id, p.clone())))
                    .collect())
            }
            fn write_batch(&mut self, pages: &[(Self::Id, &[u8])]) -> Result<(), Error> {
                for (id, data) in pages {
                    self.pages.insert(*id, Arc::from(*data));
                }
                Ok(())
            }
            fn alloc_ids(&mut self, count: usize) -> Result<Vec<Self::Id>, Error> {
                let start = self.next_id;
                self.next_id += count as u32;
                Ok((start..self.next_id).collect())
            }
            fn dealloc_ids(&mut self, ids: &[Self::Id]) -> Result<(), Error> {
                for id in ids {
                    self.pages.remove(id);
                }
                Ok(())
            }
            fn page_size_hint(&self) -> Option<usize> {
                Some(self.page_size)
            }
            fn materialize_owned(&self, bytes: &[u8]) -> Result<Self::Page, Error> {
                Ok(Arc::from(bytes))
            }
        }

        let pager = TestPager32 {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 256,
        };
        let mut tree = BPlusTree::<_, BigEndianKeyCodec<u64>, BigEndianIdCodec<u32>>::create_empty(
            pager, None,
        )
        .unwrap();

        let key1: u64 = 1;
        let key2: u64 = u64::MAX;
        let key3: u64 = 1 << 40;

        tree.insert_many(&[
            (key1, b"value_one".as_slice()),
            (key2, b"value_max".as_slice()),
            (key3, b"value_large".as_slice()),
        ])
        .unwrap();

        assert_eq!(
            tree.get(&key1).unwrap().as_deref().map(|v| &*v),
            Some(&b"value_one"[..])
        );
        assert_eq!(
            tree.get(&key2).unwrap().as_deref().map(|v| &*v),
            Some(&b"value_max"[..])
        );
        assert_eq!(
            tree.get(&key3).unwrap().as_deref().map(|v| &*v),
            Some(&b"value_large"[..])
        );
        assert!(tree.get(&(key1 + 5)).unwrap().is_none());
    }

    #[test]
    fn test_u128_logical_keys() {
        let pager = TestPager {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 256,
        };
        let mut tree =
            BPlusTree::<_, BigEndianKeyCodec<u128>, BigEndianIdCodec<u64>>::create_empty(
                pager, None,
            )
            .unwrap();

        let key1: u128 = 1;
        let key2: u128 = u128::MAX;
        let key3: u128 = 1 << 80;

        tree.insert_many(&[
            (key1, b"value_one".as_slice()),
            (key2, b"value_max".as_slice()),
            (key3, b"value_large".as_slice()),
        ])
        .unwrap();

        assert_eq!(
            tree.get(&key1).unwrap().as_deref().map(|v| &*v),
            Some(&b"value_one"[..])
        );
        assert_eq!(
            tree.get(&key2).unwrap().as_deref().map(|v| &*v),
            Some(&b"value_max"[..])
        );
        assert_eq!(
            tree.get(&key3).unwrap().as_deref().map(|v| &*v),
            Some(&b"value_large"[..])
        );
        assert!(tree.contains_key(&key2).unwrap());
        assert!(!tree.contains_key(&(key1 + 5)).unwrap());
    }

    // --- Structure validation (internal views; no payload copies) ---

    fn validate_structure<P, KC, IC>(tree: &BPlusTree<P, KC, IC>) -> Result<(), String>
    where
        P: Pager,
        KC: KeyCodec,
        IC: IdCodec<Id = P::Id>,
    {
        // empty tree OK
        if tree
            .pager
            .read_batch(&[tree.root.clone()])
            .unwrap()
            .is_empty()
            && tree.pending_writes.is_empty()
        {
            return Ok(());
        }
        let root_node = tree.read_node(&tree.root).unwrap();
        if root_node.entry_count() == 0 {
            return Ok(());
        }
        let mut leaf_depths = FxHashSet::default();
        _validate_node(tree, &tree.root_id(), 0, &mut leaf_depths, true)?;
        if leaf_depths.len() > 1 {
            return Err(format!(
                "Tree is unbalanced! Leaf depths: {:?}",
                leaf_depths
            ));
        }
        Ok(())
    }

    fn _validate_node<P, KC, IC>(
        tree: &BPlusTree<P, KC, IC>,
        node_id: &P::Id,
        depth: usize,
        leaf_depths: &mut FxHashSet<usize>,
        is_root: bool,
    ) -> Result<(), String>
    where
        P: Pager,
        KC: KeyCodec,
        IC: IdCodec<Id = P::Id>,
    {
        let node_page = tree
            .read_one(node_id)
            .map_err(|e| format!("Read error: {:?}", e))?;
        let view = super::NodeView::<P>::new(node_page).unwrap();
        let count = view.count();

        if !is_root && count == 0 {
            return Err(format!("Non-root node {:?} is empty.", node_id));
        }

        let mut prev_key: Option<KC::Key> = None;
        for i in 0..count {
            let (k_enc, _) = if view.tag().unwrap() == super::NodeTag::Internal {
                view.internal_entry_slices(i)
            } else {
                view.leaf_entry_slices(i)
            };
            let key = KC::decode_from(k_enc).unwrap();
            if let Some(prev) = &prev_key {
                if i > 0 && *prev >= key && view.tag().unwrap() != super::NodeTag::Internal {
                    return Err(format!(
                        "Keys in leaf node {:?} not sorted: {:?} >= {:?}",
                        node_id, prev, key
                    ));
                }
            }
            prev_key = Some(key.clone());
        }

        match view.tag().unwrap() {
            super::NodeTag::Leaf => {
                leaf_depths.insert(depth);
            }
            super::NodeTag::Internal => {
                if let super::Node::Internal { entries } = tree.read_node(node_id).unwrap() {
                    for (_, child_id) in entries {
                        _validate_node(tree, &child_id, depth + 1, leaf_depths, false)?;
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn point_lookup_with_string_keys() {
        let mut pager = TestPager {
            pages: FxHashMap::default(),
            next_id: 1,
            page_size: 4096,
        };
        let ids = pager.alloc_ids(3).unwrap();
        let root = ids[0];
        let left = ids[1];
        let right = ids[2];

        let apple = "apple".to_string();
        let banana = "banana".to_string();
        let grape = "grape".to_string();
        let orange = "orange".to_string();
        let pear = "pear".to_string();
        let l_entries = vec![
            (apple.clone(), b"red".to_vec()),
            (banana.clone(), b"yellow".to_vec()),
        ];
        let r_entries = vec![
            (orange.clone(), b"orange".to_vec()),
            (pear.clone(), b"green".to_vec()),
        ];
        let i_entries = vec![(grape.clone(), left), (pear.clone(), right)];

        let lbytes = super::LeafBuilder::<StringKeyCodec, BigEndianIdCodec<u64>> {
            entries: &l_entries,
            next: &Some(right),
            _p: PhantomData,
        }
        .encode();
        let rbytes = super::LeafBuilder::<StringKeyCodec, BigEndianIdCodec<u64>> {
            entries: &r_entries,
            next: &None,
            _p: PhantomData,
        }
        .encode();
        let ibytes = super::InternalBuilder::<StringKeyCodec, BigEndianIdCodec<u64>> {
            entries: &i_entries,
            _p: PhantomData,
        }
        .encode();
        pager
            .write_batch(&[(left, &lbytes), (right, &rbytes), (root, &ibytes)])
            .unwrap();
        let tree: BPlusTree<TestPager, StringKeyCodec, BigEndianIdCodec<u64>> =
            BPlusTree::new(pager, root, None);

        // Zero-copy assertions via ValueRef -> &[u8]
        assert_eq!(
            tree.get(&apple).unwrap().as_deref().map(|v| &*v),
            Some(&b"red"[..])
        );
        assert_eq!(
            tree.get(&banana).unwrap().as_deref().map(|v| &*v),
            Some(&b"yellow"[..])
        );
        assert_eq!(
            tree.get(&orange).unwrap().as_deref().map(|v| &*v),
            Some(&b"orange"[..])
        );
        assert!(tree.get(&"grape".to_string()).unwrap().is_none());
    }

    // Add inside the existing `#[cfg(test)] mod tests { ... }` block.
    // Verifies the raw on-disk byte layout for a small u64 leaf.
    // Checks header, aux(next), keys block, values block, and index table.
    //
    // Layout:
    // [0]      tag (Leaf=1)
    // [1..5)   count (u32 LE)
    // [5..9)   aux_len (u32 LE)
    // [9..]    aux payload: [u32 next_len][next_id_bytes]
    // (...)    keys block (concat of encoded keys)
    // (...)    values block (concat of values)
    // [end-16n..end) index: (k_off,k_len,v_off,v_len) * n
    #[test]
    fn layout_leaf_columnar_invariants_u64() {
        let entries = vec![
            (1u64, b"a".to_vec()),
            (2u64, b"bb".to_vec()),
            (9u64, b"zzz".to_vec()),
        ];
        let next = Some(123u64);

        let bytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &entries,
            next: &next,
            _p: core::marker::PhantomData,
        }
        .encode();
        // Header
        assert_eq!(bytes[0], super::NodeTag::Leaf as u8);
        let count = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
        assert_eq!(count, entries.len());
        let aux_len = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
        let hdr = 9usize;
        // Aux with next pointer
        let aux = &bytes[hdr..hdr + aux_len];
        let (nxt_len, pos0) = read_u32_at(aux, 0);
        assert_eq!(nxt_len as usize, 8);
        let decoded_next = u64::from_be_bytes(aux[pos0..pos0 + 8].try_into().unwrap());
        assert_eq!(decoded_next, next.unwrap());
        // Compute block boundaries using the trailing index entry.
        let index_start = bytes.len() - count * 16;
        let keys_base = hdr + aux_len;

        let last = index_start + (count - 1) * 16;
        let k_off = u32::from_le_bytes(bytes[last..last + 4].try_into().unwrap()) as usize;
        let k_len = u32::from_le_bytes(bytes[last + 4..last + 8].try_into().unwrap()) as usize;
        let keys_len = k_off + k_len;

        let values_base = keys_base + keys_len;
        let values_len = index_start - values_base;
        // Keys block must equal the concat of encoded keys.
        let mut expect_keys = Vec::new();
        for (k, _) in &entries {
            BigEndianKeyCodec::<u64>::encode_into(k, &mut expect_keys);
        }
        assert_eq!(&bytes[keys_base..keys_base + keys_len], &expect_keys[..]);
        // Values block must equal the concat of plain value bytes.
        let mut expect_vals = Vec::new();
        for (_, v) in &entries {
            expect_vals.extend_from_slice(v);
        }
        assert_eq!(
            &bytes[values_base..values_base + values_len],
            &expect_vals[..]
        );
        // Each index row must point to the correct slices.
        for (i, (k, v)) in entries.iter().enumerate() {
            let idx = index_start + i * 16;
            let ko = u32::from_le_bytes(bytes[idx..idx + 4].try_into().unwrap()) as usize;
            let kl = u32::from_le_bytes(bytes[idx + 4..idx + 8].try_into().unwrap()) as usize;
            let vo = u32::from_le_bytes(bytes[idx + 8..idx + 12].try_into().unwrap()) as usize;
            let vl = u32::from_le_bytes(bytes[idx + 12..idx + 16].try_into().unwrap()) as usize;
            let k_start = keys_base + ko;
            let v_start = values_base + vo;

            let mut k_enc = Vec::new();
            BigEndianKeyCodec::<u64>::encode_into(k, &mut k_enc);
            assert_eq!(&bytes[k_start..k_start + kl], &k_enc[..]);
            assert_eq!(&bytes[v_start..v_start + vl], &v[..]);
        }

        // Cross-check with NodeView helpers.
        let pager = TestPager {
            pages: rustc_hash::FxHashMap::default(),
            next_id: 1,
            page_size: 4096,
        };
        let page = pager.materialize_owned(&bytes).unwrap();
        let view = super::NodeView::<TestPager>::new(page).unwrap();

        assert_eq!(view.count(), entries.len());
        assert_eq!(view.leaf_fixed_key_width(), Some(8));

        let pb = view.as_slice();
        let kb = view.leaf_keys_block();
        let kb_off = kb.as_ptr() as usize - pb.as_ptr() as usize;
        assert_eq!(kb_off, keys_base);
        assert_eq!(kb.len(), keys_len);
        // Entry slices from view must match index-derived ones.
        for i in 0..entries.len() {
            let (k_enc, v_enc) = view.leaf_entry_slices(i);
            let (k, v) = &entries[i];

            let mut k2 = Vec::new();
            BigEndianKeyCodec::<u64>::encode_into(k, &mut k2);
            assert_eq!(k_enc, &k2[..]);
            assert_eq!(v_enc, &v[..]);
        }
    }

    // Verifies that fixed-width detection is disabled for variable-length
    // string keys and that keys_block still matches concat of bytes.
    #[test]
    fn layout_leaf_strings_not_fixed_width() {
        let entries = vec![
            ("a".to_string(), b"1".to_vec()),
            ("bbb".to_string(), b"22".to_vec()),
            ("cc".to_string(), b"333".to_vec()),
        ];
        let bytes = super::LeafBuilder::<StringKeyCodec, BigEndianIdCodec<u64>> {
            entries: &entries,
            next: &None,
            _p: core::marker::PhantomData,
        }
        .encode();
        // Locate regions
        let count = entries.len();
        let index_start = bytes.len() - count * 16;
        let keys_base = 9 + u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
        let last = index_start + (count - 1) * 16;
        let k_off = u32::from_le_bytes(bytes[last..last + 4].try_into().unwrap()) as usize;
        let k_len = u32::from_le_bytes(bytes[last + 4..last + 8].try_into().unwrap()) as usize;
        let keys_len = k_off + k_len;
        // Expected concatenation
        let mut expect_keys = Vec::new();
        for (k, _) in &entries {
            StringKeyCodec::encode_into(k, &mut expect_keys);
        }
        assert_eq!(&bytes[keys_base..keys_base + keys_len], &expect_keys[..]);

        // Check detection reports None for variable widths.
        let pager = TestPager {
            pages: rustc_hash::FxHashMap::default(),
            next_id: 1,
            page_size: 4096,
        };
        let page = pager.materialize_owned(&bytes).unwrap();
        let view = super::NodeView::<TestPager>::new(page).unwrap();
        assert_eq!(view.leaf_fixed_key_width(), None);
    }

    // Ensures `leaf_entry_value_range` points into the values block and
    // matches `leaf_entry_slices`'s value slice for every entry.
    #[test]
    fn layout_leaf_value_ranges_match_slices() {
        let entries = vec![
            (10u64, b"aa".to_vec()),
            (11u64, b"bbb".to_vec()),
            (12u64, b"c".to_vec()),
        ];
        let bytes = super::LeafBuilder::<BigEndianKeyCodec<u64>, BigEndianIdCodec<u64>> {
            entries: &entries,
            next: &None,
            _p: core::marker::PhantomData,
        }
        .encode();
        let pager = TestPager {
            pages: rustc_hash::FxHashMap::default(),
            next_id: 1,
            page_size: 4096,
        };
        let page = pager.materialize_owned(&bytes).unwrap();
        let view = super::NodeView::<TestPager>::new(page.clone()).unwrap();

        // Compute values block for bounds checks.
        let count = entries.len();
        let index_start = bytes.len() - count * 16;
        let keys_base = 9 + u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
        let last = index_start + (count - 1) * 16;
        let k_off = u32::from_le_bytes(bytes[last..last + 4].try_into().unwrap()) as usize;
        let k_len = u32::from_le_bytes(bytes[last + 4..last + 8].try_into().unwrap()) as usize;
        let keys_len = k_off + k_len;
        let values_base = keys_base + keys_len;
        let values_end = index_start;
        for i in 0..entries.len() {
            let r = view.leaf_entry_value_range(i);
            assert!(r.start >= values_base && r.end <= values_end);

            let (_, v) = view.leaf_entry_slices(i);
            assert_eq!(&page[r.clone()], v);
        }
    }
}
