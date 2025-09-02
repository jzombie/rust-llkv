use crate::pager::Pager;
use rustc_hash::FxHashMap;

// ================================ LRU cache ==================================
//
// Lightweight intrusive LRU with O(1) get/insert/invalidate.
// Stores pager pages (P::Page) keyed by page Id (P::Id).
//
// Design:
// - HashMap<Id, Entry> where Entry holds page and prev/next links.
// - Doubly-linked list by Ids via head/tail and per-entry links.
// - On get: move to head, return cloned Page.
// - On insert: insert at head, evict tail if over capacity.
// - On invalidate: remove entry and fix links.
// - No eviction work on hit.
// - Page type must be cheap to clone (Arc<[u8]> in tests).

struct LruEntry<Id, Page> {
    page: Page,
    prev: Option<Id>,
    next: Option<Id>,
}

pub struct NodeCache<P>
where
    P: Pager,
    P::Page: Clone,
    P::Id: Clone + Eq + std::hash::Hash,
{
    map: FxHashMap<P::Id, LruEntry<P::Id, P::Page>>,
    head: Option<P::Id>,
    tail: Option<P::Id>,
    cap: usize,
}

impl<P> Default for NodeCache<P>
where
    P: Pager,
    P::Page: Clone,
    P::Id: Clone + Eq + std::hash::Hash,
{
    fn default() -> Self {
        NodeCache::<P>::new(1024, 2)
    }
}

impl<P> NodeCache<P>
where
    P: Pager,
    P::Page: Clone,
    P::Id: Clone + Eq + std::hash::Hash,
{
    #[inline]
    pub fn new(cap: usize, _unused: usize) -> Self {
        // _unused kept for API compatibility with prior versions.
        let cap = cap.max(1);
        Self {
            map: FxHashMap::default(),
            head: None,
            tail: None,
            cap,
        }
    }

    /// Return a clone of the cached page without mutating LRU state.
    #[inline]
    pub fn peek(&self, id: &P::Id) -> Option<P::Page> {
        self.map.get(id).map(|e| e.page.clone())
    }

    /// Promote an entry to MRU if it exists (no-op if missing).
    #[inline]
    pub fn touch(&mut self, id: &P::Id) {
        if self.map.contains_key(id) {
            self.move_to_head(id);
        }
    }

    #[inline]
    pub fn set_limits(&mut self, read_cap: usize, _sweep_factor: usize) {
        self.cap = read_cap.max(1);
        while self.map.len() > self.cap {
            self.evict_one();
        }
    }

    #[inline]
    pub fn get(&mut self, id: &P::Id) -> Option<P::Page> {
        if !self.map.contains_key(id) {
            return None;
        }
        self.move_to_head(id);
        self.map.get(id).map(|e| e.page.clone())
    }

    #[inline]
    pub fn insert(&mut self, id: P::Id, page: P::Page) {
        if self.map.contains_key(&id) {
            // Update page and move to MRU.
            if let Some(e) = self.map.get_mut(&id) {
                e.page = page;
            }
            self.move_to_head(&id);
            return;
        }
        // Insert new at head.
        let old_head = self.head.clone();
        let entry = LruEntry {
            page,
            prev: None,
            next: old_head.clone(),
        };
        self.map.insert(id.clone(), entry);
        if let Some(h) = old_head
            && let Some(e) = self.map.get_mut(&h)
        {
            e.prev = Some(id.clone());
        }
        self.head = Some(id.clone());
        if self.tail.is_none() {
            self.tail = Some(id.clone());
        }
        if self.map.len() > self.cap {
            self.evict_one();
        }
    }

    #[inline]
    pub fn invalidate(&mut self, id: &P::Id) {
        if !self.map.contains_key(id) {
            return;
        }
        let (prev, next) = {
            let e = self.map.get(id).unwrap();
            (e.prev.clone(), e.next.clone())
        };
        // Fix neighbors.
        if let Some(p) = prev.clone()
            && let Some(pe) = self.map.get_mut(&p)
        {
            pe.next = next.clone();
        }
        if let Some(n) = next.clone()
            && let Some(ne) = self.map.get_mut(&n)
        {
            ne.prev = prev.clone();
        }
        // Fix head/tail.
        if self.head.as_ref() == Some(id) {
            self.head = next;
        }
        if self.tail.as_ref() == Some(id) {
            self.tail = prev;
        }
        self.map.remove(id);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
        self.head = None;
        self.tail = None;
    }

    #[inline]
    fn move_to_head(&mut self, id: &P::Id) {
        if self.head.as_ref() == Some(id) {
            return;
        }
        let (prev, next) = {
            let e = self.map.get(id).unwrap();
            (e.prev.clone(), e.next.clone())
        };
        // Detach from current spot.
        if let Some(p) = prev.clone()
            && let Some(pe) = self.map.get_mut(&p)
        {
            pe.next = next.clone();
        }
        if let Some(n) = next.clone()
            && let Some(ne) = self.map.get_mut(&n)
        {
            ne.prev = prev.clone();
        }
        if self.tail.as_ref() == Some(id) {
            self.tail = prev.clone();
        }
        // Attach at head.
        let old_head = self.head.clone();
        if let Some(e) = self.map.get_mut(id) {
            e.prev = None;
            e.next = old_head.clone();
        }
        if let Some(h) = old_head
            && let Some(he) = self.map.get_mut(&h)
        {
            he.prev = Some(id.clone());
        }
        self.head = Some(id.clone());
        if self.tail.is_none() {
            self.tail = Some(id.clone());
        }
    }

    #[inline]
    fn evict_one(&mut self) {
        let Some(tid) = self.tail.clone() else {
            return;
        };
        self.invalidate(&tid);
    }
}
