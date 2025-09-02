//! Unified, zero-alloc iterator for BPlusTree.
//!
//! Single iterator type that supports forward, reverse, range, and  prefix scanning.
//!
//! It yields (KeyRef, ValueRef) with no allocations.
//!
// TODO: Clarify this. Does the pager really control this, and doesn't `NodeCache` contribute
// to the residency size, and does `SharedBPlusTree` bounded channel size contribute as well?
// Only one leaf page is resident at a time (pager controls that).
use crate::bplus_tree::{BPlusTree, Node};
use crate::codecs::{IdCodec, KeyCodec};
use crate::errors::Error;
use crate::pager::Pager;
use crate::{
    views::key_view::KeyRef,
    views::node_view::{NodeTag, NodeView},
    views::value_view::ValueRef,
};
use core::cmp::Ordering;
use core::marker::PhantomData;
use core::ops::Bound;
use core::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::Arc;

// ------------------------------------------------------------------
// Direction and scan options
// ------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Direction {
    Forward,
    Reverse,
}

/// Configuration for a key-ordered scan over a B+Tree.
///
/// A scan walks keys in logical order and yields zero-copy `(KeyRef,
/// ValueRef)` pairs.
/// The same `ScanOpts` works for both forward and
/// reverse scans, whole-tree scans, bounded range scans, and prefix
/// scans.
/// Options are combined as an intersection (all active filters
/// must match).
///
/// Fields
/// ------
/// - `dir`: Scan direction.
///   `Forward` yields ascending keys; `Reverse` yields descending keys.
/// - `lower`: Lower bound (`Unbounded`, `Included(&K)`, or `Excluded(&K)`).
///   When `Unbounded`, the scan starts from the smallest key (or, for
///   `Reverse`, the chosen start is governed by `upper`).
/// - `upper`: Upper bound (`Unbounded`, `Included(&K)`, or `Excluded(&K)`).
///   When `Unbounded`, the scan runs to the end in the selected direction.
/// - `prefix`: Optional byte prefix filter over the *encoded* key
///   bytes. Only entries whose encoded key starts with this prefix are
///   yielded. If set together with bounds, the result is the
///   intersection of the prefix filter and the range.
/// - `frame_predicate`: Optional, stateful termination predicate. When
///   provided, the iterator captures the **first yielded key's encoded
///   bytes** as the frame head and will continue yielding **while**
///   `pred(head_enc, cur_enc)` returns true. As soon as it returns
///   false, the iterator ends. This lets you express window/partition
///   boundaries without extra branching at the call site.
///
/// Semantics
/// ---------
/// - Ordering is defined by the tree's `KeyCodec::compare_encoded`.
///   Ranges (`lower`/`upper`) are applied using the same comparison,
///   so they are correct for non-lexicographic encodings as well.
/// - Bounds are expressed symmetrically using `Bound`, matching Rust’s
///   `RangeBounds` idioms. (`Included`/`Excluded`/`Unbounded`)
/// - When `lower` > `upper` (according to `KeyCodec`), the result is
///   empty.
/// - If both `prefix` and bounds are set, both must match.
///   If the `KeyCodec`'s byte encoding is lex-ordered (e.g., UTF-8 strings),
///   a prefix typically corresponds to a contiguous key range; for
///   other encodings, prefix is still applied as a byte-filter but may
///   not map to a single contiguous range.
/// - Results reflect the tree state at iterator creation time (mutations
///   after creation are not included by that iterator).
///
/// Performance notes
/// -----------------
/// - Scans advance leaf-by-leaf. With the columnar leaf layout, key
///   comparisons are branch-light and SIMD fast paths may engage for
///   fixed-width u64 keys when available. Value payloads are returned
///   as zero-copy `ValueRef` slices.
/// - Range bounds are resolved with binary search inside the starting
///   leaf; subsequent leaves run sequentially until the bound or prefix
///   fails.
/// - Prefix scans avoid decoding full keys where possible by comparing
///   the encoded key bytes against `prefix`.
///
/// Examples
/// --------
/// Full forward scan:
/// ```ignore
/// use llkv_btree::{BPlusTreeIter, ScanOpts, Direction};
/// let it = BPlusTreeIter::with_opts(&tree, ScanOpts {
///     dir: Direction::Forward,
///     lower: core::ops::Bound::Unbounded,
///     upper: core::ops::Bound::Unbounded,
///     prefix: None,
///     frame_predicate: None,
/// });
/// for (kref, vref) in it { /* use kref.as_ref(), vref.as_ref() */ }
/// ```
///
/// Forward range `[10, 20)` (exclusive upper):
/// ```ignore
/// use llkv_btree::{BPlusTreeIter, ScanOpts, Direction};
/// use core::ops::Bound::{Included, Excluded, Unbounded};
/// let lo = 10u64;
/// let up = 20u64;
/// let it = BPlusTreeIter::with_opts(&tree, ScanOpts {
///     dir: Direction::Forward,
///     lower: Included(&lo),
///     upper: Excluded(&up),
///     prefix: None,
///     frame_predicate: None,
/// });
/// ```
///
/// Reverse range `(..=100]` (inclusive upper, unbounded lower):
/// ```ignore
/// use llkv_btree::{BPlusTreeIter, ScanOpts, Direction};
/// use core::ops::Bound::{Included, Unbounded};
/// let up = 100u64;
/// let it = BPlusTreeIter::with_opts(&tree, ScanOpts {
///     dir: Direction::Reverse,
///     lower: Unbounded,
///     upper: Included(&up),
///     prefix: None,
///     frame_predicate: None,
/// });
/// ```
///
/// Prefix-only scan (encoded-byte prefix):
/// ```ignore
/// use llkv_btree::{BPlusTreeIter, ScanOpts, Direction};
/// let it = BPlusTreeIter::with_opts(&tree, ScanOpts {
///     dir: Direction::Forward,
///     lower: core::ops::Bound::Unbounded,
///     upper: core::ops::Bound::Unbounded,
///     prefix: Some(b"ap"),
///     frame_predicate: None,
/// });
/// ```
///
/// Frame/partition scan (stop when the first 8 key bytes change):
/// ```ignore
/// use llkv_btree::{BPlusTreeIter, ScanOpts, Direction};
/// let it = BPlusTreeIter::with_opts(
///     &tree,
///     ScanOpts::forward().with_frame_predicate(|head, cur| cur.get(..8) == head.get(..8)),
/// ).unwrap();
/// ```
///
/// Errors
/// ------
/// - Returns an error if the pager fails to read a page or the page
///   bytes are corrupted.
/// - Creating an iterator on an empty tree is ok; iteration yields no
///   items.
#[derive(Clone)]
pub struct ScanOpts<'a, KC>
where
    KC: KeyCodec,
{
    pub dir: Direction,
    pub lower: Bound<&'a KC::Key>,
    pub upper: Bound<&'a KC::Key>,
    pub prefix: Option<&'a [u8]>,

    /// Optional end-of-frame predicate on **encoded** keys.
    /// The iterator saves the encoded bytes of the **first yielded** key as the
    /// head; for each subsequent candidate key `cur`, it continues while
    /// `pred(head, cur)` is true. When it returns false, the iterator ends.
    /// This is `Arc<dyn ... + Send + Sync>` so you can pass it across threads
    /// from `SharedBPlusTree::start_stream_with_opts`.
    pub frame_predicate: Option<Arc<dyn Fn(&[u8], &[u8]) -> bool + Send + Sync + 'static>>,
}

impl<'a, KC> Default for ScanOpts<'a, KC>
where
    KC: KeyCodec,
{
    fn default() -> Self {
        Self {
            dir: Direction::Forward,
            lower: Unbounded,
            upper: Unbounded,
            prefix: None,
            frame_predicate: None,
        }
    }
}

impl<'a, KC> ScanOpts<'a, KC>
where
    KC: KeyCodec,
{
    #[inline]
    pub fn forward() -> Self {
        Self {
            dir: Direction::Forward,
            ..Default::default()
        }
    }

    #[inline]
    pub fn reverse() -> Self {
        Self {
            dir: Direction::Reverse,
            ..Default::default()
        }
    }

    // -------- zero-alloc builder helpers (borrow K / prefix) --------
    #[inline]
    pub fn start_at(mut self, k: &'a KC::Key) -> Self {
        self.lower = Included(k);
        self
    }
    #[inline]
    pub fn start_after(mut self, k: &'a KC::Key) -> Self {
        self.lower = Excluded(k);
        self
    }
    #[inline]
    pub fn end_at(mut self, k: &'a KC::Key) -> Self {
        self.upper = Included(k);
        self
    }
    #[inline]
    pub fn end_before(mut self, k: &'a KC::Key) -> Self {
        self.upper = Excluded(k);
        self
    }
    #[inline]
    pub fn with_prefix(mut self, p: &'a [u8]) -> Self {
        self.prefix = Some(p);
        self
    }
    #[inline]
    pub fn with_bounds(mut self, lo: Bound<&'a KC::Key>, up: Bound<&'a KC::Key>) -> Self {
        self.lower = lo;
        self.upper = up;
        self
    }

    /// Continue the scan while `pred(head, cur)` is true. Ends the scan as
    /// soon as it returns false. Predicate receives **encoded** key bytes.
    #[inline]
    pub fn with_frame_predicate<F>(mut self, pred: F) -> Self
    where
        F: Fn(&[u8], &[u8]) -> bool + Send + Sync + 'static,
    {
        self.frame_predicate = Some(Arc::new(pred));
        self
    }
}

// ------------------------------------------------------------------
// ValueResolver: read nodes/pages and build zero-copy entry views
// ------------------------------------------------------------------

pub trait ValueResolver<'a, P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    fn read_node(&self, id: &P::Id) -> Result<Node<KC::Key, P::Id>, Error>;
    fn read_one(&self, id: &P::Id) -> Result<P::Page, Error>;
    fn root_id(&self) -> P::Id;

    /// Produce (KeyRef, ValueRef) for entry `i` without allocation.
    fn entry_at(
        &self,
        page: P::Page,
        view: &NodeView<P>,
        i: usize,
    ) -> Result<(KeyRef<P::Page>, ValueRef<P::Page>), Error>;
}

// Resolver for a plain BPlusTree: value bytes live in the leaf entry.
pub struct PlainResolver<'a, P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    tree: &'a BPlusTree<P, KC, IC>,
}

impl<'a, P, KC, IC> PlainResolver<'a, P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    #[inline]
    pub fn new(tree: &'a BPlusTree<P, KC, IC>) -> Self {
        Self { tree }
    }
}

impl<'a, P, KC, IC> ValueResolver<'a, P, KC, IC> for PlainResolver<'a, P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    #[inline]
    fn read_node(&self, id: &P::Id) -> Result<Node<KC::Key, P::Id>, Error> {
        self.tree.read_node(id)
    }

    #[inline]
    fn read_one(&self, id: &P::Id) -> Result<P::Page, Error> {
        self.tree.read_one(id)
    }

    #[inline]
    fn root_id(&self) -> P::Id {
        self.tree.root_id()
    }

    #[inline]
    fn entry_at(
        &self,
        page: P::Page,
        view: &NodeView<P>,
        i: usize,
    ) -> Result<(KeyRef<P::Page>, ValueRef<P::Page>), Error> {
        let (k_enc, _v_enc) = view.leaf_entry_slices(i);
        let kref = KeyRef::from_subslice(page.clone(), k_enc);
        let vr = view.leaf_entry_value_range(i);
        let vref = ValueRef::new(page, vr);
        Ok((kref, vref))
    }
}

// ------------------------------------------------------------------
// Unified iterator
// ------------------------------------------------------------------

/// One iterator that supports forward, reverse, range, and prefix.
/// It yields zero-copy (KeyRef, ValueRef).
pub struct ScanIter<'a, R, P, KC, IC>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    resolver: R,

    // Current leaf state
    cur_page: Option<P::Page>,
    cur_view: Option<NodeView<P>>,
    cur_count: usize,

    // Forward index (0..cur_count)
    pos: usize,

    // Reverse cursor: first index after the current element
    pos_after: usize,

    // Cached next leaf id
    leaf_id: Option<P::Id>,

    // Bounds and filters
    lower: Bound<&'a KC::Key>,
    upper: Bound<&'a KC::Key>,
    prefix: Option<&'a [u8]>,

    // Frame termination predicate and captured head key bytes.
    frame_predicate: Option<Arc<dyn Fn(&[u8], &[u8]) -> bool + Send + Sync + 'static>>,
    first_key_enc: Option<Vec<u8>>,

    dir: Direction,
    _p: PhantomData<(KC, IC)>,
}

impl<'a, R, P, KC, IC> ScanIter<'a, R, P, KC, IC>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    pub fn from_parts(resolver: R, opts: ScanOpts<'a, KC>) -> Result<Self, Error> {
        {
            let root = resolver.root_id();
            if resolver.read_node(&root)?.entry_count() == 0 {
                return Ok(Self {
                    resolver,
                    cur_page: None,
                    cur_view: None,
                    cur_count: 0,
                    pos: 0,
                    pos_after: 0,
                    leaf_id: None,
                    lower: opts.lower,
                    upper: opts.upper,
                    prefix: opts.prefix,
                    frame_predicate: opts.frame_predicate,
                    first_key_enc: None,
                    dir: opts.dir,
                    _p: PhantomData,
                });
            }
        }

        match opts.dir {
            Direction::Forward => {
                let (page, view, pos, leaf_id) =
                    descend_lower_pos::<R, P, KC, IC>(&resolver, opts.lower)?;
                let count = view.count();
                Ok(Self {
                    resolver,
                    cur_page: Some(page),
                    cur_view: Some(view),
                    cur_count: count,
                    pos,
                    pos_after: 0,
                    leaf_id,
                    lower: opts.lower,
                    upper: opts.upper,
                    prefix: opts.prefix,
                    frame_predicate: opts.frame_predicate,
                    first_key_enc: None,
                    dir: opts.dir,
                    _p: PhantomData,
                })
            }
            Direction::Reverse => {
                let (page, view, pos_after, leaf_id) =
                    descend_from_upper::<R, P, KC, IC>(&resolver, opts.upper)?;
                let count = view.count();
                Ok(Self {
                    resolver,
                    cur_page: Some(page),
                    cur_view: Some(view),
                    cur_count: count,
                    pos: 0,
                    pos_after,
                    leaf_id,
                    lower: opts.lower,
                    upper: opts.upper,
                    prefix: opts.prefix,
                    frame_predicate: opts.frame_predicate,
                    first_key_enc: None,
                    dir: opts.dir,
                    _p: PhantomData,
                })
            }
        }
    }

    #[inline]
    pub fn with_resolver(resolver: R, opts: ScanOpts<'a, KC>) -> Result<Self, Error> {
        Self::from_parts(resolver, opts)
    }

    #[inline]
    fn within_upper(&self, view: &NodeView<P>, i: usize) -> bool {
        match self.upper {
            Unbounded => true,
            Included(up) => {
                let (k_enc, _) = view.leaf_entry_slices(i);
                KC::compare_encoded(k_enc, up) != Ordering::Greater
            }
            Excluded(up) => {
                let (k_enc, _) = view.leaf_entry_slices(i);
                KC::compare_encoded(k_enc, up) == Ordering::Less
            }
        }
    }

    #[inline]
    fn within_lower(&self, view: &NodeView<P>, i: usize) -> bool {
        match self.lower {
            Unbounded => true,
            Included(lo) => {
                let (k_enc, _) = view.leaf_entry_slices(i);
                KC::compare_encoded(k_enc, lo) != Ordering::Less
            }
            Excluded(lo) => {
                let (k_enc, _) = view.leaf_entry_slices(i);
                KC::compare_encoded(k_enc, lo) == Ordering::Greater
            }
        }
    }

    #[inline]
    fn within_prefix(&self, view: &NodeView<P>, i: usize) -> bool {
        if let Some(pref) = self.prefix {
            let (k_enc, _) = view.leaf_entry_slices(i);
            k_enc.starts_with(pref)
        } else {
            true
        }
    }

    fn advance_leaf_forward(&mut self) -> Result<bool, Error> {
        let mut nid = match self.leaf_id.take() {
            Some(id) => id,
            None => {
                self.cur_page = None;
                self.cur_view = None;
                self.cur_count = 0;
                self.pos = 0;
                return Ok(false);
            }
        };

        loop {
            let page = self.resolver.read_one(&nid)?;
            let view = NodeView::<P>::new(page.clone())?;
            let count = view.count();
            let next_id = decode_next_id::<P, IC>(&view);

            self.cur_page = Some(page);
            self.cur_view = Some(view);
            self.cur_count = count;
            self.pos = 0;
            self.leaf_id = next_id;

            if self.cur_count > 0 {
                return Ok(true);
            }

            nid = match self.leaf_id.take() {
                Some(id) => id,
                None => {
                    self.cur_page = None;
                    self.cur_view = None;
                    self.cur_count = 0;
                    self.pos = 0;
                    return Ok(false);
                }
            };
        }
    }

    fn step_reverse(&mut self) -> Result<bool, Error> {
        if self.pos_after > 0 {
            self.pos_after -= 1;
            return Ok(true);
        }

        let view = match self.cur_view.as_ref() {
            Some(v) => v,
            None => return Ok(false),
        };
        if view.count() == 0 {
            return Ok(false);
        }

        let (k_enc, _) = view.leaf_entry_slices(0);
        let (page, v, pos_after, next_id) =
            descend_lt_encoded::<R, P, KC, IC>(&self.resolver, k_enc)?;
        self.cur_page = Some(page);
        self.cur_view = Some(v);
        self.cur_count = self.cur_view.as_ref().unwrap().count();
        self.pos_after = pos_after;
        self.leaf_id = next_id;
        Ok(self.cur_count > 0 && self.pos_after > 0)
    }
}

impl<'a, R, P, KC, IC> Iterator for ScanIter<'a, R, P, KC, IC>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    type Item = (KeyRef<P::Page>, ValueRef<P::Page>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_page.is_none() {
            return None;
        }

        match self.dir {
            Direction::Forward => {
                loop {
                    // Hop to the next non-empty leaf when current is exhausted.
                    if self.pos >= self.cur_count {
                        if self.advance_leaf_forward().ok()? == false {
                            return None;
                        }
                    }

                    let page = self.cur_page.as_ref()?.clone();
                    let view = self.cur_view.as_ref()?;

                    // Bounds/prefix/frame checks for current candidate.
                    if !self.within_lower(view, self.pos) {
                        return None;
                    }
                    if !self.within_upper(view, self.pos) {
                        return None;
                    }
                    if !self.within_prefix(view, self.pos) {
                        self.pos += 1;
                        continue; // ITERATIVE skip (no recursion)
                    }

                    if let Some(pred) = self.frame_predicate.as_ref() {
                        let (k_enc, _) = view.leaf_entry_slices(self.pos);
                        if let Some(head) = self.first_key_enc.as_ref() {
                            if !pred(head, k_enc) {
                                return None;
                            }
                        } else {
                            // Capture first yielded key
                            self.first_key_enc = Some(k_enc.to_vec());
                        }
                    }

                    let out = self.resolver.entry_at(page, view, self.pos).ok()?;
                    self.pos += 1;
                    return Some(out);
                }
            }
            Direction::Reverse => {
                loop {
                    // Move cursor left; when current leaf exhausted, step to previous.
                    if self.pos_after == 0 {
                        if self.step_reverse().ok()? == false {
                            return None;
                        }
                    }

                    let page = self.cur_page.as_ref()?.clone();
                    let view = self.cur_view.as_ref()?;
                    let idx = self.pos_after - 1;

                    if !self.within_lower(view, idx) {
                        return None;
                    }
                    // Upper is enforced by starting position for reverse.

                    if !self.within_prefix(view, idx) {
                        // Skip this entry, continue moving left.
                        if self.pos_after > 0 {
                            self.pos_after -= 1;
                        }
                        continue; // ITERATIVE skip (no recursion)
                    }

                    if let Some(pred) = self.frame_predicate.as_ref() {
                        let (k_enc, _) = view.leaf_entry_slices(idx);
                        if let Some(head) = self.first_key_enc.as_ref() {
                            if !pred(head, k_enc) {
                                return None;
                            }
                        } else {
                            self.first_key_enc = Some(k_enc.to_vec());
                        }
                    }

                    let out = self.resolver.entry_at(page, view, idx).ok()?;
                    if self.pos_after > 0 {
                        self.pos_after -= 1;
                    }
                    return Some(out);
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Public type aliases and compat constructors
// ------------------------------------------------------------------

pub type BPlusTreeIter<'a, P, KC, IC> = ScanIter<'a, PlainResolver<'a, P, KC, IC>, P, KC, IC>;

impl<'a, P, KC, IC> BPlusTreeIter<'a, P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    #[inline]
    pub fn new(tree: &'a BPlusTree<P, KC, IC>) -> Result<Self, Error> {
        Self::with_opts(tree, ScanOpts::forward())
    }

    #[inline]
    pub fn with_opts(
        tree: &'a BPlusTree<P, KC, IC>,
        opts: ScanOpts<'a, KC>,
    ) -> Result<Self, Error> {
        let r = PlainResolver::new(tree);
        ScanIter::with_resolver(r, opts)
    }
}

// ------------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------------

#[inline]
fn decode_next_id<P, IC>(view: &NodeView<P>) -> Option<P::Id>
where
    P: Pager,
    IC: IdCodec<Id = P::Id>,
{
    let aux = view.leaf_next_aux();
    if aux.is_empty() {
        None
    } else {
        Some(IC::decode_from(aux).ok()?.0)
    }
}

fn descend_leftmost<'a, R, P, KC, IC>(
    r: &R,
) -> Result<(P::Page, NodeView<P>, usize, Option<P::Id>), Error>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    let mut id = r.root_id();
    loop {
        match r.read_node(&id)? {
            Node::Internal { entries } => {
                let (_, child) = entries.first().expect("internal without child").clone();
                id = child;
            }
            Node::Leaf { .. } => {
                let page = r.read_one(&id)?;
                let view = NodeView::<P>::new(page.clone())?;
                let next_id = decode_next_id::<P, IC>(&view);
                return Ok((page, view, 0, next_id));
            }
        }
    }
}

fn descend_rightmost<'a, R, P, KC, IC>(
    r: &R,
) -> Result<(P::Page, NodeView<P>, Option<P::Id>), Error>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    let mut id = r.root_id();
    loop {
        match r.read_node(&id)? {
            Node::Internal { entries } => {
                let (_, child) = entries.last().expect("internal without child").clone();
                id = child;
            }
            Node::Leaf { .. } => {
                let page = r.read_one(&id)?;
                let view = NodeView::<P>::new(page.clone())?;
                let next_id = decode_next_id::<P, IC>(&view);
                return Ok((page, view, next_id));
            }
        }
    }
}

fn leaf_lower_bound<P, KC>(view: &NodeView<P>, key: &KC::Key) -> usize
where
    P: Pager,
    KC: KeyCodec,
{
    let mut lo = 0isize;
    let mut hi = view.count() as isize;
    while lo < hi {
        let mid = ((lo + hi) >> 1) as usize;
        let (k_enc, _) = view.leaf_entry_slices(mid);
        match KC::compare_encoded(k_enc, key) {
            Ordering::Less => lo = mid as isize + 1,
            _ => hi = mid as isize,
        }
    }
    lo as usize
}

fn leaf_upper_pos<P, KC>(view: &NodeView<P>, key: &KC::Key, inclusive: bool) -> usize
where
    P: Pager,
    KC: KeyCodec,
{
    let mut lo = 0isize;
    let mut hi = view.count() as isize;
    while lo < hi {
        let mid = ((lo + hi) >> 1) as usize;
        let (k_enc, _) = view.leaf_entry_slices(mid);
        let ord = KC::compare_encoded(k_enc, key);
        let is_le = if inclusive {
            ord != Ordering::Greater
        } else {
            ord == Ordering::Less
        };
        if is_le {
            lo = mid as isize + 1;
        } else {
            hi = mid as isize;
        }
    }
    lo as usize
}

fn descend_ge<'a, R, P, KC, IC>(
    r: &R,
    key: &KC::Key,
) -> Result<(P::Page, NodeView<P>, usize, Option<P::Id>), Error>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    let mut id = r.root_id();
    loop {
        let node_page = r.read_one(&id)?;
        let view = NodeView::<P>::new(node_page)?;
        if let Ok(NodeTag::Leaf) = view.tag() {
            let page = r.read_one(&id)?;
            let view = NodeView::<P>::new(page.clone())?;
            let pos = leaf_lower_bound::<P, KC>(&view, key);
            let next_id = decode_next_id::<P, IC>(&view);
            return Ok((page, view, pos, next_id));
        }

        let n = view.count();
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
        let idx = lo.min(n - 1);
        let (_, child_raw) = view.internal_entry_slices(idx);
        let (child_id, _) = IC::decode_from(child_raw)?;
        id = child_id;
    }
}

fn descend_upper_pos<'a, R, P, KC, IC>(
    r: &R,
    key: &KC::Key,
    inclusive: bool,
) -> Result<(P::Page, NodeView<P>, usize, Option<P::Id>), Error>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    let mut id = r.root_id();
    loop {
        let node_page = r.read_one(&id)?;
        let view = NodeView::<P>::new(node_page)?;
        if let Ok(NodeTag::Leaf) = view.tag() {
            let page = r.read_one(&id)?;
            let view = NodeView::<P>::new(page.clone())?;
            let pos_after = leaf_upper_pos::<P, KC>(&view, key, inclusive);
            let next_id = decode_next_id::<P, IC>(&view);
            return Ok((page, view, pos_after, next_id));
        }

        let n = view.count();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (k_enc, _) = view.internal_entry_slices(mid);
            let ord = KC::compare_encoded(k_enc, key);
            let ok = if inclusive {
                ord != Ordering::Greater
            } else {
                ord == Ordering::Less
            };
            if ok {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let chosen_idx = lo.min(n - 1);
        let (_, child_raw) = view.internal_entry_slices(chosen_idx);
        let (child_id, _) = IC::decode_from(child_raw)?;
        id = child_id;
    }
}

/// Strict less-than for an encoded key slice (reverse hop).
fn descend_lt_encoded<'a, R, P, KC, IC>(
    r: &R,
    enc: &[u8],
) -> Result<(P::Page, NodeView<P>, usize, Option<P::Id>), Error>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    // Local hex printer for quick tracing.
    let _hex8 = |b: &[u8]| -> String {
        let mut s = String::new();
        for (i, x) in b.iter().take(8).enumerate() {
            if i > 0 {
                s.push(' ');
            }
            use core::fmt::Write as _;
            let _ = write!(&mut s, "{:02x}", x);
        }
        s
    };
    // Decode once; all comparisons are numeric (LE-safe).
    let key = KC::decode_from(enc)?;
    // Keep a path of (internal_node_id, chosen_idx) to support fallback.
    let mut path: Vec<(P::Id, usize)> = Vec::new();
    let mut id = r.root_id();
    loop {
        let node_page = r.read_one(&id)?;
        let view = NodeView::<P>::new(node_page)?;

        if let Ok(NodeTag::Leaf) = view.tag() {
            // We are at a candidate leaf.
            // Count keys strictly < key.
            let page = r.read_one(&id)?;
            let view = NodeView::<P>::new(page.clone())?;
            let pos_after = leaf_upper_pos::<P, KC>(&view, &key, false);
            let next_id = decode_next_id::<P, IC>(&view);

            if pos_after > 0 {
                // This leaf has keys < key. Done.
                return Ok((page, view, pos_after, next_id));
            }

            // Fallback: climb up until we can step left to a sibling,
            // then descend to that subtree's rightmost leaf.
            while let Some((anc_id, idx)) = path.pop() {
                if idx == 0 {
                    // No left sibling at this ancestor, keep climbing.
                    continue;
                }

                // Step to the left sibling child: idx - 1.
                let anc_page = r.read_one(&anc_id)?;
                let anc_view = NodeView::<P>::new(anc_page)?;
                let (_, child_raw) = anc_view.internal_entry_slices(idx - 1);
                let (mut child_id, _) = IC::decode_from(child_raw)?;
                // Descend to rightmost leaf of that subtree.
                loop {
                    let c_page = r.read_one(&child_id)?;
                    let c_view = NodeView::<P>::new(c_page.clone())?;
                    if let Ok(NodeTag::Leaf) = c_view.tag() {
                        let next = decode_next_id::<P, IC>(&c_view);
                        let cnt = c_view.count();
                        return Ok((c_page, c_view, cnt, next));
                    }
                    let last = c_view.count().saturating_sub(1);
                    let (_, cr) = c_view.internal_entry_slices(last);
                    let (next_child, _) = IC::decode_from(cr)?;
                    child_id = next_child;
                }
            }

            // No ancestor to step left at: nothing < key in whole tree.
            return Ok((page, view, 0, next_id));
        }

        // INTERNAL: lower_bound for first sep >= key, then tentatively
        // choose that child.
        // If it has no <key, we'll use the fallback.
        let n = view.count();
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (k_enc, _) = view.internal_entry_slices(mid);
            let ord = KC::compare_encoded(k_enc, &key);
            if ord == core::cmp::Ordering::Less {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let chosen_idx = lo.min(n.saturating_sub(1));
        let (_, child_raw) = view.internal_entry_slices(chosen_idx);
        let (child_id, _) = IC::decode_from(child_raw)?;
        path.push((id, chosen_idx));
        id = child_id;
    }
}

// ------------------------------------------------------------------
// New small helpers to start from generic `Bound`s
// ------------------------------------------------------------------

fn descend_lower_pos<'a, R, P, KC, IC>(
    r: &R,
    bound: Bound<&KC::Key>,
) -> Result<(P::Page, NodeView<P>, usize, Option<P::Id>), Error>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    match bound {
        Unbounded => descend_leftmost::<R, P, KC, IC>(r),
        Included(k) => descend_ge::<R, P, KC, IC>(r, k),
        Excluded(k) => {
            let (page, view, mut pos, next) =
                descend_upper_pos::<R, P, KC, IC>(r, k, /*inclusive=*/ false)?;

            // If the search landed exactly on the key we need to exclude, advance one position.
            if pos < view.count()
                && KC::compare_encoded(view.leaf_entry_slices(pos).0, k) == Ordering::Equal
            {
                pos += 1;
            }

            Ok((page, view, pos, next))
        }
    }
}

fn descend_from_upper<'a, R, P, KC, IC>(
    r: &R,
    bound: Bound<&KC::Key>,
) -> Result<(P::Page, NodeView<P>, usize, Option<P::Id>), Error>
where
    R: ValueResolver<'a, P, KC, IC>,
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    match bound {
        Unbounded => {
            let (p, v, n) = descend_rightmost::<R, P, KC, IC>(r)?;
            let count = v.count();
            Ok((p, v, count, n))
        }
        Included(k) => descend_upper_pos::<R, P, KC, IC>(r, k, /*inclusive=*/ true),
        Excluded(k) => descend_upper_pos::<R, P, KC, IC>(r, k, /*inclusive=*/ false),
    }
}
