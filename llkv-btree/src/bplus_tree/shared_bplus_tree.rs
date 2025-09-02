use std::sync::{Mutex as StdMutex, RwLock};

use crossbeam_channel as xchan;

use crate::bplus_tree::BPlusTree;
use crate::codecs::{IdCodec, KeyCodec};
use crate::errors::Error;
use crate::iter::{BPlusTreeIter, ScanOpts};
use crate::node_cache::NodeCache;
use crate::pager::Pager;
use crate::traits::BTree;
use std::sync::Arc;

/// A façade that provides a serialized writer and snapshot readers,
/// while implementing the same `BTree` trait.
pub struct SharedBPlusTree<P, KC, IC>
where
    P: Pager + Clone,
    P::Page: Send + Sync + 'static,
    P::Id: Send + Sync + 'static,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    pager: P,
    writer: StdMutex<BPlusTree<P, KC, IC>>, // single-threaded writer
    latest_root: RwLock<P::Id>,             // published root for readers
    shared_node_cache: Arc<RwLock<NodeCache<P>>>,
}

impl<P, KC, IC> SharedBPlusTree<P, KC, IC>
where
    P: Pager + Clone,
    P::Page: Send + Sync + 'static,
    P::Id: Send + Sync + 'static,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    /// Create an empty tree + shared writer.
    pub fn create_empty(
        pager: P,
        opt_shared_node_cache: Option<Arc<RwLock<NodeCache<P>>>>,
    ) -> Result<Self, Error> {
        let shared_node_cache = BPlusTree::<P, KC, IC>::init_node_cache(opt_shared_node_cache);

        let writer = BPlusTree::<P, KC, IC>::create_empty(
            pager.clone(),
            Some(Arc::clone(&shared_node_cache)),
        )?;
        let root = writer.root_id();
        Ok(Self {
            pager,
            writer: StdMutex::new(writer),
            latest_root: RwLock::new(root),
            shared_node_cache,
        })
    }

    /// Open a shared B+Tree backed by an existing root.
    pub fn open(
        pager: P,
        root: P::Id,
        opt_shared_node_cache: Option<Arc<RwLock<NodeCache<P>>>>,
    ) -> Self {
        let shared_node_cache = BPlusTree::<P, KC, IC>::init_node_cache(opt_shared_node_cache);

        let writer = BPlusTree::<P, KC, IC>::new(
            pager.clone(),
            root.clone(),
            Some(Arc::clone(&shared_node_cache)),
        );
        Self {
            pager,
            writer: StdMutex::new(writer),
            latest_root: RwLock::new(root),
            shared_node_cache,
        }
    }

    /// Produce a **snapshot** tree at the current published root.
    /// This is a normal `BPlusTree` you can iterate with `ScanOpts`.
    pub fn snapshot(&self) -> BPlusTree<P, KC, IC> {
        let root = { self.latest_root.read().unwrap().clone() };
        let shared_node_cache = Arc::clone(&self.shared_node_cache);
        let page_size = { self.writer.lock().unwrap().page_size() };

        BPlusTree::<P, KC, IC>::with_readonly_pager_state(
            self.pager.clone(),
            root,
            shared_node_cache,
            page_size,
        )
    }

    /// Start a streaming scan at the current root, using any `ScanOpts`.
    /// Returns a crossbeam MPMC receiver; read it from as many threads as you like.
    /// Keys/values are owned (no borrows across threads).
    pub fn start_stream_with_opts(
        &self,
        opts: ScanOpts<'static, KC>,
    ) -> xchan::Receiver<(KC::Key, Vec<u8>)>
    where
        // required because we move these into a rayon worker thread
        P: Send + 'static,
        P::Id: Send + 'static,
        KC: KeyCodec + 'static,
        KC::Key: Send + Sync + 'static,
        IC: IdCodec<Id = P::Id> + 'static,
    {
        let root = { self.latest_root.read().unwrap().clone() };
        let pager = self.pager.clone();

        // TODO: Make the capacity configurable
        let (tx, rx) = xchan::bounded(1);

        let shared_node_cache = Arc::clone(&self.shared_node_cache);
        let page_size = { self.writer.lock().unwrap().page_size() };

        rayon::spawn(move || {
            let tree = BPlusTree::<P, KC, IC>::with_readonly_pager_state(
                pager,
                root,
                shared_node_cache,
                page_size,
            );
            if let Ok(mut it) = BPlusTreeIter::with_opts(&tree, opts) {
                while let Some((kref, vref)) = it.next() {
                    let key = KC::decode_from(kref.as_ref()).expect("decode key");
                    let val = vref.as_ref().to_vec();
                    if tx.send((key, val)).is_err() {
                        break;
                    }
                    // std::thread::yield_now(); // encourages interleaving in demos
                }
            }
            // drop(tx) ends the stream
        });

        rx
    }

    // Optionally expose a publish step (useful if you batch).
    // pub fn flush(&self) -> Result<(), Error> {
    //     let mut w = self.writer.lock().unwrap();
    //     w.flush()?;
    //     *self.latest_root.write().unwrap() = w.root_id();
    //     Ok(())
    // }
}

// --- Implement the public BTree trait on SharedBPlusTree ---

impl<'a, P, KC, IC> BTree<'a, KC::Key> for SharedBPlusTree<P, KC, IC>
where
    P: Pager + Clone + 'a,
    P::Page: Send + Sync + 'static,
    P::Id: Send + Sync + 'static,
    KC: KeyCodec + 'a,
    IC: IdCodec<Id = P::Id> + 'a,
{
    type Value = <BPlusTree<P, KC, IC> as BTree<'a, KC::Key>>::Value;

    fn get_many(&'a self, keys: &[KC::Key]) -> Result<Vec<Option<Self::Value>>, Error> {
        self.snapshot().get_many(keys)
    }

    fn contains_key(&'a self, key: &KC::Key) -> Result<bool, Error> {
        self.snapshot().contains_key(key)
    }

    fn insert_many(&mut self, items: &[(KC::Key, &[u8])]) -> Result<(), Error>
    where
        KC::Key: Clone,
    {
        let mut w = self.writer.lock().unwrap();
        w.insert_many(items)?;
        w.flush()?;
        *self.latest_root.write().unwrap() = w.root_id();
        Ok(())
    }

    fn delete_many(&mut self, keys: &[KC::Key]) -> Result<(), Error>
    where
        KC::Key: Clone,
    {
        let mut w = self.writer.lock().unwrap();
        w.delete_many(keys)?;
        w.flush()?;
        *self.latest_root.write().unwrap() = w.root_id();
        Ok(())
    }
}
