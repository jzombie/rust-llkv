//! Defines the public API traits for B-Tree implementations.

pub use crate::codecs::{IdCodec, KeyCodec};
use crate::errors::Error;
pub use crate::pager::Pager;

#[cfg(feature = "debug")]
pub mod graphviz_ext;
#[cfg(feature = "debug")]
pub use graphviz_ext::*;

/// A generic ordered map interface over a B+Tree.
pub trait BTree<'a, K> {
    /// Tree returns **zero-copy** views;
    /// must deref to `&[u8]`.
    type Value: AsRef<[u8]> + Clone + 'a;

    /// Batched point lookups. The output vector lines up with `keys` by index.
    fn get_many(&'a self, keys: &[K]) -> Result<Vec<Option<Self::Value>>, Error>;

    fn contains_key(&'a self, key: &K) -> Result<bool, Error>;
    fn insert_many(&mut self, items: &[(K, &[u8])]) -> Result<(), Error>
    where
        K: Clone;
    fn delete_many(&mut self, keys: &[K]) -> Result<(), Error>
    where
        K: Clone;
}
