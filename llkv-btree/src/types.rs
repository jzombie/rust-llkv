use crate::{
    errors::Error,
    pager::Pager,
    views::{key_view::KeyRef, node_view::NodeView, value_view::ValueRef},
};
use std::sync::Arc;

pub type CursorStepRes<P> = Result<
    (
        <P as Pager>::Page,
        NodeView<P>,
        usize,
        Option<<P as Pager>::Id>,
    ),
    Error,
>;

pub type NodeWithNext<P> = (<P as Pager>::Page, NodeView<P>, Option<<P as Pager>::Id>);
pub type NodeWithNextRes<P> = Result<NodeWithNext<P>, Error>;

/// Zero-copy key/value pair yielded from a page.
pub type EntryRef<P> = (KeyRef<<P as Pager>::Page>, ValueRef<<P as Pager>::Page>);

/// Result wrapper for `EntryRef`.
pub type EntryRefRes<P> = Result<EntryRef<P>, Error>;

pub type FramePred = Arc<dyn Fn(&[u8], &[u8]) -> bool + Send + Sync + 'static>;

/// Bytes on the left of the split, the promoted separator key, and the
/// bytes for the new right leaf.
pub type LeafSplitParts<K> = (Vec<u8>, K, Vec<u8>);

/// Result wrapper for leaf-split output.
pub type LeafSplitRes<K> = Result<LeafSplitParts<K>, Error>;
