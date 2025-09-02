// src/index.rs
//! Index trait returning streaming Sources via GATs.
//!
//! An index adapter exposes whether it supports a given filter and can
//! open a streaming Source over matching row-ids.

#![forbid(unsafe_code)]

use crate::expr::Filter;
use crate::source::Source;

/// Index that can stream candidate row-ids for a Filter.
///
/// Implementations must return a sorted, deduped Source. No buffering
/// of id lists is allowed in this layer.
pub trait Index {
    type Src<'a>: Source<'a>
    where
        Self: 'a;

    fn supports<'a>(&self, f: &Filter<'a>) -> bool;

    fn open<'a>(&'a self, f: &Filter<'a>) -> Option<Self::Src<'a>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Filter, Operator};
    use crate::source::{SliceSource, Source};

    struct EqIndex<'d> {
        // toy index over a single value -> fixed id set
        value: &'d [u8],
        ids: &'d [&'d [u8]],
    }

    impl<'d> Index for EqIndex<'d> {
        type Src<'a>
            = SliceSource<'a>
        where
            Self: 'a;

        fn supports<'a>(&self, f: &Filter<'a>) -> bool {
            matches!(f.op, Operator::Equals(_))
        }

        fn open<'a>(&'a self, f: &Filter<'a>) -> Option<Self::Src<'a>> {
            match f.op {
                Operator::Equals(v) if v == self.value => Some(SliceSource::new(self.ids)),
                _ => None,
            }
        }
    }

    #[test]
    fn eq_index_opens_stream() {
        let ids = [b"a".as_slice(), b"b".as_slice()];
        let idx = EqIndex {
            value: b"k",
            ids: &ids,
        };
        let f = Filter {
            field: 1,
            op: Operator::Equals(b"k"),
        };
        let mut src = idx.open(&f).expect("stream");
        assert_eq!(src.next(), Some(b"a".as_slice()));
        assert_eq!(src.next(), Some(b"b".as_slice()));
        assert_eq!(src.next(), None);
    }
}
