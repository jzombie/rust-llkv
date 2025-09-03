// llkv-table/src/index.rs

#![forbid(unsafe_code)]

use crate::expr::Filter;
use crossbeam_channel as xchan;

/// Index that can stream candidate row-ids for a Filter.
/// The stream must yield sorted, unique row-ids.
pub trait Index {
    /// Opens a stream of row IDs that match the filter.
    /// The payload in the channel is the raw row ID.
    fn open<'a>(&'a self, f: &Filter<'a>) -> Option<xchan::Receiver<Vec<u8>>>;

    /// Checks if a filter can be efficiently handled by this index.
    fn supports<'a>(&self, f: &Filter<'a>) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Filter, Operator};

    /// A dummy index for testing that uses a pre-populated channel.
    struct ChanIndex {
        value: Vec<u8>,
        ids: Vec<Vec<u8>>,
    }

    impl Index for ChanIndex {
        fn supports<'a>(&self, f: &Filter<'a>) -> bool {
            if let Operator::Equals(v) = f.op {
                v == self.value.as_slice()
            } else {
                false
            }
        }

        fn open<'a>(&'a self, f: &Filter<'a>) -> Option<xchan::Receiver<Vec<u8>>> {
            if !self.supports(f) {
                return None;
            }
            let (tx, rx) = xchan::unbounded();
            for id in self.ids.iter() {
                tx.send(id.clone()).unwrap();
            }
            Some(rx)
        }
    }

    #[test]
    fn chan_index_opens_stream() {
        let idx = ChanIndex {
            value: b"k".to_vec(),
            ids: vec![b"a".to_vec(), b"b".to_vec()],
        };
        let f = Filter {
            field: 1,
            op: Operator::Equals(b"k"),
        };
        let rx = idx.open(&f).expect("stream");
        assert_eq!(rx.recv(), Ok(b"a".to_vec()));
        assert_eq!(rx.recv(), Ok(b"b".to_vec()));
        assert!(rx.recv().is_err());
    }
}
