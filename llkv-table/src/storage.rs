//! ColumnStorage trait: project selected fields for each row-id from a
//! streaming Source. No row-id lists or row buffers are materialized.

#![forbid(unsafe_code)]

use crate::source::Source;
use crate::types::FieldId;

/// Column storage consumes a row-id Source and emits projected field
/// values as borrowed buffers. Implementations may use zero-copy paging.
pub trait ColumnStorage {
    /// Project fields for each row-id from `src`. For each row, the
    /// caller receives `row_id` and a per-field callback `on_val`
    /// invoked once per field in `projection`, in order.
    fn project_stream<'a>(
        &'a self,
        src: &mut dyn Source<'a>,
        projection: &[FieldId],
        on_row: &mut dyn FnMut(&'a [u8], &mut dyn FnMut(&'a [u8])),
    ) -> Result<(), ()>;

    /// Optional pruning: return a Source over a restricted candidate
    /// set based on the expression. Default is None (no pruning).
    type Pruned<'a>: Source<'a>
    where
        Self: 'a;

    fn open_pruned<'a>(&'a self, _expr: &crate::expr::Expr<'a>) -> Option<Self::Pruned<'a>> {
        None
    }

    /// Full scan over all live row-ids.
    type Full<'a>: Source<'a>
    where
        Self: 'a;

    fn open_full_scan<'a>(&'a self) -> Self::Full<'a>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{SliceSource, Source};
    use core::cmp::Ordering;
    use std::collections::BTreeMap;

    /// Minimal in-memory storage for tests. It stores field values per
    /// row-id as owned bytes and returns borrowed slices during project.
    struct InMem {
        // row_id -> (field_id -> value)
        rows: BTreeMap<Vec<u8>, BTreeMap<FieldId, Vec<u8>>>,
        // stable list of row-id slice views for scans
        ids: Vec<Vec<u8>>,
    }

    impl InMem {
        fn new() -> Self {
            Self {
                rows: BTreeMap::new(),
                ids: Vec::new(),
            }
        }

        fn insert(&mut self, row_id: &[u8], field: FieldId, val: &[u8]) {
            let e = self.rows.entry(row_id.to_vec()).or_default();
            e.insert(field, val.to_vec());
        }

        fn seal(&mut self) {
            self.ids = self.rows.keys().cloned().collect();
        }
    }

    impl ColumnStorage for InMem {
        fn project_stream<'a>(
            &'a self,
            src: &mut dyn Source<'a>,
            projection: &[FieldId],
            on_row: &mut dyn FnMut(&'a [u8], &mut dyn FnMut(&'a [u8])),
        ) -> Result<(), ()> {
            while let Some(rid) = src.next() {
                let vals = self.rows.get(rid).unwrap();
                let mut emit = |v: &mut dyn FnMut(&'a [u8])| {
                    for f in projection {
                        let buf = vals.get(f).map(|b| b.as_slice()).unwrap_or(&[]);
                        v(buf);
                    }
                };
                on_row(rid, &mut emit);
            }
            Ok(())
        }

        type Pruned<'a> = SliceSource<'a>;

        fn open_pruned<'a>(&'a self, _expr: &crate::expr::Expr<'a>) -> Option<Self::Pruned<'a>> {
            None
        }

        type Full<'a> = SliceSource<'a>;

        fn open_full_scan<'a>(&'a self) -> Self::Full<'a> {
            // Build a slice of slice-views over self.ids.
            let mut views: Vec<&[u8]> = Vec::with_capacity(self.ids.len());
            for id in &self.ids {
                views.push(id.as_slice());
            }
            // We need a stable container for the slice-of-slices.
            // For test simplicity, leak it. Production code should
            // return a real pager-backed Source.
            let leaked = Box::leak(views.into_boxed_slice());
            SliceSource::new(leaked)
        }
    }

    #[test]
    fn project_stream_yields_projection_order() {
        let mut s = InMem::new();
        s.insert(b"id1", 1, b"v11");
        s.insert(b"id1", 2, b"v12");
        s.insert(b"id2", 1, b"v21");
        s.insert(b"id2", 2, b"v22");
        s.seal();

        let mut src = s.open_full_scan();
        let mut rows = Vec::<(Vec<u8>, Vec<Vec<u8>>)>::new();
        let mut on_row = |rid: &[u8], on_val: &mut dyn FnMut(&[u8])| {
            let mut vs: Vec<Vec<u8>> = Vec::new();
            let mut push_val = |v: &[u8]| vs.push(v.to_vec());
            on_val(&mut push_val);
            rows.push((rid.to_vec(), vs));
        };

        let proj = [2, 1];
        s.project_stream(&mut src, &proj, &mut on_row).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, b"id1");
        assert_eq!(rows[0].1, vec![b"v12".to_vec(), b"v11".to_vec()]);
        assert_eq!(rows[1].0, b"id2");
        assert_eq!(rows[1].1, vec![b"v22".to_vec(), b"v21".to_vec()]);
    }
}
