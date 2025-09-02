//! Table orchestrator that consumes a streaming Source and delegates
//! projection to ColumnStorage. No id lists are materialized.

use crate::expr::Expr;
use crate::source::Source;
use crate::storage::ColumnStorage;
use crate::types::{FieldId, RowIdCmp};

/// Table configuration provided by the caller.
pub struct TableCfg {
    /// Comparator for opaque row-ids. This function must match the
    /// ordering of all Sources used with this Table.
    pub row_id_cmp: RowIdCmp,
}

/// Minimal table that wires a storage engine and leaves indexing to
/// the caller. It never buffers sequences of row-ids.
pub struct Table<S: ColumnStorage> {
    storage: S,
    cfg: TableCfg,
}

impl<S: ColumnStorage> Table<S> {
    pub fn new(storage: S, cfg: TableCfg) -> Self {
        Self { storage, cfg }
    }

    /// Project using a caller-provided Source. No buffering occurs.
    pub fn scan_with_source<'a>(
        &'a self,
        src: &mut dyn Source<'a>,
        projection: &[FieldId],
        on_row: &mut dyn FnMut(&'a [u8], &mut dyn FnMut(&'a [u8])),
    ) -> Result<(), ()> {
        let _ = self.cfg.row_id_cmp; // reserved for future routing
        self.storage.project_stream(src, projection, on_row)
    }

    /// Storage-only plan: use pruning when available, else full scan.
    pub fn scan_storage_only<'a>(
        &'a self,
        expr: &Expr<'a>,
        projection: &[FieldId],
        on_row: &mut dyn FnMut(&'a [u8], &mut dyn FnMut(&'a [u8])),
    ) -> Result<(), ()> {
        if let Some(mut src) = self.storage.open_pruned(expr) {
            self.storage.project_stream(&mut src, projection, on_row)
        } else {
            let mut src = self.storage.open_full_scan();
            self.storage.project_stream(&mut src, projection, on_row)
        }
    }

    /// Access the storage for adapter-specific operations.
    pub fn storage(&self) -> &S {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Filter, Operator};
    use crate::source::{SliceSource, Source};
    use crate::storage::ColumnStorage;
    use core::cmp::Ordering;
    use std::collections::BTreeMap;

    fn cmp(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    /// Small storage used only in this module to validate integration.
    struct MiniStore {
        rows: BTreeMap<Vec<u8>, BTreeMap<FieldId, Vec<u8>>>,
        ids: Vec<Vec<u8>>,
    }

    impl MiniStore {
        fn new() -> Self {
            Self {
                rows: BTreeMap::new(),
                ids: Vec::new(),
            }
        }
        fn insert(&mut self, id: &[u8], f: FieldId, v: &[u8]) {
            self.rows
                .entry(id.to_vec())
                .or_default()
                .insert(f, v.to_vec());
        }
        fn seal(&mut self) {
            self.ids = self.rows.keys().cloned().collect();
        }
    }

    impl ColumnStorage for MiniStore {
        fn project_stream<'a>(
            &'a self,
            src: &mut dyn Source<'a>,
            projection: &[FieldId],
            on_row: &mut dyn FnMut(&'a [u8], &mut dyn FnMut(&'a [u8])),
        ) -> Result<(), ()> {
            while let Some(id) = src.next() {
                let vals = self.rows.get(id).unwrap();
                let mut emit = |v: &mut dyn FnMut(&'a [u8])| {
                    for f in projection {
                        let buf = vals.get(f).map(|b| b.as_slice()).unwrap_or(&[]);
                        v(buf);
                    }
                };
                on_row(id, &mut emit);
            }
            Ok(())
        }

        type Pruned<'a> = SliceSource<'a>;
        fn open_pruned<'a>(&'a self, _expr: &Expr<'a>) -> Option<Self::Pruned<'a>> {
            None
        }

        type Full<'a> = SliceSource<'a>;
        fn open_full_scan<'a>(&'a self) -> Self::Full<'a> {
            let mut views: Vec<&[u8]> = Vec::with_capacity(self.ids.len());
            for id in &self.ids {
                views.push(id.as_slice());
            }
            let leaked = Box::leak(views.into_boxed_slice());
            SliceSource::new(leaked)
        }
    }

    #[test]
    fn table_scan_with_source_projects_rows() {
        let mut st = MiniStore::new();
        st.insert(b"id1", 1, b"v11");
        st.insert(b"id1", 2, b"v12");
        st.insert(b"id2", 1, b"v21");
        st.insert(b"id2", 2, b"v22");
        st.seal();

        let tb = Table::new(st, TableCfg { row_id_cmp: cmp });

        // Build a custom source over id2 only.
        let picks = [b"id2".as_slice()];
        let mut src = SliceSource::new(&picks);

        let mut out: Vec<(Vec<u8>, Vec<Vec<u8>>)> = Vec::new();
        let mut on_row = |rid: &[u8], on_val: &mut dyn FnMut(&[u8])| {
            let mut vals = Vec::new();
            let mut push = |v: &[u8]| vals.push(v.to_vec());
            on_val(&mut push);
            out.push((rid.to_vec(), vals));
        };
        tb.scan_with_source(&mut src, &[1, 2], &mut on_row).unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, b"id2");
        assert_eq!(out[0].1, vec![b"v21".to_vec(), b"v22".to_vec()]);
    }

    #[test]
    fn table_storage_only_uses_full_scan() {
        let mut st = MiniStore::new();
        st.insert(b"id1", 1, b"v11");
        st.insert(b"id2", 1, b"v21");
        st.seal();

        let tb = Table::new(st, TableCfg { row_id_cmp: cmp });

        // Dummy expr. Storage has no pruning, so full scan is used.
        let expr = Expr::Pred(Filter {
            field: 1,
            op: Operator::Equals(b"v11"),
        });

        let mut ids: Vec<Vec<u8>> = Vec::new();
        let mut on_row = |rid: &[u8], _on_val: &mut dyn FnMut(&[u8])| {
            ids.push(rid.to_vec());
        };
        tb.scan_storage_only(&expr, &[], &mut on_row).unwrap();
        assert_eq!(ids, vec![b"id1".to_vec(), b"id2".to_vec()]);
    }
}
